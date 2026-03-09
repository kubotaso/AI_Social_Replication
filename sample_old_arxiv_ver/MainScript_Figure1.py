"""
Automatic replication of Figure 1 from Bryson (1996) using LLM-driven iterative code generation.

What this does:
1) Summarize the paper PDF to extract Figure 1 construction details (variables, methods, plotting specifications).
2) Map codebook/variable documentation to dataset columns for variable alignment.
3) Generate Python analysis code that recreates Figure 1 and saves it as a JPEG file.
4) Execute the generated code to produce a JPEG figure file.
5) Compare the generated JPEG against the ground-truth Figure 1 JPEG using vision-enabled LLM (0-100 scoring).
6) Track the best attempt across iterations and feed best code context to retries.
7) Iterate with visual discrepancy feedback and runtime error corrections until figures match or max attempts reached.

The script uses base64-encoded data URLs to transfer JPEG images to the OpenAI Responses API for
visual comparison, avoiding file upload complications and ensuring robust image analysis.

Requirements:
- OPENAI_API_KEY must be set in the environment.
- Packages: openai>=1.51 (Responses API with vision), pandas, numpy, matplotlib, statsmodels.

Tested on Python 3.11+; should work on Python 3.13 as per your environment.
"""

from __future__ import annotations

import argparse
import sys
import re
import shutil
import time
import importlib.util
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, cast
import base64

from pathlib import Path

import pandas as pd
from openai import OpenAI, RateLimitError

# OpenAI client (reads OPENAI_API_KEY from env)
client = OpenAI()

# ──────────────────────────────────────────────────
#   Constants
# ──────────────────────────────────────────────────

ALIGNMENT_SCORE_THRESHOLD = 95  # Skip detailed check if score >= threshold (early exit)
MAX_OUTPUT_TOKENS = 20000  # Maximum output tokens for LLM responses


# ──────────────────────────────────────────────────
#   State Management
# ──────────────────────────────────────────────────


@dataclass
class RunState:
    """Track state across retry attempts."""
    # Best attempt tracking
    best_score: int = -1
    best_attempt: int = 0
    best_code_path: Optional[Path] = None
    best_img_path: Optional[Path] = None
    best_discrepancy: Optional[str] = None

    # Previous attempt tracking
    prev_attempt: int = 0
    prev_code_path: Optional[Path] = None
    prev_img_path: Optional[Path] = None
    prev_discrepancy: Optional[str] = None
    prev_error: Optional[str] = None

    # Current attempt
    last_score: int = -1
    current_code_path: Optional[Path] = None


# ──────────────────────────────────────────────────
#   Helper Utilities For Local Use
# ──────────────────────────────────────────────────


def save_artifact(output_dir: Path, filename: str, content: str) -> Path:
    """Write content to a file in output_dir and return the path."""
    path = output_dir / filename
    path.write_text(content, encoding="utf-8")
    return path


class _TeeStream:
    def __init__(self, primary, secondary) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, data) -> int:
        n1 = self._primary.write(data)
        self._secondary.write(data)
        self._secondary.flush()
        return n1

    def flush(self) -> None:
        try:
            self._primary.flush()
        finally:
            self._secondary.flush()


@contextmanager
def tee_terminal_output(log_path: Union[str, Path], *, mode: str = "a"):
    """Mirror all stdout/stderr to a log file while still printing to the terminal."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with log_path.open(mode, encoding="utf-8") as log_file:
        sys.stdout = _TeeStream(original_stdout, log_file)  # type: ignore[assignment]
        sys.stderr = _TeeStream(original_stderr, log_file)  # type: ignore[assignment]
        try:
            yield
        finally:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr


def load_data(data_path: Union[str, Path]) -> pd.DataFrame:
    """Load CSV with pragmatic defaults."""
    return pd.read_csv(
        data_path,
        sep=",",
        engine="python",
        on_bad_lines="skip",
        skip_blank_lines=True,
    )


def upload_pdf(path: Union[str, Path]) -> str:
    """Upload a PDF to the API and return its file_id."""
    p = Path(path)
    with open(p, "rb") as f:
        file_obj = client.files.create(file=f, purpose="user_data")
    return file_obj.id


def _image_to_data_url(path: Union[str, Path]) -> str:
    """Read an image and return a data URL suitable for the Responses API vision input."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif suffix == ".png":
        mime = "image/png"
    else:
        mime = "image/jpeg"
    data = p.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _llm_call(
    prompt: Optional[str] = None,
    *,
    model: str,
    instructions: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    max_retries: int = 8,
    backoff_cap: float = 10.0,
    content: Optional[List[dict]] = None,
) -> str:
    """
    Thin wrapper around the OpenAI Responses API with rate-limit backoff.
    Supports either a plain text prompt (prompt=...) or a rich content payload
    (content=[{"type": "input_text"|"input_file", ...}]).
    """
    if not prompt and not content:
        raise ValueError("_llm_call requires either `prompt` or `content`.")
    for attempt in range(1, max_retries + 1):
        try:
            if content is not None:
                resp = client.responses.create(
                    model=model,
                    instructions=instructions,
                    input=[{"role": "user", "content": content}],  # type: ignore[arg-type]
                    max_output_tokens=max_output_tokens,
                )
            else:
                resp = client.responses.create(
                    model=model,
                    instructions=instructions,
                    input=cast(str, prompt),
                    max_output_tokens=max_output_tokens,
                )
            return (resp.output_text or "").strip()
        except RateLimitError as e:
            m = re.search(r"try again in ([\d\.]+)s", str(e))
            if m:
                wait = float(m.group(1)) + 0.5
            else:
                wait = min(2 ** attempt, backoff_cap)
            print(f"[RateLimit] attempt {attempt}/{max_retries}; sleeping {wait:.1f}s")
            time.sleep(wait)
        except Exception:
            wait = min(2 ** attempt, backoff_cap)
            print(f"[LLM Error] attempt {attempt}/{max_retries}; sleeping {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError("LLM call: exceeded max retries")


def _normalize_result_to_text(result) -> str:
    """Convert various return types to a string (path text preferred)."""
    if result is None:
        return ""
    if isinstance(result, (str, Path)):
        return str(result)
    if isinstance(result, pd.DataFrame):
        return result.to_string(index=False)
    if isinstance(result, dict):
        parts = []
        for k, v in result.items():
            parts.append(f"## {k}\n{v}" if not isinstance(v, pd.DataFrame) else f"## {k}\n{v.to_string(index=False)}")
        return "\n\n".join(parts)
    if isinstance(result, list):
        return "\n\n".join(
            (x.to_string(index=False) if isinstance(x, pd.DataFrame) else str(x))
            for x in result
        )
    return str(result)


def try_run_generated_code(
    module_path: Union[str, Path],
    data_path: Union[str, Path],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Safe wrapper: returns (result_text_or_path, error_text). Expected return for this project is
    a *path string* to the saved JPEG produced by run_analysis(...).
    """
    try:
        module_path = Path(module_path)
        spec = importlib.util.spec_from_file_location("gen_mod", module_path)
        if spec is None or spec.loader is None:
            return None, f"Cannot import module from {module_path}"
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[assignment]

        if not hasattr(mod, "run_analysis"):
            return None, "Generated module has no run_analysis(...)"
        result = mod.run_analysis(str(data_path))  # type: ignore[attr-defined]
        return _normalize_result_to_text(result), None
    except Exception as e:
        tb = traceback.format_exc()
        return None, f"[RuntimeError] {e}\n{tb}"


def save_best_result(
    output_dir: Path,
    attempt: int,
    code_path: Path,
    figure_path: Path,
    score: int,
) -> None:
    """
    Save the current attempt's results as the best result.
    Creates copies with 'best_' prefix for easy identification.
    """
    shutil.copy(code_path, output_dir / "best_generated_analysis.py")
    shutil.copy(figure_path, output_dir / "best_generated_figure.jpg")
    save_artifact(output_dir, "best_result_metadata.txt",
        f"Best result from attempt: {attempt}\n"
        f"Score: {score}/100\n"
        f"Original code file: {code_path.name}\n"
        f"Original figure file: {figure_path.name}\n"
    )
    print(f"[Best] Updated best result (attempt {attempt}, score {score})")


# ──────────────────────────────────────────────────
#   Helper Utilities for LLM tasks
# ──────────────────────────────────────────────────


def summarize_and_map_pdf(
    pdf_path: Union[str, Path],
    instruction_path: Union[str, Path],
    *,
    model: str,
) -> Tuple[str, str]:
    """
    Summarize the paper to extract Figure 1 details and map to codebook.
    Returns (fig1_summary, instruction_mapping).
    """
    file_id = upload_pdf(pdf_path)

    # Summarize Figure 1
    summary_instructions = (
        "You are a concise, careful research assistant with strong quantitative skills. "
        "Extract ONLY what is needed to reproduce Figure 1 from the article."
    )
    summary_content = [
        {"type": "input_text", "text": (
            "Task: From the attached PDF, find the exact construction steps for **Figure 1**. "
            "Summarize related data sources, variables, and statistical methods."
            "Also summarize plotting details such as axes, transformations, legends, annotations, line/marker styles. "
            "Do NOT summarize tables. Focus on Figure 1 only."
        )},
        {"type": "input_file", "file_id": file_id},
    ]
    fig1_summary = _llm_call(model=model, instructions=summary_instructions, content=summary_content, max_output_tokens=MAX_OUTPUT_TOKENS)

    # Map to codebook
    inst_path = Path(instruction_path)
    inst_content = ""
    if inst_path.exists():
        try:
            with open(inst_path, "r", encoding="utf-8") as f:
                inst_content = f.read()
        except Exception as e:
            inst_content = f"(Could not read codebook: {e})"
    else:
        inst_content = "(No codebook provided.)"

    map_instructions = "You are a careful replicator who writes explicit, actionable mappings."
    map_prompt = (
        "We will reproduce **Figure 1** from the article. Using the analysis summary below and the "
        "variable documentation (if any), list the dataset variable names to use for Figure 1.\n"
        "Be concrete; name exact columns if possible. If uncertain, state robust fallbacks.\n\n"
        f"[Figure‑1 Analysis Summary]\n{fig1_summary}\n\n"
        f"[Variable Documentation]\n{inst_content}\n"
    )
    instruction_mapping = _llm_call(model=model, instructions=map_instructions, prompt=map_prompt, max_output_tokens=MAX_OUTPUT_TOKENS)

    return fig1_summary, instruction_mapping


def ask_code_with_data(
    analysis_summary: str,
    df: pd.DataFrame,
    data_path: Union[str, Path],
    data_columns: List[str],
    instruction_summary: str,
    output_dir: Path,
    *,
    model: str,
    is_retry: bool = False,
    # Best attempt info
    best_code_text: Optional[str] = None,
    best_discrepancy: Optional[str] = None,
    best_score: Optional[int] = None,
    # Previous attempt info
    prev_code_text: Optional[str] = None,
    prev_discrepancy: Optional[str] = None,
    prev_error: Optional[str] = None,
    sample_rows: int = 5,
    jpeg_basename: str = "generated_results.jpg",
) -> str:
    """
    Ask the LLM to write the Python analysis code that *produces the Figure 1 replica*.
    For retries, provides best attempt and previous attempt context.
    """
    sample_csv = df[data_columns[: min(len(data_columns), 18)]].head(sample_rows).to_csv(index=False)
    cols_str = ", ".join(data_columns[:80])
    out_dir_str = str(output_dir.resolve())

    if is_retry:
        instructions = "You fix issues precisely and write simple, correct code."
    else:
        instructions = "Return code only (no backticks, no commentary)."

    # Build the prompt
    prompt_parts = []

    if is_retry:
        prompt_parts.append("We previously ran your analysis and found discrepancies or runtime errors. Use the feedback below to correct your code.\n")

    prompt_parts.append(f"[Figure‑1 Analysis Summary]\n{analysis_summary}\n")
    prompt_parts.append(f"[Instruction Mapping]\n{instruction_summary}\n")
    prompt_parts.append(f"[Sample data ({sample_rows} rows)]\n{sample_csv}")
    prompt_parts.append(f"[All available columns]\n{cols_str}\n")
    prompt_parts.append(f"[Path to full dataset]\n{data_path}\n")

    # Add best attempt context (if exists and is_retry)
    if is_retry and best_code_text and best_score is not None:
        prompt_parts.append(
            f"[Best Attempt So Far (Score: {best_score}/100)]\n"
            f"Try to improve upon or maintain this quality.\n"
        )
        prompt_parts.append(f"[Best Code]\n{best_code_text}\n")
        if best_discrepancy:
            prompt_parts.append(f"[Best Attempt Discrepancy Report]\n{best_discrepancy}\n")

    # Add previous attempt context (if different from best)
    if is_retry and prev_code_text:
        # Check if prev is different from best (avoid duplicates)
        is_same_as_best = (best_code_text and prev_code_text == best_code_text)
        if not is_same_as_best:
            prompt_parts.append("[Previous Attempt (most recent)]\n")
            prompt_parts.append(f"[Previous Code]\n{prev_code_text}\n")
            if prev_error:
                prompt_parts.append(f"[Previous Runtime Error — MUST FIX]\n{prev_error}\n")
            elif prev_discrepancy:
                prompt_parts.append(f"[Previous Discrepancy Report — MUST FIX]\n{prev_discrepancy}\n")

    prompt_parts.append(
        "Generate a single Python function with the exact signature:\n"
        "    def run_analysis(data_source):\n"
        "- Import all required packages.\n"
        "- Read the CSV at data_source with pandas.\n"
        "- **CRITICAL**: Drop or impute NaN/missing values before fitting any statistical models.\n"
        "- Recreate Figure 1 using matplotlib (no seaborn).\n"
        "- Be careful about title and axis comparable to the article Figure 1.\n"
        f"- Save as a JPEG file (300 DPI) to '{out_dir_str}/{jpeg_basename}' and return its absolute path.\n"
        "- If unsure of a precise column, search df.columns for a plausible match (case-insensitive).\n"
        "- Do not hard-code numbers from the paper; compute from the data.\n"
        "- Output code only (no backticks, no prose).\n"
        "- The code must be self-contained (imports inside) and runnable as-is."
    )

    prompt = "\n".join(prompt_parts)
    raw_code = _llm_call(model=model, instructions=instructions, prompt=prompt, max_output_tokens=MAX_OUTPUT_TOKENS)

    # Strip any markdown fences
    lines = raw_code.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def report_discrepancy_images(true_img_path: Union[str, Path], gen_img_path: Union[str, Path], *, model: str) -> str:
    """Ask the model to describe differences and actionable fixes to the code."""
    url_true = _image_to_data_url(true_img_path)
    url_gen = _image_to_data_url(gen_img_path)
    instructions = "Write a concise, actionable discrepancy report with specific code fixes."
    content = [
        {"type": "input_text", "text": (
            "Compare the ground-truth Figure 1 to the generated figure. "
            "Identify EXACT discrepancies (axis ranges, tick labels, scales, series included/excluded, legend order, colors/linestyles, markers, annotations, smoothing/aggregation, binning, missing transformations).\n"
            "Then provide step-by-step suggestions (as bullet points) for how to modify the data processing and matplotlib calls to fix them so the replication will match exactly."
        )},
        {"type": "input_image", "image_url": url_true},
        {"type": "input_image", "image_url": url_gen},
    ]
    return _llm_call(model=model, instructions=instructions, content=content, max_output_tokens=MAX_OUTPUT_TOKENS)


def score_alignment_images(
    true_img_path: Union[str, Path],
    gen_img_path: Union[str, Path],
    *,
    model: str,
) -> int:
    """Return a 0-100 alignment score comparing the two figure images."""
    instructions = "Return only an integer from 0 to 100. No other text."
    url_true = _image_to_data_url(true_img_path)
    url_gen = _image_to_data_url(gen_img_path)

    content = [
        {
            "type": "input_text",
            "text": (
                "Score how closely the GENERATED figure matches the TRUE figure (0-100).\n\n"
                "Scoring rubric (sum to 100):\n"
                "- X-axis categories/labels and their order match: 20\n"
                "- Y-axis labels, ranges, scales, and ticks match: 30\n"
                "- Data series shapes, trends, and values visually match: 30\n"
                "- Reference lines, annotations, and legend match: 10\n"
                "- Overall layout and formatting is reasonably similar: 10\n\n"
                "Return ONLY the integer score."
            ),
        },
        {"type": "input_image", "image_url": url_true},
        {"type": "input_image", "image_url": url_gen},
    ]

    raw = _llm_call(model=model, instructions=instructions, content=content, max_output_tokens=16).strip()
    try:
        return int(raw)
    except ValueError:
        m = re.search(r"\d+", raw)
        return int(m.group(0)) if m else 0


# ──────────────────────────────────────────────────
#   Main pipeline
# ──────────────────────────────────────────────────


def _materialize_attempt_image(output_dir: Path, result_text: Optional[str], attempt_index: int) -> Optional[Path]:
    """Helper to find the generated image."""
    if not result_text:
        return None
    p = Path(result_text.strip().splitlines()[-1]).expanduser().resolve()
    if not p.exists():
        fallback = output_dir / "generated_results.jpg"
        if fallback.exists():
            p = fallback
        else:
            cands = sorted(output_dir.glob("*.jpg"), key=lambda x: x.stat().st_mtime, reverse=True)
            if cands:
                p = cands[0]
            else:
                return None
    dest = output_dir / f"generated_results_attempt_{attempt_index}.jpg"
    if str(p.resolve()) != str(dest.resolve()):
        dest.write_bytes(p.read_bytes())
    return dest


def run_single_attempt(
    attempt: int,
    state: RunState,
    output_dir: Path,
    fig1_summary: str,
    inst_summary: str,
    df: pd.DataFrame,
    data_path: Path,
    cols: List[str],
    true_fig_path: Path,
    model: str,
    max_attempts: int,
) -> str:
    """
    Execute a single attempt: generate/revise code, run it, score results.
    Returns "1" if aligned, "0" otherwise.
    """
    iter_start = time.time()
    is_retry = attempt > 1

    if is_retry:
        print(f"\n{'='*70}")
        print(f"[Retry Loop] Attempt {attempt}/{max_attempts}")
        print(f"{'='*70}")

        # Generate discrepancy report for previous attempt (if it had an image)
        if state.prev_img_path and state.prev_img_path.exists():
            state.prev_discrepancy = report_discrepancy_images(true_fig_path, state.prev_img_path, model=model)
            score_header = f"Score: {state.last_score}/100\n{'='*60}\n\n"
            save_artifact(output_dir, f"discrepancy_report_attempt_{attempt-1}.txt", score_header + state.prev_discrepancy)
            print(f"[Mismatch] Saved discrepancy_report_attempt_{attempt-1}.txt")
        else:
            state.prev_discrepancy = None
            print("[Info] Skipping discrepancy report due to previous runtime error or missing image.")

        # Read best code if available
        best_code_text = None
        best_code_file = output_dir / "best_generated_analysis.py"
        if best_code_file.exists():
            try:
                best_code_text = best_code_file.read_text(encoding="utf-8")
            except Exception:
                pass

        # Read previous code if available
        prev_code_text = None
        if state.prev_code_path and state.prev_code_path.exists():
            try:
                prev_code_text = state.prev_code_path.read_text(encoding="utf-8")
            except Exception:
                pass

        # Generate revised code with best + prev context
        revised = ask_code_with_data(
            fig1_summary, df, data_path, cols, inst_summary,
            output_dir, model=model,
            is_retry=True,
            best_code_text=best_code_text,
            best_discrepancy=state.best_discrepancy,
            best_score=state.best_score if state.best_score >= 0 else None,
            prev_code_text=prev_code_text,
            prev_discrepancy=state.prev_discrepancy,
            prev_error=state.prev_error,
            jpeg_basename="generated_results.jpg",
        )
        code_path = output_dir / f"generated_analysis_retry_{attempt}.py"
        save_artifact(output_dir, f"generated_analysis_retry_{attempt}.py", revised)
        print(f"[Retry {attempt}] Saved generated_analysis_retry_{attempt}.py")
    else:
        # Initial attempt
        print(f"\n[Attempt {attempt}] Generating initial code...")
        init_code = ask_code_with_data(
            fig1_summary, df, data_path, cols, inst_summary,
            output_dir, model=model, jpeg_basename="generated_results.jpg"
        )
        code_path = output_dir / "generated_analysis.py"
        save_artifact(output_dir, "generated_analysis.py", init_code)
        print(f"[OK] Saved generated_analysis.py")

    state.current_code_path = code_path

    # Run the generated code
    print(f"[Attempt {attempt}] Executing generated code...")
    gen_text, runtime_err = try_run_generated_code(code_path, data_path)

    if runtime_err:
        save_artifact(output_dir, f"runtime_error_attempt_{attempt}.txt", runtime_err)
        print(f"[Attempt {attempt}] Runtime error - saved to runtime_error_attempt_{attempt}.txt")
        # Store as previous attempt info for next iteration
        state.prev_attempt = attempt
        state.prev_code_path = code_path
        state.prev_img_path = None
        state.prev_error = runtime_err
        success = "0"
    else:
        attempt_img = _materialize_attempt_image(output_dir, gen_text, attempt)

        if attempt_img and attempt_img.exists():
            print(f"[Attempt {attempt}] Generated figure: {attempt_img.name}")
            state.last_score = score_alignment_images(true_fig_path, attempt_img, model=model)
            is_aligned = state.last_score >= ALIGNMENT_SCORE_THRESHOLD
            success = "1" if is_aligned else "0"
            print(f"[Score {attempt}] {state.last_score}/100, Aligned: {is_aligned}")

            # Store as previous attempt info for next iteration
            state.prev_attempt = attempt
            state.prev_code_path = code_path
            state.prev_img_path = attempt_img
            state.prev_error = None

            # Update best if needed
            if state.last_score > state.best_score:
                # Generate discrepancy for the new best (will be used in future retries)
                state.best_discrepancy = report_discrepancy_images(true_fig_path, attempt_img, model=model)
                state.best_score = state.last_score
                state.best_attempt = attempt
                state.best_code_path = code_path
                state.best_img_path = attempt_img
                save_best_result(output_dir, attempt, code_path, attempt_img, state.last_score)
        else:
            print(f"[Attempt {attempt}] No figure generated")
            # Store as previous attempt info for next iteration
            state.prev_attempt = attempt
            state.prev_code_path = code_path
            state.prev_img_path = None
            state.prev_error = "No figure was generated by the code."
            success = "0"

    print(f"[Check {attempt}/{max_attempts}] Success: {success}")
    iter_elapsed = time.time() - iter_start
    print(f"[Duration] attempt {attempt}: {iter_elapsed:.1f}s")

    return success


def main(
    pdf_path: Union[str, Path],
    true_figure_path: Union[str, Path],
    data_path: Union[str, Path],
    instruction_path: Union[str, Path],
    *,
    output_dir: Union[str, Path],
    model: str,
    max_attempts: int,
) -> str:
    """
    End‑to‑end loop to replicate Figure 1.
    Files written:
    - figure_summary.txt                  (Figure‑1 extraction)
      - instruction_summary.txt             (variable mapping)
      - generated_analysis.py               (initial code)
      - generated_analysis_retry_{i}.py     (retries)
      - generated_results_attempt_{i}.jpg   (generated figure per attempt)
      - discrepancy_report_attempt_{i}.txt  (model-written difference report)
      - runtime_error_attempt_{i}.txt       (traceback if any)
      - best_generated_analysis.py          (best code so far)
      - best_generated_figure.jpg           (best figure so far)
      - best_result_metadata.txt            (metadata about the best result)
    Returns '1' on success, '0' otherwise.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(data_path)
    true_fig_path = Path(true_figure_path).resolve()

    state = RunState()

    print(f"\n{'='*70}")
    print(f"[Config] Model: {model}")
    print(f"[Config] Max attempts: {max_attempts}")
    print(f"[Config] Output directory: {output_dir}")
    print(f"{'='*70}")

    # (1) Summarize Figure 1 from the PDF and map to codebook
    fig1_summary, inst_summary = summarize_and_map_pdf(pdf_path, instruction_path, model=model)
    save_artifact(output_dir, "figure_summary.txt", fig1_summary)
    save_artifact(output_dir, "instruction_summary.txt", inst_summary)
    print(f"[OK] Saved figure_summary.txt and instruction_summary.txt")

    # (2) Load the data
    df = load_data(data_path)
    cols = list(df.columns)
    print(f"[OK] Loaded data: {len(df)} rows, {len(cols)} columns")

    # (3) Unified attempt loop
    success = "0"
    for attempt in range(1, max_attempts + 1):
        success = run_single_attempt(
            attempt, state, output_dir, fig1_summary, inst_summary,
            df, data_path, cols, true_fig_path, model, max_attempts
        )
        if success == "1":
            break

    # Print final summary
    print(f"\n{'='*60}")
    print(f"=== FINAL SUMMARY ===")
    print(f"{'='*60}")
    print(f"Best result: attempt {state.best_attempt} with score {state.best_score}/100")
    if state.best_code_path:
        print(f"Best code saved at: {output_dir / 'best_generated_analysis.py'}")
    print(f"Total attempts: {attempt}")
    print(f"{'='*60}")

    return success


# ──────────────────────────────────────────────────
#   Main entry point
# ──────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replicate Figure 1 from a paper using PDF + Codebook + CSV with LLM assistance.")
    parser.add_argument("--pdf", dest="pdf_path", type=str,
                        default="Bryson_paper.pdf",
                        help="Path to the paper PDF.")
    parser.add_argument("--data", dest="data_path", type=str,
                        default="gss93_selected.csv",
                        help="Path to the CSV dataset.")
    parser.add_argument("--inst", dest="instruction_path", type=str,
                        default="gss_selected_variables_doc.txt",
                        help="Path to the codebook/instructions file.")
    parser.add_argument("--true-fig", dest="true_figure_path", type=str,
                        default=None,
                        help="Path to ground-truth figure image (JPEG/PNG). Defaults to Fig1.jpg in script directory.")
    parser.add_argument("--out", dest="output_dir", type=str, default="output_figure1",
                        help="Directory to store generated artifacts.")
    parser.add_argument("--model", dest="model", type=str, default="gpt-5.2",
                        help="Model for summarization, extraction, and code generation.")
    parser.add_argument("--attempts", dest="max_attempts", type=int, default=100,
                        help="Maximum number of retry iterations.")

    args = parser.parse_args()

    # Provide default ground-truth Figure 1 if not explicitly configured via CLI
    if args.true_figure_path is None:
        script_dir = Path(__file__).parent.resolve()
        args.true_figure_path = str(script_dir / "Fig1.jpg")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run_log_Figure1.txt"

    with tee_terminal_output(log_path):
        print(f"[Log] Writing terminal output to: {log_path}")
        result = main(
            pdf_path=args.pdf_path,
            true_figure_path=args.true_figure_path,
            data_path=args.data_path,
            instruction_path=args.instruction_path,
            output_dir=out_dir,
            model=args.model,
            max_attempts=args.max_attempts,
        )
        print(f"\n=== Final result: {result} ===")
