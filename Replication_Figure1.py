"""
Automatic replication of Figure 1 from Bryson (1996) using LLM-driven iterative code generation.

What this does:
1) Summarize the paper PDF to extract Figure 1 construction details (variables, methods, plotting specifications).
2) Map codebook/instruction CSV to dataset columns for variable alignment.
3) Generate Python analysis code that recreates Figure 1 and saves it as a JPEG file.
4) Execute the generated code to produce a JPEG figure file.
5) Compare the generated JPEG against the ground-truth Figure 1 JPEG using vision-enabled LLM.
6) Iterate with visual discrepancy feedback and runtime error corrections until figures match or max attempts reached.

The script uses base64-encoded data URLs to transfer JPEG images to the OpenAI Responses API for 
visual comparison, avoiding file upload complications and ensuring robust image analysis.

Requirements:
- OPENAI_API_KEY must be set in the environment.
- Packages: openai>=1.51 (Responses API with vision), pandas, numpy, matplotlib, statsmodels.

Tested on Python 3.11+; should work on Python 3.13 as per your environment.
"""

from __future__ import annotations

import os
import re
import time
import json
import types
import importlib.util
import traceback
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Dict
import base64

from pathlib import Path

import pandas as pd

# Third‑party API client
try:
    from openai import OpenAI
    from openai import RateLimitError
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore
    RateLimitError = Exception  # type: ignore

# -------------------------
# Configuration helpers
# -------------------------
CLIENT_TIMEOUT_S = 600

def _client():
    if OpenAI is None:
        raise RuntimeError("The openai package is required. pip install openai>=1.51")
    return OpenAI(timeout=CLIENT_TIMEOUT_S)

client = _client()

# -------------------------
# Data IO
# -------------------------
def load_data(data_path: Union[str, Path]) -> pd.DataFrame:
    """Load CSV with pragmatic defaults (comma separator)."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")
    if data_path.suffix.lower() != ".csv":
        raise ValueError(f"Unsupported file type for data: {data_path.suffix} (expected .csv)")
    return pd.read_csv(
        data_path,
        sep=",",
        engine="python",
        on_bad_lines="skip",
        skip_blank_lines=True,
    )

# -------------------------
# Files -> API
# -------------------------
def upload_file_for_api(path: Union[str, Path]) -> str:
    """Upload a local file (pdf/image/other) to the API and return its file_id."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    with open(p, "rb") as f:
        fobj = client.files.create(file=f, purpose="user_data")
    return fobj.id

def _image_to_data_url(path: Union[str, Path]) -> str:
    """Read an image and return a data URL suitable for the Responses API vision input.
    Supports JPEG/PNG based on file extension.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    suffix = p.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif suffix == ".png":
        mime = "image/png"
    else:
        # default to jpeg if unknown
        mime = "image/jpeg"
    data = p.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"

# -------------------------
# LLM wrapper
# -------------------------
def _llm_call(
    *,
    model: str,
    instructions: Optional[str] = None,
    prompt: Optional[str] = None,
    content: Optional[List[dict]] = None,
    max_output_tokens: Optional[int] = None,
    max_retries: int = 8,
    backoff_cap: float = 10.0,
) -> str:
    """
    Thin wrapper around the OpenAI Responses API with rate-limit backoff.
    Supports either a plain text prompt or a rich content payload (with images/PDF).
    """
    if not prompt and not content:
        raise ValueError("Provide either `prompt` or `content`.")

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
                    input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],  # type: ignore[arg-type]
                    max_output_tokens=max_output_tokens,
                )
            return (resp.output_text or "").strip()
        except RateLimitError as e:  # type: ignore[misc]
            m = re.search(r"try again in ([\d\.]+)s", str(e))
            wait = float(m.group(1)) + 0.5 if m else min(2**attempt, backoff_cap)
            print(f"[RateLimit] attempt {attempt}/{max_retries}; sleeping {wait:.1f}s")
            time.sleep(wait)
        except Exception as e:
            # Print error for diagnostics; some errors are not retryable (HTTP 400, auth, invalid content)
            err_str = str(e)
            # Bail out early on clearly non-retryable issues
            if any(k in err_str.lower() for k in ["invalid", "authentication", "unauthorized", "bad request", "unsupported"]):
                raise
            wait = min(2**attempt, backoff_cap)
            print(f"[LLM Error] attempt {attempt}/{max_retries}; sleeping {wait:.1f}s :: {err_str}")
            time.sleep(wait)
    raise RuntimeError("LLM call: exceeded max retries")

# -------------------------
# Generated code execution
# -------------------------
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
    return str(result)

def try_run_generated_code(
    module_path: Union[str, Path],
    data_path: Union[str, Path],
    *,
    sep: Optional[str] = None,
    na_values: Optional[List[str]] = None,
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
        # Ensure pandas gets an explicit separator in generated code
        effective_sep = sep if sep is not None else ","
        result = mod.run_analysis(str(data_path), sep=effective_sep, na_values=na_values)  # type: ignore[attr-defined]
        return _normalize_result_to_text(result), None
    except Exception as e:
        tb = traceback.format_exc()
        return None, f"[RuntimeError] {e}\n{tb}"

# -------------------------
# Prompting steps (Figure 1 focus)
# -------------------------

def _strip_markdown_fences(code: str) -> str:
    """
    Remove markdown code fences (```python ... ``` or ``` ... ```) if present.
    The LLM sometimes wraps code despite being told not to.
    """
    lines = code.strip().splitlines()
    # Check if first line is a fence (e.g., ```python or ```)
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    # Check if last line is a fence (e.g., ```)
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)

def summarize_pdf(pdf_path: Union[str, Path], *, model: str, max_tokens: int) -> str:
    """Ask the model to extract the Figure 1 recipe specifically."""
    print(f"\n[Step 1] Uploading PDF: {pdf_path}")
    file_id = upload_file_for_api(pdf_path)
    print(f"[Step 1] PDF uploaded successfully (file_id: {file_id[:20]}...)")
    print(f"[Step 1] Extracting Figure 1 details from PDF using model: {model}")
    instructions = (
        "You are a concise, careful research assistant with strong quantitative skills. "
        "Extract ONLY what is needed to reproduce Figure 1 from the article."
    )
    content = [
        {"type": "input_text", "text": (
            "Task: From the attached PDF, find the exact construction steps for **Figure 1**. "
            "Summarize related data sources, variables, and statistical methods."
            "Also summarize plotting details such as axes, transformations, legends, annotations, line/marker styles. "
            "Do NOT summarize tables. Focus on Figure 1 only."
        )},
        {"type": "input_file", "file_id": file_id},
    ]
    result = _llm_call(model=model, instructions=instructions, content=content, max_output_tokens=max_tokens)
    print(f"[Step 1] Figure 1 summary extracted ({len(result)} characters)")
    return result

def summarize_instruction(
    analysis_summary: str,
    instruction_csv_path: Union[str, Path],
    *,
    model: str,
    max_tokens: int,
) -> str:
    """
    Map variables in the analysis summary to the fields described in the codebook CSV file.
    """
    print(f"\n[Step 3] Reading codebook CSV: {instruction_csv_path}")
    # Read the codebook to ground the mapping (if available).
    p = Path(instruction_csv_path)
    instruction_summary = ""
    if p.exists():
        try:
            cb = pd.read_csv(p, nrows=1000)  # short sample
            instruction_summary = cb.head(20).to_csv(index=False)
            print(f"[Step 3] Codebook loaded: {len(cb)} entries, {len(cb.columns)} columns")
        except Exception as e:
            instruction_summary = f"(Could not parse codebook CSV: {e})"
            print(f"[Step 3] Warning: Could not parse codebook CSV: {e}")
    else:
        instruction_summary = "(No codebook CSV provided.)"
        print(f"[Step 3] Warning: Codebook CSV not found")

    print(f"[Step 3] Mapping variables to codebook using model: {model}")
    instructions = "You are a careful replicator who writes explicit, actionable mappings."
    prompt = (
        "We will reproduce **Figure 1** from the article. Using the analysis summary below and the "
        "codebook snippet (if any), list the dataset variable names to use for Figure 1\n"
        "Be concrete; name exact columns if possible. If uncertain, state robust fallbacks.\n\n"
        f"[Figure‑1 Analysis Summary]\n{analysis_summary}\n\n"
        f"[Codebook snippet]\n{instruction_summary}\n"
    )
    result = _llm_call(model=model, instructions=instructions, prompt=prompt, max_output_tokens=max_tokens)
    print(f"[Step 3] Variable mapping completed ({len(result)} characters)")
    return result

def ask_code_with_data(
    analysis_summary: str,
    df: pd.DataFrame,
    data_path: Union[str, Path],
    data_columns: List[str],
    instruction_summary: str,
    error_feedback: str,
    *,
    model: str,
    sample_rows: int = 5,
    jpeg_basename: str = "generated_results.jpg",
) -> str:
    """
    Ask the LLM to write the Python analysis code that *produces the Figure 1 replica*.
    The function signature MUST be:  def run_analysis(data_source, sep=None, na_values=None)
    and it MUST return the absolute path to the saved JPEG.
    """
    print(f"\n[Step 4] Generating Python code using model: {model}")
    print(f"[Step 4] Data shape: {df.shape[0]} rows, {df.shape[1]} columns")
    if error_feedback:
        print(f"[Step 4] Incorporating feedback from previous attempt ({len(error_feedback)} characters)")
    sample_csv = df[data_columns[: min(len(data_columns), 18)]].head(sample_rows).to_csv(index=False)
    cols_str = ", ".join(data_columns[:80])

    prompt = (
        "Write Python code (return code only) that defines:\n\n"
        "    def run_analysis(data_source, sep=None, na_values=None):\n"
        "        ...\n\n"
        "The function must:\n"
        "- Import all required packages.\n"
        "- Read the CSV at data_source with pandas. If sep is None, default to ',' to avoid parser warnings.\n"
        "- **CRITICAL**: Drop or impute NaN/missing values before fitting any statistical models. "
        "sklearn LogisticRegression and statsmodels Logit do NOT accept NaN. Use df.dropna(subset=...) or fillna().\n"
        "- Recreate Figure 1 using matplotlib (no seaborn).\n"
        "- Be careful about title and axis comparable to the article Figure 1.\n"
        "- Save as a JPEG file (300 DPI) to './output/" + jpeg_basename + "' and return its absolute path.\n"
        "- If unsure of a precise column, search df.columns for a plausible match (case-insensitive), "
        "  raising a clear error if not found.\n"
        "- Do not hard-code numbers from the paper; compute from the data.\n"
        "- Prefer vectorized pandas operations; avoid DataFrame.applymap on large frames.\n"
        "- Output code only (no backticks, no prose).\n"
        "- The code must be self-contained (imports inside) and runnable as-is.\n"
        "\n"
        f"[Figure‑1 Analysis Summary]\n{analysis_summary}\n\n"
        f"[Instruction Mapping]\n{instruction_summary}\n\n"
        f"[Sample data ({sample_rows} rows)]\n{sample_csv}\n"
        f"[All available columns]\n{cols_str}\n\n"
        "If the previous attempt produced errors or a wrong plot, here is feedback to fix:\n"
        f"{error_feedback}\n"
    )

    instructions = "Return code only (no backticks, no commentary)."
    raw_code = _llm_call(model=model, instructions=instructions, prompt=prompt, max_output_tokens=20000)
    # Strip any markdown fences the model might have added despite instructions
    cleaned_code = _strip_markdown_fences(raw_code)
    print(f"[Step 4] Code generation completed ({len(cleaned_code)} characters)")
    return cleaned_code

# -------------------------
# Image-based evaluation
# -------------------------
def check_alignment_images(true_img_path: Union[str, Path], gen_img_path: Union[str, Path], *, model: str) -> str:
    """
    Send both images to the model. Return '1' if the two figures depict the same content/ordering/axes,
    else '0'. No other text.
    """
    print(f"\n[Vision Check] Comparing generated figure to ground truth using model: {model}")
    # Use data URLs for robust inline vision input (avoids file id schema issues)
    url_true = _image_to_data_url(true_img_path)
    url_gen  = _image_to_data_url(gen_img_path)
    instructions = "Return strictly '1' for a strong match or '0' otherwise. No other text."
    content = [
        {"type": "input_text", "text": (
            "Determine whether these two figures are effectively the same plot of the article Figure 1. "
            "They should match in axes ranges and tick labels, axis titles, series present, legends, line/marker styles, annotations, and overall layout.\n"
        )},
        {"type": "input_image", "image_url": url_true},
        {"type": "input_image", "image_url": url_gen},
    ]
    result = _llm_call(model=model, instructions=instructions, content=content, max_output_tokens=16)
    print(f"[Vision Check] Alignment verdict: {result}")
    return result

def report_discrepancy_images(true_img_path: Union[str, Path], gen_img_path: Union[str, Path], *, model: str, max_tokens: int) -> str:
    """Ask the model to describe differences and actionable fixes to the code."""
    print(f"\n[Discrepancy Report] Analyzing differences between figures using model: {model}")
    url_true = _image_to_data_url(true_img_path)
    url_gen  = _image_to_data_url(gen_img_path)
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
    result = _llm_call(model=model, instructions=instructions, content=content, max_output_tokens=max_tokens)
    print(f"[Discrepancy Report] Report generated ({len(result)} characters)")
    return result

# -------------------------
# Orchestrator
# -------------------------
def main(
    pdf_path: Union[str, Path],
    true_figure_path: Union[str, Path],
    data_path: Union[str, Path],
    instruction_path: Union[str, Path],
    *,
    output_dir: Union[str, Path],
    summary_model: str,
    code_model: str,
    max_attempts: int,
    max_tokens: int,
) -> str:
    """
    End‑to‑end loop to replicate Figure 1.
    Files written:
      - pdf_summary.txt                     (Figure‑1 extraction)
      - instruction_summary.txt             (variable mapping)
      - generated_analysis.py               (initial code)
      - generated_analysis_retry_{i}.py     (retries)
      - generated_results_attempt_{i}.jpg   (generated figure per attempt)
      - discrepancy_report_attempt_{i}.txt  (model-written difference report)
      - runtime_error_attempt_{i}.txt       (traceback if any)
    Returns '1' on success, '0' otherwise.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"[Config] Summary model: {summary_model}")
    print(f"[Config] Code model: {code_model}")
    print(f"[Config] Max attempts: {max_attempts}")
    print(f"[Config] Output directory: {output_dir}")
    print(f"{'='*70}")

    # (1) Summarize *Figure 1* from the PDF.
    fig1_summary = summarize_pdf(pdf_path, model=summary_model, max_tokens=max_tokens)
    (output_dir / "pdf_summary.txt").write_text(fig1_summary, encoding="utf-8")
    print(f"[Step 1] Saved: pdf_summary.txt")

    # (2) Load the data to expose headers and sample values.
    print(f"\n[Step 2] Loading dataset: {data_path}")
    df = load_data(data_path)
    cols = list(df.columns)
    print(f"[Step 2] Dataset loaded: {len(df)} rows, {len(cols)} columns")

    # (3) Summarize/align instruction mapping.
    inst_summary = summarize_instruction(fig1_summary, instruction_path, model=summary_model, max_tokens=max_tokens)
    (output_dir / "instruction_summary.txt").write_text(inst_summary, encoding="utf-8")
    print(f"[Step 3] Saved: instruction_summary.txt")

    # (4) Ask for initial code that reproduces Figure 1 (returns path to saved JPEG).
    init_code = ask_code_with_data(
        fig1_summary, df, data_path, cols, inst_summary, error_feedback="",
        model=code_model, jpeg_basename="generated_results.jpg"
    )
    gen_path = output_dir / "generated_analysis.py"
    gen_path.write_text(init_code, encoding="utf-8")
    print(f"[Step 4] Saved: generated_analysis.py")

    # (5) Attempt loop
    attempt = 0
    success = "0"
    accumulated_feedback = ""

    # keep the ground truth path absolute
    true_fig_path = str(Path(true_figure_path).resolve())

    # Helper to rename/copy the most recent image to our attempt filename
    def _materialize_attempt_image(result_text: Optional[str], attempt_index: int) -> Optional[str]:
        if not result_text:
            return None
        p = Path(result_text.strip().splitlines()[-1]).expanduser().resolve()
        if not p.exists():
            # Sometimes user code prints the path but saves elsewhere; try common fallback
            fallback = output_dir / "generated_results.jpg"
            if fallback.exists():
                p = fallback
            else:
                # scan any recent image in ./output
                cands = sorted(output_dir.glob("*.jpg"), key=lambda x: x.stat().st_mtime, reverse=True)
                if cands:
                    p = cands[0]
                else:
                    return None
        dest = output_dir / f"generated_results_attempt_{attempt_index}.jpg"
        if str(p.resolve()) != str(dest.resolve()):
            dest.write_bytes(p.read_bytes())
        return str(dest.resolve())

    # (6) Run initial code
    print(f"\n[Step 5] Executing generated code (attempt 0)...")
    gen_text, runtime_err = try_run_generated_code(gen_path, data_path)
    if runtime_err:
        (output_dir / "runtime_error_attempt_0.txt").write_text(runtime_err, encoding="utf-8")
        print(f"[Step 5] Runtime error occurred - saved to runtime_error_attempt_0.txt")
    else:
        print(f"[Step 5] Code executed successfully")
    attempt_img = _materialize_attempt_image(gen_text, 0)

    if attempt_img and os.path.exists(attempt_img):
        print(f"[Step 5] Generated figure saved: {Path(attempt_img).name}")
        # Compare images
        verdict = check_alignment_images(true_fig_path, attempt_img, model=summary_model).strip()
        success = "1" if verdict == "1" else "0"
        if success == "1":
            print(f"[Step 5] ✓ Figures match! Replication successful.")
        else:
            print(f"[Step 5] ✗ Figures don't match. Will retry...")
    else:
        print(f"[Step 5] ✗ No figure generated")
        success = "0"

    # (7) Retry loop
    while success != "1" and attempt < max_attempts:
        attempt += 1
        print(f"\n{'='*70}")
        print(f"[Retry Loop] Attempt {attempt}/{max_attempts}")
        print(f"{'='*70}")

        # Produce discrepancy report if we have any image to compare
        if attempt_img and os.path.exists(attempt_img):
            disc = report_discrepancy_images(true_fig_path, attempt_img, model=summary_model, max_tokens=max_tokens)
            (output_dir / f"discrepancy_report_attempt_{attempt}.txt").write_text(disc, encoding="utf-8")
            print(f"[Retry {attempt}] Saved: discrepancy_report_attempt_{attempt}.txt")
            accumulated_feedback = disc if not accumulated_feedback else (accumulated_feedback + "\n---\n" + disc)
        else:
            accumulated_feedback = (accumulated_feedback + "\n(Note: previous run did not produce an image.)").strip()
            print(f"[Retry {attempt}] No image from previous attempt to compare")

        # Ask for revised code
        revised = ask_code_with_data(
            fig1_summary, df, data_path, cols, inst_summary, accumulated_feedback,
            model=code_model, jpeg_basename="generated_results.jpg"
        )
        rpath = output_dir / f"generated_analysis_retry_{attempt}.py"
        rpath.write_text(revised, encoding="utf-8")
        print(f"[Retry {attempt}] Saved: generated_analysis_retry_{attempt}.py")

        # Run revised code
        print(f"[Retry {attempt}] Executing revised code...")
        gen_text, runtime_err = try_run_generated_code(rpath, data_path)
        if runtime_err:
            (output_dir / f"runtime_error_attempt_{attempt}.txt").write_text(runtime_err, encoding="utf-8")
            print(f"[Retry {attempt}] Runtime error occurred - saved to runtime_error_attempt_{attempt}.txt")
            # continue; we still want to feed this error back next round
            continue
        else:
            print(f"[Retry {attempt}] Code executed successfully")

        attempt_img = _materialize_attempt_image(gen_text, attempt)
        if attempt_img and os.path.exists(attempt_img):
            print(f"[Retry {attempt}] Generated figure saved: {Path(attempt_img).name}")
            verdict = check_alignment_images(true_fig_path, attempt_img, model=summary_model).strip()
            success = "1" if verdict == "1" else "0"
            if success == "1":
                print(f"[Retry {attempt}] ✓ Figures match! Replication successful.")
            else:
                print(f"[Retry {attempt}] ✗ Figures still don't match. Continuing...")
        else:
            print(f"[Retry {attempt}] ✗ No figure generated")
            success = "0"

    if success != "1":
        print(f"\n{'='*70}")
        print(f"[Final] Failed to replicate figure after {max_attempts} attempts")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print(f"[Final] Successfully replicated figure on attempt {attempt}")
        print(f"{'='*70}")
    
    return success

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Replicate an article' Figure 1 with an LLM loop.")
    parser.add_argument("--pdf", dest="pdf_path", type=str,
                        default="Bryson_paper.pdf",
                        help="Path to the paper PDF.")
    parser.add_argument("--data", dest="data_path", type=str,
                        default="gss93_selected.csv",
                        help="Path to the CSV dataset.")
    parser.add_argument("--inst", dest="instruction_path", type=str,
                        default="gss_selected_variables_doc.csv",
                        help="Path to the codebook/instructions CSV file.")
    parser.add_argument("--out", dest="output_dir", type=str, default="output",
                        help="Directory to store generated artifacts.")
    parser.add_argument("--summary-model", dest="summary_model", type=str, default="gpt-5",
                        help="Model for summarization/extraction.")
    parser.add_argument("--code-model", dest="code_model", type=str, default="gpt-5",
                        help="Model for code generation.")
    parser.add_argument("--attempts", dest="max_attempts", type=int, default=15,
                        help="Maximum number of retry iterations.")
    parser.add_argument("--max-tokens", dest="max_tokens", type=int, default=20000,
                        help="Maximum output tokens for LLM responses.")

    args = parser.parse_args()

    # Provide default ground-truth Figure 1 if not explicitly configured via CLI
    script_dir = Path(__file__).parent.resolve()
    default_true_figure = str(script_dir / "Fig1.jpg")

    result = main(
        pdf_path=args.pdf_path,
        true_figure_path=default_true_figure,
        data_path=args.data_path,
        instruction_path=args.instruction_path,
        output_dir=args.output_dir,
        summary_model=args.summary_model,
        code_model=args.code_model,
        max_attempts=args.max_attempts,
        max_tokens=args.max_tokens,
    )
    print(f"\n=== Final result: {result} ===")
