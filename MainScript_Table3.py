"""
Automatic replication of a summary/frequency table from a research paper using LLM.

This script is designed for summary tables that show frequency distributions, counts,
and descriptive statistics (e.g., means) - NOT regression tables with coefficients
or standard errors.

What this does:
1) Summarize the paper PDF and target table (focusing on the target summary table structure).
2) Summarize/code-map a codebook/instruction file to your data.
3) Generate Python analysis code against a local CSV.
4) Run the generated code and produce a textual summary.
5) Compare generated vs. "true" results extracted from the paper.
6) Score alignment (0-100) between generated and true results based on:
   - All row items/categories present
   - Frequency counts or percentages match (within tolerance)
   - Summary statistics (means, totals, etc.) match
   - All response/column categories present
7) Track the best-scoring attempt across all iterations.
8) Iterate with discrepancy/error feedback until:
   - Score reaches threshold (default 95), OR
   - Maximum attempts exhausted
9) Save the best result and corresponding code regardless of final outcome.

Requirements:
- OPENAI_API_KEY must be set in the environment.
- Packages: openai>=1.x (Responses API), pandas.

Tested on Python 3.11+; should work on Python 3.13 as per your environment.
"""

from __future__ import annotations

import argparse
import sys
import importlib.util
import re
import shutil
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union, cast

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
    best_result_text: Optional[str] = None
    best_attempt: int = 0
    best_code_path: Optional[Path] = None
    best_discrepancy: Optional[str] = None

    # Previous attempt tracking
    prev_attempt: int = 0
    prev_code_path: Optional[Path] = None
    prev_result_text: Optional[str] = None
    prev_discrepancy: Optional[str] = None
    prev_error: Optional[str] = None

    # Current attempt
    last_score: int = -1
    current_code_path: Optional[Path] = None
    generated_text: str = ""


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
                # prompt is guaranteed non-None if we are here due to the guard above
                resp = client.responses.create(
                    model=model,
                    instructions=instructions,
                    input=cast(str, prompt),
                    max_output_tokens=max_output_tokens,
                )
            return (resp.output_text or "").strip()
        except RateLimitError as e:
            # Parse "try again in Xs" if present; otherwise exponential backoff
            m = re.search(r"try again in ([\d\.]+)s", str(e))
            if m:
                wait = float(m.group(1)) + 0.5
            else:
                wait = min(2 ** attempt, backoff_cap)
            print(f"[RateLimit] attempt {attempt}/{max_retries}; sleeping {wait:.1f}s")
            time.sleep(wait)
        except Exception:
            # Transient error; back off and retry
            wait = min(2 ** attempt, backoff_cap)
            print(f"[LLM Error] attempt {attempt}/{max_retries}; sleeping {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError("LLM call: exceeded max retries")


def _df_sample_csv(df: pd.DataFrame, cols: List[str], n: int = 5) -> str:
    """Create a small CSV snippet of selected columns to ground the model."""
    keep = [c for c in cols if c in df.columns]
    if not keep:
        keep = list(df.columns[: min(8, len(df.columns))])
    return df[keep].head(n).to_csv(index=False)


def _normalize_result_to_text(result) -> str:
    """
    Convert various possible return types into a plain string.
    """
    if result is None:
        return ""
    if isinstance(result, pd.DataFrame):
        return result.to_string(index=False)
    if isinstance(result, dict):
        parts = []
        for k, v in result.items():
            if isinstance(v, pd.DataFrame):
                parts.append(f"## {k}\n{v.to_string(index=False)}")
            else:
                parts.append(f"## {k}\n{str(v)}")
        return "\n\n".join(parts)
    if isinstance(result, list):
        return "\n\n".join(
            (x.to_string(index=False) if isinstance(x, pd.DataFrame) else str(x))
            for x in result
        )
    return str(result)


def save_best_result(
    output_dir: Path,
    attempt: int,
    code_path: Path,
    result_text: str,
    score: int,
) -> None:
    """
    Save the current attempt's results as the best result.
    Creates copies with 'best_' prefix for easy identification.
    """
    shutil.copy(code_path, output_dir / "best_generated_analysis.py")
    save_artifact(output_dir, "best_generated_results.txt", result_text)
    save_artifact(output_dir, "best_result_metadata.txt",
        f"Best result from attempt: {attempt}\n"
        f"Score: {score}/100\n"
        f"Original code file: {code_path.name}\n"
    )
    print(f"[Best] Updated best result (attempt {attempt}, score {score})")


def try_run_generated_code(
    module_path: Union[str, Path],
    data_path: Union[str, Path],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Dynamically load the generated module and execute `run_analysis`.
    Returns (result_text, error_text). If an exception occurs, capture traceback.
    """
    try:
        module_path = Path(module_path)
        data_path = Path(data_path)

        # Dynamic import of the generated module
        spec = importlib.util.spec_from_file_location("generated_analysis", str(module_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load module from: {module_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]

        if not hasattr(mod, "run_analysis"):
            raise AttributeError(
                "The generated code does not define `run_analysis(data_source)`."
            )

        # Call the generated function
        result = mod.run_analysis(str(data_path))  # type: ignore[attr-defined]
        return _normalize_result_to_text(result), None
    except Exception:
        tb = traceback.format_exc()
        return None, tb


# ──────────────────────────────────────────────────
#   Helper Utilities for LLM tasks
# ──────────────────────────────────────────────────


def summarize_and_extract_pdf(
    pdf_path: Union[str, Path],
    *,
    model: str,
) -> Tuple[str, str]:
    """
    Summarize the paper focusing on the target summary/frequency table and extract true results.
    Returns (summary, true_results).
    """
    file_id = upload_pdf(pdf_path)

    # Summarize the paper
    summary_instructions = (
        "You are a concise, careful research assistant with strong skills in "
        "quantitative methods in sociology. Extract the structure and content of summary tables."
    )
    summary_content = [
        {
            "type": "input_text",
            "text": (
                "Task: Read the attached paper PDF and identify Table 3 (the target summary/frequency table).\n"
                "List all row categories (items) in the table, identify the column categories (response options), "
                "and note what summary statistics are shown (e.g., counts/frequencies, percentages, means).\n"
                "This is a descriptive summary table, NOT a regression table with coefficients."
            ),
        },
        {"type": "input_file", "file_id": file_id},
    ]
    summary = _llm_call(model=model, instructions=summary_instructions, max_output_tokens=MAX_OUTPUT_TOKENS, content=summary_content)

    # Extract true results
    extract_instructions = "You are meticulous. Extract all counts, frequencies, and summary statistics exactly as shown in the table."
    extract_content = [
        {
            "type": "input_text",
            "text": (
                "Read the attached paper PDF and extract Table 3 (the target summary/frequency table).\n"
                "For each row item, extract:\n"
                "- Counts or frequencies for each response/column category\n"
                "- Any summary statistics shown (means, percentages, totals, etc.)\n"
                "Present the data in a clear tabular format.\n"
                "You do not need to ask questions or make suggestions. Just extract the results exactly as shown."
            ),
        },
        {"type": "input_file", "file_id": file_id},
    ]
    true_results = _llm_call(model=model, instructions=extract_instructions, max_output_tokens=MAX_OUTPUT_TOKENS, content=extract_content)

    return summary, true_results


def summarize_instruction(
    analysis_summary: str,
    instruction_path: Union[str, Path],
    *,
    model: str,
) -> str:
    """
    Map variables in the analysis summary to the fields described in the codebook file.
    """
    instructions = "You are a careful replicator who writes clear, actionable mappings."

    # Read the file containing variable documentation
    inst_path = Path(instruction_path)
    with open(inst_path, "r", encoding="utf-8") as f:
        inst_content = f.read()

    prompt = (
        "Based on the analysis summary below, produce an explanation that maps the analysis variables "
        "to the data fields described in the attached variable documentation. This mapping will be used to run the analysis.\n"
        "You do not need to ask questions or make suggestions. Just produce the mapping explanation.\n\n"
        f"[Analysis Summary]\n{analysis_summary}\n\n"
        f"[Variable Documentation]\n{inst_content}\n"
    )
    return _llm_call(prompt, model=model, instructions=instructions, max_output_tokens=MAX_OUTPUT_TOKENS)


def ask_code_with_data(
    analysis_summary: str,
    df: pd.DataFrame,
    data_path: Union[str, Path],
    data_columns: List[str],
    instruction_summary: str,
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
) -> str:
    """
    Ask the model to generate Python code implementing the analysis.
    For retries, provides best attempt and previous attempt context.
    """
    sample_csv = _df_sample_csv(df, data_columns, n=sample_rows)
    cols_str = ", ".join(data_columns)

    if is_retry:
        instructions = "You fix issues precisely and write simple, correct code for summary/frequency tables."
    else:
        instructions = (
            "You are a replication-focused researcher in quantitative methods in sociology. "
            "Produce minimal, correct, and readable code for generating summary/frequency tables."
        )

    # Build the prompt
    prompt_parts = []

    if is_retry:
        prompt_parts.append("We previously ran your analysis and found discrepancies or runtime errors. Use the feedback below to correct your code.\n")

    prompt_parts.append(f"[Analysis Summary]\n{analysis_summary}\n")
    prompt_parts.append(f"[Mapping Instruction]\n{instruction_summary}\n")
    prompt_parts.append(f"[Sample Data (first {sample_rows} rows)]\n{sample_csv}")
    prompt_parts.append(f"[Available Variables]\n{cols_str}\n")
    prompt_parts.append(f"[Path to the full dataset]\n{data_path}\n")

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
        "- Read the dataset from `data_source` using pandas (CSV).\n"
        "- For each relevant variable, compute frequency counts for each response category.\n"
        "- Compute any summary statistics shown in the target table (e.g., means, percentages).\n"
        "- Return a summary that matches the structure of the target table.\n"
        "- Import everything you need inside the code.\n"
        "- The code must run as-is. Output *code only* (no backticks, no commentary).\n"
        "- The code should save the summary table as a human-readable text file in ./output directory.\n"
        "- Never directly copy-paste numbers from the paper; always compute them from the raw data."
    )

    prompt = "\n".join(prompt_parts)
    code = _llm_call(prompt, model=model, instructions=instructions)

    # Strip markdown code fences
    code = code.strip()
    code = re.sub(r'^```(?:python|py)?\s*\n?', '', code)
    code = re.sub(r'\n?```\s*$', '', code)
    return code.strip()


def score_alignment(
    true_result_text: str,
    generated_text: str,
    *,
    model: str,
) -> int:
    """
    Score how well generated summary/frequency table results match true results (0-100).
    Scoring criteria adapted for summary/frequency tables (no coefficients or standard errors).
    """
    instructions = "Return only an integer from 0 to 100. No other text."
    prompt = (
        "Score alignment between generated and true Table 3 summary/frequency table results (0-100).\n\n"
        "SCORING CRITERIA FOR SUMMARY/FREQUENCY TABLE:\n"
        "- All row items/categories present: 20 points\n"
        "- Frequency counts or percentages match (within 5% tolerance): 35 points\n"
        "- Summary statistics (means, totals, etc.) match (within 0.1 or 5%): 25 points\n"
        "- All response/column categories present: 10 points\n"
        "- Overall structure and formatting match: 10 points\n\n"
        "Return only the integer score (0-100), nothing else.\n\n"
        f"[Generated Results]\n{generated_text}\n\n"
        f"[True Results]\n{true_result_text}"
    )
    result = _llm_call(prompt, model=model, instructions=instructions).strip()
    try:
        return int(result)
    except ValueError:
        # If parsing fails, extract first number found
        match = re.search(r'\d+', result)
        return int(match.group()) if match else 0


def report_discrepancy(
    true_result_text: str,
    generated_text: str,
    *,
    model: str,
) -> str:
    """
    Produce a human-readable discrepancy report with concrete fixes for summary/frequency table.
    """
    instructions = (
        "Be a critical reviewer. Identify all discrepancies in frequency counts, summary statistics, "
        "missing items/categories, or incorrect values and explain how to fix them."
    )
    prompt = (
        "Carefully compare the generated and true summary/frequency table results.\n"
        "Identify every mismatch in:\n"
        "- Row items/categories (names or number of items)\n"
        "- Frequency counts or percentages for each response category\n"
        "- Summary statistics (means, totals, etc.)\n"
        "- Missing or extra column categories\n"
        "Explain how to fix them so the generated analysis matches the true table.\n\n"
        f"[Generated Results]\n{generated_text}\n\n"
        f"[True Results]\n{true_result_text}"
    )
    return _llm_call(prompt, model=model, instructions=instructions, max_output_tokens=MAX_OUTPUT_TOKENS)


# ──────────────────────────────────────────────────
#   Main pipeline
# ──────────────────────────────────────────────────


def run_single_attempt(
    attempt: int,
    state: RunState,
    output_dir: Path,
    summary: str,
    true_result_text: str,
    df: pd.DataFrame,
    data_path: Path,
    cols: List[str],
    instruction: str,
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
        print(f"\n{'='*60}")
        print(f"[Retry Loop] attempt {attempt}/{max_attempts}")
        print(f"{'='*60}")

        # Generate discrepancy report for previous attempt (if it had results)
        if state.prev_result_text:
            state.prev_discrepancy = report_discrepancy(true_result_text, state.prev_result_text, model=model)
            score_header = f"Score: {state.last_score}/100\n{'='*60}\n\n"
            save_artifact(output_dir, f"discrepancy_report_attempt_{attempt-1}.txt", score_header + state.prev_discrepancy)
            print(f"[Mismatch] Saved discrepancy report → discrepancy_report_attempt_{attempt-1}.txt")
        else:
            state.prev_discrepancy = None
            print("[Info] Skipping discrepancy report due to previous runtime error.")

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
        code = ask_code_with_data(
            summary, df, data_path, cols, instruction,
            model=model,
            is_retry=True,
            best_code_text=best_code_text,
            best_discrepancy=state.best_discrepancy,
            best_score=state.best_score if state.best_score >= 0 else None,
            prev_code_text=prev_code_text,
            prev_discrepancy=state.prev_discrepancy,
            prev_error=state.prev_error,
        )
        code_path = output_dir / f"generated_analysis_retry_{attempt}.py"
        save_artifact(output_dir, f"generated_analysis_retry_{attempt}.py", code)
        print(f"[Retry] Saved revised code → {code_path.name}")
    else:
        # Initial attempt - generate first code
        code = ask_code_with_data(summary, df, data_path, cols, instruction, model=model)
        code_path = output_dir / "generated_analysis.py"
        save_artifact(output_dir, "generated_analysis.py", code)
        print(f"[OK] Saved generated code → {code_path.name}")

    state.current_code_path = code_path

    # Run the generated code
    generated_text, runtime_error = try_run_generated_code(code_path, data_path)

    if runtime_error:
        save_artifact(output_dir, f"runtime_error_attempt_{attempt}.txt", runtime_error)
        print(f"[Runtime Error] Saved traceback → runtime_error_attempt_{attempt}.txt")
        # Store as previous attempt info for next iteration
        state.prev_attempt = attempt
        state.prev_code_path = code_path
        state.prev_result_text = None
        state.prev_error = runtime_error
        state.generated_text = ""
        success = "0"
    else:
        assert generated_text is not None
        state.generated_text = generated_text
        if is_retry:
            print(f"[Retry Generated Summary] (attempt {attempt})\n{generated_text[:5000]}")
        else:
            print("[Generated Summary]\n", generated_text[:5000], sep="")

        save_artifact(output_dir, f"generated_results_attempt_{attempt}.txt", generated_text)
        print(f"[OK] Saved generated results → generated_results_attempt_{attempt}.txt")

        # Score alignment
        state.last_score = score_alignment(true_result_text, generated_text, model=model)
        is_aligned = state.last_score >= ALIGNMENT_SCORE_THRESHOLD
        success = "1" if is_aligned else "0"
        print(f"[Score {attempt}] {state.last_score}/100, Aligned: {is_aligned}")

        # Store as previous attempt info for next iteration
        state.prev_attempt = attempt
        state.prev_code_path = code_path
        state.prev_result_text = generated_text
        state.prev_error = None

        # Update best if needed
        if state.last_score > state.best_score:
            # Generate discrepancy for the new best (will be used in future retries)
            state.best_discrepancy = report_discrepancy(true_result_text, generated_text, model=model)
            state.best_score = state.last_score
            state.best_result_text = generated_text
            state.best_attempt = attempt
            state.best_code_path = code_path
            save_best_result(output_dir, attempt, code_path, generated_text, state.last_score)

    print(f"[Check {attempt}/{max_attempts}] Success: {success}")
    iter_elapsed = time.time() - iter_start
    print(f"[Duration] attempt {attempt}: {iter_elapsed:.1f}s")

    return success


def main(
    pdf_path: Union[str, Path],
    data_path: Union[str, Path],
    instruction_path: Union[str, Path],
    *,
    output_dir: Union[str, Path],
    model: str,
    max_attempts: int,
) -> str:
    """
    Orchestrates the full loop and writes intermediate artifacts to `output_dir`.

    Output files created:
    - table_summary.txt: Summary of the paper PDF and target table
    - instruction_summary.txt: Mapping from codebook to variables
    - generated_analysis.py: Initial generated code
    - generated_analysis_retry_{N}.py: Retry code for attempt N
    - generated_results_attempt_{N}.txt: Tables and results for each trial (text format)
    - runtime_error_attempt_{N}.txt: Runtime errors if any
    - discrepancy_report_attempt_{N}.txt: Comparison reports
    - best_generated_analysis.py: Best code so far
    - best_generated_results.txt: Best results so far
    - best_result_metadata.txt: Metadata about the best result

    Returns: "1" on success (aligned), "0" otherwise.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(data_path)

    state = RunState()

    print(f"\n{'='*70}")
    print(f"[Config] Model: {model}")
    print(f"[Config] Max attempts: {max_attempts}")
    print(f"[Config] Output directory: {output_dir}")
    print(f"{'='*70}")

    # (1) Summarize the paper PDF and target table and extract true results
    summary, true_result_text = summarize_and_extract_pdf(pdf_path, model=model)
    save_artifact(output_dir, "table_summary.txt", summary)
    print(f"[OK] Saved summary → table_summary.txt")
    print("[True Results Extracted]\n", true_result_text[:5000], sep="")

    # (2) Load the dataset
    df = load_data(data_path)
    cols = list(df.columns)
    print(f"[OK] Loaded data with {len(cols)} columns")

    # (3) Summarize instruction/codebook
    instruction = summarize_instruction(summary, instruction_path, model=model)
    save_artifact(output_dir, "instruction_summary.txt", instruction)
    print(f"[OK] Saved instruction summary → instruction_summary.txt")

    # (4) Unified attempt loop
    success = "0"
    for attempt in range(1, max_attempts + 1):
        success = run_single_attempt(
            attempt, state, output_dir, summary, true_result_text,
            df, data_path, cols, instruction, model, max_attempts
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
    parser = argparse.ArgumentParser(description="Replicate a summary/frequency table (Table 3) from a paper using PDF + Codebook + CSV with LLM assistance.")
    parser.add_argument("--pdf", dest="pdf_path", type=str,
                        default="Bryson_paper.pdf",
                        help="Path to the paper PDF.")
    parser.add_argument("--data", dest="data_path", type=str,
                        default="gss93_selected.csv",
                        help="Path to the CSV dataset.")
    parser.add_argument("--inst", dest="instruction_path", type=str,
                        default="gss_selected_variables_doc.txt",
                        help="Path to the codebook/instructions file.")
    parser.add_argument("--out", dest="output_dir", type=str, default="output_table3",
                        help="Directory to store generated artifacts.")
    parser.add_argument("--model", dest="model", type=str, default="gpt-5.2",
                        help="Model for summarization, extraction, and code generation.")
    parser.add_argument("--attempts", dest="max_attempts", type=int, default=100,
                        help="Maximum number of retry iterations.")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run_log_Table3.txt"

    with tee_terminal_output(log_path):
        print(f"[Log] Writing terminal output to: {log_path}")
        result = main(
            pdf_path=args.pdf_path,
            data_path=args.data_path,
            instruction_path=args.instruction_path,
            output_dir=out_dir,
            model=args.model,
            max_attempts=args.max_attempts,
        )
        print(f"\n=== Final result: {result} ===")
