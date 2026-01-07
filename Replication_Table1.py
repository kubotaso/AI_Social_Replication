"""
Automatic replication of Bryson (1996) using LLM.

What this does:
1) Summarize the paper PDF (models, variables, tables).
2) Summarize/code-map a codebook/instruction CSV to your data.
3) Generate Python analysis code against a local CSV.
4) Run the generated code and produce a textual summary.
5) Compare generated vs. "true" results extracted from the paper.
6) Iterate with discrepancy/error feedback until the results align or attempts run out.

Requirements:
- OPENAI_API_KEY must be set in the environment.
- Packages: openai>=1.x (Responses API), pandas.

Tested on Python 3.11+; should work on Python 3.13 as per your environment.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import time
import traceback
from pathlib import Path
from typing import List, Optional, Tuple, Union, cast

import pandas as pd
from openai import OpenAI, RateLimitError
from sklearn import pipeline

# OpenAI client (reads OPENAI_API_KEY from env)
client = OpenAI()


# ──────────────────────────────────────────────────
#   Helper Utilities For Local Use
# ──────────────────────────────────────────────────


def load_data(data_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load CSV with pragmatic defaults.
    """
    data_path = Path(data_path)
    ext = data_path.suffix.lower()
    if ext == ".csv":
        # Use python engine to be tolerant of inconsistent rows (on_bad_lines='skip')
        return pd.read_csv(
            data_path,
                sep=",",
            engine="python",
            on_bad_lines="skip",
            skip_blank_lines=True,
        )
    else:
        raise ValueError(f"Unsupported file type: {ext}. Only CSV files are supported.")


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


def run_generated_code(
    module_path: Union[str, Path],
    data_path: Union[str, Path],
    *,
    sep: Optional[str] = None,
    na_values: Optional[List[str]] = None,
) -> str:
    """
    Dynamically load the generated module and execute `run_analysis`.
    IMPORTANT: Ensure pandas gets an explicit CSV separator (never None).
    """
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
            "The generated code does not define `run_analysis(data_source, sep=None, na_values=None)`."
        )

    # Call the generated function with a real separator
    result = mod.run_analysis(str(data_path), sep=sep, na_values=na_values)  # type: ignore[attr-defined]
    return _normalize_result_to_text(result)


def try_run_generated_code(
    module_path: Union[str, Path],
    data_path: Union[str, Path],
    *,
    sep: Optional[str] = None,
    na_values: Optional[List[str]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Safe wrapper: returns (result_text, error_text).
    If an exception occurs (import, missing run_analysis, runtime error), capture traceback and return it as error_text.
    """
    try:
        result_text = run_generated_code(module_path, data_path, sep=sep, na_values=na_values)
        return result_text, None
    except Exception:
        tb = traceback.format_exc()
        return None, tb


# ──────────────────────────────────────────────────
#   Helper Utilities for LLM tasks
# ──────────────────────────────────────────────────


def summarize_pdf(
    pdf_path: Union[str, Path],
    *,
    model: str,
    max_tokens: int,
) -> str:
    """
    Summarize the entire paper: models, variables, tables.
    """
    instructions = (
        "You are a concise, careful research assistant with strong skills in "
        "quantitative methods in sociology. Extract models, variables, and tables."
    )
    file_id = upload_pdf(pdf_path)
    user_content = [
        {
            "type": "input_text",
            "text": (
                "Task: Read the attached paper PDF and identify Table 1 and related, data sources, variables, and statistical methods.\n"
                "List all related variables and regression models in Table 1. "
            ),
        },
        {"type": "input_file", "file_id": file_id},
    ]
    return _llm_call(model=model, instructions=instructions, max_output_tokens=max_tokens, content=user_content)


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
    instructions = "You are a careful replicator who writes clear, actionable mappings."
    
    # Read the CSV file containing variable documentation
    csv_path = Path(instruction_csv_path)
    with open(csv_path, "r", encoding="utf-8") as f:
        csv_content = f.read()
    
    prompt = (
        "Based on the analysis summary below, produce an explanation that maps the analysis variables "
        "to the data fields described in the attached variable documentation CSV. This mapping will be used to run the analysis.\n"
        "You do not need to ask questions or make suggestions. Just produce the mapping explanation.\n\n"
        f"[Analysis Summary]\n{analysis_summary}\n\n"
        f"[Variable Documentation CSV]\n{csv_content}\n"
    )
    return _llm_call(prompt, model=model, instructions=instructions, max_output_tokens=max_tokens)


def extract_result_pdf(
    pdf_path: Union[str, Path],
    *,
    model: str,
    max_tokens: int,
) -> str:
    """
    Extract the "true" regression/estimation results from the PDF (organized by table/model).
    """
    instructions = "You are meticulous. Organize results by model and table for comparison."
    file_id = upload_pdf(pdf_path)
    user_content = [
        {
            "type": "input_text",
            "text": (
                "Read the attached paper PDF and extract the statistical results (coefficients and standard errors) of Table 1.\n"
                "Be explicit about variable names used in each model.\n"
                "You do not need to ask questions or make suggestions. Just extract the results."
            ),
        },
        {"type": "input_file", "file_id": file_id},
    ]
    return _llm_call(model=model, instructions=instructions, max_output_tokens=max_tokens, content=user_content)


def ask_code_with_data(
    analysis_summary: str,
    df: pd.DataFrame,
    data_path: Union[str, Path],
    data_columns: List[str],
    instruction_summary: str,
    *,
    model: str,
    sample_rows: int = 5,
) -> str:
    """
    Ask the model to generate Python code implementing the analysis.
    """
    sample_csv = _df_sample_csv(df, data_columns, n=sample_rows)
    cols_str = ", ".join(data_columns)
    instructions = (
        "You are a replication-focused researcher in quantitative methods in sociology. Produce minimal, correct, and readable code."
    )
    prompt = (
        "Below are: (1) the analysis summary, (2) a mapping explanation from the codebook, "
        "(3) a few sample rows from the local data (CSV), and (4) the list of available variables in the dataset, and (5) path to the full dataset.\n\n"
        f"[Analysis Summary]\n{analysis_summary}\n\n"
        f"[Mapping Instruction]\n{instruction_summary}\n\n"
        f"[Sample Data (first {sample_rows} rows)]\n{sample_csv}\n"
        f"[Available Variables]\n{cols_str}\n\n"
        f"[Path to the full dataset]\n{data_path}\n\n"
        "Generate a single Python function with the exact signature:\n"
        "    def run_analysis(data_source, sep=None, na_values=None):\n"
        "- Read the dataset from `data_source` using pandas (CSV). Use `sep` and `na_values` if provided.\n"
        "- Implement preprocessing and the model(s) described (keep models as simple as possible but faithful).\n"
        "- Return either a pandas DataFrame, a dict of DataFrames, or a text summary that captures key results.\n"
        "- Import everything you need inside the code.\n"
        "- The code must run as-is. Output *code only* (no backticks, no commentary).\n"
        "- The code should save the summary and regression tables as human-readable text files in ./output directory.\n"
        "- Never directly copy-paste numbers from the paper; always compute them from the raw data."
    )
    return _llm_call(prompt, model=model, instructions=instructions)


def ask_code_with_data_retry(
    analysis_summary: str,
    df: pd.DataFrame,
    data_path: Union[str, Path],
    data_columns: List[str],
    instruction_summary: str,
    error_feedback: str,
    *,
    model: str,
    sample_rows: int = 5,
) -> str:
    """
    Retry code generation with discrepancy/runtime feedback from previous attempt(s).
    """
    sample_csv = _df_sample_csv(df, data_columns, n=sample_rows)
    cols_str = ", ".join(data_columns)
    instructions = "You fix issues precisely and write simple, correct code."
    prompt = (
        "We previously ran your analysis and found discrepancies or runtime errors. Use the feedback below to correct your code.\n\n"
        f"[Analysis Summary]\n{analysis_summary}\n\n"
        f"[Mapping Instruction]\n{instruction_summary}\n\n"
        f"[Sample Data (first {sample_rows} rows)]\n{sample_csv}\n"
        f"[Available Variables]\n{cols_str}\n\n"
        f"[Path to the full dataset]\n{data_path}\n\n"
        f"[Feedback — MUST FIX]\n{error_feedback}\n\n"
        "Generate a corrected Python function with the exact signature:\n"
        "    def run_analysis(data_source, sep=None, na_values=None):\n"
        "- Use pandas to read the CSV at `data_source` with `sep`/`na_values`.\n"
        "- Ensure variable naming, counts, and model specification align with the paper.\n"
        "- Handle potential indexing errors and missing columns robustly.\n"
        "- Return DataFrame(s) or a clear text summary of results.\n"
        "- Import all required packages.\n"
        "- Output code only (no backticks, no prose).\n"
        "- Never directly copy-paste numbers from the paper; always compute them from the raw data."
    )
    return _llm_call(prompt, model=model, instructions=instructions)


def check_alignment(
    true_result_text: str,
    generated_text: str,
    *,
    model: str,
) -> str:
    """
    Compare generated vs. true results. Return '1' if sufficiently close, else '0'.
    """
    instructions = "Return strictly '1' for close/enough match, or '0' otherwise. No other text."
    prompt = (
        "Compare the two result sets for variable names, counts, coefficients, and standard errors.\n"
        "Each number should match to the second decimal place.\n"
        "If they are sufficiently close, output '1'. Otherwise output '0'. Nothing else.\n\n"
        f"[Generated Results]\n{generated_text}\n\n"
        f"[True Results]\n{true_result_text}"
    )
    return _llm_call(prompt, model=model, instructions=instructions)


def report_discrepancy(
    true_result_text: str,
    generated_text: str,
    *,
    model: str,
    max_tokens: int,
) -> str:
    """
    Produce a human-readable discrepancy report with concrete fixes.
    """
    instructions = "Be a critical reviewer of the quantitative studies in sociology. Identify all discrepancies and how to fix them."
    prompt = (
        "Carefully compare the generated and true results. Identify every mismatch in variable names, coefficients, standard errors, or interpretation. \n"
        "Explain how to fix them so the generated analysis matches.\n\n"
        f"[Generated Results]\n{generated_text}\n\n"
        f"[True Results]\n{true_result_text}"
    )
    return _llm_call(prompt, model=model, instructions=instructions, max_output_tokens=max_tokens)


# ──────────────────────────────────────────────────
#   Main pipeline 
# ──────────────────────────────────────────────────


def main(
    pdf_path: Union[str, Path],
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
    Orchestrates the full loop and writes intermediate artifacts to `output_dir`.
    
    Output files created:
    - pdf_summary.txt: Summary of the paper PDF
    - instruction_summary.txt: Mapping from codebook to variables
    - generated_analysis.py: Initial generated code
    - generated_analysis_retry_{N}.py: Retry code for attempt N
    - generated_results_attempt_{N}.txt: Tables and results for each trial (text format)
    - runtime_error_attempt_{N}.txt: Runtime errors if any
    - discrepancy_report_attempt_{N}.txt: Comparison reports
    
    Returns: "1" on success (aligned), "0" otherwise.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # (1) Summarize the paper PDF
    summary = summarize_pdf(pdf_path, model=summary_model, max_tokens=max_tokens)
    summary_file = output_dir / "pdf_summary.txt"
    summary_file.write_text(summary, encoding="utf-8")
    print(f"[OK] Saved summary → {summary_file}")

    # (2) Load the dataset (also detects the delimiter for later use)
    df = load_data(data_path)
    cols = list(df.columns)
    print(f"[OK] Loaded data with {len(cols)} columns")

    # (3) Summarize instruction/codebook CSV
    instruction = summarize_instruction(summary, instruction_path, model=summary_model, max_tokens=max_tokens)
    instruction_file = output_dir / "instruction_summary.txt"
    instruction_file.write_text(instruction, encoding="utf-8")
    print(f"[OK] Saved instruction summary → {instruction_file}")

    # (4) Extract the 'true' results from the paper for comparison
    true_result_text = extract_result_pdf(pdf_path, model=summary_model, max_tokens=max_tokens)
    print("[True Results Extracted]\n", true_result_text[:5000], sep="")

    # (5) Ask for initial analysis code
    code = ask_code_with_data(summary, df, data_path, cols, instruction, model=code_model)
    code_file = output_dir / "generated_analysis.py"
    code_file.write_text(code, encoding="utf-8")
    print(f"[OK] Saved generated code → {code_file}")

    # (6) Run the generated code (safe wrapper)
    generated_text, runtime_error = try_run_generated_code(code_file, data_path, sep=None, na_values=None)

    attempt = 1
    accumulated_feedback = ""
    if runtime_error:
        # Save runtime error and mark as failure for this attempt
        err_file = output_dir / f"runtime_error_attempt_{attempt}.txt"
        err_file.write_text(runtime_error, encoding="utf-8")
        print(f"[Runtime Error] Saved traceback → {err_file}")
        success = "0"
        # Use the runtime error as feedback for next attempt
        accumulated_feedback = runtime_error
        generated_text = ""  # nothing to compare
    else:
        # (7) Check alignment
        assert generated_text is not None
        print("[Generated Summary]\n", generated_text[:5000], sep="")
        # Save generated summary to text file
        result_file = output_dir / f"generated_results_attempt_{attempt}.txt"
        result_file.write_text(generated_text, encoding="utf-8")
        print(f"[OK] Saved generated results → {result_file}")
        success = check_alignment(true_result_text, generated_text, model=summary_model).strip()
        print(f"[Check {attempt}/{max_attempts}] Success: {success}")

    # (8) Iterate until success == "1" or attempts exhausted
    while success != "1" and attempt < max_attempts:
        attempt += 1
        print(f"[Retry Loop] attempt {attempt}/{max_attempts}")

        # If we have actual generated text, create discrepancy report; otherwise carry runtime feedback forward.
        if generated_text:
            discrepancy = report_discrepancy(true_result_text, generated_text, model=summary_model, max_tokens=max_tokens)
            accumulated_feedback = discrepancy if not accumulated_feedback else f"{accumulated_feedback}\n\n---\n{discrepancy}"
            err_file = output_dir / f"discrepancy_report_attempt_{attempt-1}.txt"
            err_file.write_text(discrepancy, encoding="utf-8")
            print(f"[Mismatch] Saved discrepancy report → {err_file}")
        else:
            # No generated result because runtime error occurred; accumulated_feedback already contains traceback.
            print("[Info] Skipping discrepancy report due to previous runtime error.")

        # Ask for a corrected version of the analysis code using accumulated feedback (discrepancy and/or runtime error)
        revised_code = ask_code_with_data_retry(
            summary, df, data_path, cols, instruction, accumulated_feedback, model=code_model
        )
        revised_code_path = output_dir / f"generated_analysis_retry_{attempt}.py"
        revised_code_path.write_text(revised_code, encoding="utf-8")
        print(f"[Retry] Saved revised code → {revised_code_path}")

        # Run revised code safely
        generated_text, runtime_error = try_run_generated_code(revised_code_path, data_path, sep=None, na_values=None)
        if runtime_error:
            # Save runtime error and continue to next attempt (no crash)
            err_file = output_dir / f"runtime_error_attempt_{attempt}.txt"
            err_file.write_text(runtime_error, encoding="utf-8")
            print(f"[Runtime Error] Saved traceback → {err_file}")
            success = "0"
            # feed error into accumulated feedback
            accumulated_feedback = f"{accumulated_feedback}\n\n--- RUNTIME ERROR ---\n{runtime_error}" if accumulated_feedback else runtime_error
            generated_text = ""  # clear so we don't try to compare this round
            continue  # move to the next attempt

        # If we reached here, we have a generated result; check alignment
        assert generated_text is not None
        print(f"[Retry Generated Summary] (attempt {attempt})\n{generated_text[:5000]}")
        # Save generated summary to text file
        result_file = output_dir / f"generated_results_attempt_{attempt}.txt"
        result_file.write_text(generated_text, encoding="utf-8")
        print(f"[OK] Saved generated results → {result_file}")
        success = check_alignment(true_result_text, generated_text, model=summary_model).strip()
        print(f"[Check {attempt}/{max_attempts}] Success: {success}")

    return success


# ──────────────────────────────────────────────────
#   Main entry point
# ──────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replicate a paper from PDF + Codebook + CSV using LLM assistance.")
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
    parser.add_argument("--attempts", dest="max_attempts", type=int, default=25,
                        help="Maximum number of retry iterations.")
    parser.add_argument("--max-tokens", dest="max_tokens", type=int, default=20000,
                        help="Maximum output tokens for LLM responses.")

    args = parser.parse_args()

    result = main(
        pdf_path=args.pdf_path,
        data_path=args.data_path,
        instruction_path=args.instruction_path,
        output_dir=args.output_dir,
        summary_model=args.summary_model,
        code_model=args.code_model,
        max_attempts=args.max_attempts,
        max_tokens=args.max_tokens,
    )
    print(f"\n=== Final result: {result} ===")
