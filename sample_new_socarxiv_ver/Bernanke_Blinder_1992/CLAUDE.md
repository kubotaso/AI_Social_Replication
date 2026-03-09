# LLM-Assisted Paper Replication: Bernanke and Blinder (1992)

## Project Overview

This project uses Claude Code as a local AI agent to replicate all statistical tables and figures from **Bernanke, Ben S. and Alan S. Blinder (1992). "The Federal Funds Rate and the Channels of Monetary Transmission." _The American Economic Review_, 82(4), 901–921.**

The workflow has three high-level steps:

1. **Data Discovery & Acquisition** — AI reads the paper, identifies required time series, and obtains or requests them
2. **Data Validation** — AI runs a few exploratory replication attempts to verify the data is sufficient
3. **Full Replication** — Iterative replication with scoring, discrepancy analysis, and refinement

**No API keys are needed.** Claude Code itself is the AI agent — it reads files directly, generates code, and executes it locally.

---

## Required Input Files

Place these files in the same directory as this CLAUDE.md:

| File | Description |
|------|-------------|
| `Bernanke-FederalFundsRate-1992.pdf` | The paper to replicate |
| `Figure1.png` | Screenshot of the paper's Figure 1 |
| `Figure2.png` | Screenshot of the paper's Figure 2 |
| `Figure4.png` | Screenshot of the paper's Figure 4 |

The AI agent determines what time-series data is needed and either downloads it or tells you how to get it.

### Replication Targets

The following 9 targets must be replicated (in this order):

| Target | Output Directory |
|--------|-----------------|
| Table 1 | `output_table1/` |
| Table 2 | `output_table2/` |
| Table 3 | `output_table3/` |
| Table 4 | `output_table4/` |
| Table 5 | `output_table5/` |
| Table 6 | `output_table6/` |
| Figure 1 | `output_figure1/` |
| Figure 2 | `output_figure2/` |
| Figure 4 | `output_figure4/` |

Figure 3 is excluded (no reference image available).

### Using the PNG reference images for figure replication

When replicating figures, if original figure screenshots are present in the project root, the AI agent **must** read the corresponding PNG/JPG file using the Read tool to visually inspect the original figure before generating code. This allows precise matching of:
- Axis ranges, tick marks, and labels
- Line styles (solid, dashed, dotted) and line weights
- Legend placement and text
- Overall layout (single panel vs. multi-panel)
- Annotation placement and styling
- Font sizes and styles
- Y-axis scaling notation

The agent should re-read the PNG after each attempt to compare the generated figure against the original and identify specific visual discrepancies in the discrepancy report. The scoring rubric for figures should be applied by direct visual comparison with the original PNG, not just from numerical values in `figure_summary.txt`.

---

## How to Use

### Setup (do this once before your first run)

1. **Obtain the paper.** Save the paper as a PDF in this directory.
2. **Set up the Python environment.** Create a virtual environment and install the base packages:
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install pandas numpy statsmodels scipy matplotlib
   ```
   The AI agent may install additional packages as needed during the workflow (e.g., `pandas-datareader`, `fredapi`, `wbdata` for specific data sources).
3. **Install Claude Code.** Follow the instructions at https://docs.anthropic.com/en/docs/claude-code to install Claude Code. 

### Running a replication

1. Start Claude Code in this directory
2. Tell it what to replicate:
   - `"Replicate Table 1"` — a single target
   - `"Replicate all"` — all targets, sequential mode (one at a time)
   - `"Replicate all in parallel"` — all targets, parallel mode (concurrent subagents)
3. Claude Code follows the 3-step workflow below autonomously
4. All artifacts are saved to target-specific output directories (e.g., `output_table1/`)

To resume an interrupted session: say `"Resume replication"` — Claude Code reads `state.json` files from the output directories and continues from where it left off.

---

## Step 1: Data Discovery & Acquisition

The AI agent reads the paper and autonomously determines what time-series data is needed, then acquires it.

### Phase 1A: Paper Analysis

0. **Record the workflow start time** (run `date "+%Y-%m-%d %H:%M:%S"`) and save it to `workflow_start_time.txt` in the project root. This marks the beginning of the entire replication workflow and is used for wall-clock elapsed time calculation.

1. **Read the paper thoroughly.** Identify:
   - What time series the paper uses (variable names, descriptions, units)
   - The sample period (start and end dates, any sub-period analyses)
   - Data frequency (daily, weekly, monthly, quarterly, annual)
   - Data sources cited in the paper (central banks, statistical agencies, commercial databases, etc.)
   - Any data transformations (log levels, first differences, growth rates, seasonal adjustment, deflation by price index, annualization)
   - Model specification details: lag length, variable ordering, identification strategy, estimation method
   - Any supplementary data (instruments, control variables, exogenous variables)

2. **Identify all replication targets** — every table and figure in the paper. Record:
   - Target name and description
   - The statistical method used (VAR, Granger causality, IRF, FEVD, SVAR, VECM, IV, OLS, etc.)
   - Output directories to create (e.g., `output_table1/`, `output_figure1/`)

3. **Save findings** to `data_requirements.txt` in the project root. This file must include:
   - Each time series needed with description, source, and any known identifiers (e.g., FRED series codes, database mnemonics, statistical release names)
   - The sample period required
   - Data frequency and seasonal adjustment status
   - Any transformations described in the paper (logs, differences, annualized rates)
   - Source organization for each series

### Phase 1B: Automated Data Acquisition

Attempt to download or construct the required data programmatically. Try these strategies in order:

1. **Python packages** — Many time-series datasets have Python APIs. Examples:
   - `pandas-datareader`: supports FRED, World Bank, Eurostat, OECD, and other sources
   - `fredapi`: direct FRED access (for U.S. macro data)
   - `wbdata` or `wbgapi`: World Bank data
   - `yfinance`: financial market data
   - Custom APIs for central bank data (ECB, BOJ, BOE, etc.)

2. **R packages** — Some data packages exist only in R (e.g., `fredr`, `quantmod`, `rdbnomics`). Write and execute an R script if Python approaches fail.

3. **Direct downloads** — If the data is available from a public URL (e.g., government statistical agency, central bank data portal), download it using `curl` or `wget`.

4. **Web scraping** — As a last resort for publicly available data tables, scrape them programmatically.

**After successful download:**
- Save the combined dataset as a CSV file in the project root
- Ensure proper date indexing at the correct frequency
- Record in `data_requirements.txt` how each series was obtained (source, package, download method, etc.)

### Phase 1C: Manual Acquisition Instructions (if automated download fails)

If automated download is not possible for some series (e.g., historical data that predates online coverage, proprietary databases, or discontinued statistical releases):

1. **Create `data_acquisition_instructions.md`** with clear, step-by-step instructions for the human:
   - Where to go (specific URLs for data repositories)
   - What series to search for (name and/or identifiers)
   - What date range to download
   - What format to select (CSV preferred)
   - Where to save the files and what to name them
   - Any account registration needed
   - Any transformations needed before the data can be used

2. **Notify the human** and pause. Display a clear message:
   ```
   I need you to download the following data manually:
   - [Series name] from [Source]
   See data_acquisition_instructions.md for detailed instructions.
   Please let me know when the files are ready.
   ```

3. **When the human confirms the data is available**, verify the files exist and continue to Step 2.

### Phase 1D: Data Inspection

Once the data is available (whether downloaded automatically or provided by the human):

1. **Verify the data file** — Check date range, frequency, column names, missing values
2. **Verify time-series properties** — Confirm correct frequency, no large gaps, correct units (levels vs. percent vs. index), stationarity properties if relevant
3. **Check for obvious issues** — Missing observations, structural breaks, unit changes, seasonal adjustment status, base-year differences
4. **Verify the sample period matches the paper** — Confirm data coverage for all required sub-periods
5. **Save a brief inspection report** to `data_inspection.txt` summarizing what was found

### Phase 1E: Target Extraction

With the paper already read and data confirmed available, extract the detailed ground-truth information for every replication target (tables and figures).

1. For each target identified in the paper, extract **everything needed to replicate it from scratch**, including:
   - Table/figure structure (which variables, which panels, which columns/rows)
   - Model specification: variables included, ordering, lag length, sample period, estimation method
   - For Granger causality tables: which variable is tested, which equation, test statistics, p-values, significance levels
   - For impulse responses: which shock, which response variable, horizon, confidence bands, identification scheme
   - For variance decompositions: forecast horizons, percentage contributions
   - Statistical methods (unrestricted VAR, structural VAR, Choleski decomposition, long-run restrictions, sign restrictions, etc.)
   - Exact "true" results: test statistics, p-values, impulse response values at key horizons, variance decomposition percentages, coefficient estimates with standard errors
   - Data transformations applied before estimation (log levels, first differences, etc.)
   - Any footnotes or text in the paper that clarify methodology
2. Create output directories: `mkdir -p {output_dir}` for each target
3. Save the extracted information to `{output_dir}/table_summary.txt` (or `figure_summary.txt` for figures)

**Important:** Extract numerical values as precisely as possible. These are the "ground truth" that generated results will be compared against.

**Verification step:** After transcribing all values, re-read the relevant paper page(s) to double-check every number. Transcription errors in `table_summary.txt` cause false scoring penalties and wasted iterations.

**Completeness check:** The `table_summary.txt` (or `figure_summary.txt`) must contain enough detail that the code generation phase can produce correct code without re-reading the paper. Include all variable definitions, model specifications, lag lengths, sample periods, and expected results.

### Phase 1F: Variable Mapping

With the data available from Phase 1D and the target details from Phase 1E, map every variable to the dataset.

1. Read the dataset and its documentation/metadata
2. For each target, map every variable mentioned in the paper to its corresponding column name in the dataset CSV
3. Document for each variable:
   - Paper name and dataset column name
   - Units and scaling (percent, index, billions, etc.)
   - Whether the paper uses the variable in levels, logs, first differences, or growth rates
   - How to construct any derived variables (e.g., real variables deflated by price index, spreads, ratios)
   - Seasonal adjustment status (SA vs. NSA) — match the paper's specification
4. For composite or derived variables, document the full construction recipe
5. Cross-check: verify that every variable in `table_summary.txt`/`figure_summary.txt` has a corresponding dataset mapping
6. Save to `{output_dir}/instruction_summary.txt` for each target

---

## Step 2: Data Validation (Exploratory Replication)

Before committing to the full iterative replication, run a lightweight validation to confirm the data and documentation are sufficient.

### Phase 2A: Exploratory Attempts

Run up to **3 exploratory attempts** of the replication. Each attempt follows the same code-generation-and-execution cycle as Step 3 (Phases 1–4), but with relaxed expectations. The `table_summary.txt`/`figure_summary.txt` and `instruction_summary.txt` created in Step 1 (Phases 1E–1F) are already available to guide code generation.

1. Generate and run analysis code (using `table_summary.txt` and `instruction_summary.txt`)
2. Score the results against the paper
3. Write a discrepancy report

Save all artifacts to the target's output directory with the standard naming convention (see Artifact Management).

### Phase 2B: Data Sufficiency Assessment

After 3 exploratory attempts (or fewer if a score >= 70 is reached), evaluate:

1. **Are the required time series present?** Check if all variables mentioned in the paper can be found (directly or via construction) in the dataset.
2. **Does the sample period match?** Compare your date range to the paper's stated sample period. Missing observations or wrong start/end dates cause discrepancies.
3. **Are the results in the right ballpark?** Test statistics, impulse responses, and decompositions don't need to match exactly yet, but signs and rough magnitudes should be correct.
4. **Are there data vintage issues?** Note any suspected vintage effects (see Notes on Replicating Time-Series Papers, item 1).

Save the assessment to `{output_dir}/data_validation_report.txt`.

### Phase 2C: Decision Point

Based on the assessment:

- **Data is sufficient** (variables present, sample period matches, results are directionally correct):
  - Proceed to Step 3 (Full Replication)
  - The exploratory attempts carry over — their attempt numbers, scores, and artifacts continue into Step 3

- **Additional data or documentation is needed:**
  1. Create `additional_data_request.md` explaining:
     - What is missing (specific time series, historical data vintages, or discontinued releases)
     - Why it is needed
     - How to obtain it (specific download instructions, alternative sources)
  2. Notify the human and return to Step 1
  3. When the human provides the additional data, re-run Step 2 from the beginning

---

## Step 3: Full Replication

This is the core iterative replication process. It picks up from where Step 2 left off (exploratory attempts carry over).

### ⚠️ CRITICAL RULES — READ BEFORE STARTING ⚠️

> **THE MOST IMPORTANT RULE IN THIS ENTIRE DOCUMENT:**
> You MUST keep iterating until score >= 95 or 20 attempts are exhausted.
> Do NOT stop at 80. Do NOT stop at 85. Do NOT stop at 92.
> Do NOT summarize results and declare success before reaching 95.
> Do NOT move on to the next target before reaching 95 (or 20 attempts).
> If your score is below 95 and you have attempts remaining, GO BACK TO PHASE 1 AND TRY AGAIN.

1. **NEVER stop iterating on a target until score >= 95 or 20 attempts are exhausted.** A score of 80, 85, or 92 is NOT sufficient. Keep going. Do not declare victory early. Do not "wrap up" early. Do not produce the final report early. The iteration loop is: Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → back to Phase 1. This loop repeats until the exit condition is met. There are only two valid exit conditions: (a) score >= 95, or (b) 20 attempts exhausted.
2. **ALWAYS write a discrepancy report after every attempt** (Phase 4). The discrepancy report drives the next iteration; skipping it means the next attempt will be uninformed.
3. **ALWAYS save all artifacts for every attempt** — results file, discrepancy report, run_log entry, and state.json update. No exceptions.
4. **ALWAYS update best_generated_analysis.py** when a new best score is achieved (see Best Result Tracking under Artifact Management for the full procedure).

### Phase 1: Code Generation

The `table_summary.txt`/`figure_summary.txt` and `instruction_summary.txt` for this target were created in Step 1 (Phases 1E–1F). If they are missing, return to Step 1 to create them before proceeding.

1. **Record the start time** of this attempt (run `date "+%Y-%m-%d %H:%M:%S"`)
2. **On the first attempt only:** create `{output_dir}/state.json` (see Canonical state.json Schema under State Management for the format) and `{output_dir}/run_log.csv` with its header row. If these files already exist from a prior session or from Step 2, append to `run_log.csv` (do not overwrite) and resume from the attempt number recorded in `state.json`.
3. Read `{output_dir}/table_summary.txt` (or `figure_summary.txt`) and `{output_dir}/instruction_summary.txt` to understand the target and variable mappings
4. Read the first 5 rows of the dataset CSV to understand column names and data types
5. Generate a self-contained Python script following the Code Generation Contract below
6. For the first attempt: save as `{output_dir}/generated_analysis.py`
7. For retry attempts: save as `{output_dir}/generated_analysis_retry_{N}.py`

When retrying, provide this context to inform the new code:
- The best attempt so far: read `{output_dir}/best_generated_analysis.py` and its discrepancy report
- The most recent attempt: read the previous attempt's code and its error/discrepancy report
- The best score achieved so far

### Phase 2: Execution

1. Run the generated script: `.venv/bin/python {output_dir}/generated_analysis.py` (or the retry file)
2. If the script produces a **runtime error**:
   - Save the full traceback to `{output_dir}/runtime_error_attempt_{N}.txt`
   - Set the score to 0 and proceed to Phase 4 (the discrepancy report must document the error and how to fix it)
3. If the script runs **successfully**:
   - Read the output
   - Save results to `{output_dir}/generated_results_attempt_{N}.txt`
   - For figures: the script saves an image file; copy it to `{output_dir}/generated_results_attempt_{N}.jpg`

### Phase 3: Scoring (0-100)

Compare the generated results against the true results from `{output_dir}/table_summary.txt` (or `figure_summary.txt`). Assign a score from 0 to 100 using the appropriate rubric below.

**Scoring rubric for Granger causality / F-test tables:**

| Criterion | Points | How to assess |
|-----------|--------|---------------|
| Test statistic values | 30 | Each F-statistic matches within 15% relative error (data vintage effects expected) |
| Significance levels | 30 | Stars (*, **, ***) or p-value ranges match for each test |
| All variable pairs present | 15 | Every test from the paper appears in the output |
| Sample period / N | 15 | Number of observations matches within 5% of the paper |
| Correct lag specification | 10 | Lag length matches the paper's specification |

**Scoring rubric for variance decomposition (FEVD) tables:**

| Criterion | Points | How to assess |
|-----------|--------|---------------|
| Decomposition percentages | 25 | Each percentage matches within 3 percentage points of true value |
| All forecast horizons present | 20 | Every horizon from the paper (e.g., 6, 12, 24, 36, 48 months) appears |
| All variables present | 20 | Every variable's contribution is reported for each horizon |
| Rows sum to 100% | 10 | For each horizon, the decomposition shares sum to 100% within rounding tolerance (for example, 99.5-100.5) |
| Correct variable ordering | 10 | Choleski ordering matches the paper (see Notes item 4) |
| Sample period / N | 15 | Number of observations matches within 5% |

**Scoring rubric for VAR coefficient tables:**

| Criterion | Points | How to assess |
|-----------|--------|---------------|
| Coefficient signs and magnitudes | 35 | Each coefficient matches sign and is within 20% relative error (or 0.05 absolute if the true value is near zero) |
| Significance levels | 25 | Stars or t-statistics match for each coefficient |
| Sample size (N) | 15 | Number of observations matches within 5% |
| All variables / lags present | 15 | Every regressor and lag from the paper appears |
| R-squared / fit statistics | 10 | Equation-level fit measures match within 0.05 |

**Scoring rubric for reaction function / OLS estimation tables:**

| Criterion | Points | How to assess |
|-----------|--------|---------------|
| Coefficient signs and magnitudes | 30 | Each coefficient matches sign and is within 20% relative error (or 0.05 absolute if the true value is near zero) |
| Significance levels | 25 | Stars (*, **, ***) or test statistics / p-value ranges match for each coefficient |
| Sample size (N) | 15 | Generated N is within 5% of true N |
| All variables present | 15 | Every regressor from the paper appears in the output |
| R-squared / fit statistics | 15 | Goodness-of-fit measures match within 0.03 |

*Note:* This rubric also applies to latent-variable / ordered probit models (e.g., Table 5 Panel B). For such models, compare estimated thresholds/cutpoints, z-statistics or p-value ranges, and pseudo-R² in place of standard OLS t-statistics and R².

**Scoring rubric for impulse response / time-series figures:**

*Note:* When impulse responses are presented as a table of numerical values rather than a figure (e.g., Table 6), use this exact 100-point scheme instead of the visual rubric below: 15 points for all response variables / horizons present, 25 points for response shape and sign, and 60 points for data values accuracy. Omit axis-label, confidence-band, and layout scoring entirely for such table-based IRF outputs.

| Criterion | Points | How to assess |
|-----------|--------|---------------|
| Plot type and data series | 15 | Correct plot type (line plot), all response variables present |
| Response shape and sign | 25 | Responses show correct sign, timing of peak/trough, and general shape |
| Data values accuracy | 25 | Response magnitudes at key horizons match within 20% relative error (or 0.005 absolute if the true response is near zero) |
| Axis labels, ranges, scales | 15 | Axis ranges, horizon labels, and units match the paper |
| Confidence bands / error bands | 10 | Error bands present and roughly correct width (if shown in original) |
| Overall layout and appearance | 10 | Multi-panel layout, line styles, font sizes approximate the paper |

**Scoring rubric for other figures (time-series plots, scatter plots, etc.):**

| Criterion | Points | How to assess |
|-----------|--------|---------------|
| Plot type and data series | 20 | Correct plot type, all data series present |
| Data values accuracy | 25 | Computed values match within 15% relative error at key points |
| Axis labels, ranges, scales | 15 | Axis ranges, tick format, and label text match the paper |
| Key features reproduced | 15 | Peaks, troughs, trend breaks, and date range match the paper |
| Visual elements (annotations, reference lines) | 15 | Labels, annotations, recession shading, and reference lines present as in the paper |
| Overall layout and appearance | 10 | Line weights, styles, font sizes approximate the paper |

**Scoring rules — be conservative:**

Since Claude Code both generates the code and scores the results, there is a risk of inflated scores. Follow these guidelines:

- **Be conservative.** When in doubt, give a lower score — it is better to do another iteration with specific fixes than to declare premature success.
- **Compute scores numerically using the rubric, not impressionistically.** Compare each statistic, each coefficient, each significance level individually.
- **Data vintage effects are expected.** See Notes on Replicating Time-Series Papers (item 1) for details. Note suspected vintage effects rather than treating them as code errors, but general patterns (significance, signs) should still match.
- **Significance matters.** A test statistic that is significant at 1% in the paper but insignificant in your output is a real discrepancy.
- **Sample size differences indicate data issues.** If your number of observations differs by more than 5%, check your sample period and data frequency.
- **Partial credit is fine.** If most results match but a few are off, score proportionally.
- **For figures:** Compare the shape, sign, and magnitude of responses at key horizons — not just whether the figure "looks similar."
- A score of 95+ means the replication is essentially correct with only cosmetic or data-vintage differences.
- A score of 80-94 means the core results are right but some details differ.
- A score below 50 means fundamental issues (wrong model specification, missing variables, wrong sample period).

**Decision:** Proceed to Phase 4 regardless of the score. **Reminder: A score below 95 means you MUST continue iterating. Do not stop here.**

### Phase 4: Discrepancy Report and Bookkeeping — MANDATORY

This phase runs after **every** attempt, regardless of score.

1. Write a detailed report comparing generated results vs. true results
2. For each discrepancy, identify:
   - What is wrong (e.g., "F-statistic for X → Y is 3.42 but should be 4.17")
   - Why it might be wrong (e.g., "Data vintage difference" or "Wrong lag length" or "Missing variable")
   - How to fix it (e.g., "Try 6 lags instead of 12" or "Add log CPI to the model" or "Use first differences instead of levels")
3. For figures specifically, include:
   - Which responses have wrong sign or shape
   - Which horizons or time periods show the largest discrepancies
   - Whether the identification scheme / variable ordering matches the paper
   - Specific plotting fixes to try
4. Save to `{output_dir}/discrepancy_report_attempt_{N}.txt`
5. **If the score is the new best**, follow the Best Result Tracking procedure under Artifact Management
6. **Record the end time** (run `date "+%Y-%m-%d %H:%M:%S"`), compute the duration, and append a row to `{output_dir}/run_log.csv`
7. **Update `{output_dir}/state.json`** with the new attempt count, score, duration, and best score

### Phase 5: Retry — DO NOT SKIP THIS PHASE

> **CHECK BEFORE PROCEEDING:** Is your best score >= 95? If NO, you MUST return to Phase 1. No exceptions. Do not produce a final report. Do not move to the next target. Go back to Phase 1 now.

If the score is below 95 and fewer than 20 attempts have been made, **you MUST return to Phase 1** to generate improved code. Do not stop. Do not summarize. Do not move on. The retry context (best attempt, most recent attempt, discrepancy reports) is documented in Phase 1's "When retrying" block.

If the score is 95 or above, set `status` to `"completed"` in `state.json` and stop iterating on this target. If the best score has not improved for 5 or more consecutive attempts, set `status` to `"plateau"` and write a diagnostic report explaining why progress has stalled, then **continue iterating** with a different strategy. If 20 attempts are exhausted without reaching 95, set `status` to `"max_attempts_reached"` and move on.

---

## Execution Modes

When the user says **"Replicate all"**, the agent replicates all targets. Two execution modes are available: **sequential** (default) and **parallel**.

- **Sequential mode** (default): The main agent handles all targets one by one.
- **Parallel mode**: The main agent launches one subagent per target so that multiple tables/figures are replicated concurrently. Use this mode when the user says **"Replicate all in parallel"** or **"Replicate all targets in parallel"**.

### Sequential mode — How it works

1. **Step 1 (Data Discovery)** runs once at the start for the entire paper — identify all time series needed and acquire them.
2. **Step 2 (Data Validation)** runs once — pick one representative target and do 3 exploratory attempts to validate the data.
3. **Step 3 (Full Replication)** runs for each target in order:
   - The `table_summary.txt`/`figure_summary.txt` and `instruction_summary.txt` for all targets were already created in Step 1 (Phases 1E–1F).
   - For each target in the order determined during Phase 1A:
     - Run Phases 1–5 (Code Generation → Execution → Scoring → Discrepancy → Retry)
     - `state.json` and `run_log.csv` are updated automatically in Phase 4
4. **After all targets complete**, produce the combined summary.

### Parallel mode — How it works

Steps 1 and 2 are **always sequential** (they set up shared data that every target needs). Parallelism applies only to Step 3.

1. **Step 1 (Data Discovery)** — same as sequential mode. Runs once, produces the shared dataset CSV and all per-target `table_summary.txt`/`figure_summary.txt` and `instruction_summary.txt` files.
2. **Step 2 (Data Validation)** — same as sequential mode. Runs once on one representative target.
3. **Step 3 (Full Replication) — parallel dispatch:**
   - The main agent launches **one subagent per target** using the `runSubagent` tool.
   - All subagents run concurrently (call `runSubagent` for every target in the same tool-call block so they execute in parallel).
   - Each subagent receives a prompt containing:
     - The target name and output directory
     - The path to the dataset CSV
     - The paths to `table_summary.txt` (or `figure_summary.txt`) and `instruction_summary.txt`
     - The full Step 3 instructions (Phases 1–5, Critical Rules, scoring rubrics, artifact management, Code Generation Contract)
     - Instruction to return a JSON summary: `{"target", "best_score", "best_attempt", "total_attempts", "status", "duration_seconds"}`
   - Each subagent independently iterates Phases 1–5 until score ≥ 95 or 20 attempts are exhausted.
   - Each subagent writes all artifacts to its own `{output_dir}/` — no cross-contamination.
4. **After all subagents return**, the main agent **verifies** each target before producing the final report:
   - Read `{output_dir}/state.json` for each target
   - If `best_score < 95` AND `current_attempt < 20`, the subagent stopped prematurely — **re-launch** a new subagent for that target with resume instructions (it will read `state.json` and `run_log.csv` to continue from where it left off)
   - Repeat verification after each re-launched subagent returns
   - Only proceed to the Final Report once every target has either `best_score >= 95` or `current_attempt >= 20`
   - Once all targets are verified, collect the JSON summaries from each subagent
   - Read `state.json` and `run_log.csv` from each output directory
   - Produce the combined Final Report and `run_log_all.csv`

### Parallel mode — Important rules

- **Shared data is read-only during Step 3.** The dataset CSV and `data_requirements.txt` must not be modified by any subagent. Each subagent only writes to its own `{output_dir}/`.
- **No inter-target dependencies.** Each target's replication is fully independent. A subagent must never read from or write to another target's output directory.
- **Use `.venv/bin/python` for all scripts.** The virtual environment is shared (read-only) across subagents.
- **Error isolation.** If one subagent fails or hits max attempts, the others continue unaffected. The main agent reports the failure in the Final Report.
- **The stopping criterion from Critical Rule #1 applies independently to each subagent/target.**

### Subagent prompt template (parallel mode)

When launching a subagent for a target, the main agent must include this information in the prompt:

```
You are replicating {target_name} from Bernanke and Blinder (1992).

Output directory: {output_dir}/
Dataset: {dataset_csv}
Target specification: {output_dir}/table_summary.txt (or figure_summary.txt)
Variable mapping: {output_dir}/instruction_summary.txt

[Insert full Step 3 instructions here: Phases 1–5, Critical Rules, scoring rubrics,
 Code Generation Contract, artifact management rules, time tracking format]

When finished, return a JSON summary:
{
  "target": "{target_name}",
  "best_score": <int>,
  "best_attempt": <int>,
  "total_attempts": <int>,
  "status": "completed|plateau|max_attempts_reached",
  "duration_seconds": <int>
}
```

### Important rules (both modes)

- Each target gets its **own output directory** — no cross-contamination.
- Use `.venv/bin/python` to run all scripts.
- The stopping criterion from Critical Rule #1 applies independently to each target.

### Final Report

After all targets are complete (or max attempts exhausted), create `replication_summary.md` in the **project root** as a comprehensive markdown report. This single file should contain all of the following sections:

1. **Overview** — Best score per target, best attempt number, total attempts, total duration, completion status. Include:

   **Time metrics** (compute from `workflow_start_time.txt` and `run_log_all.csv`):
   - **Wall-clock elapsed time**: difference between the `workflow_start_time` (recorded in `workflow_start_time.txt` at the beginning of Step 1) and the latest `end_time` across all rows in `run_log_all.csv`. This covers the entire replication workflow — including Step 1 (paper analysis, data acquisition, target extraction) and Steps 2–3 (validation and iterative replication).
   - **Step 1 duration**: difference between `workflow_start_time` and the earliest `start_time` in `run_log_all.csv`. This is the time spent on data discovery and acquisition before any replication attempts began.
   - **Sum of attempt durations**: sum of all `duration_seconds` values. This is the total compute time across all attempts (may exceed wall-clock time when targets run in parallel).

   **Summary table:**

   ```
   | Target   | Best Score | Attempts | Status    | Best Attempt |
   |----------|------------|----------|-----------|--------------|
   | Table 1  | 97/100     | 6        | completed | 6            |
   | Table 2  | 95/100     | 8        | completed | 8            |
   | Figure 1 | 95/100     | 4        | completed | 4            |
   ```

   *Note: The scores above are illustrative. Your results will vary.*

2. **Results Comparison** — For each target, a side-by-side comparison of original vs. replicated values. For test-statistic tables: F-statistics and significance levels. For VAR/regression tables: coefficient estimates, variance decompositions, or fit statistics. For figures: response magnitudes at key horizons or other quantitative comparisons.

3. **Figure Comparison** — For each figure target, include a visual side-by-side comparison showing the original figure from the paper and the best replicated figure. Use an HTML table to place them next to each other:

   ```markdown
   ### Figure X: [Title]

   <table><tr>
   <td><strong>Original (from paper)</strong><br><img src="FigureX.png" alt="Original Figure X" width="400"></td>
   <td><strong>Replicated (best attempt)</strong><br><img src="output_figureX/best_generated_figure.jpg" alt="Replicated Figure X" width="400"></td>
   </tr></table>
   ```

   Below each pair, include a brief narrative noting key visual similarities and differences.

4. **Scoring Breakdown** — Points per rubric component with detail on which items matched and which did not.

5. **Best Configuration** — Key methodological choices that produced the best result: model specification, lag length, variable ordering, sample period, data transformations, and any adjustments that improved the score.

6. **Score Progression** — Table showing all attempts with score and key strategy per attempt, plus narrative description of methodological phases and what was learned.

7. **Article vs. Replication: Detailed Comparison** — See the "Article vs. Replication Comparison" section below for required content. Pay special attention to data vintage effects for older papers.

8. **Environment** — Collect system and software information by running appropriate commands (`sw_vers`, `sysctl`, `uname -m`, `.venv/bin/python --version`, library versions). Present as a markdown table:

   | Component | Detail |
   |---|---|
   | AI Agent | model name and ID |
   | Interface | Claude Code (CLI), running in VSCode extension |
   | Date | run date |
   | Machine | architecture |
   | CPU | CPU model |
   | RAM | RAM in GB |
   | OS | OS version (build) |
   | Kernel | kernel version |
   | Python | version |
   | pandas | version |
   | numpy | version |
   | statsmodels | version |
   | matplotlib | version |
   | scipy | version |

9. **Combined run log and time summary** — Read `run_log.csv` from each output directory and create a combined `run_log_all.csv` in the project root. Same columns as `run_log.csv` (see Time Tracking) with a `target` column prepended.

   After creating `run_log_all.csv`, compute and report the following time metrics:
   - **Wall-clock elapsed time** = max(`end_time`) − `workflow_start_time` (from `workflow_start_time.txt`). This covers all steps including Step 1.
   - **Step 1 duration** = min(`start_time`) − `workflow_start_time`. This is the time spent on paper analysis, data acquisition, and target extraction before any replication attempts began.
   - **Sum of attempt durations** = sum of all `duration_seconds` values
   - Per-target duration breakdown (sum of `duration_seconds` grouped by `target`)

---

## Time Tracking

Every attempt must be timed. Maintain `{output_dir}/run_log.csv` as a cumulative log of all attempts across all sessions.

Three time metrics are tracked:
- **Wall-clock elapsed time**: real-world time from the workflow start (beginning of Step 1) to the last attempt's end (computed from `workflow_start_time.txt` and `run_log_all.csv` in the Final Report). This covers all three steps: data discovery, validation, and full replication.
- **Step 1 duration**: time from workflow start to the first replication attempt. This captures the data discovery and acquisition phase.
- **Sum of attempt durations**: total of all individual `duration_seconds` values. Represents cumulative compute effort across all attempts and targets.

### run_log.csv format

```csv
attempt,start_time,end_time,duration_seconds,score,result,code_file
1,2025-06-01 14:30:00,2025-06-01 14:32:15,135,59,discrepancy,generated_analysis.py
2,2025-06-01 14:32:15,2025-06-01 14:35:42,207,72,discrepancy,generated_analysis_retry_2.py
3,2025-06-01 14:35:42,2025-06-01 14:38:10,148,70,discrepancy,generated_analysis_retry_3.py
```

Column definitions:
| Column | Description |
|--------|-------------|
| `attempt` | Attempt number (1, 2, 3, ...) |
| `start_time` | ISO-like timestamp when the attempt started (beginning of code generation) |
| `end_time` | Timestamp when the attempt ended (after scoring/discrepancy report) |
| `duration_seconds` | Wall-clock seconds from start to end of this attempt |
| `score` | Alignment score (0-100), or `error` if runtime error |
| `result` | One of: `success` (score >= 95), `discrepancy` (score < 95), `runtime_error` |
| `code_file` | Filename of the generated code for this attempt |

### How to record time

Record timestamps using `date "+%Y-%m-%d %H:%M:%S"` at the start of each attempt (Phase 1 step 1) and at the end (Phase 4 step 6). Compute `duration_seconds` as the difference.

---

## Artifact Management

All artifacts are saved to `{output_dir}/` (e.g., `output_table1/` for Table 1):

**Step 1 artifacts (project root and per-target output directories):**

| File | Description |
|------|-------------|
| `data_requirements.txt` | What data the paper needs and how it was obtained |
| `data_acquisition_instructions.md` | Instructions for human to download data (if automated download failed) |
| `data_inspection.txt` | Summary of data file inspection results |
| `{output_dir}/table_summary.txt` | Extracted paper info (tables) — created in Phase 1E |
| `{output_dir}/figure_summary.txt` | Extracted paper info (figures) — created in Phase 1E |
| `{output_dir}/instruction_summary.txt` | Variable mapping from paper to dataset — created in Phase 1F |

**Step 2 artifacts (project root or output directory):**

| File | Description |
|------|-------------|
| `{output_dir}/data_validation_report.txt` | Assessment of data sufficiency after exploratory attempts |
| `additional_data_request.md` | Request for additional data/documentation (if needed) |

**Step 3 artifacts (per-target output directory):**

| File | Description |
|------|-------------|
| `generated_analysis.py` | First attempt Python code |
| `generated_analysis_retry_{N}.py` | Retry attempt N Python code |
| `generated_results_attempt_{N}.txt` | Results output from attempt N |
| `generated_results_attempt_{N}.jpg` | Generated figure from attempt N (figures only) |
| `runtime_error_attempt_{N}.txt` | Python traceback if code failed |
| `discrepancy_report_attempt_{N}.txt` | **MANDATORY** comparison report for attempt N |
| `best_generated_analysis.py` | Copy of highest-scoring code |
| `best_generated_results.txt` | Results from best attempt |
| `best_generated_figure.jpg` | Best generated figure (figures only) |
| `best_result_metadata.txt` | Metadata: attempt number, score, source file |
| `state.json` | Session state for resumability |
| `run_log.csv` | Cumulative time log for all attempts |

**Project-root files (after all targets complete):**

| File | Description |
|------|-------------|
| `workflow_start_time.txt` | Timestamp recorded at the start of Step 1 (for wall-clock calculation) |
| `replication_summary.md` | Comprehensive final report (see Final Report section) |
| `run_log_all.csv` | Combined time log across all targets (adds `target` column) |

**Best result tracking:** Whenever a new attempt achieves a higher score than the previous best:
1. Copy the code to `best_generated_analysis.py`
2. Save results to `best_generated_results.txt`
3. For figures: copy the figure to `best_generated_figure.jpg`
4. Write `best_result_metadata.txt` with the attempt number and score:
   ```
   Best result from attempt: {N}
   Score: {score}/100
   Original code file: {filename}
   ```

---

## State Management and Resume Protocol

### Canonical state.json schema

The authoritative format for `state.json` (first created in Step 3 Phase 1 step 2, updated in Phase 4 step 7):

```json
{
  "target": "table1",
  "current_attempt": 3,
  "best_score": 72,
  "best_attempt": 2,
  "status": "in_progress",
  "total_duration_seconds": 490,
  "attempts": [
    {"attempt": 1, "score": 59, "duration_seconds": 135},
    {"attempt": 2, "score": 72, "duration_seconds": 207},
    {"attempt": 3, "score": 70, "duration_seconds": 148}
  ]
}
```

**When resuming a session:**
1. Check if `data_requirements.txt` exists — if not, start from Step 1 (record `workflow_start_time.txt` first)
2. Check if the dataset CSV exists — if not, resume Step 1 (Phase 1B or 1C)
3. Check if `{output_dir}/table_summary.txt` (or `figure_summary.txt`) and `{output_dir}/instruction_summary.txt` exist — if not, resume Step 1 (Phase 1E)
4. Check if `{output_dir}/data_validation_report.txt` exists — if not, start Step 2
5. Check if `{output_dir}/state.json` exists — if yes, read it to determine the current attempt number and best score. Also check `{output_dir}/run_log.csv` — append to it (do not overwrite).
6. Read `best_generated_analysis.py` and the most recent discrepancy report
7. Continue from the next attempt number

---

## Article vs. Replication Comparison (Final Report Requirement)

The `replication_summary.md` must include an **"Article vs. Replication: Detailed Comparison"** section. This section should document discrepancies between the article's stated methodology and what was actually necessary to reproduce the results.

The comparison section should cover:

1. **What the article says vs. what was found:** For each methodological element (model specification, variable definitions, sample period, data source), compare the article's description with what the replication encountered. Note where the stated methodology could or could not reproduce the reported results.

2. **Data vintage effects:** Document any cases where current publicly available data produces different results from what the paper reports, likely due to data revisions since original publication. Identify which series appear most affected by revisions.

3. **Information missing from the article:** List all details that were necessary for replication but not provided or underspecified in the paper (e.g., exact data series identifiers, treatment of seasonal adjustment, handling of data transformations, exact start/end dates for estimation).

4. **Contradictions in the article:** Document any cases where different parts of the paper give conflicting information about the same variable or method.

5. **Quantitative comparison:** Report the best score achievable by faithfully following the stated methodology, the magnitude of data-vintage-driven discrepancies, and which results were hardest to match.

---

## Code Generation Contract

Every generated Python script must follow these rules:

1. **Function signature:** `def run_analysis(data_source):` as the main entry point
2. **Self-contained:** All imports at the top of the file (pandas, numpy, statsmodels, etc.)
3. **Data loading:** Read CSV from the `data_source` path using `pd.read_csv()`, parse dates, set DatetimeIndex
4. **Preprocessing:** Implement all data transformations (logs, differences, deflation), handle missing values, set correct sample period
5. **Statistical models:** Run the appropriate time-series analysis (VAR, Granger causality, impulse responses, variance decompositions, or other methods)
6. **Results:** Return results as a string, pandas DataFrame, or dict of DataFrames
7. **Main block:** Include `if __name__ == "__main__":` that calls `run_analysis("<dataset_filename>.csv")` and prints the results
8. **No hardcoded numbers:** Never copy values from the paper. Always compute from the raw data.
9. **Standard libraries only:** Use pandas, numpy, statsmodels, scipy, matplotlib (no exotic dependencies)
10. **Automated scoring:** Include a `score_against_ground_truth()` function that computes the score programmatically. The ground truth values extracted during Step 1 Phase 1E (from `table_summary.txt` or `figure_summary.txt`) should be embedded as Python dictionaries in the script. This ensures consistent, reproducible scoring across all attempts and eliminates the risk of subjective score inflation. The function must use the same tolerance thresholds specified in the scoring rubrics (e.g., 15% relative error for F-statistics, 3 pp for FEVD, 20% relative error for coefficients and IRF values, 5% for sample size). Return both the total score and a per-criterion breakdown. For figures, compute the numeric criteria programmatically and emit a separate manual-review checklist for visual criteria such as layout, line styles, labels, and confidence-band appearance.

**Time-series-specific requirements:**
- Use `statsmodels.tsa.api.VAR` for VAR estimation
- Use `statsmodels.tsa.stattools` for Granger causality tests, unit root tests (ADF, KPSS)
- Set the appropriate frequency on the DatetimeIndex (e.g., `freq='MS'` for monthly, `freq='QS'` for quarterly)
- Handle any NaN values from differencing or lagging properly
- For structural identification: implement the correct scheme (Choleski, long-run restrictions, sign restrictions, etc.)

Example skeleton:
```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt

def run_analysis(data_source):
    df = pd.read_csv(data_source, parse_dates=['date'], index_col='date')
    df.index.freq = 'MS'  # set appropriate frequency
    # ... data transformations (logs, differences) ...
    # ... model estimation ...
    # ... format results ...
    return results_text

if __name__ == "__main__":
    result = run_analysis("dataset.csv")
    print(result)
```

---

## Notes on Replicating Time-Series Papers

1. **Data vintages:** Economic data is frequently revised after initial release. The data available today from public repositories may differ from what the authors used. This is a known challenge — exact replication of point estimates may not be possible, but qualitative patterns (significance, signs, relative magnitudes) should be reproducible.

2. **Seasonal adjustment:** Many data sources offer both SA and NSA versions. Ensure consistency with the paper (checked in Phases 1A, 1D, 1F).

3. **Identification:** Structural identification schemes (Choleski ordering, long-run restrictions, sign restrictions, external instruments) are critical for impulse response results. The paper should specify the ordering or restrictions — follow them exactly.

4. **FEVD ordering sensitivity:** Variance decomposition results depend heavily on the variable ordering in Choleski factorization. If the paper presents multiple orderings (e.g., the same decomposition with a variable placed first vs. last), each ordering must be replicated separately — the results will differ substantially. Pay close attention to the column ordering in the paper's variance decomposition tables, as this reveals the Choleski ordering used.

5. **Lag selection:** The paper typically specifies the lag length used. If not, try common choices (e.g., 4 or 8 for quarterly data, 6 or 12 for monthly data) and compare results. Information criteria (AIC, BIC) can guide the choice.

6. **Sample period sensitivity:** Time-series results can be sensitive to the exact sample period. Match the paper's dates as closely as possible. Sub-period analyses and structural breaks matter.

7. **Unit roots and stationarity:** Check whether the paper estimates in levels or differences. Using the wrong transformation is a common source of large discrepancies.

8. **Software differences:** Different software packages (Stata, EViews, RATS, SAS) may use slightly different defaults for VAR estimation, lag exclusion tests, or impulse response computation. Document any suspected software-specific differences.

---

## Reference

For understanding the approach and expected output formats, see other replication directories in this workspace which may contain:
- Complete worked examples with pre-prepared data
- Example outputs from previous runs

These are optional reference material and are not required for the current replication.
