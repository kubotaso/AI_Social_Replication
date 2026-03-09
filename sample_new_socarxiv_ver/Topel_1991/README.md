# LLM-Assisted Replication: Topel (1991) "Specific Capital, Mobility, and Wages"

This repository contains code and outputs from an automated replication of:

> Topel, Robert (1991). "Specific Capital, Mobility, and Wages: Wages Rise with Job Seniority." *Journal of Political Economy*, 99(1), 145–176.

The replication was performed by Claude Code (an AI coding agent) following the instructions in `CLAUDE.md`.

## Required Files Not Included

The following files are not included in this repository due to copyright or licensing restrictions. You must obtain them yourself and place them in this directory before running the replication.

### Paper

**`Topel-SpecificCapitalMobility-1991.pdf`**

- **JSTOR**: <https://www.jstor.org/stable/2937716>
- **Your institution's library**: Search for the title or DOI
- **DOI**: `10.1086/261744`

Save the PDF as `Topel-SpecificCapitalMobility-1991.pdf` in this directory.

### Data

This replication requires data from the **Panel Study of Income Dynamics (PSID)**, waves 1968–1983, and the **CPS Displaced Workers Survey** (1984, 1986). Both may require free registration to download. The AI agent will instruct you on what to download during the replication workflow.

## How to Run

See `CLAUDE.md` for full instructions. In brief:

1. Place `Topel-SpecificCapitalMobility-1991.pdf` in this directory
2. Set up the Python environment: `python3 -m venv .venv && .venv/bin/pip install pandas numpy statsmodels scipy matplotlib`
3. Start Claude Code and say `"Replicate all"`
4. The AI agent will analyze the paper and instruct you to download the required data during the workflow

## Replication Targets

- **Tables 1–7** and **Table A1** (Appendix): Wage equations, returns to seniority, IV estimates, displaced worker analysis

> **Tip:** A `settings.json` with pre-configured permissions is provided in this directory. Copy it to your `.claude/` folder to avoid repeated permission prompts during the replication.

## Results

See `replication_summary.md` for the full replication report with scores, comparisons, and methodology notes.
