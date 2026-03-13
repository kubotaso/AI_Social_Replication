# LLM-Assisted Replication: Bernanke and Blinder (1992) "The Federal Funds Rate and the Channels of Monetary Transmission"

This repository contains code and outputs from an automated replication of:

> Bernanke, Ben S. and Alan S. Blinder (1992). "The Federal Funds Rate and the Channels of Monetary Transmission." *The American Economic Review*, 82(4), 901–921.

The replication was performed by Claude Code (an AI coding agent) following the instructions in `CLAUDE.md`.

## Required Files Not Included

The following files are not included in this repository due to copyright restrictions. You must obtain them yourself and place them in this directory before running the replication.

### Paper

**`Bernanke-FederalFundsRate-1992.pdf`**

- **JSTOR**: <https://www.jstor.org/stable/2117350>
- **DOI**: `10.2307/2117350`

Save the PDF as `Bernanke-FederalFundsRate-1992.pdf` in this directory.

### Figure screenshots

**`Figure1.png`**, **`Figure2.png`**, **`Figure4.png`** — Screenshots of the original figures from the paper, used for visual comparison during replication. Take screenshots of each figure from the PDF and save them with the corresponding filenames in this directory. (Figure 3 is excluded from replication.)

## How to Run

See `CLAUDE.md` for full instructions. In brief:

1. Place `Bernanke-FederalFundsRate-1992.pdf` and figure screenshots in this directory
3. Set up the Python environment: `python3 -m venv .venv && .venv/bin/pip install pandas numpy statsmodels scipy matplotlib pandas-datareader`
4. Start Claude Code and say `"Replicate all"`

## Data

The data is automatically downloaded by the AI agent during Step 1 (Data Discovery) from **FRED (Federal Reserve Economic Data)** via `pandas-datareader` (no API key needed).

> **Tip:** A `settings.json` with pre-configured permissions is provided in this directory. Copy it to your `.claude/` folder to avoid repeated permission prompts during the replication.

## Results

See `replication_summary.md` for the full replication report with scores, comparisons, and methodology notes.
