# LLM-Assisted Replication: Bartels (2000) "Partisanship and Voting Behavior"

This repository contains code and outputs from an automated replication of:

> Bartels, Larry M. (2000). "Partisanship and Voting Behavior, 1952–1996." *American Journal of Political Science*, 44(1), 35–50.

The replication was performed by Claude Code (an AI coding agent) following the instructions in `CLAUDE.md`.

## Required Files Not Included

The following files are not included in this repository due to copyright restrictions. You must obtain them yourself and place them in this directory before running the replication.

### Paper

**`Bartels_2000.pdf`**

- **JSTOR**: <https://www.jstor.org/stable/2669282>
- **Your institution's library**: Search for the title or DOI
- **DOI**: `10.2307/2669282`

Save the PDF as `Bartels_2000.pdf` in this directory.

### Figure screenshots

**`Figure1.png`** through **`Figure6.png`** — Screenshots of the original figures from the paper, used for visual comparison during replication. Take screenshots of each figure from the PDF and save them as `Figure1.png`, `Figure2.png`, ..., `Figure6.png` in this directory.

## How to Run

See `CLAUDE.md` for full instructions. In brief:

1. Place `Bartels_2000.pdf` and figure screenshots in this directory
2. Install R and the `anesr` package (for data download)
3. Set up the Python environment: `python3 -m venv .venv && .venv/bin/pip install pandas numpy statsmodels scipy matplotlib`
4. Start Claude Code and say `"Replicate all"`

## Data

The data is automatically downloaded by the AI agent during Step 1 (Data Discovery) via the `anesr` R package. It uses data from the **American National Election Studies (ANES)**, which is publicly available.

> **Tip:** A `settings.json` with pre-configured permissions is provided in this directory. Copy it to your `.claude/` folder to avoid repeated permission prompts during the replication.

## Results

See `replication_summary.md` for the full replication report with scores, comparisons, and methodology notes.
