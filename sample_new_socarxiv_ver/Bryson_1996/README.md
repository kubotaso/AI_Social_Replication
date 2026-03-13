# LLM-Assisted Replication: Bryson (1996) "'Anything But Heavy Metal'"

This repository contains code and outputs from an automated replication of:

> Bryson, Bethany (1996). "'Anything But Heavy Metal': Symbolic Exclusion and Musical Dislikes." *American Sociological Review*, 61(5), 884–899.

The replication was performed by Claude Code (an AI coding agent) following the instructions in `CLAUDE.md`.

## Required Files Not Included

The following files are not included in this repository due to copyright restrictions. You must obtain them yourself and place them in this directory before running the replication.

### Paper

**`Bryson_paper.pdf`**

- **JSTOR**: <https://www.jstor.org/stable/2096460>
- **DOI**: `10.2307/2096460`

Save the PDF as `Bryson_paper.pdf` in this directory.

### Figure screenshots

**`Fig1.jpg`** — Screenshot of the original figure from the paper, used for visual comparison during replication. Take a screenshot of Figure 1 from the PDF and save it as `Fig1.jpg` in this directory.

## Data

`gss1993.csv` contains variables extracted from the 1993 General Social Survey (GSS). The GSS is publicly available from [NORC at the University of Chicago](https://gss.norc.org/) and is distributed free of charge. The extraction script (`download_data.R` or equivalent) can regenerate this file from the `gssr` R package if needed.

## How to Run

See `CLAUDE.md` for full instructions. In brief:

1. Place `Bryson_paper.pdf` and figure screenshot in this directory
2. Set up the Python environment: `python3 -m venv .venv && .venv/bin/pip install pandas numpy statsmodels scipy matplotlib`
3. Start Claude Code and say `"Replicate all"`

> **Tip:** A `settings.json` with pre-configured permissions is provided in this directory. Copy it to your `.claude/` folder to avoid repeated permission prompts during the replication.

## Results

See `replication_summary.md` for the full replication report with scores, comparisons, and methodology notes.
