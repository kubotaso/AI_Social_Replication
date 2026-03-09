# LLM-Assisted Replication: Inglehart & Baker (2000) "Modernization, Cultural Change, and the Persistence of Traditional Values"

This directory contains code and outputs from an automated replication of:

> Inglehart, R. and Baker, W. E. (2000). "Modernization, Cultural Change, and the Persistence of Traditional Values." *American Sociological Review*, 65(1), 19–51.

The replication was performed by Claude Code (an AI coding agent) following the instructions in `CLAUDE.md`.

## Required Files Not Included

The following files are not included due to copyright or licensing restrictions. You must obtain them yourself before running the replication.

### Paper

**`Inglehart_Baker_2000.pdf`** — the paper to replicate.

- **JSTOR**: <https://www.jstor.org/stable/2657288>
- **DOI**: `10.2307/2657288`

Save the PDF as `Inglehart_Baker_2000.pdf` in this directory.

### Figure screenshots

**`Figure1.jpg`** through **`Figure8.jpg`** — Screenshots of the original figures from the paper, used for visual comparison during replication. Take screenshots of each figure from the PDF and save them as `Figure1.jpg`, `Figure2.jpg`, ..., `Figure8.jpg` in this directory.

### Data

This replication requires multiple datasets. Some are downloaded automatically by the AI agent; others require manual download due to registration requirements.

**Manual download required** — place in the `data/` directory (see `data_acquisition_instructions.md`):

| File | Source | Notes |
|------|--------|-------|
| WVS Time-Series CSV | [worldvaluessurvey.org](https://www.worldvaluessurvey.org/WVSDocumentationWVL.jsp) | Free registration required |
| EVS 1990 data | [GESIS](https://search.gesis.org/) | Study ZA4460; free registration required |
| WVS codebook (PDF) | Same as WVS download page | Variable documentation |
| EVS codebook (PDF) | Same as EVS download page | Variable documentation |
| World Development Report 1983 | [World Bank](https://www.worldbank.org/en/publication/wdr/wdr-archive) | Industry share of GDP by country |
| World Development Report 1997 | [World Bank](https://www.worldbank.org/en/publication/wdr/wdr-archive) | Higher education enrollment by country |

**Downloaded automatically** by the AI agent during the replication workflow:

| File | Source | Notes |
|------|--------|-------|
| World Bank indicators | [World Bank Open Data](https://data.worldbank.org/) | Industry share, higher education enrollment |
| Penn World Table 5.6 | [NBER/Penn](https://www.rug.nl/ggdc/productivity/pwt/) | GDP per capita data |

## How to Run

See `CLAUDE.md` for full instructions. In brief:

1. Place `Inglehart_Baker_2000.pdf` and figure screenshots in this directory
2. Download all required data files to `data/` (see `data_acquisition_instructions.md`)
3. Set up the Python environment: `python3 -m venv .venv && .venv/bin/pip install pandas numpy statsmodels scipy matplotlib`
4. Start Claude Code and say `"Replicate all"`

## Replication Targets

- **Tables 1–8** (including 5a and 5b): Factor analyses, correlations, and regressions
- **Figures 1–8**: Value maps, scatterplots, and cultural zone visualizations

> **Tip:** A `settings.json` with pre-configured permissions is provided in this directory. Copy it to your `.claude/` folder to avoid repeated permission prompts during the replication.

## Results

See `replication_summary.md` for the full replication report with scores, comparisons, and methodology notes.
