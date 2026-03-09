# Sample Outputs: arXiv Version (OpenAI API)

This directory contains sample outputs from the earlier API-based replication approach described in the arXiv version of the paper:

Kubota, Yakura, Coavoux, Yamada, Nakamura. "LLM-Assisted Replication for Quantitative Social Science." arXiv:2602.18453. [http://arxiv.org/abs/2602.18453](http://arxiv.org/abs/2602.18453)

## Overview

The scripts use the OpenAI Responses API to replicate tables and figures from Bryson (1996). The workflow:
1. Summarize the paper PDF and target table/figure
2. Map variables from the paper to a dataset codebook
3. Generate Python analysis code against a local CSV
4. Run the generated code and compare results against the original paper
5. Score alignment (0–100) and iterate with discrepancy feedback

## Prerequisites

### OpenAI API Account

You need an OpenAI API account with access to the Responses API. Set `OPENAI_API_KEY` in your environment.

### Required Files

Place the following files in the same directory as the scripts:

- **`Bryson_paper.pdf`** — The research paper to replicate (Bryson 1996)
- **`Fig1.jpg`** — Ground-truth image of Figure 1 from the paper (required only for Figure 1 replication)

### Data Preparation

Run `CustomDocData.R` to download the GSS 1993 data and create the necessary files. The script uses the `gssr` and `gssrdoc` R packages.
- `gss93_selected.csv` — Dataset with selected variables
- `gss_selected_variables_doc.txt` — Variable documentation/codebook

## Scripts

| Script | Target |
|--------|--------|
| `MainScript_Table1.py` | Table 1 (frequency/summary statistics) |
| `MainScript_Table2.py` | Table 2 (OLS regression) |
| `MainScript_Table3.py` | Table 3 (OLS regression) |
| `MainScript_Figure1.py` | Figure 1 (bar chart) |

## Output Directories

Each script saves its outputs to a corresponding directory:

- `Table1/` — Generated code, results, and discrepancy reports for Table 1
- `Table2/` — Generated code, results, and discrepancy reports for Table 2
- `Table3/` — Generated code, results, and discrepancy reports for Table 3
- `Figure1/` — Generated code, results, and discrepancy reports for Figure 1

## Reference

Bryson, B. (1996). "Anything But Heavy Metal": Symbolic Exclusion and Musical Dislikes. *American Sociological Review*, 61(5), 884-899.
