# LLM-Assisted Replication for Quantitative Social Science

## [Download the paper here (SocArXiv)](https://osf.io/preprints/socarxiv/qtgx8_v1)

Kubota, Yakura, Coavoux, Yamada, Nakamura. "LLM-Assisted Replication as Scientific Infrastructure."

A detailed version is also available here: [AI_Replication_Paper.pdf](AI_Replication_Paper.pdf)

This repository provides tools for automatically replicating statistical analyses (tables and figures) from Bryson (1996) using Large Language Models (LLMs).

## Overview

The scripts in this repository use the OpenAI API to:
1. Read and summarize a research paper PDF
2. Map variables from the paper to a dataset codebook
3. Generate Python code to replicate the analysis
4. Compare the generated results against the original paper
5. Iteratively refine the code until the results match

## Prerequisites

### OpenAI API Account

You need an OpenAI API account with access to the Responses API.

### Required Files

Place the following files in the same directory as the scripts:

- **`Bryson_paper.pdf`** - The research paper to replicate (Bryson 1996)
- **`Fig1.jpg`** - Ground-truth image of Figure 1 from the paper (required only for Figure 1 replication)

### Data Preparation

Run `CustomDocData.R` to download the GSS 1993 data and create the necessary files. The script uses the `gssr` and `gssrdoc` packages.
- `gss93_selected.csv` - Dataset with selected variables
- `gss_selected_variables_doc.txt` - Variable documentation/codebook

## Sample Outputs

The `SampleOutputs/` directory contains example outputs from running each script.

## Reference

Bryson, B. (1996). "Anything But Heavy Metal": Symbolic Exclusion and Musical Dislikes. *American Sociological Review*, 61(5), 884-899.
