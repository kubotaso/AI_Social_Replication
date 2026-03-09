# LLM-Assisted Replication as Scientific Infrastructure

**Kubota, Yakura, Yamada, Nakamura and Coavoux (2026)**

[Download the paper (SocArXiv)](https://osf.io/preprints/socarxiv/qtgx8_v1) | [Older API-based version (arXiv)](http://arxiv.org/abs/2602.18453)

---

This repository provides tools for automatically replicating statistical analyses (tables and figures) from published research using Large Language Models (LLMs). The approach has been applied to five papers across economics, political science, and sociology.

## How It Works

The replication workflow uses [Claude Code](https://docs.anthropic.com/en/docs/claude-code) as a local AI agent. Given a research paper PDF, it autonomously:

1. **Discovers and acquires data** — reads the paper, identifies required datasets, and downloads them
2. **Validates the data** — runs exploratory replication attempts to confirm sufficiency
3. **Iteratively replicates** — generates Python code, scores results against the original, and refines until convergence

Each paper has a `CLAUDE.md` file containing the full replication protocol. See `sample_new_socarxiv_ver/` for examples.

## Quick Start

1. Place the research paper (PDF) and `CLAUDE.md` in a project directory
2. Add figure screenshots for visual comparison (recommended)
3. Start Claude Code and tell it what to replicate:
   - `"Replicate Table 1"` — a single target
   - `"Replicate all"` — all targets in parallel
4. Results are saved to target-specific directories (e.g., `output_table1/`)
5. A final report is saved to `replication_summary.md`

## Replicated Papers

See each paper's directory under `sample_new_socarxiv_ver/` for detailed setup instructions, required files, and replication results.

- Bartels, L. M. (2000). "Partisanship and Voting Behavior, 1952-1996." *American Journal of Political Science*, 44(1), 35-50.
- Bernanke, B. S. and Blinder, A. S. (1992). "The Federal Funds Rate and the Channels of Monetary Transmission." *American Economic Review*, 82(4), 901-921.
- Bryson, B. (1996). "Anything But Heavy Metal": Symbolic Exclusion and Musical Dislikes. *American Sociological Review*, 61(5), 884-899.
- Inglehart, R. and Baker, W. E. (2000). "Modernization, Cultural Change, and the Persistence of Traditional Values." *American Sociological Review*, 65(1), 19-51.
- Topel, R. (1991). "Specific Capital, Mobility, and Wages: Wages Rise with Job Seniority." *Journal of Political Economy*, 99(1), 145-176.

## Reference

Kubota, Yakura, Yamada, Nakamura, Coavoux. "LLM-Assisted Replication as Scientific Infrastructure." SocArXiv, DOI: 10.31235/osf.io/qtgx8_v1. [https://osf.io/preprints/socarxiv/qtgx8_v1](https://osf.io/preprints/socarxiv/qtgx8_v1)

Kubota, Yakura, Yamada, Nakamura, Coavoux. "LLM-Assisted Replication for Quantitative Social Science." arXiv:2602.18453. [http://arxiv.org/abs/2602.18453](http://arxiv.org/abs/2602.18453)
