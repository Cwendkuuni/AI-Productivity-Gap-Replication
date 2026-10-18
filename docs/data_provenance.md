# Data Provenance and Reuse

## Overview

This study is a secondary analysis of two METR developer-productivity
datasets and an existing systematic-review artifact.

The original source data are kept separate from the authors'
derived analyses.

## METR Early-2025

**Source:** Measuring the Impact of Early-2025 AI on Experienced
Open-Source Developer Productivity.

Repository:
https://github.com/METR/Measuring-Early-2025-AI-on-Exp-OSS-Devs

Paper:
https://arxiv.org/abs/2507.09089

Local source files:
- data/raw/metr_early/data_complete.csv
- data/raw/metr_early/README.md
- data/raw/metr_early/regression.py

Main use:
- RQ1: expected versus experimentally estimated productivity
- Task-time calibration and estimation bias

Reuse note:
This artifact originates from the METR research repository.
The original authors retain their applicable rights.

## METR Late-2025

**Source:** METR Late-2025 open-source developer productivity study.

Repository:
https://github.com/METR/Measuring-Late-2025-AI-on-OSS-Devs

Local source files:
- data/raw/metr_late/data.csv
- data/raw/metr_late/README.md
- data/raw/metr_late/regression.py

Main use:
- RQ1: pre-task expectations, post-task perceptions,
  and experimentally estimated productivity
- RQ2: measurement-boundary and matched-task comparisons

Reuse note:
This artifact originates from the METR research repository.
The original authors retain their applicable rights.

## Systematic-Review Artifact

**Source:** Mohamed, Assi, and Guizani,
"The impact of LLM-assistants on software developer productivity:
A systematic review and mapping study."

Publication:
ACM Transactions on Software Engineering and Methodology (2026).

Local source files:
- data/raw/slr_artifact/SLR_all_included_studies.xlsx
- data/raw/slr_artifact/Tables.pdf
- data/raw/slr_artifact/Readme.md

Main use:
- Historical SPACE mapping and snapshot reconciliation
- RQ3: workflow-boundary classification

The historical SPACE summary contains 37 studies.
The current workbook contains 39 studies.
The two snapshots share 33 studies.

Reuse note:
This artifact originates from the systematic-review study.
The original authors retain their applicable rights.

## Original Secondary Analysis

The authors' derived analysis includes:
- productivity-gap calculations;
- developer-cluster bootstrap results;
- matched-task measurement-boundary comparisons;
- SLR snapshot reconciliation;
- evidence-supported workflow-boundary classification.

The final RQ3 coding is stored in:

data/derived/slr_boundary_current39_final.csv

Its final classification fields are:
- verified_eligibility
- verified_boundary

## Integrity

The source hashes recorded in DATA_SHA256.txt identify the
source artifacts used for the documented reproduction.

These hashes establish file identity, not redistribution rights.

## Reproduction and Redistribution

Local reproduction has been validated with the documented
Python environment.

The replication package preserves source artifacts separately
from the authors' secondary analyses.

Source attribution is provided above. Inclusion in this
repository does not change the rights applicable to
third-party materials.
