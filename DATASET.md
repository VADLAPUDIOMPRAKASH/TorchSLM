# Data, Attribution, and Artifact Policy

## What this repository contains

TorchSLM distributes source code and documentation only. It does **not** distribute training data, cleaned data, encoded token streams, SentencePiece tokenizer artifacts, or model checkpoints.

The following generated or third-party artifacts are intentionally excluded from version control:

- `data/raw/`
- `data/cleaned/`
- `data/encoded/`
- `data/tokenizer/*.model` and `data/tokenizer/*.vocab`
- `checkpoints/` and `*.pt`

## Using third-party training data

Obtain any training corpus directly from its original publisher. Before downloading, processing, training on, sharing, or redistributing data, confirm its license, terms of use, attribution requirements, privacy obligations, and any restrictions on derivative artifacts.

The MIT License in this repository applies only to TorchSLM's original source code and documentation. It does not grant rights to third-party datasets, their contents, tokenizer files trained on them, or model checkpoints trained from them.

## Reproducibility record

For each training run, record the following outside the dataset itself (for example, in a release note or an experiment log):

| Item | Record |
| --- | --- |
| Dataset provider | Original organization or publisher |
| Dataset name and version | Exact release identifier or revision |
| Source URL | Canonical download or dataset card URL |
| Access date | Date the data was obtained |
| License / terms | License name and link to applicable terms |
| Attribution | Required citation or attribution text |
| Processing | Cleaning, filtering, deduplication, and split details |
| Training configuration | Commit SHA and `GPTConfig` values |

## IndicCorpV2 note

If a run uses AI4Bharat IndicCorpV2, document the exact dataset subset and release used. AI4Bharat states that datasets created as part of the IndicBERT work are released under CC0, but users must verify the terms for the exact data they downloaded before publishing derived artifacts or making licensing claims.

- Dataset project: <https://github.com/AI4Bharat/IndicBERT>
- IndicCorpV2 landing page: <https://huggingface.co/datasets/ai4bharat/IndicCorpV2>

This document is operational guidance, not legal advice.
