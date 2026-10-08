# DL4NLP: Codeforces algorithmic tags and difficulty prediction

Multitask fine-tuning of [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) to
predict, from the statement of a Codeforces problem, its **algorithmic tags** (multi-label
classification) and its **difficulty rating** (regression).

Individual project for the *Deep Learning for NLP* course (Master's in Artificial Intelligence).
Author: Alex Sierra Alcalá.

## Approach

- **Data:** [`open-r1/codeforces`](https://huggingface.co/datasets/open-r1/codeforces). Only the
  statement fields are read (column projection over the parquet files); editorials, tests and
  checkers are never downloaded.
- **Preprocessing:** LaTeX cleanup that keeps numbers, deduplication, and a temporal 80/10/10
  split by contest date.
- **Model:** ModernBERT-base encoder with mean pooling and two heads, trained with a weighted
  binary cross-entropy plus a masked MSE on the normalized rating.
- **Evaluation:** per-tag decision thresholds tuned on validation; comparison against a trivial
  baseline, TF-IDF + linear models, and a linear probe on the frozen encoder.

The notebook itself is written in Spanish, as required by the course.

## Files

| File | Contents |
|---|---|
| `Sierra_Alcala_Alex.ipynb` | Deliverable notebook (self-contained, meant for Colab with a T4 GPU) |
| `notebook.py` | Notebook source in jupytext *percent* format |
| `requirements-local.txt` | Library versions used for the local smoke test |

## Usage

Open `Sierra_Alcala_Alex.ipynb` in Google Colab, select a T4 GPU and run all cells. Data and
models are downloaded from the Hugging Face Hub. Optionally, add a Hugging Face token as the Colab
secret `HF_TOKEN` to push the fine-tuned model to the Hub.

To regenerate the notebook from its source and run the quick local check:

```bash
jupytext --to ipynb notebook.py -o Sierra_Alcala_Alex.ipynb
CF_SMOKE_TEST=1 jupyter nbconvert --to notebook --execute Sierra_Alcala_Alex.ipynb --output smoke.ipynb
```
