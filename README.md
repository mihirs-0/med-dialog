# med-dialog

Fine-tune **FLAN-T5-base** on the **HealthCareMagic-100k-en** corpus to reproduce
physician response patterns in clinical dialogue, and evaluate whether the
fine-tuned model generates doctor-like responses.

## Goal

Test whether a small seq2seq model (250M params), trained on ~100K real
patient → doctor exchanges, learns to reproduce the *distributional* style of
physician replies — including when a doctor asks a clarifying question versus
offering a statement — rather than imposing structured slot annotations on top
of the text. Evaluation combines standard similarity metrics (BERTScore, BLEU,
ROUGE-L) with a post-hoc **question vs. statement** breakdown.

## Pipeline

```
  HealthCareMagic-100k-en  ──►  data.py   ──►  tokenized train/val/test
                                   │             (90/5/5, seed 42)
                                   ▼
                               train.py   ──►  flan-t5-meddialog-finetuned/
                                   │             (best checkpoint)
                                   ▼
                              generate.py ──►  test_results.csv
                                   │             (input, reference, generated)
                                   ▼
                              evaluate.py ──►  evaluation_results.json
                                                 + summary table
```

`run.py` is a thin orchestrator that chains all three stages.

## Files

| File | Role |
|---|---|
| `config.py` | All hyperparameters, paths, and constants. Single source of truth — edit here to change batch size, learning rate, generation settings, BERTScore model, etc. |
| `data.py` | Loads `wangrongsheng/HealthCareMagic-100k-en`, cleans HTML and whitespace, strips "Chat Doctor" branding artifacts (openers, sign-offs, inline mentions), filters empty rows, shuffles and splits 90/5/5 with seed 42, tokenizes with the FLAN-T5 tokenizer, and prints dataset statistics. Runnable standalone to inspect cleaned examples. |
| `train.py` | Full fine-tune of `google/flan-t5-base` via `Seq2SeqTrainer`. Handles bf16/fp16 selection, eval + checkpoint every 0.5 epoch, loads best model at end, saves to `flan-t5-meddialog-finetuned/`. `--smoke` runs 10 steps on 100 examples for end-to-end pipeline validation. |
| `generate.py` | Loads the best checkpoint and runs beam-search generation over the test split (`num_beams=4`, `max_new_tokens=192`, `no_repeat_ngram_size=3`), batched at 32, writing `test_results.csv` with columns `input`, `reference`, `generated`. `--limit N` for smoke runs. |
| `evaluate.py` | Computes **BERTScore** (default `microsoft/deberta-xlarge-mnli`), **ROUGE-1/2/L**, corpus **BLEU**, average generated vs. reference token lengths, and a **question vs. statement** breakdown — percentages plus BERTScore F1 split by (a) generated type and (b) reference type. Writes `evaluation_results.json` and prints a summary table. `--bertscore-model` lets you swap in a smaller model for smoke testing. |
| `run.py` | End-to-end orchestrator: `train → generate → evaluate`. Supports `--skip-train`, `--skip-generate`, `--skip-evaluate` so you can re-run individual stages. |
| `requirements.txt` | Pinned lower bounds for torch, transformers, datasets, accelerate, bert-score, rouge-score, sacrebleu, pandas, numpy, tqdm, sentencepiece. |

## Dataset

- **Source**: [`wangrongsheng/HealthCareMagic-100k-en`](https://huggingface.co/datasets/wangrongsheng/HealthCareMagic-100k-en)
- **Size**: 112,165 raw examples → ~112,164 after filtering empty targets
- **Structure**: Single-turn QA. Each row has `input` (patient question) and `output` (doctor response). The `instruction` field is ignored.
- **Cleaning**: HTML unescape + tag strip, whitespace normalization, and removal of "Chat Doctor" branding artifacts. Legitimate conversational openers from physicians (e.g., "Hi, thank you for posting your query") are preserved as part of the style we want the model to learn.
- **Splits**: 90/5/5 random (seed 42) → 100,947 train / 5,608 val / 5,609 test.

## Setup

```bash
pip install -r requirements.txt
```

Expects a single GPU with ≥16 GB (A40 48 GB or A100 40 GB recommended). Full
fine-tuning FLAN-T5-base (250M params) fits comfortably.

## Usage

### End-to-end

```bash
python run.py
```

### Individual stages

```bash
python data.py --show-examples 5        # inspect cleaned examples
python train.py                         # full fine-tune (~3 epochs)
python train.py --smoke                 # 10-step dry run
python generate.py                      # beam-search on full test set
python generate.py --limit 20           # just 20 examples
python evaluate.py                      # all metrics on test_results.csv
python evaluate.py --bertscore-model distilbert-base-uncased   # fast eval
```

### Re-run partial pipeline

```bash
python run.py --skip-train              # skip training, only generate + evaluate
python run.py --skip-train --skip-generate   # only re-score existing CSV
```

## Outputs

| Path | Description |
|---|---|
| `flan-t5-meddialog-finetuned/` | Best checkpoint + tokenizer |
| `trainer_output/` | Intermediate checkpoints during training |
| `test_results.csv` | One row per test example: input, reference, generated |
| `evaluation_results.json` | All metrics + per-subset BERTScore |

## Key hyperparameters

| | Value |
|---|---|
| Base model | `google/flan-t5-base` (250M params) |
| Task prefix | `"Respond as a doctor to the following patient conversation:\n\n"` |
| Max input tokens | 512 |
| Max target tokens | 192 |
| Learning rate | 5e-5 (AdamW, default) |
| Batch size | 16 per device |
| Epochs | 3 |
| Warmup ratio | 0.05 |
| Weight decay | 0.01 |
| Precision | bf16 (preferred) / fp16 fallback |
| Beam width | 4 |
| No-repeat n-gram | 3 |
| Seed | 42 |

## Question vs. statement breakdown

A generated response is classified as a **question** if it contains `?`,
otherwise a **statement**. The same is computed for the reference. Reported:

- % of generated responses that are questions, vs. % of reference responses
- BERTScore F1 mean split by (a) generated question/statement, (b) reference question/statement

This exposes whether the model learned the *interrogative* physician pattern
(asking for more info) or only the *declarative* one (giving advice directly).
