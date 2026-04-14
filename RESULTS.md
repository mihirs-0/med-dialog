# Results

Fine-tune of `google/flan-t5-base` on `wangrongsheng/HealthCareMagic-100k-en`
(3 epochs, batch 16, lr 5e-5, bf16, seed 42). Evaluated on a held-out
**5,609-example** test split. Hardware: RTX Pro 4500 Blackwell, ~58 min
training + ~18 min generation + ~3 min scoring.

See [`README.md`](./README.md) for the full pipeline and hyperparameters.
Raw artifacts: [`test_results.csv`](./test_results.csv),
[`evaluation_results.json`](./evaluation_results.json), [`run.log`](./run.log).

---

## TL;DR

The fine-tuned model reproduces the *style* of physician responses
(BERTScore F1 **0.597**) but almost completely fails to reproduce the
*interrogative* pattern — asking clarifying questions — that appears in
roughly 6% of real doctor replies.

**Headline finding:**

| | Generated | Reference |
|---|---|---|
| % of responses that are questions | **0.04%** (2 of 5,609) | **6.15%** (345 of 5,609) |

The model collapsed into a declarative, advice-giving mode. Cross-entropy
fine-tuning on this corpus smoothed out the minority interrogative behavior.

---

## Similarity metrics (n = 5,609)

| Metric | Value |
|---|---|
| **BERTScore** (microsoft/deberta-xlarge-mnli) | |
| &nbsp;&nbsp;Precision | 0.6153 |
| &nbsp;&nbsp;Recall | 0.5817 |
| &nbsp;&nbsp;**F1** | **0.5973** |
| **ROUGE** (F1) | |
| &nbsp;&nbsp;ROUGE-1 | 0.2990 |
| &nbsp;&nbsp;ROUGE-2 | 0.0806 |
| &nbsp;&nbsp;ROUGE-L | 0.1870 |
| **BLEU** (corpus) | 6.59 |

BERTScore is the headline metric here. BLEU is low by design on open-ended
dialogue — there is no single correct response to a patient query, so n-gram
overlap undersells real quality. ROUGE-L and ROUGE-1 show real but modest
lexical alignment.

### How to read BERTScore F1 0.60

- Chance alignment of unrelated clinical text hovers around 0.40–0.50 on
  `deberta-xlarge-mnli` embeddings (without baseline rescaling).
- Identical text scores 1.00.
- **0.60 indicates substantial semantic alignment** — the model produces
  responses that are topically and medically relevant, not random text.
  Not state-of-the-art for a specialist dialogue model (which would be
  0.70+), but solid for a 250M-parameter general encoder-decoder trained
  for ~1 hour.

---

## The declarative collapse

Raw counts on the 5,609-example test set:

| | Questions (contains `?`) | Statements |
|---|---|---|
| Generated | **2** | 5,607 |
| Reference | 345 | 5,264 |

345 of the 5,609 reference doctor responses (6.15%) asked the patient
something — follow-up symptoms, clarifying context, whether they've seen
another specialist, etc. The model produced questions in **2** cases.
For practical purposes, **it never asks**.

### Why this matters clinically

Clarifying questions are not decorative. "Can you tell me more about when
this started?" or "Are you on any other medications?" are high-information
turns — they reduce uncertainty before advice is given. A model that
always launches into advice without ever asking for more information is
overconfident by construction, even when its advice is plausible.

### Why this probably happened

- **Corpus imbalance**: 94% of reference responses are declarative. Cross-entropy loss fit the dominant mode.
- **Single-turn structure**: HealthCareMagic data is mostly one patient turn → one doctor turn. The doctor typically wraps up the case in a single reply rather than opening a dialogue. Multi-turn datasets might surface more clarification.
- **FLAN-T5 pre-training bias**: The base model was instruction-tuned to *answer*, not to *ask back*. Our prompt ("Respond as a doctor to the following patient conversation") reinforces that framing.

### Supporting evidence: BERTScore by response type

| | BERTScore F1 |
|---|---|
| **By generated type** | |
| &nbsp;&nbsp;Generated = question (n=2) | 0.684 |
| &nbsp;&nbsp;Generated = statement (n=5,607) | 0.597 |
| **By reference type** | |
| &nbsp;&nbsp;Reference = question (n=345) | **0.574** |
| &nbsp;&nbsp;Reference = statement (n=5,264) | **0.599** |

The bottom rows are the interesting ones. When the reference doctor asked
a question, the model's reply scored **0.574** (worse than average).
When the reference doctor made a statement, the model's reply scored
**0.599** (better). The gap is the quantitative cost of the model's
failure to match interrogative references — it produced a declarative
answer where the reference asked a clarifying question, and BERTScore
penalized the semantic mismatch.

The n=2 for generated-questions is too small to read anything into.

---

## Length analysis

| | Avg tokens |
|---|---|
| Generated | 107.5 |
| Reference | 147.7 |

The model under-generates by roughly **27%**. Two plausible drivers:

1. `MAX_TARGET_LENGTH = 192` during training truncates the tail of real
   physician responses, implicitly training the model to stop earlier.
2. Beam search with `length_penalty = 1.0` and `no_repeat_ngram_size = 3`
   has a mild bias toward shorter completions.

Fixable in a follow-up by raising max-target to 256 and nudging length
penalty to 1.2. Not considered a primary finding.

---

## Caveats

- **Question detection is a `?` presence check.** It's a proxy. Some
  questions end with a period in casual writing, and some statements
  rhetorically include `?`. A classifier-based detector would sharpen the
  numbers but the ~150× gap (6.15% vs 0.04%) is large enough to be robust
  to detection noise.
- **No human evaluation.** All metrics are automatic. A blind physician
  review of 50 sampled (reference, generated) pairs would be a natural
  next step.
- **Single seed.** Results are from seed 42. Multi-seed runs would give
  confidence intervals, but the qualitative findings are unlikely to move.
- **No safety filtering.** Some generated responses recommend specific
  drug names and dosages (inherited from HealthCareMagic style). These are
  not clinically vetted. The model is a research artifact, not a decision
  support tool.

---

## Artifacts

| File | Contents |
|---|---|
| `test_results.csv` | 5,609 rows of `(input, reference, generated)`. ~8 MB. The substrate for every metric in `evaluation_results.json`. |
| `evaluation_results.json` | All metrics — BERTScore, ROUGE, BLEU, length, question/statement breakdown and counts. |
| `run.log` | Full training log: per-step loss, per-eval validation loss, HF downloads, checkpoint saves. Useful for reproducing or diagnosing. |

### Reproducing

Re-scoring from the existing CSV (fast, no GPU needed for ROUGE/BLEU;
a GPU helps BERTScore):

```bash
python evaluate.py --csv test_results.csv --out-json evaluation_results.json
```

Re-generating from scratch requires the trained model weights (~1 GB, not
in the repo — see next section).

---

## What about the trained model weights?

The fine-tuned model (`flan-t5-meddialog-finetuned/`, ~1 GB) is **not in
this repo**. If you want to use it:

- **Preferred**: push to the HuggingFace Hub (`mihirs-0/flan-t5-meddialog-finetuned`) and load via `AutoModelForSeq2SeqLM.from_pretrained(...)`. One-time setup, trivial reuse.
- **Alternative**: keep it on the RunPod persistent volume and resume the same pod when needed.
- **Not recommended**: Git LFS or committing to the repo directly — the file is large enough to make clones painful.

The intermediate `trainer_output/` directory (~3 GB of checkpoints +
optimizer state) is **not worth preserving**: it's only useful for
resuming training from a partial run, which we don't need, and the best
checkpoint is already saved separately.
