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

## Trained model weights

Published to the HuggingFace Hub at
[**mihir-s/flan-t5-meddialog-finetuned**](https://huggingface.co/mihir-s/flan-t5-meddialog-finetuned)
(~1 GB). Loads in two lines:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("mihir-s/flan-t5-meddialog-finetuned")
tokenizer = AutoTokenizer.from_pretrained("mihir-s/flan-t5-meddialog-finetuned")
```

The model is kept on the Hub rather than in this git repo because ~1 GB
files make clones painful. All code in this repo points at the HF Hub
copy for re-generation or downstream use.

The intermediate `trainer_output/` directory (~3 GB of checkpoints +
optimizer state) is **not preserved**: it's only useful for resuming
training from a partial run, and the best checkpoint is already in the
Hub-published model.

---

## Experiment 1: Prompt ablation — catastrophic prompt brittleness

**Goal.** Test whether the missing interrogative behavior (0.04% vs 6.15%
question rate in the main run) was *erased at the weight level* during
fine-tuning, or merely *suppressed by the default prompt* and recoverable
with a different instruction. If the behavior survived training, it
should surface under explicit prompting; if it was erased, no prompt
should elicit it.

**Setup.** Same trained model, same first 500 examples of the test split,
same decoding config (beam 4, max_new_tokens 192, no_repeat_ngram 3).
Only the task prefix changes:

| Prompt | String |
|---|---|
| Baseline | `"Respond as a doctor to the following patient conversation:\n\n"` (the exact training prefix) |
| V1 (rephrased) | `"Respond as a doctor. If the patient's description is missing critical information, ask a clarifying question before giving advice.\n\n"` |
| V2 (training prefix + appended instruction) | `"Respond as a doctor to the following patient conversation. If the description is missing critical information, ask a clarifying question before giving advice:\n\n"` |

V2 preserves the exact 56-character training trigger and only **appends**
the clarify instruction — the smallest possible perturbation.

### Result

| Prompt | Question rate (`?`) | Coherent English | Avg length |
|---|---|---|---|
| Reference (ground truth) | 5.20% | — | — |
| **Baseline** (training prefix verbatim) | 0.40% | coherent | 78 words |
| **V1 clarify** (rephrased prefix) | **0.00%** | **0% coherent** | 97 words |
| **V2 clarify** (prefix + append) | **0.00%** | **0% coherent** | 97 words |

*"Coherent English"* is operationalized as "no 3+ consecutive identical
tokens" — the diagnostic failure mode of broken generation. **100% of V1
and V2 outputs (500/500 each) exhibit this degenerate pattern.**

### Failure mode (typical V2 output)

> `gefühlgefühl fühlt fühlt fühlt Adelaide Adelaide Adelaide Byron Byron Byron möbel möbel möbel impun impun impun dés dés dés universitaire universitaire weil weil weil Liege Liege Liege broyeur broyeur broyeur …`

A multilingual loop over tokens from FLAN-T5's pre-training distribution
(German, French, Romanian). Under beam search with `no_repeat_ngram=3`,
the decoder settles on a cycle of 3-token blocks, each repeated twice
before advancing — the weakest possible satisfaction of the no-repeat
constraint.

### Interpretation

The fine-tuned model memorized the training prefix as a **decoding
trigger**, not as meaningful English. Changing the colon to a period and
appending one sentence is enough to knock decoding entirely off the
manifold; rather than degrading gracefully, the model falls into an
attractor built from untouched pre-training residuals.

This has two consequences:

1. **The original question-rate experiment cannot be probed via prompt
   variation on this model.** We cannot distinguish "interrogative
   capability erased at the weight level" from "interrogative capability
   survived but needs different elicitation" using prompts — no alternate
   prompt produces usable output at all. The correct follow-up is
   **Experiment 2: upsample interrogative training examples and
   retrain**, which manipulates the weights directly.
2. **The catastrophic specialization is itself a first-class finding.**
   A 250M-parameter encoder-decoder, full-fine-tuned for 3 epochs on 100K
   single-turn medical QA pairs with a fixed prefix, becomes so narrowly
   specialized that the smallest possible prompt perturbation (keeping
   the entire 56-char trigger, appending one clause) produces 100%
   degenerate output. This is a sharp cliff, not a gradient. It's an
   argument for **parameter-efficient fine-tuning** (LoRA/adapters) or
   **instruction-mixing regularization** in future work: full
   fine-tuning at this corpus size and prompt uniformity overwrites
   general-purpose instruction following.

### Artifacts

| File | Contents |
|---|---|
| `test_results_clarify_500.csv` | 500 rows of `(input, reference, generated)` from the V1 prompt. |
| `test_results_clarify_v2_500.csv` | 500 rows from the V2 prompt. |
| `clarify_gen.log`, `clarify_v2_gen.log` | Generation logs for the two ablations. |

### Reproducing

Both prompts are defined in `config.py` (`CLARIFY_PROMPT`,
`CLARIFY_PROMPT_V2`) and exposed via `--prompt-preset` on `generate.py`:

```bash
python generate.py \
    --model-dir mihir-s/flan-t5-meddialog-finetuned \
    --out-csv test_results_clarify_500.csv \
    --prompt-preset clarify \
    --limit 500 --device cpu

python generate.py \
    --model-dir mihir-s/flan-t5-meddialog-finetuned \
    --out-csv test_results_clarify_v2_500.csv \
    --prompt-preset clarify_v2 \
    --limit 500 --device cpu
```

Roughly 20 min each on an M-series Mac CPU. MPS is auto-skipped because
beam-search seq2seq hangs on the MPS backend; pass `--device mps`
explicitly if you want to try.
