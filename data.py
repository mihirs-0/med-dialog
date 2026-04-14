"""Load HealthCareMagic-100k-en, clean, split, and tokenize for FLAN-T5.

Single-turn patient -> doctor QA pairs. Strips "Chat Doctor" branding
artifacts left over from the ChatDoctor release of this dataset, plus HTML
entities and whitespace noise. 90/5/5 random split with seed 42.
"""
from __future__ import annotations

import argparse
import html
import logging
import re

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

import config

log = logging.getLogger(__name__)

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

# Phrase-level "Chat Doctor" artifacts — applied in order. Longest first so
# we consume full openers ("Thanks for using Chat Doctor.") rather than just
# the brand token and leave "Thanks for using" dangling.
_CHATDOCTOR_PHRASE_RES = [
    # Multi-word openers that wrap the brand. "you" is optional ("Thanks for
    # using ..." and "Thank you for using ..." both occur), and an intermediate
    # vocative ("dear", "user") is optional too.
    re.compile(
        r"thank(?:s)?(?:\s+you)?"
        r"(?:\s+(?:dear|user|you))?"
        r"\s+(?:for|to)\s+"
        r"(?:using|consulting(?:\s+in)?|posting(?:\s+on)?|contacting|"
        r"reaching\s+out\s+to|your\s+(?:question|query)\s+on|"
        r"asking\s+(?:question|query)\s+on|query\s+on|"
        r"posting\s+(?:your\s+)?query\s+on)\s+"
        r"chat[-\s]?doctor(?:\s*\.\s*com)?[\s.,!:;]*",
        re.IGNORECASE,
    ),
    # "Welcome to Chat Doctor" (no intermediate verb).
    re.compile(
        r"welcome\s+to\s+chat[-\s]?doctor(?:\s*\.\s*com)?[\s.,!:;]*",
        re.IGNORECASE,
    ),
    # Sign-offs
    re.compile(
        r"regards[,.\s]*chat[-\s]?doctor(?:\s*\.\s*com)?[\s.,!:;]*",
        re.IGNORECASE,
    ),
    # Standalone brand mention — final fallback.
    re.compile(
        r"\bchat[-\s]?doctor(?:\s*\.\s*com)?\b[\s.,!:;]*",
        re.IGNORECASE,
    ),
]
_PUNCT_CLEANUP_RES = [
    (re.compile(r"\s+([.,;:!?])"), r"\1"),
    (re.compile(r"([.,;:!?]){2,}"), r"\1"),
    (re.compile(r"\s+"), " "),
]


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def strip_chatdoctor_artifacts(text: str) -> str:
    if not text:
        return ""
    for rgx in _CHATDOCTOR_PHRASE_RES:
        text = rgx.sub(" ", text)
    for rgx, repl in _PUNCT_CLEANUP_RES:
        text = rgx.sub(repl, text)
    return text.strip()


def preprocess_example(row: dict) -> dict:
    inp = clean_text(row.get("input", ""))
    out = strip_chatdoctor_artifacts(clean_text(row.get("output", "")))
    return {"input_text": inp, "target_text": out}


def split_dataset(ds: Dataset, seed: int = config.SEED) -> DatasetDict:
    ds = ds.shuffle(seed=seed)
    n = len(ds)
    n_train = int(n * config.TRAIN_FRAC)
    n_val = int(n * config.VAL_FRAC)
    return DatasetDict(
        {
            "train": ds.select(range(n_train)),
            "validation": ds.select(range(n_train, n_train + n_val)),
            "test": ds.select(range(n_train + n_val, n)),
        }
    )


def tokenize_splits(splits: DatasetDict, tokenizer) -> DatasetDict:
    def tokenize(batch):
        prefixed = [config.TASK_PREFIX + t for t in batch["input_text"]]
        model_inputs = tokenizer(
            prefixed,
            max_length=config.MAX_INPUT_LENGTH,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=config.MAX_TARGET_LENGTH,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return splits.map(tokenize, batched=True, desc="Tokenizing")


def print_statistics(splits: DatasetDict, tokenizer) -> None:
    total = sum(len(s) for s in splits.values())
    print("\n=== Dataset statistics ===")
    print(f"Total examples: {total}")
    for name, s in splits.items():
        print(f"  {name}: {len(s)}")

    sample = splits["train"].select(range(min(5000, len(splits["train"]))))
    in_lens = [
        len(tokenizer.encode(config.TASK_PREFIX + x, add_special_tokens=True))
        for x in sample["input_text"]
    ]
    tgt_lens = [
        len(tokenizer.encode(x, add_special_tokens=True)) for x in sample["target_text"]
    ]
    print(f"Avg input tokens (sample of {len(sample)}): {sum(in_lens)/len(in_lens):.1f}")
    print(f"Avg target tokens (sample of {len(sample)}): {sum(tgt_lens)/len(tgt_lens):.1f}")
    print("Format: single-turn QA (100% single-turn)")
    print("===========================\n")


def load_and_prepare(tokenizer=None, tokenize: bool = True):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    log.info("Loading %s", config.DATASET_NAME)
    raw = load_dataset(config.DATASET_NAME, cache_dir=str(config.DATA_CACHE_DIR))
    if "train" not in raw:
        raise RuntimeError(f"Expected 'train' split, got: {list(raw.keys())}")
    log.info("Raw examples: %d", len(raw["train"]))

    processed = raw["train"].map(
        preprocess_example,
        remove_columns=raw["train"].column_names,
        desc="Cleaning",
    )

    before = len(processed)
    processed = processed.filter(
        lambda r: bool(r["input_text"].strip()) and bool(r["target_text"].strip()),
        desc="Filtering empties",
    )
    log.info("Filtered %d empty examples (%d -> %d)", before - len(processed), before, len(processed))

    splits = split_dataset(processed)
    print_statistics(splits, tokenizer)

    if tokenize:
        splits = tokenize_splits(splits, tokenizer)
    return splits, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-tokenize", action="store_true")
    parser.add_argument("--show-examples", type=int, default=5,
                        help="Print N random cleaned (input, target) pairs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    splits, tokenizer = load_and_prepare(tokenize=not args.no_tokenize)

    if args.show_examples and not args.no_tokenize:
        return
    if args.show_examples:
        import random
        random.seed(config.SEED)
        train = splits["train"]
        idxs = random.sample(range(len(train)), min(args.show_examples, len(train)))
        print(f"\n=== {len(idxs)} random cleaned training examples ===")
        for i in idxs:
            ex = train[i]
            print(f"\n--- idx {i} ---")
            print(f"INPUT:  {ex['input_text'][:400]}")
            print(f"TARGET: {ex['target_text'][:400]}")


if __name__ == "__main__":
    main()
