"""Run beam-search generation on the test split and save CSV of predictions."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import config
from data import load_and_prepare

log = logging.getLogger(__name__)


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def generate_test_predictions(model_dir=config.OUTPUT_DIR,
                              out_csv=config.TEST_RESULTS_CSV,
                              limit: int | None = None,
                              prompt: str | None = None) -> Path:
    if prompt is None:
        prompt = config.TASK_PREFIX
    device = _pick_device()
    log.info("Loading model from %s (device=%s)", model_dir, device)
    log.info("Prompt: %r", prompt)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir)).to(device)
    model.eval()

    splits, _ = load_and_prepare(tokenizer=tokenizer, tokenize=False)
    test = splits["test"]
    if limit is not None:
        test = test.select(range(min(limit, len(test))))
        log.info("Limited test set to %d examples", len(test))
    inputs = list(test["input_text"])
    refs = list(test["target_text"])

    generated: list[str] = []
    for start in tqdm(range(0, len(inputs), config.GEN_BATCH_SIZE), desc="Generating"):
        batch = inputs[start:start + config.GEN_BATCH_SIZE]
        prefixed = [prompt + x for x in batch]
        enc = tokenizer(
            prefixed,
            return_tensors="pt",
            max_length=config.MAX_INPUT_LENGTH,
            truncation=True,
            padding=True,
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                num_beams=config.GEN_NUM_BEAMS,
                max_new_tokens=config.GEN_MAX_NEW_TOKENS,
                length_penalty=config.GEN_LENGTH_PENALTY,
                early_stopping=config.GEN_EARLY_STOPPING,
                no_repeat_ngram_size=config.GEN_NO_REPEAT_NGRAM,
            )
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        generated.extend(decoded)

    df = pd.DataFrame({"input": inputs, "reference": refs, "generated": generated})
    df.to_csv(out_csv, index=False)
    log.info("Wrote %d rows to %s", len(df), out_csv)
    return out_csv


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=str(config.OUTPUT_DIR),
                        help="Local path or HuggingFace Hub repo id")
    parser.add_argument("--out-csv", type=Path, default=config.TEST_RESULTS_CSV)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only generate on first N test examples")
    parser.add_argument("--prompt-preset", choices=list(config.PROMPT_PRESETS),
                        default="default",
                        help="Which prompt preset to prepend to each input")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Override with an arbitrary prompt string (takes precedence over --prompt-preset)")
    args = parser.parse_args()
    prompt = args.prompt if args.prompt is not None else config.PROMPT_PRESETS[args.prompt_preset]
    generate_test_predictions(args.model_dir, args.out_csv, args.limit, prompt)


if __name__ == "__main__":
    main()
