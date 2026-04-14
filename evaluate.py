"""Compute BERTScore, ROUGE, BLEU, and question/statement breakdowns."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from statistics import mean

import pandas as pd
import sacrebleu
import bert_score.utils as _bsu
from bert_score import score as bertscore_fn
from rouge_score import rouge_scorer
from transformers import AutoTokenizer

import config

# Workaround for bert_score 0.3.13 + newer tokenizers/transformers: when a
# tokenizer's model_max_length is the sentinel ~1e30, the Rust tokenizer
# raises OverflowError. Clamp to the model's real positional limit (512 for
# BERT/DeBERTa-family models used as scorers).
_orig_sent_encode = _bsu.sent_encode


def _safe_sent_encode(tokenizer, sent):
    if tokenizer.model_max_length > 10_000:
        tokenizer.model_max_length = 512
    return _orig_sent_encode(tokenizer, sent)


_bsu.sent_encode = _safe_sent_encode

log = logging.getLogger(__name__)


def is_question(text: str) -> bool:
    return "?" in (text or "")


def compute_rouge(preds: list[str], refs: list[str]) -> dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for p, r in zip(preds, refs):
        s = scorer.score(r, p)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    return {
        "rouge1_f": mean(r1) if r1 else 0.0,
        "rouge2_f": mean(r2) if r2 else 0.0,
        "rougeL_f": mean(rl) if rl else 0.0,
    }


def compute_bleu(preds: list[str], refs: list[str]) -> float:
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    return float(bleu.score)


def compute_bertscore(preds: list[str], refs: list[str],
                      model_type: str = config.BERTSCORE_MODEL
                      ) -> tuple[list[float], list[float], list[float]]:
    P, R, F = bertscore_fn(
        preds,
        refs,
        model_type=model_type,
        lang="en",
        rescale_with_baseline=False,
        verbose=True,
    )
    return P.tolist(), R.tolist(), F.tolist()


def mean_or_none(xs: list[float]) -> float | None:
    return mean(xs) if xs else None


def average_tokens(tokenizer, texts: list[str]) -> float:
    if not texts:
        return 0.0
    lens = [len(tokenizer.encode(t, add_special_tokens=False)) for t in texts]
    return sum(lens) / len(lens)


def evaluate_csv(csv_path: Path = config.TEST_RESULTS_CSV,
                 out_json: Path = config.EVAL_RESULTS_JSON,
                 bertscore_model: str = config.BERTSCORE_MODEL) -> dict:
    df = pd.read_csv(csv_path).fillna("")
    preds = df["generated"].astype(str).tolist()
    refs = df["reference"].astype(str).tolist()
    log.info("Loaded %d (pred, ref) pairs from %s", len(df), csv_path)

    log.info("Computing BERTScore (%s)...", bertscore_model)
    P, R, F = compute_bertscore(preds, refs, model_type=bertscore_model)

    log.info("Computing ROUGE...")
    rouge = compute_rouge(preds, refs)

    log.info("Computing BLEU...")
    bleu = compute_bleu(preds, refs)

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    avg_gen_tokens = average_tokens(tokenizer, preds)
    avg_ref_tokens = average_tokens(tokenizer, refs)

    gen_is_q = [is_question(p) for p in preds]
    ref_is_q = [is_question(r) for r in refs]

    pct_gen_questions = sum(gen_is_q) / len(gen_is_q) if gen_is_q else 0.0
    pct_ref_questions = sum(ref_is_q) / len(ref_is_q) if ref_is_q else 0.0

    f_by_gen_q = [f for f, q in zip(F, gen_is_q) if q]
    f_by_gen_s = [f for f, q in zip(F, gen_is_q) if not q]
    f_by_ref_q = [f for f, q in zip(F, ref_is_q) if q]
    f_by_ref_s = [f for f, q in zip(F, ref_is_q) if not q]

    results = {
        "n_examples": len(df),
        "bertscore": {
            "model": bertscore_model,
            "precision_mean": mean(P),
            "recall_mean": mean(R),
            "f1_mean": mean(F),
        },
        "rouge": rouge,
        "bleu_corpus": bleu,
        "length": {
            "avg_generated_tokens": avg_gen_tokens,
            "avg_reference_tokens": avg_ref_tokens,
        },
        "question_vs_statement": {
            "pct_generated_questions": pct_gen_questions,
            "pct_reference_questions": pct_ref_questions,
            "bertscore_f1": {
                "generated_questions": mean_or_none(f_by_gen_q),
                "generated_statements": mean_or_none(f_by_gen_s),
                "reference_questions": mean_or_none(f_by_ref_q),
                "reference_statements": mean_or_none(f_by_ref_s),
            },
            "counts": {
                "generated_questions": sum(gen_is_q),
                "generated_statements": len(gen_is_q) - sum(gen_is_q),
                "reference_questions": sum(ref_is_q),
                "reference_statements": len(ref_is_q) - sum(ref_is_q),
            },
        },
    }

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Wrote metrics to %s", out_json)

    print_summary(results)
    return results


def print_summary(r: dict) -> None:
    bs = r["bertscore"]
    rg = r["rouge"]
    qs = r["question_vs_statement"]
    bsq = qs["bertscore_f1"]
    ln = r["length"]
    lines = [
        "",
        "================ Evaluation Summary ================",
        f"N examples:              {r['n_examples']}",
        "",
        f"--- BERTScore ({bs.get('model', 'unknown')}) ---",
        f"  Precision:             {bs['precision_mean']:.4f}",
        f"  Recall:                {bs['recall_mean']:.4f}",
        f"  F1:                    {bs['f1_mean']:.4f}",
        "",
        "--- ROUGE (F1) ---",
        f"  ROUGE-1:               {rg['rouge1_f']:.4f}",
        f"  ROUGE-2:               {rg['rouge2_f']:.4f}",
        f"  ROUGE-L:               {rg['rougeL_f']:.4f}",
        "",
        "--- BLEU ---",
        f"  Corpus BLEU:           {r['bleu_corpus']:.4f}",
        "",
        "--- Length (tokens) ---",
        f"  Avg generated:         {ln['avg_generated_tokens']:.1f}",
        f"  Avg reference:         {ln['avg_reference_tokens']:.1f}",
        "",
        "--- Question vs Statement ---",
        f"  % generated = question: {qs['pct_generated_questions']:.2%}",
        f"  % reference = question: {qs['pct_reference_questions']:.2%}",
        "",
        "  BERTScore F1 by generated type:",
        f"    questions:            {_fmt(bsq['generated_questions'])}",
        f"    statements:           {_fmt(bsq['generated_statements'])}",
        "  BERTScore F1 by reference type:",
        f"    questions:            {_fmt(bsq['reference_questions'])}",
        f"    statements:           {_fmt(bsq['reference_statements'])}",
        "===================================================",
        "",
    ]
    print("\n".join(lines))


def _fmt(v: float | None) -> str:
    return f"{v:.4f}" if v is not None else "n/a"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=config.TEST_RESULTS_CSV)
    parser.add_argument("--out-json", type=Path, default=config.EVAL_RESULTS_JSON)
    parser.add_argument("--bertscore-model", type=str, default=config.BERTSCORE_MODEL)
    args = parser.parse_args()
    evaluate_csv(args.csv, args.out_json, args.bertscore_model)


if __name__ == "__main__":
    main()
