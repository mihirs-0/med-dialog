"""Full fine-tuning of FLAN-T5-base on MedDialog."""
from __future__ import annotations

import argparse
import logging
import math

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

import config
from data import load_and_prepare

log = logging.getLogger(__name__)


def build_training_args(num_train_examples: int, smoke: bool = False) -> Seq2SeqTrainingArguments:
    steps_per_epoch = max(
        1,
        math.ceil(num_train_examples / (config.PER_DEVICE_BATCH_SIZE * config.GRAD_ACCUM_STEPS)),
    )
    eval_steps = max(1, int(steps_per_epoch * config.EVAL_FRACTION_PER_EPOCH))

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    kwargs = dict(
        output_dir=str(config.TRAINER_OUTPUT_DIR),
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=config.PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        warmup_ratio=config.WARMUP_RATIO,
        logging_steps=config.LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_num_workers=config.DATALOADER_WORKERS,
        predict_with_generate=False,
        report_to=["none"],
        seed=config.SEED,
        remove_unused_columns=True,
    )

    if smoke:
        kwargs.update(
            output_dir=str(config.ROOT / "smoke_output"),
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            max_steps=10,
            num_train_epochs=1,
            eval_steps=5,
            save_steps=5,
            logging_steps=2,
            warmup_ratio=0.0,
            save_total_limit=1,
            fp16=False,
            bf16=False,
            dataloader_num_workers=0,
        )

    return Seq2SeqTrainingArguments(**kwargs)


def train(smoke: bool = False) -> None:
    set_seed(config.SEED)
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    splits, _ = load_and_prepare(tokenizer=tokenizer, tokenize=True)

    if smoke:
        log.info("Smoke mode: restricting to 100 train / 20 val examples")
        splits["train"] = splits["train"].select(range(100))
        splits["validation"] = splits["validation"].select(range(20))

    model = AutoModelForSeq2SeqLM.from_pretrained(config.MODEL_NAME)

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
        label_pad_token_id=-100,
    )

    args = build_training_args(len(splits["train"]), smoke=smoke)
    log.info("Training args built. Eval/save every %d steps.", args.eval_steps)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=splits["train"],
        eval_dataset=splits["validation"],
        processing_class=tokenizer,
        data_collator=collator,
    )

    trainer.train()

    out_dir = (config.ROOT / "smoke_model") if smoke else config.OUTPUT_DIR
    log.info("Saving best model to %s", out_dir)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    metrics = trainer.evaluate(eval_dataset=splits["validation"])
    log.info("Final validation metrics: %s", metrics)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny dataset + 10 steps for pipeline validation")
    args = parser.parse_args()
    train(smoke=args.smoke)


if __name__ == "__main__":
    main()
