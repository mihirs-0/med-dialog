"""End-to-end orchestrator: train -> generate -> evaluate."""
from __future__ import annotations

import argparse
import logging

from evaluate import evaluate_csv
from generate import generate_test_predictions
from train import train

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    args = parser.parse_args()

    if not args.skip_train:
        log.info(">>> Stage 1: training")
        train()
    if not args.skip_generate:
        log.info(">>> Stage 2: generation")
        generate_test_predictions()
    if not args.skip_evaluate:
        log.info(">>> Stage 3: evaluation")
        evaluate_csv()


if __name__ == "__main__":
    main()
