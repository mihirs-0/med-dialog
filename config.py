"""Configuration constants for the MedDialog fine-tuning pipeline."""
from pathlib import Path

SEED = 42

DATASET_NAME = "wangrongsheng/HealthCareMagic-100k-en"
DATASET_CONFIG = None

MODEL_NAME = "google/flan-t5-base"
TASK_PREFIX = "Respond as a doctor to the following patient conversation:\n\n"

# Prompt-ablation experiment: does explicit instruction to ask clarifying
# questions elicit interrogative behavior from the fine-tuned model?
CLARIFY_PROMPT = (
    "Respond as a doctor. If the patient's description is missing critical "
    "information, ask a clarifying question before giving advice.\n\n"
)

PROMPT_PRESETS = {
    "default": TASK_PREFIX,
    "clarify": CLARIFY_PROMPT,
}

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 192

TRAIN_FRAC = 0.90
VAL_FRAC = 0.05
TEST_FRAC = 0.05

LEARNING_RATE = 5e-5
PER_DEVICE_BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 1
NUM_EPOCHS = 3
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 100
EVAL_FRACTION_PER_EPOCH = 0.5
DATALOADER_WORKERS = 4
SAVE_TOTAL_LIMIT = 2

GEN_NUM_BEAMS = 4
GEN_MAX_NEW_TOKENS = 128
GEN_LENGTH_PENALTY = 1.0
GEN_EARLY_STOPPING = True
GEN_NO_REPEAT_NGRAM = 3
GEN_BATCH_SIZE = 32

BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "flan-t5-meddialog-finetuned"
DATA_CACHE_DIR = ROOT / "data_cache"
TEST_RESULTS_CSV = ROOT / "test_results.csv"
EVAL_RESULTS_JSON = ROOT / "evaluation_results.json"
TRAINER_OUTPUT_DIR = ROOT / "trainer_output"
