"""Project-wide constants. Import from here; never hard-code these elsewhere."""

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_JSON = RESULTS_DIR / "results.json"
LLM_CACHE_DIR = DATA_DIR / "cache" / "llm"

DATASETS = {
    "s2l_dev": DATA_DIR / "speech2latex" / "en_dev",
    "s2l_test500": DATA_DIR / "speech2latex" / "en_test_500",
    "s2l_test": DATA_DIR / "speech2latex" / "en_test",
    "mathspeech": DATA_DIR / "mathspeech" / "test",
}
MATHBRIDGE_POOL = DATA_DIR / "mathbridge" / "train.jsonl"
PROCESSBENCH_DIR = DATA_DIR / "processbench"

# Models
WHISPER_MODEL = "openai/whisper-large-v3-turbo"
PHONEME_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"
LLM_MODEL = "gpt-4.1-mini"

# Reproducibility / hardware
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
SAMPLE_RATE = 16_000

# Method hyperparameters
ALPHA = 0.05          # target silent-correction rate for delta calibration
BETA = 0.5            # acoustic mix: beta * whisper + (1 - beta) * phoneme
K_BEAMS = 5           # Whisper n-best transcripts
M_CANDIDATES = 3      # LaTeX candidates per transcript
N_FEWSHOT = 8         # MathBridge few-shot pairs in the LLM prompt
N_CHECK_POINTS = 20   # random points for numeric equivalence / residual checks
NUM_TOL = 1e-6        # numeric tolerance
SYMPY_TIMEOUT_S = 5   # per-pair timeout for symbolic checks

# Phase 4: confusable-number pool augmentation (src/augment.py); shipped system = True.
# Phase 2 single-equation results (results/tuned.json, ablation) are pre-augmentation.
AUGMENT_CONFUSABLE = True
