import random

import numpy as np
import torch

SEED = 42


def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACKBONES = {
    "lateon": {"checkpoint": "lightonai/LateOn"},
    "colbert-v2": {"checkpoint": "colbert-ir/colbertv2.0"},
    "colbert-small": {"checkpoint": "answerdotai/answerai-colbert-small-v1"},
    "moderncolbert": {"checkpoint": "lightonai/GTE-ModernColBERT-v1"},
}
BEIR_TASKS = ["scifact", "nfcorpus"]
BACKBONE_LR_DEFAULT = 1e-6
VARIANTS = [
    {"name": "signed", "use_sign": True, "backbone_lr": BACKBONE_LR_DEFAULT},
    {"name": "magnitude_only", "use_sign": False, "backbone_lr": BACKBONE_LR_DEFAULT},
    {"name": "signed_frozen", "use_sign": True, "backbone_lr": 0.0},
]
EMBEDDING_DIM = 128
DROPOUT = 0.1
BATCH_SIZE = 16
HEAD_LR = 3e-5
WEIGHT_DECAY = 0.01
MAX_STEPS = 2500
EVAL_EVERY = 30
PATIENCE_EVALS = 3
MIN_STEPS_BEFORE_STOP = 1000
DEFAULT_N_GENERAL_SAMPLES = 12_000
DEFAULT_NEVIR_OVERSAMPLE = 6
RESULTS_PATH = "results/signed_maxsim_results.json"
