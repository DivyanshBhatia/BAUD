"""
BAUD Configuration — All hyperparameters and paths.
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
EXTERNAL_DIR = os.path.join(PROJECT_ROOT, "external")
OPENGRAPHAU_DIR = os.path.join(EXTERNAL_DIR, "OpenGraphAU")
AU_CHECKPOINT = os.path.join(OPENGRAPHAU_DIR, "checkpoints",
                              "OpenGprahAU-ResNet50_second_stage.pth")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── OpenGraphAU Model ──
AU_NUM_MAIN = 27        # Main AU classes in checkpoint
AU_NUM_SUB = 14         # Sub AU classes in checkpoint
NUM_AUS = 41            # Total output (main + sub)

# ── AU Mapping ──
PAIN_AU_INDICES = [2, 4, 5, 6, 7, 17]   # AU4, AU6, AU7, AU9, AU10, AU43
PAIN_AU_NAMES = ["AU4", "AU6", "AU7", "AU9", "AU10", "AU43"]

# FACS-annotation AU columns (PEMF xlsx)
FACS_AU_COLS = ["AU4", "AU6", "AU7", "AU9", "AU10", "AU12",
                "AU20", "AU25", "AU26", "AU27", "AU43", "AU45"]

# ── BAUD Hyperparameters ──
BASELINE_EPSILON = 1e-4
TEMPORAL_WINDOW = 5
PAIN_SIGMOID_SHIFT = 2.0
PAIN_AU_PRIOR_WEIGHT = 3.0

# ── Meta-Learning ──
META_LR = 1e-3
META_EPOCHS = 100
META_BATCH_SIZE = 4
DEVIATION_HIDDEN = 64
GRU_HIDDEN = 32
LAMBDA_PAIN = 1.0
LAMBDA_RANK = 0.5
LAMBDA_TEMPORAL = 0.1

# ── Data ──
RANDOM_SEED = 42
CALIBRATION_DURATIONS = [5, 10, 15, 20]
