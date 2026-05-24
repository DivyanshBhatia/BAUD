"""
BAUD Configuration — All hyperparameters and paths in one place.
"""
import os

# ============================================================================
# Paths
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
EXTERNAL_DIR = os.path.join(PROJECT_ROOT, "external")

# OpenGraphAU paths
OPENGRAPHAU_DIR = os.path.join(EXTERNAL_DIR, "OpenGraphAU")
AU_CHECKPOINT = os.path.join(
    OPENGRAPHAU_DIR, "checkpoints", "OpenGprahAU-ResNet50_second_stage.pth"
)

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# AU Configuration
# ============================================================================
NUM_AUS = 41  # OpenGraphAU outputs 41 AUs

# AU index → name mapping (OpenGraphAU ordering)
AU_NAMES_FULL = {
    0: "AU1", 1: "AU2", 2: "AU4", 3: "AU5", 4: "AU6",
    5: "AU7", 6: "AU9", 7: "AU10", 8: "AU12", 9: "AU14",
    10: "AU15", 11: "AU17", 12: "AU20", 13: "AU23", 14: "AU24",
    15: "AU25", 16: "AU26", 17: "AU43",
}

# Pain-related AU indices (PSPI formula: AU4 + max(AU6,AU7) + max(AU9,AU10) + AU43)
PAIN_AU_INDICES = [2, 4, 5, 6, 7, 17]  # AU4, AU6, AU7, AU9, AU10, AU43
PAIN_AU_NAMES = ["AU4", "AU6", "AU7", "AU9", "AU10", "AU43"]

# All 12 commonly evaluated AUs (BP4D standard)
EVAL_AU_INDICES = list(range(18))
EVAL_AU_NAMES = [AU_NAMES_FULL[i] for i in range(18)]

# ============================================================================
# BAUD Hyperparameters
# ============================================================================
# Calibration
DEFAULT_CALIBRATION_FRAMES = 100     # Frames of baseline for calibration
MIN_CALIBRATION_FRAMES = 10          # Minimum frames needed
BASELINE_EPSILON = 1e-6              # Prevent division by zero in std

# Temporal smoothing
TEMPORAL_WINDOW_SIZE = 10            # Sliding window for z-score smoothing

# Pain scoring
PAIN_SIGMOID_SHIFT = 2.0            # Sigmoid center point
PAIN_THRESHOLD_BINARY = 0.5         # Binary pain/no-pain threshold

# Prior weights for pain-related AUs
PAIN_AU_PRIOR_WEIGHT = 3.0          # How much more to weight pain AUs

# ============================================================================
# Meta-Learning Hyperparameters
# ============================================================================
META_LR = 1e-3                       # Learning rate for deviation weights
META_EPOCHS = 50                     # Meta-training epochs
META_BATCH_SIZE = 8                  # Subjects per meta-batch
DEVIATION_NET_HIDDEN = 64           # Hidden size for deviation weight MLP
GRU_HIDDEN_SIZE = 32                # GRU hidden size for temporal aggregation
GRU_NUM_LAYERS = 1                  # GRU layers

# Loss weights
LAMBDA_PAIN = 1.0                   # Weight for pain classification loss
LAMBDA_RANK = 0.5                   # Weight for ranking loss
LAMBDA_TEMPORAL = 0.1               # Weight for temporal smoothness loss

# ============================================================================
# Synthetic Data
# ============================================================================
NUM_SYNTHETIC_PATIENTS = 20
SYNTHETIC_BASELINE_FRAMES = 100
SYNTHETIC_PAIN_FRAMES = 80

# ============================================================================
# Evaluation
# ============================================================================
CALIBRATION_DURATIONS = [10, 20, 50, 100, 200]  # Frames to test
RANDOM_SEED = 42
