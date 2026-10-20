"""
Canonical Project Configuration
Robust Crop Recommendation Using Synthetic Data, TabNet, Random Forest, XGBoost, and Explainable AI
"""

from pathlib import Path

# Paths
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = WORKSPACE_ROOT / "config"
DATA_DIR = WORKSPACE_ROOT / "data"
SRC_DIR = WORKSPACE_ROOT / "src"
TESTS_DIR = WORKSPACE_ROOT / "tests"
ARTIFACTS_DIR = WORKSPACE_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PREPROC_DIR = ARTIFACTS_DIR / "preprocessing"
MANIFESTS_DIR = ARTIFACTS_DIR / "manifests"
RESULTS_DIR = WORKSPACE_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"
ROBUSTNESS_DIR = RESULTS_DIR / "robustness"
EXPLAIN_DIR = RESULTS_DIR / "explainability"
REPORTS_DIR = WORKSPACE_ROOT / "reports"
ARCHIVE_DIR = WORKSPACE_ROOT / "archive"

# Primary Data Files
RAW_DATA_FILE = DATA_DIR / "Crop_recommendation.csv"
REAL_TRAIN_FILE = DATA_DIR / "REAL_TRAIN.csv"
REAL_TEST_FILE = DATA_DIR / "REAL_TEST.csv"
INTERNAL_TRAIN_FILE = DATA_DIR / "INTERNAL_TRAIN.csv"
INTERNAL_VAL_FILE = DATA_DIR / "INTERNAL_VAL.csv"

SYNTHETIC_MODE_A_FILE = DATA_DIR / "SYNTHETIC_Mode_A.csv"
SYNTHETIC_MODE_B_FILE = DATA_DIR / "SYNTHETIC_Mode_B.csv"
SYNTHETIC_MODE_C_FILE = DATA_DIR / "SYNTHETIC_Mode_C.csv"

# Canonical Features and Targets
CANONICAL_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COLUMN = "crop"

EXPECTED_CROPS = sorted([
    "rice", "maize", "chickpea", "kidneybeans", "pigeonpeas",
    "mothbeans", "mungbean", "blackgram", "lentil", "pomegranate",
    "banana", "mango", "grapes", "watermelon", "muskmelon",
    "apple", "orange", "papaya", "coconut", "cotton", "jute", "coffee"
])

NUM_CLASSES = len(EXPECTED_CROPS)

# Benchmark Constants
BENCHMARK_RAW_ROWS = 2200
BENCHMARK_REAL_TRAIN_ROWS = 1760  # 80 per crop
BENCHMARK_REAL_TEST_ROWS = 440    # 20 per crop
INTERNAL_TRAIN_ROWS = 1408        # 64 per crop
INTERNAL_VAL_ROWS = 352           # 16 per crop
SYNTHETIC_PER_CROP = 7500
SYNTHETIC_TOTAL_PER_MODE = 165000 # 7500 * 22

RANDOM_SEED = 42

# Agronomic Feature Physical Invariant Ranges
PHYSICAL_RANGES = {
    "N": (0.0, 300.0),
    "P": (0.0, 300.0),
    "K": (0.0, 300.0),
    "temperature": (-10.0, 60.0),
    "humidity": (0.0, 100.0),
    "ph": (0.0, 14.0),
    "rainfall": (0.0, 1000.0)
}

# 12 Model Configurations: 3 Architectures x 4 Regimes
ARCHITECTURES = ["TabNet", "Random Forest", "XGBoost"]
REGIMES = [
    ("Real-Only", None, "real_only", 0),
    ("Real + Mode A", SYNTHETIC_MODE_A_FILE, "mode_a", SYNTHETIC_TOTAL_PER_MODE),
    ("Real + Mode B", SYNTHETIC_MODE_B_FILE, "mode_b", SYNTHETIC_TOTAL_PER_MODE),
    ("Real + Mode C", SYNTHETIC_MODE_C_FILE, "mode_c", SYNTHETIC_TOTAL_PER_MODE),
]

# Model Default Hyperparameters
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 12,
    "random_state": RANDOM_SEED,
    "n_jobs": -1
}

XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.10,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "objective": "multi:softprob",
    "num_class": NUM_CLASSES,
    "random_state": RANDOM_SEED,
    "eval_metric": "mlogloss",
    "n_jobs": -1
}

TABNET_PARAMS = {
    "n_d": 16,
    "n_a": 16,
    "n_steps": 4,
    "gamma": 1.3,
    "lambda_sparse": 1e-4,
    "seed": RANDOM_SEED,
    "verbose": 0
}
