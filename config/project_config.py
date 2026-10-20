"""
Canonical Project Configuration
Robust Crop Recommendation Using Synthetic Data, TabNet, Random Forest, XGBoost, and Explainable AI
"""

from pathlib import Path

# Base Paths
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = WORKSPACE_ROOT / "config"
DATA_DIR = WORKSPACE_ROOT / "data"
SRC_DIR = WORKSPACE_ROOT / "src"
TESTS_DIR = WORKSPACE_ROOT / "tests"
ARTIFACTS_DIR = WORKSPACE_ROOT / "artifacts"
RESULTS_DIR = WORKSPACE_ROOT / "results"
REPORTS_DIR = WORKSPACE_ROOT / "reports"
ARCHIVE_DIR = WORKSPACE_ROOT / "archive"

# Primary Canonical Data Files
RAW_DATA_FILE = DATA_DIR / "Crop_recommendation.csv"
REAL_TRAIN_FILE = DATA_DIR / "REAL_TRAIN.csv"
REAL_TEST_FILE = DATA_DIR / "REAL_TEST.csv"
INTERNAL_TRAIN_FILE = DATA_DIR / "INTERNAL_TRAIN.csv"
INTERNAL_VAL_FILE = DATA_DIR / "INTERNAL_VAL.csv"

# Generated Synthetic Datasets (Generated v1 - Leak-Free, fitted on INTERNAL_TRAIN only)
GENERATED_DATA_DIR = DATA_DIR / "generated_v1"
SYNTHETIC_MODE_A_FILE = GENERATED_DATA_DIR / "SYNTHETIC_Mode_A.csv"
SYNTHETIC_MODE_B_FILE = GENERATED_DATA_DIR / "SYNTHETIC_Mode_B.csv"
SYNTHETIC_MODE_C_FILE = GENERATED_DATA_DIR / "SYNTHETIC_Mode_C.csv"
SYNTHETIC_MASTER_FILE = GENERATED_DATA_DIR / "SYNTHETIC_CROP_DATA_MASTER.csv"

# Historical Synthetic Datasets (Preserved for lineage disclosure)
HISTORICAL_SYNTHETIC_DIR = DATA_DIR / "historical_synthetic"

# Retrained Artifacts Directories (retrained_v1)
RETRAINED_ARTIFACTS_DIR = ARTIFACTS_DIR / "retrained_v1"
RETRAINED_MODELS_DIR = RETRAINED_ARTIFACTS_DIR / "models"
RETRAINED_PREPROC_DIR = RETRAINED_ARTIFACTS_DIR / "preprocessing"
RETRAINED_MANIFESTS_DIR = RETRAINED_ARTIFACTS_DIR / "manifests"

# Retrained Results Directories (retrained_v1)
RETRAINED_RESULTS_DIR = RESULTS_DIR / "retrained_v1"
RETRAINED_METRICS_DIR = RETRAINED_RESULTS_DIR / "metrics"
RETRAINED_PRED_VAL_DIR = RETRAINED_RESULTS_DIR / "predictions_val"
RETRAINED_PRED_BENCH_DIR = RETRAINED_RESULTS_DIR / "predictions_benchmark"
RETRAINED_CONFUSION_DIR = RETRAINED_RESULTS_DIR / "confusion_matrices"
RETRAINED_ROBUSTNESS_DIR = RETRAINED_RESULTS_DIR / "robustness"
RETRAINED_EXPLAIN_DIR = RETRAINED_RESULTS_DIR / "explainability"

# Legacy/Canonical Aliases for backwards compatibility
MODELS_DIR = RETRAINED_MODELS_DIR
PREPROC_DIR = RETRAINED_PREPROC_DIR
MANIFESTS_DIR = RETRAINED_MANIFESTS_DIR
METRICS_DIR = RETRAINED_METRICS_DIR
PREDICTIONS_DIR = RETRAINED_PRED_BENCH_DIR
CONFUSION_DIR = RETRAINED_CONFUSION_DIR
ROBUSTNESS_DIR = RETRAINED_ROBUSTNESS_DIR
EXPLAIN_DIR = RETRAINED_EXPLAIN_DIR

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
FORCE_RETRAIN = True

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
