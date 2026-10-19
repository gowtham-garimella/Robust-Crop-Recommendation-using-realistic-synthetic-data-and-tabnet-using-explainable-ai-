"""
Unified Configuration, Reproducibility & Path Management Module
Project: Robust Crop Recommendation Using Synthetic Data, TabNet, and Explainable AI
"""

import os
import sys
import random
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any

# ------------------------------------------------------------------------------
# 1. CORE REPRODUCIBILITY & PROJECT METADATA
# ------------------------------------------------------------------------------
PROJECT_TITLE = "Robust Crop Recommendation Using Realistic Synthetic Data, TabNet, and Explainable AI"
RANDOM_SEED = 42

def set_deterministic_seed(seed: int = RANDOM_SEED):
    """Sets deterministic seeds across Python, NumPy, and PyTorch runtimes."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_deterministic_seed(RANDOM_SEED)

# ------------------------------------------------------------------------------
# 2. FEATURE SCHEMA & CROP CLASSES
# ------------------------------------------------------------------------------
CANONICAL_FEATURES: List[str] = [
    "N", "P", "K", "temperature", "humidity", "ph", "rainfall"
]

TARGET_COLUMN = "crop"

EXPECTED_CROPS: List[str] = sorted([
    "apple", "banana", "blackgram", "chickpea", "coconut", "coffee", "cotton",
    "grapes", "jute", "kidneybeans", "lentil", "maize", "mango", "mothbeans",
    "mungbean", "muskmelon", "orange", "papaya", "pigeonpeas", "pomegranate",
    "rice", "watermelon"
])

EXPECTED_CROPS_COUNT = len(EXPECTED_CROPS)  # 22

# ------------------------------------------------------------------------------
# 3. DATA QUOTAS & BENCHMARK DIMENSIONS
# ------------------------------------------------------------------------------
BENCHMARK_TOTAL_ROWS = 2200
BENCHMARK_REAL_TRAIN_ROWS = 1760  # 80 per crop * 22
BENCHMARK_REAL_TEST_ROWS = 440    # 20 per crop * 22

SAMPLES_PER_CROP = 7500
EXPECTED_TOTAL_PER_MODE = SAMPLES_PER_CROP * EXPECTED_CROPS_COUNT  # 165,000

# ------------------------------------------------------------------------------
# 4. PATH RESOLUTION (CROSS-PLATFORM: LOCAL & COLAB)
# ------------------------------------------------------------------------------
IN_COLAB = "google.colab" in sys.modules

# Workspace base detection
_curr_file = Path(__file__).resolve()
WORKSPACE_ROOT = _curr_file.parent.parent.parent
AUDITED_RESULTS_DIR = WORKSPACE_ROOT / "audited_final_results"

DATA_DIR = WORKSPACE_ROOT / "data"
PRIMARY_RAW_FILE = DATA_DIR / "Crop_recommendation.csv"

# Dedicated subdirectories
SCRIPTS_DIR = AUDITED_RESULTS_DIR / "01_scripts_and_config"
MODELS_DIR = AUDITED_RESULTS_DIR / "02_model_artifacts"
PREPROC_DIR = AUDITED_RESULTS_DIR / "03_preprocessing_artifacts"
METRICS_DIR = AUDITED_RESULTS_DIR / "04_evaluation_metrics"
STRESS_DIR = AUDITED_RESULTS_DIR / "05_stress_testing"
XAI_DIR = AUDITED_RESULTS_DIR / "06_xai_and_reliability"
PLOTS_DIR = AUDITED_RESULTS_DIR / "07_figures_and_plots"
TESTS_DIR = AUDITED_RESULTS_DIR / "08_automated_tests"
DEFECTS_DIR = AUDITED_RESULTS_DIR / "09_defect_register"
MANIFEST_DIR = AUDITED_RESULTS_DIR / "10_manifest_and_metadata"

for p in [DATA_DIR, AUDITED_RESULTS_DIR, SCRIPTS_DIR, MODELS_DIR, PREPROC_DIR,
          METRICS_DIR, STRESS_DIR, XAI_DIR, PLOTS_DIR, TESTS_DIR, DEFECTS_DIR, MANIFEST_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Compute Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
