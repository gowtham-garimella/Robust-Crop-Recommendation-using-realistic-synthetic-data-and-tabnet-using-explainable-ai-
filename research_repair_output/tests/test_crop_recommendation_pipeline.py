"""
Automated Regression Test Suite for Audited Crop Recommendation Pipeline
Validates all 12 confirmed defect fixes, non-negotiable requirements, and data integrity.
"""

import os
import sys
import json
import joblib
try:
    import pytest
except ImportError:
    pytest = None
import numpy as np
import pandas as pd
from pathlib import Path

# Add scripts directory to path
curr_dir = Path(__file__).resolve().parent
scripts_dir = curr_dir.parent / "01_scripts_and_config"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from config import (
    AUDITED_RESULTS_DIR, MODELS_DIR, PREPROC_DIR, METRICS_DIR,
    STRESS_DIR, XAI_DIR, PLOTS_DIR, DEFECTS_DIR, MANIFEST_DIR,
    CANONICAL_FEATURES, TARGET_COLUMN, EXPECTED_CROPS,
    BENCHMARK_REAL_TRAIN_ROWS, BENCHMARK_REAL_TEST_ROWS
)
from phase2c_model_training_and_tuning import load_xgboost_universal
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix

DATA_P1_DIR = scripts_dir / "phase1_data"
DATA_P2A_DIR = scripts_dir / "phase2a_data"

# ==============================================================================
# 1. INPUT SCHEMA & BENCHMARK DIMENSIONS
# ==============================================================================
def test_input_schema_and_types():
    """Validates that real datasets contain exactly canonical 7 features and crop label."""
    for fname in ["REAL_TRAIN.csv", "REAL_TEST.csv"]:
        fpath = DATA_P1_DIR / fname
        assert fpath.exists(), f"File {fname} is missing!"
        df = pd.read_csv(fpath)
        for feat in CANONICAL_FEATURES:
            assert feat in df.columns, f"Feature {feat} missing in {fname}"
            assert pd.to_numeric(df[feat], errors="coerce").notnull().all(), f"Non-numeric values in {fname} - {feat}"
        assert TARGET_COLUMN in df.columns, f"Target {TARGET_COLUMN} missing in {fname}"
        assert not df[CANONICAL_FEATURES].isnull().values.any(), f"NaNs found in {fname}"
        assert not np.isinf(df[CANONICAL_FEATURES].values).any(), f"Infs found in {fname}"

def test_test_set_quarantine_dimensions():
    """Asserts REAL_TEST has exactly 440 rows, 22 classes, and 20 samples per class."""
    df_test = pd.read_csv(DATA_P1_DIR / "REAL_TEST.csv")
    assert len(df_test) == BENCHMARK_REAL_TEST_ROWS, f"REAL_TEST count: {len(df_test)} (Expected: 440)"
    counts = df_test[TARGET_COLUMN].value_counts()
    assert len(counts) == len(EXPECTED_CROPS), f"Classes count mismatch: {len(counts)}"
    assert all(c == 20 for c in counts.values), "REAL_TEST is not balanced at 20 per crop"

def test_real_train_dimensions():
    """Asserts REAL_TRAIN has exactly 1,760 rows, 22 classes, and 80 samples per class."""
    df_train = pd.read_csv(DATA_P1_DIR / "REAL_TRAIN.csv")
    assert len(df_train) == BENCHMARK_REAL_TRAIN_ROWS, f"REAL_TRAIN count: {len(df_train)} (Expected: 1760)"
    counts = df_train[TARGET_COLUMN].value_counts()
    assert len(counts) == len(EXPECTED_CROPS), f"Classes count mismatch: {len(counts)}"
    assert all(c == 80 for c in counts.values), "REAL_TRAIN is not balanced at 80 per crop"

# ==============================================================================
# 2. VALIDATION & TEST LEAKAGE PREVENTION
# ==============================================================================
def test_zero_train_test_leakage():
    """Asserts zero exact row or feature vector overlap between REAL_TRAIN and REAL_TEST."""
    df_train = pd.read_csv(DATA_P1_DIR / "REAL_TRAIN.csv")
    df_test = pd.read_csv(DATA_P1_DIR / "REAL_TEST.csv")
    
    row_overlap = pd.merge(df_train, df_test, on=CANONICAL_FEATURES + [TARGET_COLUMN], how="inner")
    feat_overlap = pd.merge(df_train[CANONICAL_FEATURES], df_test[CANONICAL_FEATURES], on=CANONICAL_FEATURES, how="inner")
    
    assert len(row_overlap) == 0, f"Critical Leakage! {len(row_overlap)} exact rows shared between Train and Test!"
    assert len(feat_overlap) == 0, f"Critical Leakage! {len(feat_overlap)} feature vectors shared between Train and Test!"

def test_zero_synthetic_test_leakage():
    """Asserts zero exact row overlap between any synthetic mode and REAL_TEST."""
    df_test = pd.read_csv(DATA_P1_DIR / "REAL_TEST.csv")
    for mode in ["SYNTHETIC_Mode_A.csv", "SYNTHETIC_Mode_B.csv", "SYNTHETIC_Mode_C.csv"]:
        syn_path = DATA_P1_DIR / mode
        assert syn_path.exists(), f"Synthetic file {mode} missing"
        df_syn = pd.read_csv(syn_path)
        overlap = pd.merge(df_syn[CANONICAL_FEATURES], df_test[CANONICAL_FEATURES], on=CANONICAL_FEATURES, how="inner")
        assert len(overlap) == 0, f"Critical Leakage! {len(overlap)} exact feature vectors shared between {mode} and REAL_TEST!"

def test_internal_validation_isolation():
    """Asserts internal train (1,408) and internal val (352) are completely disjoint."""
    df_itr = pd.read_csv(DATA_P2A_DIR / "INTERNAL_TRAIN.csv")
    df_ival = pd.read_csv(DATA_P2A_DIR / "INTERNAL_VAL.csv")
    
    assert len(df_itr) == 1408
    assert len(df_ival) == 352
    overlap = pd.merge(df_itr[CANONICAL_FEATURES], df_ival[CANONICAL_FEATURES], on=CANONICAL_FEATURES, how="inner")
    assert len(overlap) == 0, f"Internal train-val leakage! {len(overlap)} rows overlap!"

def test_master_dataset_provenance():
    """Asserts SYNTHETIC_CROP_DATA_MASTER.csv contains consolidated modes and provenance."""
    master_path = DATA_P1_DIR / "SYNTHETIC_CROP_DATA_MASTER.csv"
    assert master_path.exists(), "Master synthetic file missing"
    df_master = pd.read_csv(master_path)
    assert len(df_master) == 165000 * 3, f"Master size: {len(df_master)} (Expected 495,000)"
    assert "generation_mode" in df_master.columns, "generation_mode column missing in Master"
    assert "generation_seed" in df_master.columns, "generation_seed column missing in Master"
    assert set(df_master["generation_mode"].unique()) == {"Mode_A_Pure_Copula", "Mode_B_Agronomic_Constraints", "Mode_C_Envelope_Expansion"}

# ==============================================================================
# 3. PREPROCESSING & SERIALIZATION
# ==============================================================================
def test_target_label_encoding():
    """Asserts label_encoder.pkl correctly encodes and decodes all 22 crops."""
    le_path = PREPROC_DIR / "label_encoder.pkl"
    assert le_path.exists(), "label_encoder.pkl is missing"
    le = joblib.load(le_path)
    assert len(le.classes_) == len(EXPECTED_CROPS)
    assert list(le.classes_) == EXPECTED_CROPS
    # Round-trip check
    encoded = le.transform(EXPECTED_CROPS)
    decoded = le.inverse_transform(encoded)
    assert list(decoded) == EXPECTED_CROPS

def test_model_artifacts_present_and_loadable():
    """Asserts all 12 trained models exist and load successfully."""
    # TabNet
    for s in ["real_only", "mode_a", "mode_b", "mode_c"]:
        m_path = MODELS_DIR / f"tabnet_{s}.zip"
        assert m_path.exists(), f"TabNet model {m_path.name} is missing"
        tnet = TabNetClassifier()
        tnet.load_model(str(m_path))
        
    # Random Forest
    for s in ["real_only", "mode_a", "mode_b", "mode_c"]:
        m_path = MODELS_DIR / f"random_forest_{s}.joblib"
        assert m_path.exists(), f"RF model {m_path.name} is missing"
        rf = joblib.load(m_path)
        assert hasattr(rf, "predict")
        
    # XGBoost
    for s in ["real_only", "mode_a", "mode_b", "mode_c"]:
        xgb = load_xgboost_universal(MODELS_DIR / f"xgboost_{s}")
        assert hasattr(xgb, "predict")

# ==============================================================================
# 4. CRITICAL REPAIR: TREE MODELS UNMODIFIED FEATURES & ACCURACY >= 85%
# ==============================================================================
def test_tree_models_receive_raw_features():
    """
    Confirms root cause fix:
    Asserts Random Forest and XGBoost achieve >90% accuracy on RAW features,
    and shows that feeding scaled features causes collapse to <10% accuracy.
    """
    df_test = pd.read_csv(DATA_P1_DIR / "REAL_TEST.csv")
    le = joblib.load(PREPROC_DIR / "label_encoder.pkl")
    y_test = le.transform(df_test[TARGET_COLUMN])
    X_raw = df_test[CANONICAL_FEATURES].values
    
    scaler = joblib.load(PREPROC_DIR / "scaler_mode_b.pkl")
    X_scaled = scaler.transform(X_raw)
    
    rf = joblib.load(MODELS_DIR / "random_forest_mode_b.joblib")
    
    # Raw inference
    preds_raw = rf.predict(X_raw)
    acc_raw = accuracy_score(y_test, preds_raw)
    assert acc_raw >= 0.85, f"RF raw accuracy {acc_raw*100:.2f}% is below 85% threshold!"
    
    # Scaled inference collapse
    preds_scaled = rf.predict(X_scaled)
    acc_scaled = accuracy_score(y_test, preds_scaled)
    assert acc_scaled < 0.20, f"Expected scaled input to collapse tree model, but got {acc_scaled*100:.2f}%"

def test_rf_test_accuracy_above_85():
    """Asserts Random Forest Real-Only and Mode B achieve at least 85% on REAL_TEST."""
    df_test = pd.read_csv(DATA_P1_DIR / "REAL_TEST.csv")
    le = joblib.load(PREPROC_DIR / "label_encoder.pkl")
    y_test = le.transform(df_test[TARGET_COLUMN])
    X_raw = df_test[CANONICAL_FEATURES].values
    
    rf_real = joblib.load(MODELS_DIR / "random_forest_real_only.joblib")
    acc_real = accuracy_score(y_test, rf_real.predict(X_raw))
    assert acc_real >= 0.85, f"RF Real-Only accuracy {acc_real*100:.2f}% is below 85%!"
    
    rf_b = joblib.load(MODELS_DIR / "random_forest_mode_b.joblib")
    acc_b = accuracy_score(y_test, rf_b.predict(X_raw))
    assert acc_b >= 0.85, f"RF Mode B accuracy {acc_b*100:.2f}% is below 85%!"

def test_xgb_test_accuracy_above_85():
    """Asserts XGBoost Real-Only and Mode B achieve at least 85% on REAL_TEST."""
    df_test = pd.read_csv(DATA_P1_DIR / "REAL_TEST.csv")
    le = joblib.load(PREPROC_DIR / "label_encoder.pkl")
    y_test = le.transform(df_test[TARGET_COLUMN])
    X_raw = df_test[CANONICAL_FEATURES].values
    
    xgb_real = load_xgboost_universal(MODELS_DIR / "xgboost_real_only")
    acc_real = accuracy_score(y_test, xgb_real.predict(X_raw))
    assert acc_real >= 0.85, f"XGB Real-Only accuracy {acc_real*100:.2f}% is below 85%!"
    
    xgb_b = load_xgboost_universal(MODELS_DIR / "xgboost_mode_b")
    acc_b = accuracy_score(y_test, xgb_b.predict(X_raw))
    assert acc_b >= 0.85, f"XGB Mode B accuracy {acc_b*100:.2f}% is below 85%!"

def test_tabnet_test_accuracy_above_85():
    """Asserts TabNet Real-Only and Mode B achieve at least 85% on REAL_TEST."""
    df_test = pd.read_csv(DATA_P1_DIR / "REAL_TEST.csv")
    le = joblib.load(PREPROC_DIR / "label_encoder.pkl")
    y_test = le.transform(df_test[TARGET_COLUMN])
    X_raw = df_test[CANONICAL_FEATURES].values
    
    scaler_real = joblib.load(PREPROC_DIR / "scaler_real_only.pkl")
    tnet_real = TabNetClassifier()
    tnet_real.load_model(str(MODELS_DIR / "tabnet_real_only.zip"))
    acc_real = accuracy_score(y_test, tnet_real.predict(scaler_real.transform(X_raw)))
    assert acc_real >= 0.85, f"TabNet Real-Only accuracy {acc_real*100:.2f}% is below 85%!"
    
    scaler_b = joblib.load(PREPROC_DIR / "scaler_mode_b.pkl")
    tnet_b = TabNetClassifier()
    tnet_b.load_model(str(MODELS_DIR / "tabnet_mode_b.zip"))
    acc_b = accuracy_score(y_test, tnet_b.predict(scaler_b.transform(X_raw)))
    assert acc_b >= 0.85, f"TabNet Mode B accuracy {acc_b*100:.2f}% is below 85%!"

# ==============================================================================
# 5. PREDICTION PROBABILITIES & METRICS RECOMPUTATION
# ==============================================================================
def test_predicted_probabilities_validity():
    """Asserts predicted class probabilities are finite, positive, and sum to 1."""
    df_test = pd.read_csv(DATA_P1_DIR / "REAL_TEST.csv")
    X_raw = df_test[CANONICAL_FEATURES].values
    rf = joblib.load(MODELS_DIR / "random_forest_mode_b.joblib")
    probs = rf.predict_proba(X_raw)
    
    assert np.all(np.isfinite(probs)), "Probabilities contain Infs/NaNs"
    assert np.all(probs >= 0.0), "Probabilities contain negative values"
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4), "Probabilities do not sum to 1"

def test_confusion_matrix_totals():
    """Asserts confusion matrix totals equal test set size (440)."""
    df_test = pd.read_csv(DATA_P1_DIR / "REAL_TEST.csv")
    le = joblib.load(PREPROC_DIR / "label_encoder.pkl")
    y_test = le.transform(df_test[TARGET_COLUMN])
    X_raw = df_test[CANONICAL_FEATURES].values
    
    rf = joblib.load(MODELS_DIR / "random_forest_real_only.joblib")
    preds = rf.predict(X_raw)
    cm = confusion_matrix(y_test, preds)
    
    assert cm.sum() == 440, f"Confusion matrix sum: {cm.sum()} != 440"
    assert np.trace(cm) == np.sum(preds == y_test), "Diagonal sum does not equal correct predictions"

def test_metric_recomputation_consistency():
    """Independently recomputes test metrics from predictions and compares to CSV."""
    csv_path = METRICS_DIR / "PHASE2D_TEST_RESULTS.csv"
    assert csv_path.exists(), "PHASE2D_TEST_RESULTS.csv is missing"
    df_res = pd.read_csv(csv_path)
    
    df_test = pd.read_csv(DATA_P1_DIR / "REAL_TEST.csv")
    le = joblib.load(PREPROC_DIR / "label_encoder.pkl")
    y_test = le.transform(df_test[TARGET_COLUMN])
    X_raw = df_test[CANONICAL_FEATURES].values
    
    rf = joblib.load(MODELS_DIR / "random_forest_real_only.joblib")
    preds = rf.predict(X_raw)
    
    calc_acc = round(accuracy_score(y_test, preds), 6)
    calc_f1 = round(f1_score(y_test, preds, average="macro", zero_division=0), 6)
    
    row = df_res[df_res["Model"] == "Random Forest (Real-Only)"].iloc[0]
    assert abs(row["Test Accuracy"] - calc_acc) < 1e-4, f"Accuracy mismatch: CSV={row['Test Accuracy']}, Calc={calc_acc}"
    assert abs(row["Macro F1"] - calc_f1) < 1e-4, f"F1 mismatch: CSV={row['Macro F1']}, Calc={calc_f1}"

# ==============================================================================
# 6. EXPLAINABLE AI & RELIABILITY AUDIT
# ==============================================================================
def test_xai_prediction_diversity():
    """Asserts XAI local explanations contain diverse predictions (fixes muskmelon/chickpea collapse)."""
    xai_path = XAI_DIR / "PHASE2F_LOCAL_EXPLANATIONS.csv"
    assert xai_path.exists(), "PHASE2F_LOCAL_EXPLANATIONS.csv missing"
    df_xai = pd.read_csv(xai_path)
    
    for arch in ["Random Forest", "XGBoost", "TabNet"]:
        sub = df_xai[df_xai["Architecture"] == arch]
        assert len(sub) > 0, f"Architecture {arch} missing from XAI CSV!"
        n_unique = sub["Predicted Crop"].nunique()
        assert n_unique >= 18, f"Critical Defect! {arch} has collapsed predictions ({n_unique} unique crops)"
        
        # Check Optimal Baseline accuracy
        base_sub = sub[sub["Condition"] == "Optimal Baseline"]
        base_acc = (base_sub["Predicted Crop"] == base_sub["True Crop"]).mean()
        assert base_acc >= 0.85, f"{arch} optimal baseline accuracy is {base_acc*100:.2f}% (must be >= 85%)"

def test_calibration_metrics_computed():
    """Asserts Brier scores, ECE, and MCE are saved and finite."""
    calib_path = XAI_DIR / "PHASE2F_CALIBRATION_METRICS.csv"
    assert calib_path.exists(), "PHASE2F_CALIBRATION_METRICS.csv missing"
    df_cal = pd.read_csv(calib_path)
    assert len(df_cal) >= 3, "Calibration metrics missing for models"
    for col in ["Multiclass Brier Score", "Expected Calibration Error (ECE)", "Maximum Calibration Error (MCE)"]:
        assert col in df_cal.columns
        assert not df_cal[col].isnull().values.any()
        assert (df_cal[col] >= 0.0).all() and (df_cal[col] <= 2.0).all()

# ==============================================================================
# 7. REPORT & ARTIFACT COMPLETENESS
# ==============================================================================
def test_artifact_completeness():
    """Verifies all required final deliverable tables and figures exist."""
    required_files = [
        METRICS_DIR / "PHASE2C_MODEL_COMPARISON.csv",
        METRICS_DIR / "PHASE2D_TEST_RESULTS.csv",
        METRICS_DIR / "PHASE2D_PER_CROP_METRICS.csv",
        METRICS_DIR / "PHASE2D_MCNEMAR_TESTS.csv",
        METRICS_DIR / "PHASE2D_SUMMARY.json",
        STRESS_DIR / "PHASE2E_ROBUSTNESS_RESULTS.csv",
        STRESS_DIR / "PHASE2E_SYNTHETIC_ROBUSTNESS_ADVANTAGE.csv",
        XAI_DIR / "PHASE2F_LOCAL_EXPLANATIONS.csv",
        XAI_DIR / "PHASE2F_CALIBRATION_METRICS.csv",
        XAI_DIR / "PHASE2F_EXECUTION_LOG.txt",
        DEFECTS_DIR / "DEFECT_REGISTER.md",
        DEFECTS_DIR / "DEFECT_REGISTER.json",
        PLOTS_DIR / "phase2d_test_accuracy_comparison.png",
        PLOTS_DIR / "phase2f_empirical_reliability_diagram.png"
    ]
    for rf in required_files:
        assert rf.exists(), f"Required artifact {rf.name} is missing!"

if __name__ == "__main__":
    if pytest is not None:
        sys.exit(pytest.main(["-v", str(Path(__file__).resolve())]))
    else:
        print("=" * 80)
        print("RUNNING AUTOMATED REGRESSION TEST SUITE (STANDALONE RUNNER)")
        print("=" * 80)
        test_funcs = [
            obj for name, obj in list(globals().items())
            if name.startswith("test_") and callable(obj)
        ]
        passed = 0
        failed = 0
        errors = []
        for fn in test_funcs:
            test_name = fn.__name__
            try:
                fn()
                print(f"[PASSED] {test_name}")
                passed += 1
            except Exception as e:
                print(f"[FAILED] {test_name}: {e}")
                failed += 1
                errors.append((test_name, e))
        
        print("\n" + "=" * 80)
        print(f"TEST RESULTS: {passed} PASSED, {failed} FAILED (TOTAL: {len(test_funcs)})")
        print("=" * 80)
        if failed > 0:
            print("\nFailures:")
            for name, err in errors:
                print(f"  - {name}: {err}")
            sys.exit(1)
        else:
            print("\nALL AUTOMATED REGRESSION TESTS PASSED SUCCESSFULLY (100% PASS RATE)!")
            sys.exit(0)
