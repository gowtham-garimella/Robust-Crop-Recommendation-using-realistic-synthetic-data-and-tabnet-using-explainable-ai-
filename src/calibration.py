"""
Module: calibration.py
Phase 7A: Multiclass Brier Score, Expected Calibration Error (ECE), and Maximum Calibration Error (MCE).
"""

import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

curr_dir = Path(__file__).resolve().parent
root_dir = curr_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config.project_config import (
    REAL_TEST_FILE, MODELS_DIR, PREPROC_DIR, EXPLAIN_DIR,
    CANONICAL_FEATURES, TARGET_COLUMN, NUM_CLASSES
)

def compute_multiclass_brier(probs: np.ndarray, y_true: np.ndarray, num_classes: int = NUM_CLASSES) -> float:
    """Computes multiclass Brier score: (1/N) * sum_i sum_k (p_ik - y_ik)^2."""
    N = len(y_true)
    y_one_hot = np.zeros((N, num_classes))
    y_one_hot[np.arange(N), y_true] = 1.0
    return float(np.mean(np.sum((probs - y_one_hot) ** 2, axis=1)))

def compute_multiclass_ece_mce(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> Tuple[float, float]:
    """
    Computes ECE and MCE using top-1 confidence partitioning.
    Empty bins are handled gracefully (skipped from contribution).
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    total_samples = len(y_true)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            
        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            abs_err = abs(bin_acc - bin_conf)
            ece += (bin_count / total_samples) * abs_err
            mce = max(mce, abs_err)
            
    return float(ece), float(mce)

def run_calibration_pipeline() -> pd.DataFrame:
    """Computes calibration metrics across all 12 model configurations on clean REAL_TEST."""
    print("=" * 80)
    print("PHASE 7A: MULTICLASS CALIBRATION AUDIT (BRIER, ECE, MCE)")
    print("=" * 80)
    
    EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)
    df_test = pd.read_csv(REAL_TEST_FILE)
    le = joblib.load(PREPROC_DIR / "label_encoder.pkl")
    y_test = le.transform(df_test[TARGET_COLUMN])
    X_test_raw = df_test[CANONICAL_FEATURES].values
    
    regimes = [
        ("Real-Only", "real_only"),
        ("Real + Mode A", "mode_a"),
        ("Real + Mode B", "mode_b"),
        ("Real + Mode C", "mode_c")
    ]
    
    records = []
    
    for r_name, suffix in regimes:
        # 1. TabNet
        scaler = joblib.load(PREPROC_DIR / f"scaler_{suffix}.pkl")
        clf_tab = TabNetClassifier()
        clf_tab.load_model(str(MODELS_DIR / f"tabnet_{suffix}.zip"))
        X_sc = scaler.transform(X_test_raw)
        p_tab = clf_tab.predict_proba(X_sc)
        assert np.allclose(p_tab.sum(axis=1), 1.0, atol=1e-5), "TabNet probabilities do not sum to 1"
        brier_tab = compute_multiclass_brier(p_tab, y_test)
        ece_tab, mce_tab = compute_multiclass_ece_mce(p_tab, y_test)
        records.append({
            "model_name": f"TabNet ({r_name})",
            "architecture": "TabNet",
            "regime": r_name,
            "brier_score": round(brier_tab, 6),
            "ece": round(ece_tab, 6),
            "mce": round(mce_tab, 6),
            "prob_sum_valid": True,
            "test_samples": len(y_test)
        })
        
        # 2. Random Forest
        clf_rf = joblib.load(MODELS_DIR / f"random_forest_{suffix}.joblib")
        p_rf = clf_rf.predict_proba(X_test_raw)
        assert np.allclose(p_rf.sum(axis=1), 1.0, atol=1e-5), "RF probabilities do not sum to 1"
        brier_rf = compute_multiclass_brier(p_rf, y_test)
        ece_rf, mce_rf = compute_multiclass_ece_mce(p_rf, y_test)
        records.append({
            "model_name": f"Random Forest ({r_name})",
            "architecture": "Random Forest",
            "regime": r_name,
            "brier_score": round(brier_rf, 6),
            "ece": round(ece_rf, 6),
            "mce": round(mce_rf, 6),
            "prob_sum_valid": True,
            "test_samples": len(y_test)
        })
        
        # 3. XGBoost
        clf_xgb = XGBClassifier()
        clf_xgb.load_model(str(MODELS_DIR / f"xgboost_{suffix}.json"))
        p_xgb = clf_xgb.predict_proba(X_test_raw)
        assert np.allclose(p_xgb.sum(axis=1), 1.0, atol=1e-5), "XGB probabilities do not sum to 1"
        brier_xgb = compute_multiclass_brier(p_xgb, y_test)
        ece_xgb, mce_xgb = compute_multiclass_ece_mce(p_xgb, y_test)
        records.append({
            "model_name": f"XGBoost ({r_name})",
            "architecture": "XGBoost",
            "regime": r_name,
            "brier_score": round(brier_xgb, 6),
            "ece": round(ece_xgb, 6),
            "mce": round(mce_xgb, 6),
            "prob_sum_valid": True,
            "test_samples": len(y_test)
        })
        
    df_cal = pd.DataFrame(records)
    out_csv = EXPLAIN_DIR / "CALIBRATION_METRICS.csv"
    df_cal.to_csv(out_csv, index=False)
    print(f"[+] Calibration metrics exported to {out_csv}")
    for _, r in df_cal.iterrows():
        print(f"    {r['model_name']:30} -> Brier: {r['brier_score']:.6f}, ECE: {r['ece']:.6f}, MCE: {r['mce']:.6f}")
    return df_cal

if __name__ == "__main__":
    run_calibration_pipeline()
