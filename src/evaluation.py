"""
Module: evaluation.py
Phase 4 & Phase 5: Frozen Holdout Benchmark Evaluation & Complete Metric Recalculation.
"""

import sys
import json
import joblib
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

curr_dir = Path(__file__).resolve().parent
root_dir = curr_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, matthews_corrcoef, confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

from config.project_config import (
    REAL_TEST_FILE, MODELS_DIR, PREPROC_DIR,
    METRICS_DIR, PREDICTIONS_DIR, CONFUSION_DIR,
    CANONICAL_FEATURES, TARGET_COLUMN, EXPECTED_CROPS,
    BENCHMARK_REAL_TEST_ROWS
)

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def run_evaluation_pipeline() -> pd.DataFrame:
    """Evaluates all 12 model configurations on the frozen historical benchmark (REAL_TEST.csv)."""
    print("=" * 80)
    print("PHASE 4 & 5: BENCHMARK EVALUATION & ROW PREDICTION EXPORT")
    print("=" * 80)
    
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    CONFUSION_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    
    df_test = pd.read_csv(REAL_TEST_FILE)
    test_hash = compute_sha256(REAL_TEST_FILE)
    assert len(df_test) == BENCHMARK_REAL_TEST_ROWS
    
    le = joblib.load(PREPROC_DIR / "label_encoder.pkl")
    X_test_raw = df_test[CANONICAL_FEATURES].values
    y_test = le.transform(df_test[TARGET_COLUMN])
    
    regimes = [
        ("Real-Only", "real_only"),
        ("Real + Mode A", "mode_a"),
        ("Real + Mode B", "mode_b"),
        ("Real + Mode C", "mode_c")
    ]
    
    model_configs = []
    # 1. TabNet
    for r_name, suffix in regimes:
        model_configs.append((
            f"TabNet ({r_name})", "TabNet", r_name,
            MODELS_DIR / f"tabnet_{suffix}.zip",
            PREPROC_DIR / f"scaler_{suffix}.pkl",
            "tabnet"
        ))
    # 2. Random Forest
    for r_name, suffix in regimes:
        model_configs.append((
            f"Random Forest ({r_name})", "Random Forest", r_name,
            MODELS_DIR / f"random_forest_{suffix}.joblib",
            None,
            "rf"
        ))
    # 3. XGBoost
    for r_name, suffix in regimes:
        model_configs.append((
            f"XGBoost ({r_name})", "XGBoost", r_name,
            MODELS_DIR / f"xgboost_{suffix}.json",
            None,
            "xgb"
        ))
        
    benchmark_records = []
    
    for model_name, arch, regime, model_path, preproc_path, m_type in model_configs:
        assert model_path.exists(), f"Missing model artifact: {model_path}"
        model_sha256 = compute_sha256(model_path)
        
        # Load and predict
        if m_type == "tabnet":
            scaler = joblib.load(preproc_path)
            clf = TabNetClassifier()
            clf.load_model(str(model_path))
            X_eval = scaler.transform(X_test_raw)
            probs = clf.predict_proba(X_eval)
            preds = np.argmax(probs, axis=1)
            confidences = np.max(probs, axis=1)
        elif m_type == "rf":
            clf = joblib.load(model_path)
            probs = clf.predict_proba(X_test_raw)
            preds = clf.predict(X_test_raw)
            confidences = np.max(probs, axis=1)
        elif m_type == "xgb":
            clf = XGBClassifier()
            clf.load_model(str(model_path))
            probs = clf.predict_proba(X_test_raw)
            preds = clf.predict(X_test_raw)
            confidences = np.max(probs, axis=1)
            
        # Verify prediction count
        assert len(preds) == BENCHMARK_REAL_TEST_ROWS
        n_correct = int((preds == y_test).sum())
        n_errors = int((preds != y_test).sum())
        
        # Row-level predictions dataframe
        pred_df = pd.DataFrame({
            "record_id": list(range(len(y_test))),
            "true_crop": df_test[TARGET_COLUMN].values,
            "true_label_idx": y_test,
            "pred_crop": le.inverse_transform(preds),
            "pred_label_idx": preds,
            "is_correct": (preds == y_test),
            "confidence": np.round(confidences, 6),
            "model_name": model_name,
            "model_sha256": model_sha256,
            "eval_dataset_sha256": test_hash
        })
        
        clean_model_tag = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
        pred_csv_path = PREDICTIONS_DIR / f"predictions_{clean_model_tag}.csv"
        pred_df.to_csv(pred_csv_path, index=False)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, preds, labels=list(range(len(EXPECTED_CROPS))))
        assert cm.sum() == BENCHMARK_REAL_TEST_ROWS, f"Confusion matrix total mismatch: {cm.sum()}"
        assert np.trace(cm) == n_correct, f"Confusion matrix diagonal mismatch: {np.trace(cm)} vs {n_correct}"
        
        cm_df = pd.DataFrame(cm, index=EXPECTED_CROPS, columns=EXPECTED_CROPS)
        cm_csv_path = CONFUSION_DIR / f"cm_{clean_model_tag}.csv"
        cm_df.to_csv(cm_csv_path)
        
        # Comprehensive Metrics
        acc = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average="macro")
        weighted_f1 = f1_score(y_test, preds, average="weighted")
        macro_prec = precision_score(y_test, preds, average="macro", zero_division=0)
        macro_rec = recall_score(y_test, preds, average="macro", zero_division=0)
        bal_acc = balanced_accuracy_score(y_test, preds)
        mcc = matthews_corrcoef(y_test, preds)
        
        benchmark_records.append({
            "model_name": model_name,
            "architecture": arch,
            "regime": regime,
            "eval_split": "REAL_TEST (Frozen Benchmark)",
            "test_samples": BENCHMARK_REAL_TEST_ROWS,
            "correct_samples": n_correct,
            "incorrect_samples": n_errors,
            "accuracy": round(acc, 6),
            "macro_f1": round(macro_f1, 6),
            "weighted_f1": round(weighted_f1, 6),
            "macro_precision": round(macro_prec, 6),
            "macro_recall": round(macro_rec, 6),
            "balanced_accuracy": round(bal_acc, 6),
            "matthews_corrcoef": round(mcc, 6),
            "cm_diagonal_sum": int(np.trace(cm)),
            "cm_total_samples": int(cm.sum()),
            "artifact_file": model_path.name,
            "artifact_sha256": model_sha256,
            "dataset_sha256": test_hash
        })
        
        print(f"[+] {model_name:30} -> Acc: {acc:.4f} ({n_correct}/{BENCHMARK_REAL_TEST_ROWS}), Macro F1: {macro_f1:.4f}, Balanced Acc: {bal_acc:.4f}")
        
    df_results = pd.DataFrame(benchmark_records)
    
    # Save Benchmark Metrics and Final Comparison
    out_bm = METRICS_DIR / "BENCHMARK_METRICS.csv"
    out_cmp = METRICS_DIR / "FINAL_MODEL_COMPARISON.csv"
    df_results.to_csv(out_bm, index=False)
    df_results.to_csv(out_cmp, index=False)
    
    print(f"\n[+] Final comparison metrics exported to:\n    - {out_bm}\n    - {out_cmp}")
    return df_results

if __name__ == "__main__":
    run_evaluation_pipeline()
