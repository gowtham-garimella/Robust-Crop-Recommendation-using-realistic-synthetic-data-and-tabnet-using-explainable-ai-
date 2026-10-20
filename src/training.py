"""
Module: training.py
Phase 3: Clean, Forced Model Retraining on Strictly Quarantined Training Data.

CORRECTIVE ACTION IMPLEMENTED:
1. FORCE_RETRAIN is enforced. No existing checkpoints are ever reused.
2. Real-only models are trained on exactly 1,408 records (INTERNAL_TRAIN.csv).
3. Preprocessing scalers are fitted exclusively on the permitted training arrays (n_samples_seen_ = 1408.0).
4. Augmented regimes stack INTERNAL_TRAIN.csv with newly generated synthetic data (166,408 rows).
5. INTERNAL_VAL.csv (352 rows) is strictly quarantined and used solely for validation scoring.
6. Complete training manifests with hashes, row counts, and parameters are exported.
"""

import sys
import os
import time
import json
import hashlib
import joblib
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, matthews_corrcoef
)
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

curr_dir = Path(__file__).resolve().parent
root_dir = curr_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config.project_config import (
    INTERNAL_TRAIN_FILE, INTERNAL_VAL_FILE, REAL_TEST_FILE,
    SYNTHETIC_MODE_A_FILE, SYNTHETIC_MODE_B_FILE, SYNTHETIC_MODE_C_FILE,
    RETRAINED_MODELS_DIR, RETRAINED_PREPROC_DIR, RETRAINED_METRICS_DIR,
    RETRAINED_PRED_VAL_DIR, RETRAINED_MANIFESTS_DIR,
    CANONICAL_FEATURES, TARGET_COLUMN, EXPECTED_CROPS,
    INTERNAL_TRAIN_ROWS, INTERNAL_VAL_ROWS, BENCHMARK_REAL_TEST_ROWS,
    SYNTHETIC_TOTAL_PER_MODE, RANDOM_SEED, FORCE_RETRAIN,
    RF_PARAMS, XGB_PARAMS, TABNET_PARAMS
)

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def verify_dataset_quarantine(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame):
    """Enforces mathematical disjointness and schema integrity before any training begins."""
    assert len(df_train) == INTERNAL_TRAIN_ROWS, f"Expected {INTERNAL_TRAIN_ROWS} train rows, got {len(df_train)}"
    assert len(df_val) == INTERNAL_VAL_ROWS, f"Expected {INTERNAL_VAL_ROWS} val rows, got {len(df_val)}"
    assert len(df_test) == BENCHMARK_REAL_TEST_ROWS, f"Expected {BENCHMARK_REAL_TEST_ROWS} test rows, got {len(df_test)}"
    
    # Check disjointness
    val_overlap = pd.merge(df_train[CANONICAL_FEATURES], df_val[CANONICAL_FEATURES], how="inner")
    assert len(val_overlap) == 0, f"Critical leakage: {len(val_overlap)} rows overlap between train and val!"
    
    test_overlap = pd.merge(df_train[CANONICAL_FEATURES], df_test[CANONICAL_FEATURES], how="inner")
    assert len(test_overlap) == 0, f"Critical leakage: {len(test_overlap)} rows overlap between train and test!"
    
    val_test_overlap = pd.merge(df_val[CANONICAL_FEATURES], df_test[CANONICAL_FEATURES], how="inner")
    assert len(val_test_overlap) == 0, f"Critical leakage: {len(val_test_overlap)} rows overlap between val and test!"

def run_clean_training_pipeline() -> Dict[str, Any]:
    """Executes clean retraining of all 12 model configurations from scratch."""
    pipeline_t0 = time.time()
    print("=" * 80)
    print("PHASE 3: CLEAN, FORCED MODEL RETRAINING (RETRAINED_V1)")
    print("=" * 80)

    # 1. Prepare clean directories
    RETRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RETRAINED_PREPROC_DIR.mkdir(parents=True, exist_ok=True)
    RETRAINED_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    RETRAINED_PRED_VAL_DIR.mkdir(parents=True, exist_ok=True)
    RETRAINED_MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Load and verify real partitions
    df_train = pd.read_csv(INTERNAL_TRAIN_FILE)
    df_val = pd.read_csv(INTERNAL_VAL_FILE)
    df_test = pd.read_csv(REAL_TEST_FILE)

    verify_dataset_quarantine(df_train, df_val, df_test)

    train_hash = compute_sha256(INTERNAL_TRAIN_FILE)
    val_hash = compute_sha256(INTERNAL_VAL_FILE)
    test_hash = compute_sha256(REAL_TEST_FILE)

    print(f"[+] Loaded strictly quarantined real datasets:")
    print(f"    - INTERNAL_TRAIN: {len(df_train)} rows | SHA-256: {train_hash[:16]}...")
    print(f"    - INTERNAL_VAL  : {len(df_val)} rows | SHA-256: {val_hash[:16]}... (QUARANTINED)")
    print(f"    - REAL_TEST     : {len(df_test)} rows | SHA-256: {test_hash[:16]}... (BENCHMARK)")

    # 3. Fit and Save Canonical Label Encoder
    le = LabelEncoder().fit(EXPECTED_CROPS)
    le_path = RETRAINED_PREPROC_DIR / "label_encoder.pkl"
    joblib.dump(le, le_path)
    le_hash = compute_sha256(le_path)
    print(f"[+] LabelEncoder saved: {len(le.classes_)} classes | SHA-256: {le_hash[:16]}...")

    # 4. Prepare Validation Arrays (strictly for scoring)
    X_val_raw = df_val[CANONICAL_FEATURES].values.astype(np.float32)
    y_val = le.transform(df_val[TARGET_COLUMN])

    regimes = [
        ("Real-Only", None, "real_only", 0),
        ("Real + Mode A", SYNTHETIC_MODE_A_FILE, "mode_a", SYNTHETIC_TOTAL_PER_MODE),
        ("Real + Mode B", SYNTHETIC_MODE_B_FILE, "mode_b", SYNTHETIC_TOTAL_PER_MODE),
        ("Real + Mode C", SYNTHETIC_MODE_C_FILE, "mode_c", SYNTHETIC_TOTAL_PER_MODE)
    ]

    training_manifest_entries = []
    val_metric_rows = []

    for regime_name, syn_file, suffix, n_syn in regimes:
        print("\n" + "=" * 70)
        print(f"REGIME: {regime_name.upper()} ({suffix})")
        print("=" * 70)

        # -------------------------------------------------------------
        # A. Build Training Arrays
        # -------------------------------------------------------------
        X_real = df_train[CANONICAL_FEATURES].values.astype(np.float32)
        y_real = le.transform(df_train[TARGET_COLUMN])

        if syn_file is None:
            X_train = X_real
            y_train = y_real
            expected_total = INTERNAL_TRAIN_ROWS
            dataset_desc = "INTERNAL_TRAIN.csv"
            syn_hash = "NONE"
        else:
            assert syn_file.exists(), f"Synthetic file {syn_file} not found! Run synthetic_generation.py first."
            df_syn = pd.read_csv(syn_file)
            assert len(df_syn) == n_syn, f"Expected {n_syn} synthetic rows, got {len(df_syn)}"
            
            X_syn = df_syn[CANONICAL_FEATURES].values.astype(np.float32)
            y_syn = le.transform(df_syn[TARGET_COLUMN])
            
            X_train = np.vstack([X_real, X_syn])
            y_train = np.concatenate([y_real, y_syn])
            expected_total = INTERNAL_TRAIN_ROWS + n_syn
            dataset_desc = f"INTERNAL_TRAIN.csv + {syn_file.name}"
            syn_hash = compute_sha256(syn_file)

        assert len(X_train) == expected_total, f"Training count mismatch! Expected {expected_total}, got {len(X_train)}"
        print(f"[*] Training Data Assembly:")
        print(f"    - Source: {dataset_desc}")
        print(f"    - Total rows: {len(X_train):,} (Real: {len(X_real)}, Synthetic: {n_syn})")
        print(f"    - Features: {CANONICAL_FEATURES}")

        # -------------------------------------------------------------
        # B. Fit and Save Preprocessing Scaler
        # -------------------------------------------------------------
        scaler = StandardScaler()
        scaler.fit(X_train)
        
        # Verify n_samples_seen_
        assert scaler.n_samples_seen_ == len(X_train), (
            f"Scaler sample count mismatch: {scaler.n_samples_seen_} vs {len(X_train)}"
        )
        if syn_file is None:
            assert scaler.n_samples_seen_ == 1408.0, (
                f"Real-only scaler must see exactly 1408 observations, got {scaler.n_samples_seen_}"
            )
            
        scaler_file = RETRAINED_PREPROC_DIR / f"scaler_{suffix}.pkl"
        joblib.dump(scaler, scaler_file)
        scaler_hash = compute_sha256(scaler_file)
        print(f"[+] Fitted StandardScaler saved: {scaler_file.name}")
        print(f"    - n_samples_seen_: {scaler.n_samples_seen_} (VERIFIED)")
        print(f"    - SHA-256: {scaler_hash[:16]}...")

        # -------------------------------------------------------------
        # C. Retrain Model 1: TabNet
        # -------------------------------------------------------------
        tabnet_file = RETRAINED_MODELS_DIR / f"tabnet_{suffix}.zip"
        print(f"\n[*] Training TabNet ({regime_name}) from scratch...")
        t0 = time.time()
        
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val_raw)
        
        tabnet_clf = TabNetClassifier(**TABNET_PARAMS)
        
        if syn_file is None:
            tabnet_clf.fit(
                X_train=X_train_scaled, y_train=y_train,
                max_epochs=40, patience=10,
                batch_size=128, virtual_batch_size=32
            )
        else:
            tabnet_clf.fit(
                X_train=X_train_scaled, y_train=y_train,
                max_epochs=8, patience=5,
                batch_size=2048, virtual_batch_size=256
            )
            
        tabnet_duration = round(time.time() - t0, 2)
        tabnet_clf.save_model(str(RETRAINED_MODELS_DIR / f"tabnet_{suffix}"))
        tabnet_hash = compute_sha256(tabnet_file)
        print(f"[+] TabNet trained in {tabnet_duration}s | Artifact: {tabnet_file.name} | SHA-256: {tabnet_hash[:16]}...")

        # Evaluate TabNet on internal validation
        tabnet_preds = tabnet_clf.predict(X_val_scaled)
        tabnet_probs = tabnet_clf.predict_proba(X_val_scaled)
        
        # Save validation predictions
        df_t_val_pred = pd.DataFrame({
            "sample_id": range(len(df_val)),
            "true_crop": df_val[TARGET_COLUMN],
            "true_label": y_val,
            "pred_crop": le.inverse_transform(tabnet_preds),
            "pred_label": tabnet_preds,
            "is_correct": (tabnet_preds == y_val).astype(int),
            "confidence": np.max(tabnet_probs, axis=1)
        })
        t_val_pred_file = RETRAINED_PRED_VAL_DIR / f"val_predictions_tabnet_{suffix}.csv"
        df_t_val_pred.to_csv(t_val_pred_file, index=False)

        t_acc = accuracy_score(y_val, tabnet_preds)
        t_f1 = f1_score(y_val, tabnet_preds, average="macro")
        t_prec = precision_score(y_val, tabnet_preds, average="macro", zero_division=0)
        t_rec = recall_score(y_val, tabnet_preds, average="macro", zero_division=0)
        t_bal = balanced_accuracy_score(y_val, tabnet_preds)
        t_mcc = matthews_corrcoef(y_val, tabnet_preds)
        print(f"    TabNet Val Accuracy: {t_acc:.4f}, Macro F1: {t_f1:.4f}, MCC: {t_mcc:.4f}")

        val_metric_rows.append({
            "model_name": f"TabNet ({regime_name})",
            "architecture": "TabNet",
            "regime": regime_name,
            "split": "INTERNAL_VAL",
            "val_samples": len(y_val),
            "val_accuracy": round(t_acc, 6),
            "val_macro_f1": round(t_f1, 6),
            "val_macro_precision": round(t_prec, 6),
            "val_macro_recall": round(t_rec, 6),
            "val_balanced_accuracy": round(t_bal, 6),
            "val_mcc": round(t_mcc, 6),
            "artifact_file": tabnet_file.name,
            "artifact_sha256": tabnet_hash
        })

        training_manifest_entries.append({
            "model_name": f"TabNet ({regime_name})",
            "architecture": "TabNet",
            "regime": regime_name,
            "training_samples": len(X_train),
            "real_samples": len(X_real),
            "synthetic_samples": n_syn,
            "synthetic_file": syn_file.name if syn_file else "None",
            "synthetic_sha256": syn_hash,
            "scaler_file": scaler_file.name,
            "scaler_sha256": scaler_hash,
            "scaler_n_samples_seen": float(scaler.n_samples_seen_),
            "training_duration_s": tabnet_duration,
            "random_seed": RANDOM_SEED,
            "hyperparameters": TABNET_PARAMS,
            "artifact_file": tabnet_file.name,
            "artifact_sha256": tabnet_hash
        })

        # -------------------------------------------------------------
        # D. Retrain Model 2: Random Forest
        # -------------------------------------------------------------
        rf_file = RETRAINED_MODELS_DIR / f"random_forest_{suffix}.joblib"
        print(f"\n[*] Training Random Forest ({regime_name}) from scratch...")
        t0 = time.time()
        
        rf_clf = RandomForestClassifier(**RF_PARAMS)
        rf_clf.fit(X_train, y_train)
        
        # Verify root node weighted sample count
        rf_root_weight = float(rf_clf.estimators_[0].tree_.weighted_n_node_samples[0])
        assert rf_root_weight == float(len(X_train)), (
            f"RF tree sample count mismatch: {rf_root_weight} vs {len(X_train)}"
        )
        if syn_file is None:
            assert rf_root_weight == 1408.0, f"RF real-only must be 1408.0, got {rf_root_weight}"

        joblib.dump(rf_clf, rf_file)
        rf_duration = round(time.time() - t0, 2)
        rf_hash = compute_sha256(rf_file)
        print(f"[+] RF trained in {rf_duration}s | Artifact: {rf_file.name} | SHA-256: {rf_hash[:16]}...")
        print(f"    - Tree root node sample weight: {rf_root_weight} (VERIFIED)")

        # Evaluate Random Forest on internal validation
        rf_preds = rf_clf.predict(X_val_raw)
        rf_probs = rf_clf.predict_proba(X_val_raw)

        df_rf_val_pred = pd.DataFrame({
            "sample_id": range(len(df_val)),
            "true_crop": df_val[TARGET_COLUMN],
            "true_label": y_val,
            "pred_crop": le.inverse_transform(rf_preds),
            "pred_label": rf_preds,
            "is_correct": (rf_preds == y_val).astype(int),
            "confidence": np.max(rf_probs, axis=1)
        })
        rf_val_pred_file = RETRAINED_PRED_VAL_DIR / f"val_predictions_random_forest_{suffix}.csv"
        df_rf_val_pred.to_csv(rf_val_pred_file, index=False)

        rf_acc = accuracy_score(y_val, rf_preds)
        rf_f1 = f1_score(y_val, rf_preds, average="macro")
        rf_prec = precision_score(y_val, rf_preds, average="macro", zero_division=0)
        rf_rec = recall_score(y_val, rf_preds, average="macro", zero_division=0)
        rf_bal = balanced_accuracy_score(y_val, rf_preds)
        rf_mcc = matthews_corrcoef(y_val, rf_preds)
        print(f"    RF Val Accuracy: {rf_acc:.4f}, Macro F1: {rf_f1:.4f}, MCC: {rf_mcc:.4f}")

        val_metric_rows.append({
            "model_name": f"Random Forest ({regime_name})",
            "architecture": "Random Forest",
            "regime": regime_name,
            "split": "INTERNAL_VAL",
            "val_samples": len(y_val),
            "val_accuracy": round(rf_acc, 6),
            "val_macro_f1": round(rf_f1, 6),
            "val_macro_precision": round(rf_prec, 6),
            "val_macro_recall": round(rf_rec, 6),
            "val_balanced_accuracy": round(rf_bal, 6),
            "val_mcc": round(rf_mcc, 6),
            "artifact_file": rf_file.name,
            "artifact_sha256": rf_hash
        })

        training_manifest_entries.append({
            "model_name": f"Random Forest ({regime_name})",
            "architecture": "Random Forest",
            "regime": regime_name,
            "training_samples": len(X_train),
            "real_samples": len(X_real),
            "synthetic_samples": n_syn,
            "synthetic_file": syn_file.name if syn_file else "None",
            "synthetic_sha256": syn_hash,
            "rf_root_sample_weight": rf_root_weight,
            "training_duration_s": rf_duration,
            "random_seed": RANDOM_SEED,
            "hyperparameters": RF_PARAMS,
            "artifact_file": rf_file.name,
            "artifact_sha256": rf_hash
        })

        # -------------------------------------------------------------
        # E. Retrain Model 3: XGBoost
        # -------------------------------------------------------------
        xgb_json_file = RETRAINED_MODELS_DIR / f"xgboost_{suffix}.json"
        xgb_joblib_file = RETRAINED_MODELS_DIR / f"xgboost_{suffix}.joblib"
        print(f"\n[*] Training XGBoost ({regime_name}) from scratch...")
        t0 = time.time()
        
        xgb_clf = XGBClassifier(**XGB_PARAMS)
        xgb_clf.fit(X_train, y_train)
        
        xgb_clf.save_model(str(xgb_json_file))
        joblib.dump(xgb_clf, xgb_joblib_file)
        xgb_duration = round(time.time() - t0, 2)
        xgb_hash = compute_sha256(xgb_json_file)
        print(f"[+] XGBoost trained in {xgb_duration}s | Artifact: {xgb_json_file.name} | SHA-256: {xgb_hash[:16]}...")

        # Evaluate XGBoost on internal validation
        xgb_preds = xgb_clf.predict(X_val_raw)
        xgb_probs = xgb_clf.predict_proba(X_val_raw)

        df_xgb_val_pred = pd.DataFrame({
            "sample_id": range(len(df_val)),
            "true_crop": df_val[TARGET_COLUMN],
            "true_label": y_val,
            "pred_crop": le.inverse_transform(xgb_preds),
            "pred_label": xgb_preds,
            "is_correct": (xgb_preds == y_val).astype(int),
            "confidence": np.max(xgb_probs, axis=1)
        })
        xgb_val_pred_file = RETRAINED_PRED_VAL_DIR / f"val_predictions_xgboost_{suffix}.csv"
        df_xgb_val_pred.to_csv(xgb_val_pred_file, index=False)

        xgb_acc = accuracy_score(y_val, xgb_preds)
        xgb_f1 = f1_score(y_val, xgb_preds, average="macro")
        xgb_prec = precision_score(y_val, xgb_preds, average="macro", zero_division=0)
        xgb_rec = recall_score(y_val, xgb_preds, average="macro", zero_division=0)
        xgb_bal = balanced_accuracy_score(y_val, xgb_preds)
        xgb_mcc = matthews_corrcoef(y_val, xgb_preds)
        print(f"    XGBoost Val Accuracy: {xgb_acc:.4f}, Macro F1: {xgb_f1:.4f}, MCC: {xgb_mcc:.4f}")

        val_metric_rows.append({
            "model_name": f"XGBoost ({regime_name})",
            "architecture": "XGBoost",
            "regime": regime_name,
            "split": "INTERNAL_VAL",
            "val_samples": len(y_val),
            "val_accuracy": round(xgb_acc, 6),
            "val_macro_f1": round(xgb_f1, 6),
            "val_macro_precision": round(xgb_prec, 6),
            "val_macro_recall": round(xgb_rec, 6),
            "val_balanced_accuracy": round(xgb_bal, 6),
            "val_mcc": round(xgb_mcc, 6),
            "artifact_file": xgb_json_file.name,
            "artifact_sha256": xgb_hash
        })

        training_manifest_entries.append({
            "model_name": f"XGBoost ({regime_name})",
            "architecture": "XGBoost",
            "regime": regime_name,
            "training_samples": len(X_train),
            "real_samples": len(X_real),
            "synthetic_samples": n_syn,
            "synthetic_file": syn_file.name if syn_file else "None",
            "synthetic_sha256": syn_hash,
            "training_duration_s": xgb_duration,
            "random_seed": RANDOM_SEED,
            "hyperparameters": XGB_PARAMS,
            "artifact_file": xgb_json_file.name,
            "artifact_sha256": xgb_hash
        })

    # Export Validation Metrics
    df_val_metrics = pd.DataFrame(val_metric_rows)
    val_csv_path = RETRAINED_METRICS_DIR / "VALIDATION_METRICS.csv"
    df_val_metrics.to_csv(val_csv_path, index=False)
    print(f"\n[+] Validation metrics exported to {val_csv_path}")

    # Export Training Manifest
    master_training_manifest = {
        "manifest_type": "retrained_v1_training_manifest",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_internal_train": {
            "file": INTERNAL_TRAIN_FILE.name,
            "rows": INTERNAL_TRAIN_ROWS,
            "sha256": train_hash
        },
        "quarantined_validation": {
            "file": INTERNAL_VAL_FILE.name,
            "rows": INTERNAL_VAL_ROWS,
            "sha256": val_hash
        },
        "models_retrained_count": len(training_manifest_entries),
        "models": training_manifest_entries,
        "total_training_duration_s": round(time.time() - pipeline_t0, 2)
    }
    manifest_out = RETRAINED_MANIFESTS_DIR / "RETRAINED_TRAINING_MANIFEST.json"
    with open(manifest_out, "w") as f:
        json.dump(master_training_manifest, f, indent=2)
    print(f"[+] Master training manifest saved to {manifest_out}")

    return master_training_manifest

# Backwards compatible alias
run_training_pipeline = run_clean_training_pipeline

if __name__ == "__main__":
    run_clean_training_pipeline()

