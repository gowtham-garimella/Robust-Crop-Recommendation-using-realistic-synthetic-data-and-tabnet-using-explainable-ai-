"""
Phase 2A: Real-Data Splitting, Quality Audit, and Validation Leakage Prevention
Project: Robust Crop Recommendation Using Synthetic Data, TabNet, and Explainable AI
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from sklearn.model_selection import train_test_split

curr_dir = Path(__file__).resolve().parent
if str(curr_dir) not in sys.path:
    sys.path.insert(0, str(curr_dir))

from config import (
    AUDITED_RESULTS_DIR, CANONICAL_FEATURES, TARGET_COLUMN,
    EXPECTED_CROPS, RANDOM_SEED, BENCHMARK_REAL_TRAIN_ROWS, BENCHMARK_REAL_TEST_ROWS
)

def run_phase2a_audit(
    real_train_path: Optional[Path] = None,
    real_test_path: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Runs Phase 2A quality, balance, and leakage audit."""
    if output_dir is None:
        output_dir = AUDITED_RESULTS_DIR / "01_scripts_and_config" / "phase2a_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if real_train_path is None or not real_train_path.exists():
        real_train_path = AUDITED_RESULTS_DIR / "01_scripts_and_config" / "phase1_data" / "REAL_TRAIN.csv"
    if real_test_path is None or not real_test_path.exists():
        real_test_path = AUDITED_RESULTS_DIR / "01_scripts_and_config" / "phase1_data" / "REAL_TEST.csv"
        
    print("\n" + "=" * 80)
    print("PHASE 2A: DATA SPLITS & VALIDATION LEAKAGE AUDIT")
    print("=" * 80)
    
    df_train = pd.read_csv(real_train_path)
    df_test = pd.read_csv(real_test_path)
    
    # 1. Schema & Shape Validation
    assert len(df_train) == BENCHMARK_REAL_TRAIN_ROWS, f"REAL_TRAIN size mismatch: {len(df_train)}"
    assert len(df_test) == BENCHMARK_REAL_TEST_ROWS, f"REAL_TEST size mismatch: {len(df_test)}"
    
    for df, name in [(df_train, "REAL_TRAIN"), (df_test, "REAL_TEST")]:
        for feat in CANONICAL_FEATURES:
            assert feat in df.columns, f"Missing feature {feat} in {name}"
            assert pd.to_numeric(df[feat], errors="coerce").notnull().all(), f"Non-numeric values in {name} - {feat}"
        assert not df[CANONICAL_FEATURES].isnull().values.any(), f"NaNs detected in {name}"
        assert not np.isinf(df[CANONICAL_FEATURES].values).any(), f"Infs detected in {name}"
        assert TARGET_COLUMN in df.columns, f"Missing target column {TARGET_COLUMN} in {name}"
        
    # 2. Class Balance
    train_counts = df_train[TARGET_COLUMN].value_counts().to_dict()
    test_counts = df_test[TARGET_COLUMN].value_counts().to_dict()
    
    assert len(train_counts) == len(EXPECTED_CROPS), f"Train classes mismatch: {len(train_counts)}"
    assert len(test_counts) == len(EXPECTED_CROPS), f"Test classes mismatch: {len(test_counts)}"
    assert all(v == 80 for v in train_counts.values()), "REAL_TRAIN is not balanced (80 samples/crop required)"
    assert all(v == 20 for v in test_counts.values()), "REAL_TEST is not balanced (20 samples/crop required)"
    
    # 3. Leakage Checks
    # A. Exact row overlap
    overlap_rows = pd.merge(df_train, df_test, on=CANONICAL_FEATURES + [TARGET_COLUMN], how="inner")
    # B. Feature-only overlap (duplicate feature vectors with potentially conflicting labels)
    overlap_features = pd.merge(df_train[CANONICAL_FEATURES], df_test[CANONICAL_FEATURES], on=CANONICAL_FEATURES, how="inner")
    
    assert len(overlap_rows) == 0, f"Critical Leakage! {len(overlap_rows)} exact rows shared between Train and Test!"
    assert len(overlap_features) == 0, f"Critical Leakage! {len(overlap_features)} feature vectors shared between Train and Test!"
    
    # 4. Strict Internal Train/Validation Split for Isolated Tuning
    # Real Train (1,760) -> Internal Train (1,408: 64/crop) + Internal Val (352: 16/crop)
    df_internal_train, df_internal_val = train_test_split(
        df_train,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=df_train[TARGET_COLUMN],
        shuffle=True
    )
    
    assert len(df_internal_train) == 1408
    assert len(df_internal_val) == 352
    
    internal_overlap = pd.merge(
        df_internal_train[CANONICAL_FEATURES],
        df_internal_val[CANONICAL_FEATURES],
        on=CANONICAL_FEATURES,
        how="inner"
    )
    assert len(internal_overlap) == 0, f"Internal train-val leakage! {len(internal_overlap)} rows overlap!"
    
    # Save Internal Splits
    df_internal_train.to_csv(output_dir / "INTERNAL_TRAIN.csv", index=False)
    df_internal_val.to_csv(output_dir / "INTERNAL_VAL.csv", index=False)
    
    # Export Audit Reports
    audit_summary = {
        "real_train_rows": len(df_train),
        "real_test_rows": len(df_test),
        "num_classes": len(EXPECTED_CROPS),
        "train_samples_per_crop": 80,
        "test_samples_per_crop": 20,
        "train_test_exact_matches": len(overlap_rows),
        "train_test_feature_overlap": len(overlap_features),
        "internal_train_rows": len(df_internal_train),
        "internal_val_rows": len(df_internal_val),
        "internal_train_val_overlap": len(internal_overlap),
        "leakage_audit_status": "PASSED_ZERO_LEAKAGE"
    }
    
    audit_df = pd.DataFrame([audit_summary])
    audit_df.to_csv(output_dir / "PHASE2A_LEAKAGE_AUDIT.csv", index=False)
    
    # Schema validation table
    schema_records = []
    for f in CANONICAL_FEATURES:
        schema_records.append({
            "Feature": f,
            "Train Mean": round(df_train[f].mean(), 4),
            "Train Std": round(df_train[f].std(), 4),
            "Train Min": round(df_train[f].min(), 4),
            "Train Max": round(df_train[f].max(), 4),
            "Test Mean": round(df_test[f].mean(), 4),
            "Test Std": round(df_test[f].std(), 4),
            "Null Count": int(df_train[f].isnull().sum() + df_test[f].isnull().sum())
        })
    pd.DataFrame(schema_records).to_csv(output_dir / "PHASE2A_SCHEMA_VALIDATION.csv", index=False)
    
    print("[+] Phase 2A Audit Completed Successfully:")
    print(f"    - Leakage Status: PASSED (0 exact matches, 0 feature overlaps)")
    print(f"    - Class Balance : PASSED (80 train, 20 test across all 22 classes)")
    print(f"    - Internal Train: {len(df_internal_train)} rows | Internal Val: {len(df_internal_val)} rows")
    return audit_summary

if __name__ == "__main__":
    run_phase2a_audit()
