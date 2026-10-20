"""
Module: data_splitting.py
Strict, deterministic, leak-free dataset partitioning.
"""

import sys
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split

curr_dir = Path(__file__).resolve().parent
root_dir = curr_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config.project_config import (
    RAW_DATA_FILE, REAL_TRAIN_FILE, REAL_TEST_FILE,
    INTERNAL_TRAIN_FILE, INTERNAL_VAL_FILE,
    CANONICAL_FEATURES, TARGET_COLUMN, EXPECTED_CROPS,
    BENCHMARK_RAW_ROWS, BENCHMARK_REAL_TRAIN_ROWS, BENCHMARK_REAL_TEST_ROWS,
    INTERNAL_TRAIN_ROWS, INTERNAL_VAL_ROWS, RANDOM_SEED
)

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def partition_and_quarantine_datasets(
    raw_path: Path = RAW_DATA_FILE,
    seed: int = RANDOM_SEED
) -> Dict[str, Any]:
    """
    1. Loads raw dataset (2,200 rows).
    2. Splits into REAL_TRAIN (1,760 rows, 80/crop) and REAL_TEST (440 rows, 20/crop, quarantined).
    3. Splits REAL_TRAIN strictly into INTERNAL_TRAIN (1,408 rows, 64/crop) and INTERNAL_VAL (352 rows, 16/crop).
    4. Confirms zero leakage, disjointness, and exact union.
    """
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {raw_path}")
        
    df_raw = pd.read_csv(raw_path)
    if "label" in df_raw.columns and "crop" not in df_raw.columns:
        df_raw = df_raw.rename(columns={"label": "crop"})
        
    assert len(df_raw) == BENCHMARK_RAW_ROWS, f"Raw row count mismatch: {len(df_raw)}"
    
    # 1. Partition Raw -> REAL_TRAIN (1760) and REAL_TEST (440)
    # Check if canonical files exist in DATA_DIR to preserve established split
    if REAL_TRAIN_FILE.exists() and REAL_TEST_FILE.exists():
        df_real_train = pd.read_csv(REAL_TRAIN_FILE)
        df_real_test = pd.read_csv(REAL_TEST_FILE)
        for d in [df_real_train, df_real_test]:
            if "label" in d.columns and "crop" not in d.columns:
                d.rename(columns={"label": "crop"}, inplace=True)
    else:
        df_real_train, df_real_test = train_test_split(
            df_raw,
            test_size=0.20,
            random_state=seed,
            stratify=df_raw[TARGET_COLUMN],
            shuffle=True
        )
        
    assert len(df_real_train) == BENCHMARK_REAL_TRAIN_ROWS
    assert len(df_real_test) == BENCHMARK_REAL_TEST_ROWS
    
    # 2. Partition REAL_TRAIN -> INTERNAL_TRAIN (1408) and INTERNAL_VAL (352)
    # Deterministic stratified split on REAL_TRAIN with fixed seed
    df_internal_train, df_internal_val = train_test_split(
        df_real_train,
        test_size=0.20,
        random_state=seed,
        stratify=df_real_train[TARGET_COLUMN],
        shuffle=True
    )
    
    assert len(df_internal_train) == INTERNAL_TRAIN_ROWS
    assert len(df_internal_val) == INTERNAL_VAL_ROWS
    
    # 3. Disjointness and Leakage Verifications
    feat_cols = CANONICAL_FEATURES
    t_raw = set(tuple(round(v, 4) for v in x) for x in df_raw[feat_cols].values)
    t_train = set(tuple(round(v, 4) for v in x) for x in df_real_train[feat_cols].values)
    t_test = set(tuple(round(v, 4) for v in x) for x in df_real_test[feat_cols].values)
    t_int_train = set(tuple(round(v, 4) for v in x) for x in df_internal_train[feat_cols].values)
    t_int_val = set(tuple(round(v, 4) for v in x) for x in df_internal_val[feat_cols].values)
    
    assert len(t_train & t_test) == 0, "Critical Leakage: REAL_TRAIN and REAL_TEST overlap!"
    assert (t_train | t_test) == t_raw, "Union of REAL_TRAIN and REAL_TEST does not match raw dataset!"
    
    assert len(t_int_train & t_int_val) == 0, "Critical Leakage: INTERNAL_TRAIN and INTERNAL_VAL overlap!"
    assert len(t_int_train & t_test) == 0, "Critical Leakage: INTERNAL_TRAIN and REAL_TEST overlap!"
    assert len(t_int_val & t_test) == 0, "Critical Leakage: INTERNAL_VAL and REAL_TEST overlap!"
    assert (t_int_train | t_int_val) == t_train, "Union of internal train and val does not equal REAL_TRAIN!"
    
    # 4. Save to data/
    REAL_TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_real_train.to_csv(REAL_TRAIN_FILE, index=False)
    df_real_test.to_csv(REAL_TEST_FILE, index=False)
    df_internal_train.to_csv(INTERNAL_TRAIN_FILE, index=False)
    df_internal_val.to_csv(INTERNAL_VAL_FILE, index=False)
    
    hashes = {
        "RAW_DATA": compute_sha256(raw_path),
        "REAL_TRAIN": compute_sha256(REAL_TRAIN_FILE),
        "REAL_TEST": compute_sha256(REAL_TEST_FILE),
        "INTERNAL_TRAIN": compute_sha256(INTERNAL_TRAIN_FILE),
        "INTERNAL_VAL": compute_sha256(INTERNAL_VAL_FILE)
    }
    
    print(f"[+] Dataset partitioning complete and leak-free:")
    print(f"    - REAL_TRAIN:     {len(df_real_train)} rows (SHA-256: {hashes['REAL_TRAIN'][:12]}...)")
    print(f"    - REAL_TEST:      {len(df_real_test)} rows (SHA-256: {hashes['REAL_TEST'][:12]}...)")
    print(f"    - INTERNAL_TRAIN: {len(df_internal_train)} rows (SHA-256: {hashes['INTERNAL_TRAIN'][:12]}...)")
    print(f"    - INTERNAL_VAL:   {len(df_internal_val)} rows (SHA-256: {hashes['INTERNAL_VAL'][:12]}...)")
    
    return hashes

if __name__ == "__main__":
    partition_and_quarantine_datasets()
