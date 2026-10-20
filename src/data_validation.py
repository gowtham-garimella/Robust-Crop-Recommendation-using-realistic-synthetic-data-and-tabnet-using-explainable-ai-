"""
Module: data_validation.py
Comprehensive data quality, physical invariant, and leakage validation.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

curr_dir = Path(__file__).resolve().parent
root_dir = curr_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config.project_config import (
    RAW_DATA_FILE, REAL_TRAIN_FILE, REAL_TEST_FILE,
    INTERNAL_TRAIN_FILE, INTERNAL_VAL_FILE,
    SYNTHETIC_MODE_A_FILE, SYNTHETIC_MODE_B_FILE, SYNTHETIC_MODE_C_FILE,
    CANONICAL_FEATURES, TARGET_COLUMN, EXPECTED_CROPS,
    PHYSICAL_RANGES, DATA_DIR
)

def validate_dataframe(df: pd.DataFrame, name: str, expected_rows: int = None, expected_classes: int = 22) -> Dict[str, Any]:
    """Validates schema, nulls, infs, ranges, and classes for a given dataframe."""
    target_col = TARGET_COLUMN if TARGET_COLUMN in df.columns else "label"
    assert target_col in df.columns, f"Target column '{TARGET_COLUMN}' missing in {name}"
    
    for feat in CANONICAL_FEATURES:
        assert feat in df.columns, f"Feature '{feat}' missing in {name}"
        assert pd.to_numeric(df[feat], errors="coerce").notnull().all(), f"Non-numeric values in {name} - {feat}"
        
    assert not df[CANONICAL_FEATURES].isnull().values.any(), f"NaNs found in {name}"
    assert not np.isinf(df[CANONICAL_FEATURES].values).any(), f"Infs found in {name}"
    
    # Check physical bounds
    for feat, (low, high) in PHYSICAL_RANGES.items():
        assert (df[feat] >= low).all(), f"Feature {feat} below physical minimum {low} in {name}"
        assert (df[feat] <= high).all(), f"Feature {feat} exceeds physical maximum {high} in {name}"
        
    # Check classes
    unique_crops = sorted(df[target_col].unique())
    assert len(unique_crops) == expected_classes, f"Expected {expected_classes} classes, got {len(unique_crops)} in {name}"
    assert unique_crops == EXPECTED_CROPS, f"Unexpected crop labels in {name}"
    
    if expected_rows is not None:
        assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)} in {name}"
        
    return {
        "name": name,
        "rows": len(df),
        "columns": list(df.columns),
        "classes": len(unique_crops),
        "status": "VALID"
    }

def run_comprehensive_validation() -> Dict[str, Any]:
    """Executes full audit across raw, split, and synthetic datasets."""
    print("=" * 80)
    print("RUNNING COMPREHENSIVE DATA VALIDATION & LEAKAGE AUDIT")
    print("=" * 80)
    
    reports = {}
    reports["RAW"] = validate_dataframe(pd.read_csv(RAW_DATA_FILE), "Raw Dataset", 2200)
    reports["REAL_TRAIN"] = validate_dataframe(pd.read_csv(REAL_TRAIN_FILE), "REAL_TRAIN", 1760)
    reports["REAL_TEST"] = validate_dataframe(pd.read_csv(REAL_TEST_FILE), "REAL_TEST", 440)
    reports["INTERNAL_TRAIN"] = validate_dataframe(pd.read_csv(INTERNAL_TRAIN_FILE), "INTERNAL_TRAIN", 1408)
    reports["INTERNAL_VAL"] = validate_dataframe(pd.read_csv(INTERNAL_VAL_FILE), "INTERNAL_VAL", 352)
    
    for mode_name, mode_file in [("Mode A", SYNTHETIC_MODE_A_FILE), ("Mode B", SYNTHETIC_MODE_B_FILE), ("Mode C", SYNTHETIC_MODE_C_FILE)]:
        reports[mode_name] = validate_dataframe(pd.read_csv(mode_file), mode_name, 165000)
        
    # Leakage check: zero exact feature matches with REAL_TEST
    df_test = pd.read_csv(REAL_TEST_FILE)
    t_test = set(tuple(round(v, 4) for v in x) for x in df_test[CANONICAL_FEATURES].values)
    
    for split_name, split_file in [
        ("REAL_TRAIN", REAL_TRAIN_FILE),
        ("INTERNAL_TRAIN", INTERNAL_TRAIN_FILE),
        ("INTERNAL_VAL", INTERNAL_VAL_FILE),
        ("Mode A", SYNTHETIC_MODE_A_FILE),
        ("Mode B", SYNTHETIC_MODE_B_FILE),
        ("Mode C", SYNTHETIC_MODE_C_FILE),
    ]:
        df_curr = pd.read_csv(split_file)
        t_curr = set(tuple(round(v, 4) for v in x) for x in df_curr[CANONICAL_FEATURES].values)
        overlap = len(t_curr & t_test)
        assert overlap == 0, f"Critical Leakage! {split_name} shares {overlap} samples with REAL_TEST!"
        reports[split_name]["leakage_with_real_test"] = 0
        
    # Disjointness between internal train and internal val
    df_itrain = pd.read_csv(INTERNAL_TRAIN_FILE)
    df_ival = pd.read_csv(INTERNAL_VAL_FILE)
    t_itrain = set(tuple(round(v, 4) for v in x) for x in df_itrain[CANONICAL_FEATURES].values)
    t_ival = set(tuple(round(v, 4) for v in x) for x in df_ival[CANONICAL_FEATURES].values)
    internal_overlap = len(t_itrain & t_ival)
    assert internal_overlap == 0, f"Internal train-val overlap: {internal_overlap}"
    
    out_path = DATA_DIR / "manifests" / "DATA_VALIDATION_REPORT.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)
        
    print(f"[+] All datasets validated successfully. Zero leakage confirmed. Report saved to {out_path}")
    return reports

if __name__ == "__main__":
    run_comprehensive_validation()
