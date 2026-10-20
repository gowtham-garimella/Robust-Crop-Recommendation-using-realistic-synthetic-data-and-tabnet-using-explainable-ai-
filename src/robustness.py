"""
Module: robustness.py
Phase 6: Distribution-Shift Stress-Testing & Robustness Advantage Quantification.
"""

import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

curr_dir = Path(__file__).resolve().parent
root_dir = curr_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config.project_config import (
    REAL_TEST_FILE, MODELS_DIR, PREPROC_DIR, ROBUSTNESS_DIR,
    CANONICAL_FEATURES, TARGET_COLUMN, EXPECTED_CROPS, RANDOM_SEED
)

def run_robustness_pipeline() -> pd.DataFrame:
    """Executes distribution-shift stress testing across 5 agronomic scenarios."""
    print("=" * 80)
    print("PHASE 6: DISTRIBUTION-SHIFT STRESS TESTING & ROBUSTNESS ADVANTAGE")
    print("=" * 80)
    
    ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)
    df_test = pd.read_csv(REAL_TEST_FILE)
    le = joblib.load(PREPROC_DIR / "label_encoder.pkl")
    y_test = le.transform(df_test[TARGET_COLUMN])
    X_base = df_test[CANONICAL_FEATURES].copy()
    
    # Define 5 Agronomic Stress Scenarios
    rng = np.random.RandomState(RANDOM_SEED)
    
    scenarios = {}
    
    # 1. Temperature Heat Stress (+5°C)
    df_heat = X_base.copy()
    df_heat["temperature"] = df_heat["temperature"] + 5.0
    scenarios["Heat Stress (+5C)"] = df_heat
    
    # 2. Rainfall Deficit / Drought (-30% precipitation)
    df_drought = X_base.copy()
    df_drought["rainfall"] = np.maximum(0.0, df_drought["rainfall"] * 0.70)
    scenarios["Drought (-30% Rain)"] = df_drought
    
    # 3. Combined Climate Volatility (+4°C, -25% rainfall)
    df_climate = X_base.copy()
    df_climate["temperature"] = df_climate["temperature"] + 4.0
    df_climate["rainfall"] = np.maximum(0.0, df_climate["rainfall"] * 0.75)
    scenarios["Extreme Climate (+4C, -25% Rain)"] = df_climate
    
    # 4. Soil Degradation / Nutrient Depletion (-25% N, P, K, +0.5 pH)
    df_soil = X_base.copy()
    df_soil["N"] = np.maximum(0.0, df_soil["N"] * 0.75)
    df_soil["P"] = np.maximum(0.0, df_soil["P"] * 0.75)
    df_soil["K"] = np.maximum(0.0, df_soil["K"] * 0.75)
    df_soil["ph"] = np.minimum(14.0, df_soil["ph"] + 0.5)
    scenarios["Soil Degradation (-25% NPK, +0.5 pH)"] = df_soil
    
    # 5. Sensor Noise (Gaussian noise sigma = 0.15 * feature std)
    df_noise = X_base.copy()
    for col in CANONICAL_FEATURES:
        col_std = df_noise[col].std()
        noise = rng.normal(0, 0.15 * col_std, size=len(df_noise))
        df_noise[col] = np.maximum(0.0, df_noise[col] + noise)
    scenarios["Sensor Noise (15% Jitter)"] = df_noise
    
    # Architectures and regimes
    architectures = ["TabNet", "Random Forest", "XGBoost"]
    regimes = [
        ("Real-Only", "real_only"),
        ("Real + Mode A", "mode_a"),
        ("Real + Mode B", "mode_b"),
        ("Real + Mode C", "mode_c")
    ]
    
    tidy_records = []
    
    for arch in architectures:
        # Load all 4 regime models for this architecture
        models = {}
        scalers = {}
        for r_name, suffix in regimes:
            if arch == "TabNet":
                clf = TabNetClassifier()
                clf.load_model(str(MODELS_DIR / f"tabnet_{suffix}.zip"))
                models[r_name] = clf
                scalers[r_name] = joblib.load(PREPROC_DIR / f"scaler_{suffix}.pkl")
            elif arch == "Random Forest":
                models[r_name] = joblib.load(MODELS_DIR / f"random_forest_{suffix}.joblib")
            elif arch == "XGBoost":
                clf = XGBClassifier()
                clf.load_model(str(MODELS_DIR / f"xgboost_{suffix}.json"))
                models[r_name] = clf
                
        # 1. Compute unshifted baseline macro F1 on clean REAL_TEST
        clean_scores = {}
        for r_name, suffix in regimes:
            if arch == "TabNet":
                X_sc = scalers[r_name].transform(X_base.values)
                preds = models[r_name].predict(X_sc)
            else:
                preds = models[r_name].predict(X_base.values)
            clean_scores[r_name] = f1_score(y_test, preds, average="macro")
            
        # 2. Evaluate under each stress scenario
        for scen_name, scen_df in scenarios.items():
            scen_f1s = {}
            for r_name, suffix in regimes:
                if arch == "TabNet":
                    X_sc = scalers[r_name].transform(scen_df[CANONICAL_FEATURES].values)
                    preds = models[r_name].predict(X_sc)
                else:
                    preds = models[r_name].predict(scen_df[CANONICAL_FEATURES].values)
                scen_f1s[r_name] = f1_score(y_test, preds, average="macro")
                
            real_only_f1 = scen_f1s["Real-Only"]
            
            for r_name, suffix in regimes:
                f1_val = scen_f1s[r_name]
                clean_f1 = clean_scores[r_name]
                retention = f1_val / clean_f1 if clean_f1 > 0 else 0.0
                advantage = f1_val - real_only_f1
                
                tidy_records.append({
                    "architecture": arch,
                    "regime": r_name,
                    "scenario": scen_name,
                    "metric": "Macro F1",
                    "shifted_score": round(f1_val, 6),
                    "unshifted_baseline": round(clean_f1, 6),
                    "relative_retention": round(retention, 6),
                    "real_only_shifted_score": round(real_only_f1, 6),
                    "robustness_advantage": round(advantage, 6),
                    "test_samples": len(y_test)
                })
                
    df_tidy = pd.DataFrame(tidy_records)
    out_csv = ROBUSTNESS_DIR / "ROBUSTNESS_ADVANTAGE_TIDY.csv"
    df_tidy.to_csv(out_csv, index=False)
    print(f"[+] Robustness advantage results saved to {out_csv} ({len(df_tidy)} rows)")
    return df_tidy

if __name__ == "__main__":
    run_robustness_pipeline()
