"""
Phase 2E: Distribution-Shift Stress Testing & Agronomic Suitability Verification
Project: Robust Crop Recommendation Using Synthetic Data, TabNet, and Explainable AI
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, matthews_corrcoef

curr_dir = Path(__file__).resolve().parent
if str(curr_dir) not in sys.path:
    sys.path.insert(0, str(curr_dir))

from config import (
    AUDITED_RESULTS_DIR, MODELS_DIR, PREPROC_DIR, STRESS_DIR, PLOTS_DIR,
    CANONICAL_FEATURES, TARGET_COLUMN, EXPECTED_CROPS, RANDOM_SEED
)
from phase2c_model_training_and_tuning import load_xgboost_universal
from pytorch_tabnet.tab_model import TabNetClassifier

# ------------------------------------------------------------------------------
# 9 CONTROLLED STRESS SHIFT SCENARIOS
# ------------------------------------------------------------------------------
def shift_temp_mod(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["temperature"] = d["temperature"] + 3.0
    return d

def shift_temp_sev(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["temperature"] = d["temperature"] + 5.0
    return d

def shift_drought_mod(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["rainfall"] = np.maximum(5.0, d["rainfall"] * 0.70)
    d["humidity"] = np.maximum(10.0, d["humidity"] * 0.85)
    return d

def shift_drought_sev(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["rainfall"] = np.maximum(5.0, d["rainfall"] * 0.55)
    d["humidity"] = np.maximum(10.0, d["humidity"] * 0.75)
    return d

def shift_acid_mod(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ph"] = np.maximum(3.5, d["ph"] - 1.0)
    return d

def shift_acid_sev(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ph"] = np.maximum(3.5, d["ph"] - 1.5)
    return d

def shift_alk_mod(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ph"] = np.minimum(9.5, d["ph"] + 1.0)
    return d

def shift_alk_sev(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ph"] = np.minimum(9.5, d["ph"] + 1.5)
    return d

def shift_compound(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["temperature"] = d["temperature"] + 4.0
    d["rainfall"] = np.maximum(5.0, d["rainfall"] * 0.65)
    d["humidity"] = np.maximum(10.0, d["humidity"] * 0.80)
    d["ph"] = np.maximum(3.5, d["ph"] - 1.0)
    return d

STRESS_SCENARIOS = [
    {"name": "Moderate Warming (+3°C)", "fn": shift_temp_mod, "category": "Climate Stress"},
    {"name": "Severe Climate Warming (+5°C)", "fn": shift_temp_sev, "category": "Climate Stress"},
    {"name": "Moderate Drought (-30% Rain)", "fn": shift_drought_mod, "category": "Water Deficit"},
    {"name": "Severe Drought (-45% Rain)", "fn": shift_drought_sev, "category": "Water Deficit"},
    {"name": "Soil Acidification (-1.0 pH)", "fn": shift_acid_mod, "category": "Edaphic Stress"},
    {"name": "Severe Acidification (-1.5 pH)", "fn": shift_acid_sev, "category": "Edaphic Stress"},
    {"name": "Soil Alkalinization (+1.0 pH)", "fn": shift_alk_mod, "category": "Edaphic Stress"},
    {"name": "Severe Alkalinization (+1.5 pH)", "fn": shift_alk_sev, "category": "Edaphic Stress"},
    {"name": "Severe Compound Multi-Stress", "fn": shift_compound, "category": "Multi-Compound Stress"},
]

# Physical survival envelopes from institutional ICAR/TNAU compendia
PHYSIOLOGICAL_LIMITS = {
    "rice": {"temp_max": 42.0, "rain_min": 100.0, "ph_min": 4.5, "ph_max": 8.5},
    "maize": {"temp_max": 38.0, "rain_min": 40.0, "ph_min": 5.0, "ph_max": 8.0},
    "chickpea": {"temp_max": 35.0, "rain_min": 25.0, "ph_min": 5.5, "ph_max": 8.5},
    "kidneybeans": {"temp_max": 32.0, "rain_min": 40.0, "ph_min": 5.0, "ph_max": 7.5},
    "pigeonpeas": {"temp_max": 40.0, "rain_min": 35.0, "ph_min": 5.0, "ph_max": 8.0},
    "mothbeans": {"temp_max": 45.0, "rain_min": 15.0, "ph_min": 5.0, "ph_max": 8.5},
    "mungbean": {"temp_max": 42.0, "rain_min": 30.0, "ph_min": 5.5, "ph_max": 8.0},
    "blackgram": {"temp_max": 42.0, "rain_min": 35.0, "ph_min": 5.5, "ph_max": 8.5},
    "lentil": {"temp_max": 35.0, "rain_min": 25.0, "ph_min": 5.5, "ph_max": 8.2},
    "pomegranate": {"temp_max": 45.0, "rain_min": 25.0, "ph_min": 5.5, "ph_max": 8.5},
    "banana": {"temp_max": 42.0, "rain_min": 70.0, "ph_min": 5.0, "ph_max": 8.0},
    "mango": {"temp_max": 44.0, "rain_min": 40.0, "ph_min": 5.5, "ph_max": 8.0},
    "grapes": {"temp_max": 42.0, "rain_min": 35.0, "ph_min": 5.5, "ph_max": 8.0},
    "watermelon": {"temp_max": 40.0, "rain_min": 30.0, "ph_min": 5.5, "ph_max": 7.5},
    "muskmelon": {"temp_max": 42.0, "rain_min": 30.0, "ph_min": 6.0, "ph_max": 7.5},
    "apple": {"temp_max": 32.0, "rain_min": 60.0, "ph_min": 5.5, "ph_max": 6.8},
    "orange": {"temp_max": 40.0, "rain_min": 50.0, "ph_min": 5.5, "ph_max": 7.5},
    "papaya": {"temp_max": 40.0, "rain_min": 60.0, "ph_min": 5.5, "ph_max": 7.5},
    "coconut": {"temp_max": 40.0, "rain_min": 80.0, "ph_min": 5.0, "ph_max": 8.0},
    "cotton": {"temp_max": 42.0, "rain_min": 40.0, "ph_min": 5.5, "ph_max": 8.5},
    "jute": {"temp_max": 42.0, "rain_min": 90.0, "ph_min": 5.5, "ph_max": 8.5},
    "coffee": {"temp_max": 35.0, "rain_min": 80.0, "ph_min": 4.5, "ph_max": 6.5}
}

def verify_agronomic_suitability(crop: str, sample: pd.Series) -> bool:
    """Verifies whether recommended crop is physically viable under the shifted environment."""
    lims = PHYSIOLOGICAL_LIMITS.get(crop)
    if not lims:
        return True
    if sample["temperature"] > lims.get("temp_max", 50.0):
        return False
    if sample["rainfall"] < lims.get("rain_min", 0.0):
        return False
    if sample["ph"] < lims.get("ph_min", 0.0) or sample["ph"] > lims.get("ph_max", 14.0):
        return False
    return True

def run_phase2e_stress_testing(
    test_path: Optional[Path] = None,
    models_dir: Optional[Path] = None,
    preproc_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Runs 9-scenario distribution shift testing across 12 models."""
    if test_path is None:
        test_path = AUDITED_RESULTS_DIR / "01_scripts_and_config" / "phase1_data" / "REAL_TEST.csv"
    if models_dir is None:
        models_dir = MODELS_DIR
    if preproc_dir is None:
        preproc_dir = PREPROC_DIR
    if output_dir is None:
        output_dir = STRESS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("PHASE 2E: DISTRIBUTION-SHIFT STRESS TESTING (9 SCENARIOS x 12 MODELS)")
    print("=" * 80)
    
    test_df_orig = pd.read_csv(test_path)
    if "label" in test_df_orig.columns and "crop" not in test_df_orig.columns:
        test_df_orig.rename(columns={"label": "crop"}, inplace=True)
        
    le = joblib.load(preproc_dir / "label_encoder.pkl")
    y_test_orig = le.transform(test_df_orig[TARGET_COLUMN])
    
    # Baseline accuracies from unshifted REAL_TEST
    phase2d_res = pd.read_csv(AUDITED_RESULTS_DIR / "04_evaluation_metrics" / "PHASE2D_TEST_RESULTS.csv")
    baseline_acc_map = dict(zip(phase2d_res["Model"], phase2d_res["Test Accuracy"]))
    baseline_f1_map = dict(zip(phase2d_res["Model"], phase2d_res["Macro F1"]))
    
    # Model Configurations
    model_configs = [
        ("TabNet (Real-Only)", "TabNet", "Real-Only", "tabnet_real_only", "scaler_real_only"),
        ("TabNet (Real + Mode A)", "TabNet", "Real + Mode A", "tabnet_mode_a", "scaler_mode_a"),
        ("TabNet (Real + Mode B)", "TabNet", "Real + Mode B", "tabnet_mode_b", "scaler_mode_b"),
        ("TabNet (Real + Mode C)", "TabNet", "Real + Mode C", "tabnet_mode_c", "scaler_mode_c"),
        ("Random Forest (Real-Only)", "Random Forest", "Real-Only", "random_forest_real_only", "none"),
        ("Random Forest (Real + Mode A)", "Random Forest", "Real + Mode A", "random_forest_mode_a", "none"),
        ("Random Forest (Real + Mode B)", "Random Forest", "Real + Mode B", "random_forest_mode_b", "none"),
        ("Random Forest (Real + Mode C)", "Random Forest", "Real + Mode C", "random_forest_mode_c", "none"),
        ("XGBoost (Real-Only)", "XGBoost", "Real-Only", "xgboost_real_only", "none"),
        ("XGBoost (Real + Mode A)", "XGBoost", "Real + Mode A", "xgboost_mode_a", "none"),
        ("XGBoost (Real + Mode B)", "XGBoost", "Real + Mode B", "xgboost_mode_b", "none"),
        ("XGBoost (Real + Mode C)", "XGBoost", "Real + Mode C", "xgboost_mode_c", "none"),
    ]
    
    # Preload models and scalers
    loaded_models = {}
    scalers_cache = {}
    
    for display_name, arch, regime, m_key, preproc_key in model_configs:
        if arch == "TabNet":
            m = TabNetClassifier()
            m.load_model(str(models_dir / f"{m_key}.zip"))
            loaded_models[display_name] = m
            if preproc_key not in scalers_cache:
                scalers_cache[preproc_key] = joblib.load(preproc_dir / f"{preproc_key}.pkl")
        elif arch == "Random Forest":
            loaded_models[display_name] = joblib.load(models_dir / f"{m_key}.joblib")
        elif arch == "XGBoost":
            loaded_models[display_name] = load_xgboost_universal(models_dir / m_key)
            
    robustness_records = []
    degradation_records = []
    
    for s_idx, s_info in enumerate(STRESS_SCENARIOS):
        s_name = s_info["name"]
        s_cat = s_info["category"]
        df_shifted = s_info["fn"](test_df_orig)
        
        # Verify feature actual shift
        diffs = (df_shifted[CANONICAL_FEATURES] != test_df_orig[CANONICAL_FEATURES]).any()
        assert diffs.any(), f"Scenario {s_name} failed to perturb features!"
        
        X_shifted_raw = df_shifted[CANONICAL_FEATURES].values
        
        for display_name, arch, regime, m_key, preproc_key in model_configs:
            model = loaded_models[display_name]
            
            if arch == "TabNet":
                X_eval = scalers_cache[preproc_key].transform(X_shifted_raw)
            else:
                X_eval = X_shifted_raw
                
            preds = model.predict(X_eval)
            probs = model.predict_proba(X_eval)
            pred_crops = le.inverse_transform(preds)
            
            # Experiment A: Classification Robustness (holding label fixed)
            exp_a_acc = accuracy_score(y_test_orig, preds)
            exp_a_f1 = f1_score(y_test_orig, preds, average="macro", zero_division=0)
            
            # Experiment B: Agronomic Ground Truth Suitability
            # Proportion of predictions that are physiologically viable under the shifted environment
            viable_count = 0
            for i in range(len(df_shifted)):
                if verify_agronomic_suitability(pred_crops[i], df_shifted.iloc[i]):
                    viable_count += 1
            exp_b_suitability = viable_count / len(df_shifted)
            
            # Confidence
            max_probs = np.max(probs, axis=1)
            mean_conf = float(np.mean(max_probs))
            
            # Degradation relative to unshifted holdout baseline
            base_acc = baseline_acc_map[display_name]
            base_f1 = baseline_f1_map[display_name]
            acc_drop = base_acc - exp_a_acc
            f1_drop = base_f1 - exp_a_f1
            
            robustness_records.append({
                "Scenario": s_name,
                "Category": s_cat,
                "Model": display_name,
                "Architecture": arch,
                "Training Regime": regime,
                "Baseline Accuracy": round(base_acc, 6),
                "Shifted Accuracy (Exp A)": round(exp_a_acc, 6),
                "Accuracy Degradation (Exp A)": round(acc_drop, 6),
                "Shifted Macro F1 (Exp A)": round(exp_a_f1, 6),
                "Macro F1 Degradation (Exp A)": round(f1_drop, 6),
                "Agronomic Suitability (Exp B)": round(exp_b_suitability, 6),
                "Mean Confidence": round(mean_conf, 4)
            })
            
    df_robustness = pd.DataFrame(robustness_records)
    df_robustness.to_csv(output_dir / "PHASE2E_ROBUSTNESS_RESULTS.csv", index=False)
    
    # Calculate Synthetic Robustness Advantage over Real-Only baseline
    advantage_records = []
    for arch in ["TabNet", "Random Forest", "XGBoost"]:
        real_only_name = f"{arch} (Real-Only)"
        df_arch = df_robustness[df_robustness["Architecture"] == arch]
        
        for regime in ["Real + Mode A", "Real + Mode B", "Real + Mode C"]:
            aug_name = f"{arch} ({regime})"
            df_aug = df_arch[df_arch["Model"] == aug_name]
            df_real = df_arch[df_arch["Model"] == real_only_name]
            
            mean_aug_f1 = df_aug["Shifted Macro F1 (Exp A)"].mean()
            mean_real_f1 = df_real["Shifted Macro F1 (Exp A)"].mean()
            mean_aug_suit = df_aug["Agronomic Suitability (Exp B)"].mean()
            mean_real_suit = df_real["Agronomic Suitability (Exp B)"].mean()
            
            advantage_records.append({
                "Architecture": arch,
                "Augmented Regime": regime,
                "Mean Shifted Macro F1 (Augmented)": round(mean_aug_f1, 4),
                "Mean Shifted Macro F1 (Real-Only)": round(mean_real_f1, 4),
                "Robustness F1 Advantage": round(mean_aug_f1 - mean_real_f1, 4),
                "Agronomic Suitability (Augmented)": round(mean_aug_suit, 4),
                "Agronomic Suitability (Real-Only)": round(mean_real_suit, 4),
                "Agronomic Suitability Advantage": round(mean_aug_suit - mean_real_suit, 4)
            })
            
    df_advantage = pd.DataFrame(advantage_records)
    df_advantage.to_csv(output_dir / "PHASE2E_SYNTHETIC_ROBUSTNESS_ADVANTAGE.csv", index=False)
    
    # Figures: Degradation Curves across Scenarios
    plt.figure(figsize=(14, 6), dpi=300)
    sns.lineplot(data=df_robustness, x="Scenario", y="Shifted Macro F1 (Exp A)", hue="Model", marker="o")
    plt.title("Phase 2E: Classification Macro F1 Under Controlled Distribution Shifts", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Macro F1")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "phase2e_macro_f1_degradation_all_models.png")
    plt.close()
    
    summary = {
        "status": "STRESS_TESTING_SUCCESSFUL",
        "scenarios_evaluated": len(STRESS_SCENARIOS),
        "models_evaluated": len(model_configs),
        "synthetic_advantage_summary": advantage_records,
        "distinction_note": "Experiment A measures prediction invariance holding the historical label fixed. Experiment B measures physiological viability under shifted environmental parameters."
    }
    
    with open(output_dir / "PHASE2E_SUMMARY.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print("\n[+] Phase 2E Stress Testing Completed. Results saved to PHASE2E_ROBUSTNESS_RESULTS.csv")
    print(df_advantage.to_string())
    return summary

if __name__ == "__main__":
    run_phase2e_stress_testing()
