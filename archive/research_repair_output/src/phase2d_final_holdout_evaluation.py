"""
Phase 2D: Unseen Real-Data Robustness Evaluation (Held-Out Benchmark: 440 Samples)
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

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, balanced_accuracy_score, confusion_matrix
)
import scipy.stats as stats

curr_dir = Path(__file__).resolve().parent
if str(curr_dir) not in sys.path:
    sys.path.insert(0, str(curr_dir))

from config import (
    AUDITED_RESULTS_DIR, MODELS_DIR, PREPROC_DIR, METRICS_DIR, PLOTS_DIR,
    CANONICAL_FEATURES, TARGET_COLUMN, EXPECTED_CROPS, RANDOM_SEED,
    BENCHMARK_REAL_TEST_ROWS
)
from phase2c_model_training_and_tuning import load_xgboost_universal
from pytorch_tabnet.tab_model import TabNetClassifier

def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05
) -> Dict[str, Tuple[float, float]]:
    """Computes empirical bootstrap confidence intervals for key metrics."""
    np.random.seed(RANDOM_SEED)
    n = len(y_true)
    accs, f1s, mccs = [], [], []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        yt_b = y_true[idx]
        yp_b = y_pred[idx]
        accs.append(accuracy_score(yt_b, yp_b))
        f1s.append(f1_score(yt_b, yp_b, average="macro", zero_division=0))
        mccs.append(matthews_corrcoef(yt_b, yp_b))
        
    low_pct = (alpha / 2.0) * 100
    high_pct = (1.0 - alpha / 2.0) * 100
    
    return {
        "acc_ci": (round(float(np.percentile(accs, low_pct)), 4), round(float(np.percentile(accs, high_pct)), 4)),
        "f1_ci": (round(float(np.percentile(f1s, low_pct)), 4), round(float(np.percentile(f1s, high_pct)), 4)),
        "mcc_ci": (round(float(np.percentile(mccs, low_pct)), 4), round(float(np.percentile(mccs, high_pct)), 4))
    }

def run_phase2d_evaluation(
    test_path: Optional[Path] = None,
    models_dir: Optional[Path] = None,
    preproc_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Runs final evaluation on the quarantined REAL_TEST dataset."""
    if test_path is None:
        test_path = AUDITED_RESULTS_DIR / "01_scripts_and_config" / "phase1_data" / "REAL_TEST.csv"
    if models_dir is None:
        models_dir = MODELS_DIR
    if preproc_dir is None:
        preproc_dir = PREPROC_DIR
    if output_dir is None:
        output_dir = METRICS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("PHASE 2D: FINAL EVALUATION ON QUARANTINED REAL_TEST BENCHMARK (440 SAMPLES)")
    print("=" * 80)
    
    # 1. Load REAL_TEST
    test_df = pd.read_csv(test_path)
    if "label" in test_df.columns and "crop" not in test_df.columns:
        test_df.rename(columns={"label": "crop"}, inplace=True)
        
    assert len(test_df) == BENCHMARK_REAL_TEST_ROWS, f"REAL_TEST size mismatch: {len(test_df)}"
    assert not test_df[CANONICAL_FEATURES].isnull().values.any(), "REAL_TEST contains NaNs"
    
    le = joblib.load(preproc_dir / "label_encoder.pkl")
    y_test = le.transform(test_df[TARGET_COLUMN])
    X_test_raw = test_df[CANONICAL_FEATURES].values
    
    # 2. Define 12 Model Configurations & Proper Preprocessing
    # Model entries: (display_name, architecture, regime, model_key, preproc_key)
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
    
    scalers_cache = {}
    test_results = []
    per_crop_records = []
    predictions_map = {}
    cm_records = []
    
    for display_name, arch, regime, m_key, preproc_key in model_configs:
        # Load Model
        if arch == "TabNet":
            model = TabNetClassifier()
            model_zip = models_dir / f"{m_key}.zip"
            model.load_model(str(model_zip))
            
            # Load Scaler
            if preproc_key not in scalers_cache:
                scalers_cache[preproc_key] = joblib.load(preproc_dir / f"{preproc_key}.pkl")
            X_eval = scalers_cache[preproc_key].transform(X_test_raw)
            
        elif arch == "Random Forest":
            model_path = models_dir / f"{m_key}.joblib"
            model = joblib.load(model_path)
            # TREE MODEL RECEIVES RAW UNMODIFIED FEATURES
            X_eval = X_test_raw
            
        elif arch == "XGBoost":
            model = load_xgboost_universal(models_dir / m_key)
            # TREE MODEL RECEIVES RAW UNMODIFIED FEATURES
            X_eval = X_test_raw
            
        preds = model.predict(X_eval)
        probs = model.predict_proba(X_eval)
        predictions_map[display_name] = preds
        
        # Calculate Metrics
        acc = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
        macro_prec = precision_score(y_test, preds, average="macro", zero_division=0)
        macro_rec = recall_score(y_test, preds, average="macro", zero_division=0)
        bal_acc = balanced_accuracy_score(y_test, preds)
        mcc = matthews_corrcoef(y_test, preds)
        
        # Confidence Metrics
        max_probs = np.max(probs, axis=1)
        mean_conf = float(np.mean(max_probs))
        correct_mask = (preds == y_test)
        mean_correct_conf = float(np.mean(max_probs[correct_mask])) if correct_mask.any() else 0.0
        mean_error_conf = float(np.mean(max_probs[~correct_mask])) if (~correct_mask).any() else 0.0
        
        # Entropy
        eps = 1e-12
        entropy = -np.sum(probs * np.log(probs + eps), axis=1)
        mean_entropy = float(np.mean(entropy))
        
        # Bootstrap 95% Confidence Intervals
        ci_dict = compute_bootstrap_ci(y_test, preds, n_bootstrap=1000)
        
        test_results.append({
            "Model": display_name,
            "Architecture": arch,
            "Training Regime": regime,
            "Test Accuracy": round(acc, 6),
            "Accuracy 95% CI": f"[{ci_dict['acc_ci'][0]:.4f}, {ci_dict['acc_ci'][1]:.4f}]",
            "Macro F1": round(macro_f1, 6),
            "Macro F1 95% CI": f"[{ci_dict['f1_ci'][0]:.4f}, {ci_dict['f1_ci'][1]:.4f}]",
            "Weighted F1": round(weighted_f1, 6),
            "Macro Precision": round(macro_prec, 6),
            "Macro Recall": round(macro_rec, 6),
            "Balanced Accuracy": round(bal_acc, 6),
            "MCC": round(mcc, 6),
            "MCC 95% CI": f"[{ci_dict['mcc_ci'][0]:.4f}, {ci_dict['mcc_ci'][1]:.4f}]",
            "Mean Confidence": round(mean_conf, 4),
            "Mean Correct Confidence": round(mean_correct_conf, 4),
            "Mean Error Confidence": round(mean_error_conf, 4),
            "Mean Entropy": round(mean_entropy, 4)
        })
        
        # Per-crop metrics
        cm = confusion_matrix(y_test, preds, labels=np.arange(len(EXPECTED_CROPS)))
        for c_idx, c_name in enumerate(EXPECTED_CROPS):
            tp = cm[c_idx, c_idx]
            fn = cm[c_idx, :].sum() - tp
            fp = cm[:, c_idx].sum() - tp
            support = int(cm[c_idx, :].sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            
            per_crop_records.append({
                "Model": display_name,
                "Crop": c_name,
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1-Score": round(f1, 4),
                "Support": support
            })
            
            cm_records.append({
                "Model": display_name,
                "True Class": c_name,
                "Predicted Class": c_name,
                "Count": int(tp)
            })
            
        print(f"[{display_name:30s}] Acc: {acc*100:6.2f}% (CI: [{ci_dict['acc_ci'][0]*100:.1f}%, {ci_dict['acc_ci'][1]*100:.1f}%]) | F1: {macro_f1:.4f} | Conf: {mean_conf:.3f}")
        
    df_results = pd.DataFrame(test_results)
    df_results.to_csv(output_dir / "PHASE2D_TEST_RESULTS.csv", index=False)
    
    df_per_crop = pd.DataFrame(per_crop_records)
    df_per_crop.to_csv(output_dir / "PHASE2D_PER_CROP_METRICS.csv", index=False)
    
    pd.DataFrame(cm_records).to_csv(output_dir / "PHASE2D_CONFUSION_MATRICES.csv", index=False)
    
    # 3. Pairwise McNemar Significance Tests with Holm-Bonferroni Correction
    mcnemar_records = []
    comparisons = [
        ("TabNet (Real-Only)", "TabNet (Real + Mode B)"),
        ("Random Forest (Real-Only)", "Random Forest (Real + Mode B)"),
        ("XGBoost (Real-Only)", "XGBoost (Real + Mode B)"),
        ("TabNet (Real + Mode B)", "Random Forest (Real + Mode B)"),
        ("TabNet (Real + Mode B)", "XGBoost (Real + Mode B)"),
        ("Random Forest (Real-Only)", "XGBoost (Real-Only)")
    ]
    
    raw_p_values = []
    for m1, m2 in comparisons:
        p1 = predictions_map[m1] == y_test
        p2 = predictions_map[m2] == y_test
        n_both_correct = np.sum(p1 & p2)
        n_m1_only = np.sum(p1 & (~p2))
        n_m2_only = np.sum((~p1) & p2)
        n_both_wrong = np.sum((~p1) & (~p2))
        
        table = [[n_both_correct, n_m1_only], [n_m2_only, n_both_wrong]]
        b, c = int(n_m1_only), int(n_m2_only)
        if b + c == 0:
            p_val = 1.0
        else:
            p_val = float(stats.binomtest(b, b + c, 0.5, alternative='two-sided').pvalue)
        raw_p_values.append(p_val)
        mcnemar_records.append({
            "Model 1": m1,
            "Model 2": m2,
            "M1 Correct Only": int(n_m1_only),
            "M2 Correct Only": int(n_m2_only),
            "Both Correct": int(n_both_correct),
            "Both Wrong": int(n_both_wrong),
            "Raw P-Value": round(float(p_val), 6)
        })
        
    # Apply Holm-Bonferroni correction
    sorted_indices = np.argsort(raw_p_values)
    m = len(raw_p_values)
    for rank, orig_idx in enumerate(sorted_indices):
        adjusted_p = min(1.0, raw_p_values[orig_idx] * (m - rank))
        mcnemar_records[orig_idx]["Adjusted P-Value (Holm)"] = round(float(adjusted_p), 6)
        mcnemar_records[orig_idx]["Statistically Significant (p < 0.05)"] = bool(adjusted_p < 0.05)
        
    df_mcnemar = pd.DataFrame(mcnemar_records)
    df_mcnemar.to_csv(output_dir / "PHASE2D_MCNEMAR_TESTS.csv", index=False)
    
    # 4. Generate Publication Figures
    plt.figure(figsize=(12, 6), dpi=300)
    sns.barplot(data=df_results, x="Model", y="Test Accuracy", hue="Architecture", palette="viridis")
    plt.axhline(0.85, color="red", linestyle="--", lw=1.5, label="85% Minimum Acceptance Threshold")
    plt.title("Holdout Test Accuracy on Quarantined REAL_TEST (440 Samples)", fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0.80, 1.01)
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "phase2d_test_accuracy_comparison.png")
    plt.close()
    
    # Macro F1 Plot
    plt.figure(figsize=(12, 6), dpi=300)
    sns.barplot(data=df_results, x="Model", y="Macro F1", hue="Architecture", palette="magma")
    plt.axhline(0.85, color="red", linestyle="--", lw=1.5, label="85% Acceptance Threshold")
    plt.title("Holdout Macro F1-Score Across 12 Model Configurations", fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0.80, 1.01)
    plt.ylabel("Macro F1")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "phase2d_test_macro_f1_comparison.png")
    plt.close()
    
    summary = {
        "status": "EVALUATION_SUCCESSFUL",
        "dataset": "REAL_TEST.csv",
        "total_test_samples": len(test_df),
        "target_accuracy_requirement_passed": bool(all(r["Test Accuracy"] >= 0.85 for r in test_results if r["Architecture"] in ["Random Forest", "XGBoost"])),
        "models_evaluated": len(test_results),
        "results": test_results
    }
    
    with open(output_dir / "PHASE2D_SUMMARY.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print("\n[+] Phase 2D Final Evaluation Completed. Results saved to PHASE2D_TEST_RESULTS.csv")
    return summary

if __name__ == "__main__":
    run_phase2d_evaluation()
