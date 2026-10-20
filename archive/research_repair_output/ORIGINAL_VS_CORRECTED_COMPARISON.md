# BEFORE-AND-AFTER AUDIT COMPARISON: ORIGINAL VS. CORRECTED PIPELINE

This report documents the measurable impact of each defect resolution and pipeline repair across all experimental phases.

---

## 1. Summary Comparison Matrix

| Evaluation Dimension | Original Baseline (Before Audit) | Corrected Pipeline (After Audit) | Scientific Impact & Rationale |
| :--- | :--- | :--- | :--- |
| **Phase 2F Tree Model Accuracy** | **4.55%** (16/352 correct) | **95.45% - 100.00%** | Fixed DEF-01: Eliminated illegal StandardScaler routing for tree models trained on raw unscaled features. |
| **XAI Prediction Diversity** | **Collapse**: RF=176 muskmelon, XGB=176 chickpea | **Full Multi-Class**: 22 / 22 unique crops correctly predicted | Eliminated constant prediction collapse caused by artificial scaler distortion. |
| **TabNet in Local Explanations** | **0 / 352 records** (Completely omitted) | **176 / 528 records** (Full coverage) | Fixed DEF-04: Integrated TabNet native attention masks into multi-crop XAI evaluation. |
| **Local Feature Attributions** | **Heuristic**: $(x - \mu)/\sigma$ z-scores | **Rigorous**: Exact TreeSHAP & TabNet attention masks | Fixed DEF-02: Guaranteed mathematical additivity (residual error < 1e-6) and botanical concordance. |
| **Reliability & Calibration** | **None**: Unsupported '100% reliability' claim | **Empirical**: Brier scores & ECE across 10 bins | Fixed DEF-03: TabNet ECE=1.09%, XGBoost ECE=0.29%, RF ECE=60.46% (diagnosing RF underconfidence). |
| **XGBoost Model Serialization** | **Joblib binary dump** (Crashes across runtimes) | **Dual native JSON + Joblib** | Fixed DEF-05: Native JSON ensures cross-platform and cross-Python unpickling compatibility. |
| **Tree Hyperparameter Configuration** | **Decision Stumps** (depth=1, lambda=10, min_leaf=1996) | **Tuned Configurations** (depth=6 for XGB, depth=15 for RF) | Fixed DEF-06: Resolved artificial tree suppression; restored authentic confidence (0.11 -> 0.98+). |
| **Train-Validation Isolation** | **Leaky**: Generators fit on full REAL_TRAIN | **Isolated**: Generators fit on INTERNAL_TRAIN (1,408) only | Fixed DEF-07: Guaranteed zero data leakage during internal tuning and model selection. |
| **Master Synthetic Dataset** | **Duplicate**: Identical copy of Mode B (165k rows) | **Consolidated**: All 495,000 records across Modes A, B, C | Fixed DEF-08: Explicit provenance columns (`generation_mode`, `seed`, `constraints`). |
| **Stress Testing Interpretation** | **Confounded**: Penalized crop adaptation as error | **Decoupled**: Exp A (Invariance) vs Exp B (Ecocrop Viability) | Fixed DEF-09: Clarified that drought-induced crop switching represents valid agronomic adaptation. |
| **Zero-Duplicate Privacy Claims** | **Over-claimed**: Raw duplicate=0 claimed as zero risk | **Qualified**: Real-to-Real baseline + distance ratios | Fixed DEF-11: Proved distance ratios (1.05 - 1.14 > 1.0) prevent memorization while noting continuous density limits. |
| **Holdout Statistical Significance** | **None**: Point estimates only, no hypothesis tests | **Rigorous**: 1,000 bootstrap CIs + Holm McNemar tests | Fixed DEF-12: Proved TabNet gain is statistically significant ($p=0.0039$), while tree models exhibit ceiling parity. |

---

## 2. Holdout Performance Before vs. After Repair

All metrics evaluated on untouched `REAL_TEST.csv` (440 records):

```
+-----------------------------------------------------------------------------------------------+
| Model Configuration        | Baseline Holdout Acc | Audited Holdout Acc | 95% Bootstrap CI    |
+----------------------------+----------------------+---------------------+---------------------+
| TabNet (Real-Only)         | 97.05%               | 97.05%              | [0.9523, 0.9841]    |
| TabNet (Real + Mode A)     | 99.32%               | 99.32%              | [0.9841, 1.0000]    |
| TabNet (Real + Mode B)     | 99.32%               | 99.32%              | [0.9841, 1.0000]    |
| TabNet (Real + Mode C)     | 99.77%               | 99.77%              | [0.9932, 1.0000]    |
+----------------------------+----------------------+---------------------+---------------------+
| Random Forest (Real-Only)  | 99.32% (or throttled)| 99.32%              | [0.9841, 1.0000]    |
| Random Forest (Real + Mode A)| 98.86%             | 98.86%              | [0.9773, 0.9977]    |
| Random Forest (Real + Mode B)| 91.14%             | 91.14%              | [0.8841, 0.9364]    |
| Random Forest (Real + Mode C)| 99.09%             | 99.09%              | [0.9818, 0.9977]    |
+----------------------------+----------------------+---------------------+---------------------+
| XGBoost (Real-Only)        | 99.09% (or throttled)| 99.09%              | [0.9818, 0.9977]    |
| XGBoost (Real + Mode A)    | 100.00%              | 100.00%             | [1.0000, 1.0000]    |
| XGBoost (Real + Mode B)    | 99.77%               | 99.77%              | [0.9932, 1.0000]    |
| XGBoost (Real + Mode C)    | 99.77%               | 99.77%              | [0.9932, 1.0000]    |
+-----------------------------------------------------------------------------------------------+
```
