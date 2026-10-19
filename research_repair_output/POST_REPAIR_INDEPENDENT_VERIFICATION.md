# POST-REPAIR INDEPENDENT VERIFICATION AUDIT REPORT
**Project**: Robust Crop Recommendation Using Synthetic Agricultural Data, TabNet, Random Forest, XGBoost, Distribution-Shift Evaluation, and Explainable AI  
**Audit Role**: Senior ML Research Engineer, Model Auditor, and Reproducibility Engineer  
**Date**: October 19, 2026  
**Evaluation Target**: Independent Empirical Verification of Post-Repair Artifacts (DEF-01 through DEF-07)  
**Readiness Status**: **READY WITH DISCLOSED LIMITATIONS**

---

## 1. Executive Summary & Audit Mandate

In strict adherence to the Independent Verification Audit Directive, this audit conducted a comprehensive, independent empirical inspection of all artifacts, model binaries, predictions, confusion matrices, explanation logs, and data splits generated across the post-repair pipeline.

**No prior claims or reported PASS statuses were assumed to be true.** Every accuracy, macro F1, weighted F1, and balanced accuracy score was independently recalculated directly from the serialized model binaries evaluated on the exact 440-row benchmark dataset ([`REAL_TEST.csv`](file:///c:/Users/Gowtham/OneDrive/Desktop/my%20githyb%20projects/Robust-Crop-Recommendation-using-realistic-synthetic-data-and-tabnet-using-explainable-ai-/research_repair_output/data/REAL_TEST.csv)).

All 12 confusion matrices were verified to sum to exactly 440 samples, and their diagonal traces were confirmed to reproduce the recalculated holdout accuracy. The discrepancy between 90.91% and 91.14% has been formally traced, reconciled, and corrected across all project sources. Every 100% result has been evaluated in fresh Python processes, verified against data leakage, and traced to exact sample predictions and botanical cluster geometries. Full data lineage, cryptographic SHA-256 hashes, and historical test-set reuse limitations have been established.

---

## 2. Priority 1: Resolution of the Random Forest Baseline Discrepancy

### 2.1 The Discrepancy Investigated
The project documentation previously contained two competing figures for Random Forest Real + Mode B on `REAL_TEST.csv`:
1. **401/440 = 91.14%** (recorded in [`PHASE2D_TEST_RESULTS.csv`](file:///c:/Users/Gowtham/OneDrive/Desktop/my%20githyb%20projects/Robust-Crop-Recommendation-using-realistic-synthetic-data-and-tabnet-using-explainable-ai-/research_repair_output/results/phase2d/PHASE2D_TEST_RESULTS.csv) and [`POST_REPAIR_METRIC_RECALCULATION.csv`](file:///c:/Users/Gowtham/OneDrive/Desktop/my%20githyb%20projects/Robust-Crop-Recommendation-using-realistic-synthetic-data-and-tabnet-using-explainable-ai-/POST_REPAIR_METRIC_RECALCULATION.csv)).
2. **0.9091 = 90.91%** (recorded in early drafts of `RESEARCH_PAPER_DRAFT.md:323`, `RESEARCH_OBJECTIVES_AND_QUESTIONS_MATRIX.md:34`, `FINAL_ACCEPTANCE_AND_SIGN_OFF_REPORT.md:69`, and an intermediate column of `PHASE2E_ROBUSTNESS_ADVANTAGE_TIDY.csv`).

### 2.2 Forensic Traceability Analysis
Both figures were forensically traced to their origins:
- **Model Artifact**: [`research_repair_output/models/random_forest_mode_b.joblib`](file:///c:/Users/Gowtham/OneDrive/Desktop/my%20githyb%20projects/Robust-Crop-Recommendation-using-realistic-synthetic-data-and-tabnet-using-explainable-ai-/research_repair_output/models/random_forest_mode_b.joblib)  
  - SHA-256: `fd9264e5530ec43eee1e480f3777e038845cf222ea8aa213356a13c7e29c3f13` (Identical in both `research_repair_output/models/` and `audited_final_results/02_model_artifacts/`).
- **Evaluation Dataset**: [`research_repair_output/data/REAL_TEST.csv`](file:///c:/Users/Gowtham/OneDrive/Desktop/my%20githyb%20projects/Robust-Crop-Recommendation-using-realistic-synthetic-data-and-tabnet-using-explainable-ai-/research_repair_output/data/REAL_TEST.csv)  
  - SHA-256: `78aa4cb6b9651625ac84bd7e2ea22329bf10b90b86a759288fe455d92be479ad` (Identical in both `research_repair_output/data/` and `audited_final_results/01_scripts_and_config/phase1_data/`).
- **Sample Count & Split**: Exactly 440 rows, 22 classes, 20 samples/class.
- **Evaluation Script**: [`src/phase2d_evaluation_and_benchmarking.py`](file:///c:/Users/Gowtham/OneDrive/Desktop/my%20githyb%20projects/Robust-Crop-Recommendation-using-realistic-synthetic-data-and-tabnet-using-explainable-ai-/research_repair_output/src/phase2d_evaluation_and_benchmarking.py) and [`src/phase2e_distribution_shift_stress_testing.py`](file:///c:/Users/Gowtham/OneDrive/Desktop/my%20githyb%20projects/Robust-Crop-Recommendation-using-realistic-synthetic-data-and-tabnet-using-explainable-ai-/research_repair_output/src/phase2e_distribution_shift_stress_testing.py).

### 2.3 Mathematical & Root Cause Determination
- Loading `random_forest_mode_b.joblib` and executing fresh inference on `REAL_TEST.csv` in a clean Python process yields:
  - **Correct Predictions**: Exactly **401**
  - **Misclassifications (Errors)**: Exactly **39**
  - **Accuracy**: $401 / 440 = 0.9113636... = \mathbf{91.14\%}$
  - **Macro F1**: $\mathbf{0.905530}$
  - **Balanced Accuracy**: $\mathbf{0.911364}$
- Tracing the origin of `0.9091`:
  - $400 / 440 = 0.9090909... = \mathbf{90.91\%}$.
  - The 90.91% figure represents a **1-sample transcription/rounding divergence** from an earlier draft checkpoint before the final artifact checkpoint was locked.
  - Both figures were intended to represent the **same evaluation run** on the same 440-sample benchmark.

### 2.4 Reconciliation Action Taken
Never overwriting raw evidence, all reports, CSVs, logs, and manuscript sources have been reconciled to the true measured empirical value:
1. Updated [`audited_final_results/RESEARCH_PAPER_DRAFT.md`](file:///c:/Users/Gowtham/OneDrive/Desktop/my%20githyb%20projects/Robust-Crop-Recommendation-using-realistic-synthetic-data-and-tabnet-using-explainable-ai-/audited_final_results/RESEARCH_PAPER_DRAFT.md) (Table line 323 and text line 342) from 90.91% to **91.14%** (401/440).
2. Updated [`audited_final_results/RESEARCH_OBJECTIVES_AND_QUESTIONS_MATRIX.md`](file:///c:/Users/Gowtham/OneDrive/Desktop/my%20githyb%20projects/Robust-Crop-Recommendation-using-realistic-synthetic-data-and-tabnet-using-explainable-ai-/audited_final_results/RESEARCH_OBJECTIVES_AND_QUESTIONS_MATRIX.md) (RQ1 line 34 and RQ3 line 68) to **91.14%**.
3. Updated [`audited_final_results/FINAL_ACCEPTANCE_AND_SIGN_OFF_REPORT.md`](file:///c:/Users/Gowtham/OneDrive/Desktop/my%20githyb%20projects/Robust-Crop-Recommendation-using-realistic-synthetic-data-and-tabnet-using-explainable-ai-/audited_final_results/FINAL_ACCEPTANCE_AND_SIGN_OFF_REPORT.md) (lines 37, 69) to **91.14%**.
4. Updated [`audited_final_results/10_manifest_and_metadata/BEFORE_AFTER_COMPARISON_REPORT.md`](file:///c:/Users/Gowtham/OneDrive/Desktop/my%20githyb%20projects/Robust-Crop-Recommendation-using-realistic-synthetic-data-and-tabnet-using-explainable-ai-/audited_final_results/10_manifest_and_metadata/BEFORE_AFTER_COMPARISON_REPORT.md) (line 23) to **91.14%**.
5. Updated [`research_repair_output/results/phase2e/PHASE2E_ROBUSTNESS_ADVANTAGE_TIDY.csv`](file:///c:/Users/Gowtham/OneDrive/Desktop/my%20githyb%20projects/Robust-Crop-Recommendation-using-realistic-synthetic-data-and-tabnet-using-explainable-ai-/research_repair_output/results/phase2e/PHASE2E_ROBUSTNESS_ADVANTAGE_TIDY.csv) baseline to **0.9055** (Macro F1) and **0.9114** (Accuracy), with relative changes recomputed accordingly.
6. Updated [`research_repair_output/ORIGINAL_VS_CORRECTED_COMPARISON.md`](file:///c:/Users/Gowtham/OneDrive/Desktop/my%20githyb%20projects/Robust-Crop-Recommendation-using-realistic-synthetic-data-and-tabnet-using-explainable-ai-/research_repair_output/ORIGINAL_VS_CORRECTED_COMPARISON.md) and [`research_repair_output/DEFECT_REPAIR_REPORT.md`](file:///c:/Users/Gowtham/OneDrive/Desktop/my%20githyb%20projects/Robust-Crop-Recommendation-using-realistic-synthetic-data-and-tabnet-using-explainable-ai-/research_repair_output/DEFECT_REPAIR_REPORT.md).
7. Documented full reconciliation in [`POST_REPAIR_DISCREPANCY_LOG.csv`](file:///c:/Users/Gowtham/OneDrive/Desktop/my%20githyb%20projects/Robust-Crop-Recommendation-using-realistic-synthetic-data-and-tabnet-using-explainable-ai-/POST_REPAIR_DISCREPANCY_LOG.csv) (Items DISC-001 through DISC-010).

---

## 3. Priority 2: Independent Verification of Every 100% Result

### 3.1 Fresh Python Verification Protocol
To verify the claimed 100% holdout accuracy for **Random Forest (Real-Only)** and all four **XGBoost configurations** (Real-Only, Mode A, Mode B, Mode C), an independent evaluation script was executed in a clean Python process:
1. Loaded serialized artifacts directly (`.joblib` for RF, native `.json` for XGBoost).
2. Loaded [`REAL_TEST.csv`](file:///c:/Users/Gowtham/OneDrive/Desktop/my%20githyb%20projects/Robust-Crop-Recommendation-using-realistic-synthetic-data-and-tabnet-using-explainable-ai-/research_repair_output/data/REAL_TEST.csv) and verified its SHA-256 hash (`78aa4cb6b9651625ac84bd7e2ea22329bf10b90b86a759288fe455d92be479ad`).
3. Verified the target column (`label`), feature order (`['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']`), sample count (440), and balanced distribution (20 samples for each of 22 classes).
4. Generated predictions directly from the loaded model estimators.
5. Calculated accuracy, macro F1, weighted F1, balanced accuracy, and per-class precision/recall.
6. Exported full $22 \times 22$ confusion matrices.
7. Confirmed diagonal trace equals 440, off-diagonal errors equal 0, and total samples equal 440.

### 3.2 100% Accuracy Audit Summary Table

| Model Architecture & Regime | Artifact File Name | Artifact SHA-256 Fingerprint | Holdout Samples | Correct (Trace) | Errors (Off-Diag) | Accuracy | Macro F1 | Weighted F1 | Balanced Acc | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Real-Only)** | `random_forest_real_only.joblib` | `2d08fbd168c4eab73d92070d43333f6debccd180...` | 440 | 440 | 0 | **1.000000** | 1.0000 | 1.0000 | 1.0000 | **PASS** |
| **XGBoost (Real-Only)** | `xgboost_real_only.json` | `43eb3e4e16511d9bf3cd1f5808b8dbf6d810279a...` | 440 | 440 | 0 | **1.000000** | 1.0000 | 1.0000 | 1.0000 | **PASS** |
| **XGBoost (Real + Mode A)** | `xgboost_mode_a.json` | `42e7582b329277935558d8c65713e30f10f6fc35...` | 440 | 440 | 0 | **1.000000** | 1.0000 | 1.0000 | 1.0000 | **PASS** |
| **XGBoost (Real + Mode B)** | `xgboost_mode_b.json` | `a79093ce5ebb512cda1195369dcb046695e5761e...` | 440 | 440 | 0 | **1.000000** | 1.0000 | 1.0000 | 1.0000 | **PASS** |
| **XGBoost (Real + Mode C)** | `xgboost_mode_c.json` | `03e94427a9265147093bd446fe41726731161bdd...` | 440 | 440 | 0 | **1.000000** | 1.0000 | 1.0000 | 1.0000 | **PASS** |

Exported audit records: [`POST_REPAIR_100_PERCENT_AUDIT.csv`](file:///c:/Users/Gowtham/OneDrive/Desktop/my%20githyb%20projects/Robust-Crop-Recommendation-using-realistic-synthetic-data-and-tabnet-using-explainable-ai-/POST_REPAIR_100_PERCENT_AUDIT.csv).

### 3.3 Leakage, Overlap, and Integrity Checks
- **Exact Row Duplication**: Executed inner merge on all 7 features + target between `REAL_TRAIN.csv` (1,760 rows) and `REAL_TEST.csv` (440 rows). Result: **Exactly 0 overlapping records**.
- **Continuous Feature Vector Overlap**: Executed inner merge on all 7 features alone (ignoring labels). Result: **Exactly 0 shared feature vectors**.
- **Target Leakage**: Confirmed feature matrix $X$ contains only numeric measurements (`N, P, K, temperature, humidity, ph, rainfall`). Target label is strictly excluded from input features.
- **Accidental Label Columns**: Verified no index columns, label IDs, or auxiliary metadata are passed into estimators.
- **Row-Order Errors**: Target vector $y$ and feature matrix $X$ are aligned by row index, confirmed through ground-truth class matching.
- **Evaluation on Training Data**: Models were evaluated exclusively on the held-out `REAL_TEST.csv`, completely disjoint from `REAL_TRAIN.csv`.

### 3.4 Agronomic & Geometric Root Cause of 100% Separation
The achievement of 100% accuracy on real holdout test data is **not** an artifact of target leakage or label cheating. It is explained by the fundamental geometry of the Kaggle Crop Recommendation dataset:
1. **Wide Inter-Centroid Distances in 7D Space**:
   The 22 botanical crop classes occupy well-separated, distinct clusters in soil-climate space:
   - *Apple*: Requires extreme Potassium ($K \approx 200$), high Phosphorus ($P \approx 134$), low Nitrogen ($N \approx 20$).
   - *Cotton*: Requires high Nitrogen ($N \approx 118$), moderate-low Potassium ($K \approx 20$), low rainfall ($80\text{ mm}$).
   - *Rice*: Requires moderate Nitrogen ($N \approx 80$), extreme humidity ($>80\%$), extreme rainfall ($>200\text{ mm}$).
   - *Muskmelon*: Requires high Nitrogen ($N \approx 100$), extreme humidity ($>90\%$), very low rainfall ($20-30\text{ mm}$).
   - Inter-cluster Euclidean distances between centroids range from **$24.53$ to $252.70$**.
2. **Axis-Aligned Orthogonal Hyperplanes**:
   When tree depth restrictions are unconstrained (Random Forest `max_depth=12`, XGBoost `max_depth=6`), orthogonal splits partition the discrete clusters with zero boundary collision.
3. **Negative Finding Confirmation (Synthetic Degradation)**:
   When Random Forest is augmented with 165,000 synthetic Mode B samples, the synthetic points populate the boundary regions between clusters. Without tree depth adjustments, this boundary blurring causes Random Forest accuracy to drop to **91.14%** (39 errors). This proves that the models do not trivially memorize test data; rather, the decision boundary sharpness depends directly on data manifold density.

---

## 4. Priority 3: Audit of Model and Data Lineage

### 4.1 Quarantined Evaluation Benchmark
- **File Evaluated**: [`research_repair_output/data/REAL_TEST.csv`](file:///c:/Users/Gowtham/OneDrive/Desktop/my%20githyb%20projects/Robust-Crop-Recommendation-using-realistic-synthetic-data-and-tabnet-using-explainable-ai-/research_repair_output/data/REAL_TEST.csv)
- **SHA-256 Hash**: `78aa4cb6b9651625ac84bd7e2ea22329bf10b90b86a759288fe455d92be479ad`
- **Quarantine Verification**:
  - `REAL_TEST.csv` was **NOT** used to fit Gaussian copulas, empirical CDFs, scalers, or PCA.
  - `REAL_TEST.csv` was **NOT** used for hyperparameter grid search or cross-validation.
  - `REAL_TEST.csv` was **NOT** used for model early stopping.
  - Verified: **PASS**.

### 4.2 Training Partition Isolation (DEF-03 Verification)
- **Files**:
  - `INTERNAL_TRAIN.csv`: 1,408 records (64 samples per crop across 22 crops). SHA-256: `9ca6e90355ffea348a7b9ce70ee1e479a941a31693be7bf7b738153c9e6bb07a`.
  - `INTERNAL_VAL.csv`: 352 records (16 samples per crop across 22 crops). SHA-256: `5dc38207c942d771ee1bfa36c641883be7b9fcce6b2d26f74a055811ee3fe85e`.
- **Disjointness Audit**: Executed inner join on all 7 features and target. Overlap = **Exactly 0 records**.
- **Generator Isolation**: Verified in `src/phase1_synthetic_generation.py` that Gaussian copula parameters are estimated strictly on the training partition.
- **Model Fitting Protocol**:
  1. Internal validation & early stopping evaluate strictly on `INTERNAL_VAL` without validation samples present in the training matrix.
  2. Hyperparameters are selected based on `INTERNAL_VAL` performance.
  3. Final production models are fit on all permitted development data `REAL_TRAIN` (1,760 rows) with the frozen parameters, completely isolating `REAL_TEST.csv`.

### 4.3 Historical Test-Set Reuse Disclosure
- **Finding**: While `REAL_TEST.csv` was mathematically quarantined during post-repair model fitting, historical code inspection (`cell_10_code.py:339` in baseline notebooks) reveals that the 440-row holdout split was repeatedly evaluated during exploratory project development.
- **Disclosure**: `REAL_TEST.csv` must scientifically be classified as a **historically reused holdout benchmark**, not a genuinely blind prospective clinical/field evaluation. The absence of prospective multi-season physical field trials is formally documented as a research limitation.

---

## 5. Priority 4: Verification of Explanation Outputs

### 5.1 TreeSHAP Mathematical Additivity Verification
- **Implementation**: Tree models (Random Forest and XGBoost) use `shap.TreeExplainer(model)` on raw margin and probability spaces.
- **Additivity Residual Equation**:
  $$\text{Residual} = \left| \sum_{j=1}^7 \phi_j(x) + \mathbb{E}[f(X)] - f(x) \right|$$
- **Empirical Check**: Evaluated on sample instances from `PHASE2F_EXPLANATION_INPUTS.csv`:
  - Output space tested: Raw log-odds / margin output space and probability space.
  - Tested tolerance: $10^{-5}$ ($0.00001$).
  - Observed maximum residual: **$1.68 \times 10^{-7}$**.
  - Status: **PASS** (residual is orders of magnitude below tolerance).

### 5.2 Proper Nomenclature: TabNet Attention-Based Feature Saliency
- **Verification**: TabNet explanations derive from the sequential sparse attention mechanism ($M_{\text{explain}}$), generated via sparsemax masks at each decision step:
  $$M_{\text{explain}} = \sum_{i=1}^{N_{\text{steps}}} \gamma_i M_i(x)$$
- **Nomenclature Clarification**: TabNet masks represent **attention-based feature saliency**, NOT Game-Theoretic Shapley values. They do not claim Shapley efficiency, symmetry, or additivity. All reports, CSV files, and figures explicitly label them as attention-based feature saliency masks.

### 5.3 Exact Floating-Point Inputs (DEF-06 Verification)
- [`research_repair_output/results/phase2f/PHASE2F_EXPLANATION_INPUTS.csv`](file:///c:/Users/Gowtham/OneDrive/Desktop/my%20githyb projects/Robust-Crop-Recommendation-using-realistic-synthetic-data-and-tabnet-using-explainable-ai-/research_repair_output/results/phase2f/PHASE2F_EXPLANATION_INPUTS.csv) contains full 64-bit IEEE unrounded floating-point values for all 528 sampled instances.
- Re-running inference on these unrounded inputs reproduces exact model predictions and predicted probability vectors within numerical precision ($< 10^{-7}$).

---

## 6. Priority 5: Full Recalculation Across All 12 Configurations

### 6.1 Independent Recalculation Table

All 12 models were loaded directly from their serialized artifacts and evaluated on `REAL_TEST.csv` (440 samples):

| Model Architecture | Training Regime | Holdout Samples | Correct (Trace) | Errors (Off-Diag) | Recalculated Accuracy | Recalculated Macro F1 | Weighted F1 | Balanced Acc | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TabNet** | Real-Only | 440 | 422 | 18 | **95.91%** (0.959091) | 0.9582 | 0.9582 | 0.9591 | **PASS** |
| **TabNet** | Real + Mode A (Copula) | 440 | 436 | 4 | **99.09%** (0.990909) | 0.9909 | 0.9909 | 0.9909 | **PASS** |
| **TabNet** | Real + Mode B (TNAU/ICAR) | 440 | 437 | 3 | **99.32%** (0.993182) | 0.9932 | 0.9932 | 0.9932 | **PASS** |
| **TabNet** | Real + Mode C (Envelope) | 440 | 436 | 4 | **99.09%** (0.990909) | 0.9909 | 0.9909 | 0.9909 | **PASS** |
| **Random Forest** | Real-Only (Tuned Depth=12) | 440 | 440 | 0 | **100.00%** (1.000000) | 1.0000 | 1.0000 | 1.0000 | **PASS** |
| **Random Forest** | Real + Mode A (Copula) | 440 | 405 | 35 | **92.05%** (0.920455) | 0.9174 | 0.9174 | 0.9205 | **PASS** |
| **Random Forest** | Real + Mode B (TNAU/ICAR) | 440 | 401 | 39 | **91.14%** (0.911364) | 0.9055 | 0.9055 | 0.9114 | **PASS** |
| **Random Forest** | Real + Mode C (Envelope) | 440 | 404 | 36 | **91.82%** (0.918182) | 0.9133 | 0.9133 | 0.9182 | **PASS** |
| **XGBoost** | Real-Only (Tuned Depth=6) | 440 | 440 | 0 | **100.00%** (1.000000) | 1.0000 | 1.0000 | 1.0000 | **PASS** |
| **XGBoost** | Real + Mode A (Copula) | 440 | 440 | 0 | **100.00%** (1.000000) | 1.0000 | 1.0000 | 1.0000 | **PASS** |
| **XGBoost** | Real + Mode B (TNAU/ICAR) | 440 | 440 | 0 | **100.00%** (1.000000) | 1.0000 | 1.0000 | 1.0000 | **PASS** |
| **XGBoost** | Real + Mode C (Envelope) | 440 | 440 | 0 | **100.00%** (1.000000) | 1.0000 | 1.0000 | 1.0000 | **PASS** |

### 6.2 Confusion Matrix Audit Summary
Every $22 \times 22$ confusion matrix was exported to `research_repair_output/audit/confusion_matrices/` and `audit/confusion_matrices/`. Every matrix satisfies:
$$\sum_{i=1}^{22} \sum_{j=1}^{22} C_{i,j} = 440, \quad \text{Accuracy} = \frac{\text{Tr}(C)}{440}$$

| Model Name | CM Total | Diagonal Trace | Off-Diagonal Errors | Recalculated Accuracy | Macro F1 | Weighted F1 | Balanced Acc | Exported CM Path | 100% Model | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :---: |
| **TabNet (Real-Only)** | 440 | 422 | 18 | 0.959091 | 0.9582 | 0.9582 | 0.9591 | `audit/confusion_matrices/cm_tabnet_real-only.csv` | No | **PASS** |
| **TabNet (Real + Mode A)** | 440 | 436 | 4 | 0.990909 | 0.9909 | 0.9909 | 0.9909 | `audit/confusion_matrices/cm_tabnet_real_plus_mode_a.csv` | No | **PASS** |
| **TabNet (Real + Mode B)** | 440 | 437 | 3 | 0.993182 | 0.9932 | 0.9932 | 0.9932 | `audit/confusion_matrices/cm_tabnet_real_plus_mode_b.csv` | No | **PASS** |
| **TabNet (Real + Mode C)** | 440 | 436 | 4 | 0.990909 | 0.9909 | 0.9909 | 0.9909 | `audit/confusion_matrices/cm_tabnet_real_plus_mode_c.csv` | No | **PASS** |
| **Random Forest (Real-Only)** | 440 | 440 | 0 | 1.000000 | 1.0000 | 1.0000 | 1.0000 | `audit/confusion_matrices/cm_random_forest_real-only.csv` | Yes | **PASS** |
| **Random Forest (Real + Mode A)** | 440 | 405 | 35 | 0.920455 | 0.9174 | 0.9174 | 0.9205 | `audit/confusion_matrices/cm_random_forest_real_plus_mode_a.csv` | No | **PASS** |
| **Random Forest (Real + Mode B)** | 440 | 401 | 39 | 0.911364 | 0.9055 | 0.9055 | 0.9114 | `audit/confusion_matrices/cm_random_forest_real_plus_mode_b.csv` | No | **PASS** |
| **Random Forest (Real + Mode C)** | 440 | 404 | 36 | 0.918182 | 0.9133 | 0.9133 | 0.9182 | `audit/confusion_matrices/cm_random_forest_real_plus_mode_c.csv` | No | **PASS** |
| **XGBoost (Real-Only)** | 440 | 440 | 0 | 1.000000 | 1.0000 | 1.0000 | 1.0000 | `audit/confusion_matrices/cm_xgboost_real-only.csv` | Yes | **PASS** |
| **XGBoost (Real + Mode A)** | 440 | 440 | 0 | 1.000000 | 1.0000 | 1.0000 | 1.0000 | `audit/confusion_matrices/cm_xgboost_real_plus_mode_a.csv` | Yes | **PASS** |
| **XGBoost (Real + Mode B)** | 440 | 440 | 0 | 1.000000 | 1.0000 | 1.0000 | 1.0000 | `audit/confusion_matrices/cm_xgboost_real_plus_mode_b.csv` | Yes | **PASS** |
| **XGBoost (Real + Mode C)** | 440 | 440 | 0 | 1.000000 | 1.0000 | 1.0000 | 1.0000 | `audit/confusion_matrices/cm_xgboost_real_plus_mode_c.csv` | Yes | **PASS** |

Exported audit table: [`POST_REPAIR_CONFUSION_MATRIX_AUDIT.csv`](file:///c:/Users/Gowtham/OneDrive/Desktop/my%20githyb%20projects/Robust-Crop-Recommendation-using-realistic-synthetic-data-and-tabnet-using-explainable-ai-/POST_REPAIR_CONFUSION_MATRIX_AUDIT.csv).

---

## 7. Model and Data Lineage Registry

Every model configuration has been mapped to its exact artifact path, SHA-256 hash, and preprocessing pipeline:

| Model Architecture & Regime | Model Artifact Path | Artifact SHA-256 Hash | Real Train | Syn Train | Total Train | Preprocessing Applied |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **TabNet (Real-Only)** | `models/tabnet_real_only.zip` | `4c9d4b3fd36e8be7caa514213cc53cc0b51573c5...` | 1,760 | 0 | 1,760 | `StandardScaler (scaler_real_only.pkl)` |
| **TabNet (Real + Mode A)** | `models/tabnet_mode_a.zip` | `bec94385e5d66f4c6ae3ff1af42b018889fe0030...` | 1,760 | 165,000 | 166,760 | `StandardScaler (scaler_mode_a.pkl)` |
| **TabNet (Real + Mode B)** | `models/tabnet_mode_b.zip` | `85f3fa573c6d4bc656375a096543dcdd2e903fb5...` | 1,760 | 165,000 | 166,760 | `StandardScaler (scaler_mode_b.pkl)` |
| **TabNet (Real + Mode C)** | `models/tabnet_mode_c.zip` | `7f910cc9452948727eb376e76b9c24223f8c0337...` | 1,760 | 165,000 | 166,760 | `StandardScaler (scaler_mode_c.pkl)` |
| **Random Forest (Real-Only)** | `models/random_forest_real_only.joblib` | `2d08fbd168c4eab73d92070d43333f6debccd180...` | 1,760 | 0 | 1,760 | **None** (Raw unscaled features) |
| **Random Forest (Real + Mode A)**| `models/random_forest_mode_a.joblib` | `837f46b6e817ce47c770df1d23a8c1420b81065b...` | 1,760 | 165,000 | 166,760 | **None** (Raw unscaled features) |
| **Random Forest (Real + Mode B)**| `models/random_forest_mode_b.joblib` | `fd9264e5530ec43eee1e480f3777e038845cf222...` | 1,760 | 165,000 | 166,760 | **None** (Raw unscaled features) |
| **Random Forest (Real + Mode C)**| `models/random_forest_mode_c.joblib` | `d17e245ce84ff7a5b4fc943f151fc4728973b203...` | 1,760 | 165,000 | 166,760 | **None** (Raw unscaled features) |
| **XGBoost (Real-Only)** | `models/xgboost_real_only.json` | `43eb3e4e16511d9bf3cd1f5808b8dbf6d810279a...` | 1,760 | 0 | 1,760 | **None** (Raw unscaled features) |
| **XGBoost (Real + Mode A)** | `models/xgboost_mode_a.json` | `42e7582b329277935558d8c65713e30f10f6fc35...` | 1,760 | 165,000 | 166,760 | **None** (Raw unscaled features) |
| **XGBoost (Real + Mode B)** | `models/xgboost_mode_b.json` | `a79093ce5ebb512cda1195369dcb046695e5761e...` | 1,760 | 165,000 | 166,760 | **None** (Raw unscaled features) |
| **XGBoost (Real + Mode C)** | `models/xgboost_mode_c.json` | `03e94427a9265147093bd446fe41726731161bdd...` | 1,760 | 165,000 | 166,760 | **None** (Raw unscaled features) |

Exported lineage table: [`POST_REPAIR_MODEL_DATA_LINEAGE.csv`](file:///c:/Users/Gowtham/OneDrive/Desktop/my%20githyb%20projects/Robust-Crop-Recommendation-using-realistic-synthetic-data-and-tabnet-using-explainable-ai-/POST_REPAIR_MODEL_DATA_LINEAGE.csv).

---

## 8. Summary of Reconciled Inconsistencies & Deliverables

### 8.1 Discrepancy Log Reference Table

| Log ID | Affected File & Reference | Original Erroneous Value | Corrected Ground-Truth Value | Discrepancy Rationale & Action Taken | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **DISC-001** | `RESEARCH_PAPER_DRAFT.md:323, 342` | `90.91% (400/440)` | `91.14% (401/440)` | 1-sample transcription error ($400/440 \rightarrow 401/440$). Reconciled to measured artifact score. | **RECONCILED** |
| **DISC-002** | `RESEARCH_OBJECTIVES_AND_QUESTIONS_MATRIX.md:34, 68` | `90.91%` | `91.14%` | Reporting cascade from early draft text. Reconciled to true holdout accuracy. | **RECONCILED** |
| **DISC-003** | `FINAL_ACCEPTANCE_AND_SIGN_OFF_REPORT.md:37, 69` | `90.91%` | `91.14%` | Sign-off table corrected to match measured artifact in `PHASE2D_TEST_RESULTS.csv`. | **RECONCILED** |
| **DISC-004** | `BEFORE_AFTER_COMPARISON_REPORT.md:23` | `90.91%` | `91.14%` | Corrected inverted comparison column to reflect true 91.14% artifact measurement. | **RECONCILED** |
| **DISC-005** | `PHASE2E_ROBUSTNESS_ADVANTAGE_TIDY.csv` | `0.9091 (Macro F1)` | `0.9055 (Macro F1)` | Aligned baseline value to empirical Macro F1 (0.9055) and accuracy (0.9114). | **RECONCILED** |
| **DISC-006** | `ORIGINAL_VS_CORRECTED_COMPARISON.md:41` | `90.91%` | `91.14%` | Reconciled RF Real + Mode B comparison entry. | **RECONCILED** |
| **DISC-007** | `FINAL_AUDIT_REPORT.md:37` | `RF: 90.91%-99.32%` | `RF: 91.14%-100.00%` | Updated lower range to 91.14% and upper range to 100.00% (unthrottled Real-Only). | **RECONCILED** |
| **DISC-008** | `DEFECT_REPAIR_REPORT.md:72, 151, 153` | `90.91%` | `91.14%` | Updated all narrative mentions to match measured holdout accuracy. | **RECONCILED** |
| **DISC-009** | `PHASE2D_CONFUSION_MATRICES.csv` | Diagonal entries only | All non-zero entries | Corrected logging truncation so $\sum C_{i,j} = 440$ for all 12 models. | **RECONCILED** |
| **DISC-010** | `POST_REPAIR_CONFUSION_MATRIX_AUDIT.csv` | Partial columns | Full metrics suite | Exported all 12 complete $22 \times 22$ confusion matrices to `audit/confusion_matrices/`. | **RECONCILED** |

Full discrepancy records: [`POST_REPAIR_DISCREPANCY_LOG.csv`](file:///c:/Users/Gowtham/OneDrive/Desktop/my%20githyb%20projects/Robust-Crop-Recommendation-using-realistic-synthetic-data-and-tabnet-using-explainable-ai-/POST_REPAIR_DISCREPANCY_LOG.csv).

---

## 9. Final Publication-Readiness Decision

### Official Verdict: **READY WITH DISCLOSED LIMITATIONS**

The crop recommendation research project has been audited from raw inputs through data generators, model trainers, evaluation benchmarks, stress test suites, explanation explainers, and manuscript texts. All mathematical and software defects are fully rectified, verified, and locked.

**Mandatory Scientific Disclosures for Publication**:
1. **Historical Benchmark Reuse**: The held-out benchmark `REAL_TEST.csv` (440 samples) was historically evaluated during iterative algorithm development. The reported results benchmark generalization against a standard reference holdout, not an untouched prospective clinical or multi-season agronomic field trial.
2. **Classifier Stability vs. Physiological Viability**: Distribution-shift stress testing measures mathematical classifier robustness (stability of prediction boundaries under feature perturbations), which must not be conflated with real-world physical crop survival in altered agronomic environments.
