# Independent Post-Repair Verification Report

**Target Commit:** `8cded76aa2774c4531c89b1971eb5d98cf2b44c5`  
**Repository:** `gowtham-garimella/Robust-Crop-Recommendation-using-realistic-synthetic-data-and-tabnet-using-explainable-ai-`  
**Evaluation Date:** October 20, 2026  
**Auditor:** Senior ML Research Engineer, Data Auditor & Reproducibility Specialist  
**Standard of Verification:** Strict Empirical Inspection (No Audit Trust, Direct Artifact Dissection)

---

## Executive Summary & Final Verdict

| Metric / Check Category | Result | Key Finding |
|---|---|---|
| **Git Working Tree & Commit** | **VERIFIED** | Clean tree at commit `8cded76aa2774c4531c89b1971eb5d98cf2b44c5` |
| **Claim 1 (Leak-Free Training Script)** | **FAILED / REFUTED** | `src/training.py` code specifies 1,408 rows, but skipped retraining because checkpoints existed; committed models were trained on 1,760 rows. |
| **Claim 2 (Internal Validation Quarantine)** | **FAILED / REFUTED** | `INTERNAL_VAL.csv` rows are present in the training set of the committed `random_forest_real_only.joblib`, `xgboost_*.json`, and scalers. |
| **Claim 3 (Synthetic Generator Provenance)** | **NOT VERIFIED / REFUTED** | No generator fitting code in `src/`; synthetic data in `data/` was copied from baseline, which was originally fitted on full `REAL_TRAIN.csv` (1,760 rows). |
| **Claim 4 (Model Loading & Predictions)** | **VERIFIED** | All 12 models load in fresh Python process and reproduce 440 predictions with 0 mismatches. |
| **Claim 5 (Validation Metrics Recomputation)**| **VERIFIED (With Caveat)** | Metrics match `VALIDATION_METRICS.csv` to 6 decimals, but evaluate models trained on those same validation rows. |
| **Claim 6 (Benchmark Metrics Recomputation)** | **VERIFIED** | Recomputed benchmark metrics match `FINAL_MODEL_COMPARISON.csv` and are labeled `REAL_TEST (Frozen Benchmark)`. |
| **Claim 7 (Confusion Matrices & 100% Results)**| **VERIFIED** | 0 test duplicate rows, 0 test/train overlap, confusion matrix sums = 440, valid non-trivial probability distributions. |
| **Claim 8 (Robustness Scenarios)** | **VERIFIED** | 60 scenario results verified, 0 NaNs, correct degradation and retention formulas. |
| **Claim 9 (TreeSHAP Additivity)** | **VERIFIED** | RF probability residual $\le 4.44 \times 10^{-16}$, XGBoost raw margin residual $\le 3.34 \times 10^{-06}$, exact float inputs verified. |
| **Claim 10 (Manifest Hashes & Cleanup)** | **VERIFIED** | All 28 manifest entries match SHA-256 hashes on disk; 433 cleanup entries tracked. |
| **Claim 11 (Automated Test Suite)** | **VERIFIED** | `python -m unittest -v tests/test_audit_suite.py` -> 14/14 tests PASSED in 27.732s. |

### Final Scientific Verdict
**NOT YET DEFENSIBLE FOR PEER-REVIEWED PUBLICATION WITHOUT FULL RETRAINING**  
While the repository restructuring, benchmark reproducibility, TreeSHAP additivity, and test suites are technically sound, **the committed model checkpoints and preprocessing scalers in `artifacts/` still contain historical internal validation leakage from `REAL_TRAIN.csv` (1,760 rows)**. Before submitting to a peer-reviewed journal or conference, the models and scalers must be forcibly retrained from scratch on `INTERNAL_TRAIN.csv` (1,408 rows).

---

## Detailed Claim-by-Claim Audit Findings

### Claim 1: `src/training.py` trains models using only `INTERNAL_TRAIN.csv` for real observations
- **Status:** **FAILED / REFUTED**
- **Empirical Evidence:**
  1. In `src/training.py`, execution guards check if the artifact exists before fitting:
     ```python
     rf_clf = joblib.load(rf_path) if rf_path.exists() else RandomForestClassifier(**RF_PARAMS)
     if xgb_json_path.exists(): xgb_clf.load_model(str(xgb_json_path))
     ```
     Because the models already existed in the directory, retraining was bypassed.
  2. Inspection of `artifacts/models/random_forest_real_only.joblib`:
     - Individual tree estimators reveal: `tree.weighted_n_node_samples[0] = 1760.0`.
     - In scikit-learn, a model trained on 1,408 rows produces `weighted_n_node_samples[0] = 1408.0`. A count of 1,760.0 proves the saved artifact was trained on the entire `REAL_TRAIN.csv` (1,760 rows).
  3. Inspection of `artifacts/models/xgboost_real_only.json`:
     - Tree 0 root node `sum_hessian = 128.6033`.
     - Experimental retraining of XGBoost on `INTERNAL_TRAIN.csv` (1,408 rows) produces `sum_hessian = 103.17769`.
     - Retraining on `REAL_TRAIN.csv` (1,760 rows) produces exactly `sum_hessian = 128.6033`.
  4. Inspection of `artifacts/models/xgboost_mode_a.json`, `mode_b`, `mode_c`:
     - Tree 0 root node `sum_hessian = 12289.339`, which mathematically equals training on $1,760 + 165,000 = 166,760$ rows (training on $1,408 + 165,000$ yields `12263.046`).

### Claim 2: `INTERNAL_VAL.csv` never enters model fitting, preprocessing fitting, generator fitting, or hyperparameter selection
- **Status:** **FAILED / REFUTED**
- **Empirical Evidence:**
  1. `artifacts/preprocessing/scaler_real_only.pkl` has `n_samples_seen_ = 1760.0`.
  2. `artifacts/preprocessing/scaler_mode_a.pkl` has `n_samples_seen_ = 166760.0`.
  3. `artifacts/models/random_forest_real_only.joblib` and `artifacts/models/xgboost_*.json` were fitted on 1,760 real observations.
  4. Because `REAL_TRAIN.csv` (1,760) = `INTERNAL_TRAIN.csv` (1,408) $\cup$ `INTERNAL_VAL.csv` (352), all 352 internal validation rows were present during model fitting and scaler fitting for these artifacts.
  5. Consequently, the internal validation scores in `results/metrics/VALIDATION_METRICS.csv` evaluate models on records included in their training histories.

### Claim 3: Synthetic-data generator fitting and provenance are supported by actual code and reproducible evidence
- **Status:** **NOT VERIFIED / REFUTED**
- **Empirical Evidence:**
  1. The canonical `src/` directory contains no synthetic generation script.
  2. The legacy script `archive/research_repair_output/src/phase1_synthetic_generation.py` merely copies pre-existing CSVs from `baseline_artifacts/01_Phase1_Synthetic_Generation/`.
  3. In `new_updated__crop_recommendation.ipynb`, lines 852 and 1749 explicitly document:
     `"REAL_TRAIN: 1760 rows (Used for Copula fitting & Envelope estimation)"`.
  4. This proves that the synthetic datasets in `data/` were fitted on the complete 1,760 rows, meaning `INTERNAL_VAL.csv` was also present during copula estimation.

### Claim 4: All 12 saved model artifacts load successfully and reproduce recorded predictions
- **Status:** **VERIFIED / PASSED**
- **Empirical Evidence:**
  - Fresh Python process loaded all 12 models from `artifacts/models/` and scalers from `artifacts/preprocessing/`.
  - Predictions on all 440 rows of `data/REAL_TEST.csv` matched `results/predictions/predictions_<model>.csv` with exactly 0/440 mismatches across all 12 configurations.

### Claim 5: Independently recompute validation metrics from regenerated predictions
- **Status:** **VERIFIED / PASSED (With Provenance Caveat)**
- **Empirical Evidence:**
  - Evaluating the saved models against `data/INTERNAL_VAL.csv` (352 rows) produces exact matches to `results/metrics/VALIDATION_METRICS.csv`:
    - TabNet Real-Only: 0.960227
    - TabNet Mode A: 0.994318
    - TabNet Mode B: 0.985795
    - TabNet Mode C: 0.991477
    - RF Real-Only: 0.997159
    - RF Mode A: 0.923295
    - RF Mode B: 0.906250
    - RF Mode C: 0.920455
    - XGBoost Real-Only: 0.994318
    - XGBoost Mode A: 1.000000
    - XGBoost Mode B: 0.997159
    - XGBoost Mode C: 1.000000
  - Caveat: As proven under Claims 1 & 2, these scores reflect models trained on those validation rows.

### Claim 6: Independently recompute historical benchmark metrics and label benchmark as historically reused
- **Status:** **VERIFIED / PASSED**
- **Empirical Evidence:**
  - Recomputed metrics from `results/predictions/` match `results/metrics/FINAL_MODEL_COMPARISON.csv` to 6 decimals.
  - Formally labeled as `REAL_TEST (Frozen Benchmark)`.

### Claim 7: Inspect confusion matrices, probability outputs, and 100% accuracy results
- **Status:** **VERIFIED / PASSED**
- **Empirical Evidence:**
  - Zero duplicate rows in `data/REAL_TEST.csv`.
  - Zero exact feature vector matches between `data/REAL_TEST.csv` and `data/REAL_TRAIN.csv`.
  - All 12 confusion matrices sum to exactly 440, and matrix traces equal `correct_samples`.
  - Predicted class probabilities for 100% models are continuous and non-trivial (mean confidence between 0.9682 and 0.9972; minimum confidence ranges from 0.4978 to 0.8654).

### Claim 8: Verify all 60 robustness scenario results against actual implementation
- **Status:** **VERIFIED / PASSED**
- **Empirical Evidence:**
  - `results/robustness/ROBUSTNESS_ADVANTAGE_TIDY.csv` contains exactly 60 rows across 3 architectures $\times$ 4 regimes $\times$ 5 environmental stress scenarios.
  - Zero NaN values; verified relative retention and robustness advantage calculations.

### Claim 9: Independently verify TreeSHAP additivity calculations and exact-input reproducibility
- **Status:** **VERIFIED / PASSED**
- **Empirical Evidence:**
  - Random Forest TreeSHAP evaluated in probability output space: maximum additivity residual $\le 4.44 \times 10^{-16}$.
  - XGBoost TreeSHAP evaluated in raw margin output space: maximum additivity residual $\le 3.34 \times 10^{-06}$.
  - Both pass the tolerance threshold ($\le 1.0 \times 10^{-5}$).
  - Full IEEE 754 floating-point precision preserved in `results/explainability/EXPLAINABILITY_INPUT_SAMPLES.csv`.

### Claim 10: Verify manifest hashes and cleanup integrity
- **Status:** **VERIFIED / PASSED**
- **Empirical Evidence:**
  - All 28 artifacts listed in `artifacts/manifests/FINAL_ARTIFACT_MANIFEST.json` match their SHA-256 hashes on disk.
  - `CLEANUP_MANIFEST.csv` tracks 433 entries (360 ARCHIVE, 73 KEEP, 0 unarchived deletions).

### Claim 11: Run automated test suite
- **Status:** **VERIFIED / PASSED**
- **Execution Command:** `python -m unittest -v tests/test_audit_suite.py`
- **Output:**
  ```text
  Ran 14 tests in 27.732s
  OK
  ```
- **Failures / Errors:** 0 failures, 0 errors.
- **Environment Versions:**
  - Python: 3.13.15 (AMD64)
  - scikit-learn: 1.9.1
  - PyTorch: 2.14.1+cpu
  - pytorch-tabnet: 4.1.0
  - XGBoost: 3.4.1
  - SHAP: 0.54.1
  - pandas: 3.0.5
  - numpy: 2.5.3

---

## Remaining Blockers for Research Paper Publication

1. **Forced Retraining Required**: The model checkpoints (`artifacts/models/`) and scalers (`artifacts/preprocessing/`) must be re-fitted from scratch on `data/INTERNAL_TRAIN.csv` (1,408 rows) by removing artifact existence guards in `src/training.py`.
2. **Synthetic Generator Fitting Script**: A reproducible script (`src/synthetic_generation.py`) must be added to fit the empirical Gaussian-rank copula strictly on `data/INTERNAL_TRAIN.csv` rather than copying legacy files fitted on 1,760 rows.
3. **Disclosure of Reused Benchmark**: When reporting results on `data/REAL_TEST.csv`, the paper must disclose that this 440-row partition is a frozen historical benchmark.
