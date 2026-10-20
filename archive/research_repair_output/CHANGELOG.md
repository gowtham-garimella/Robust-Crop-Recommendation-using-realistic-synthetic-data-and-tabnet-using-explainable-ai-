# CHANGELOG — AUDITED RESEARCH REPAIR & DEFECT RESOLUTION

All modifications, fixes, and regenerations performed across the project workspace are cataloged below.

---

### [2026-10-19] — Comprehensive Scientific Audit & Repair Release

#### 1. Phase 2F (Reliability, Calibration & XAI)
- **Fixed DEF-01 (Critical)**: Decoupled feature scaling from tree model inference path. Route raw unscaled feature vectors to Random Forest and XGBoost, while preserving `StandardScaler` strictly for TabNet.
  - *Effect*: Restored Random Forest XAI baseline accuracy from 4.55% (constant muskmelon prediction) to 95.45% (22/22 unique crops). Restored XGBoost from 4.55% to 100.00%.
- **Fixed DEF-02 (Critical)**: Replaced heuristic standardized deviations `(x - mean)/std` with genuine `shap.TreeExplainer` for tree models and native attention masks for TabNet.
  - *Effect*: 528 explanation records generated with verified mathematical additivity (residual error < 1e-6).
- **Fixed DEF-03 (High)**: Excised unsubstantiated text claims of '100% reliability'. Computed Multiclass Brier scores, Expected Calibration Error (ECE), and Maximum Calibration Error (MCE) across 10 bins.
  - *Effect*: Generated `PHASE2F_CALIBRATION_METRICS.csv` and `phase2f_empirical_reliability_diagram.png`.
- **Fixed DEF-04 (High)**: Integrated TabNet into the multi-scenario local explanation pipeline across all 22 crops (176 instances).
- **Added Exact Explanation Inputs**: Created `PHASE2F_EXPLANATION_INPUTS.csv` storing unrounded 7D float vectors.

#### 2. Phase 2C (Model Training & Serialization)
- **Fixed DEF-05 (High)**: Implemented dual-serialization for XGBoost (`*.json` + `*.joblib`) with a universal loader to prevent unpickling corruption across Python runtimes.
- **Fixed DEF-06 (Medium)**: Removed artificial decision stump constraints (depth=1, reg_lambda=10) on tree models. Tuned models on internal validation to reflect legitimate, unthrottled performance.

#### 3. Phase 2A & 2C (Validation Isolation & Leakage Prevention)
- **Fixed DEF-07 (High)**: Enforced a strict internal train-validation protocol (`INTERNAL_TRAIN`: 1,408; `INTERNAL_VAL`: 352). Synthetic generators and scalers are fitted strictly on `INTERNAL_TRAIN` during tuning experiments.

#### 4. Phase 1 & 2B (Synthetic Data Generation & Fidelity Audit)
- **Fixed DEF-08 (Medium)**: Replaced identical Mode B duplicate in `SYNTHETIC_CROP_DATA_MASTER.csv` with a consolidated 495,000-record dataset containing all three modes and full provenance metadata.
- **Fixed DEF-11 (Low)**: Established an empirical Real-to-Real nearest-neighbor baseline distance (0.4154) and calculated distance ratios (1.054 - 1.141) to qualify zero-duplicate privacy guarantees.

#### 5. Phase 2D (Holdout Evaluation & Statistical Significance)
- **Fixed DEF-12 (Medium)**: Added 1,000-fold bootstrap 95% confidence intervals and pairwise exact McNemar discordance tests with Holm-Bonferroni correction across all 12 model configurations.

#### 6. Phase 2E (Distribution-Shift Stress Testing)
- **Fixed DEF-09 (Medium)**: Decoupled Experiment A (Classification Robustness / Prediction Invariance under fixed labels) from Experiment B (Agronomic Ground Truth Adaptation). Created tidy schema in `PHASE2E_ROBUSTNESS_ADVANTAGE_TIDY.csv`.

#### 7. Project Infrastructure & Reproducibility
- **Fixed DEF-10 (Medium)**: Removed hardcoded Google Colab paths and interactive upload blocks. Implemented environment-aware path resolution in `config.py`.
- **Added Windows One-Command Batch Runner**: Created `run_all_phases.bat` for turnkey reproduction.
- **Created Pinned Environment Spec**: Created `environment/requirements.txt`.
- **Generated Machine-Readable Inventory**: Created `PROJECT_INVENTORY.json` and `MANIFEST.json`.
