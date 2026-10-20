# Robust Crop Recommendation Using Realistic Synthetic Data, TabNet, and Explainable AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Audit Status: Passed](https://img.shields.io/badge/Audit-Passed%2014%2F14%20Tests-brightgreen.svg)]()
[![Model Formats: Native JSON / Joblib / Zip](https://img.shields.io/badge/Artifacts-Validated-blue.svg)]()

Canonical repository for the research study: **"Robust Crop Recommendation Using Realistic Synthetic Data, TabNet, and Explainable AI"**.

This project provides an end-to-end, scientifically defensible machine learning pipeline evaluating Deep Tabular Learning (**TabNet**), Random Forest (**RF**), and Extreme Gradient Boosting (**XGBoost**) across real and synthetic data regimes. The system includes rigorous data partitioning, distribution-shift stress testing, multiclass calibration, TreeSHAP additivity verification, and TabNet attention saliency masks.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Research Findings](#key-research-findings)
3. [Repository Structure](#repository-structure)
4. [Environment Setup](#environment-setup)
5. [Reproducing the Pipeline](#reproducing-the-pipeline)
6. [Automated Test Suite](#automated-test-suite)
7. [Artifacts & Outputs](#artifacts--outputs)
8. [Scientific Disclosures & Limitations](#scientific-disclosures--limitations)
9. [Citation & License](#citation--license)

---

## Project Overview

Agricultural decision-support systems require high predictive accuracy, robustness to environmental stress (e.g., climate change and soil degradation), and explainability for farmer trust. This study addresses three core questions:
1. **Model Architecture Comparison**: How does deep tabular attention (TabNet) compare with tree ensembles (Random Forest, XGBoost) on multiclass crop recommendation?
2. **Synthetic Data Augmentation**: Do domain-constrained synthetic datasets (Modes A, B, C) improve generalization or distribution-shift robustness?
3. **Trust & Explainability**: Can TreeSHAP and TabNet attention masks provide consistent, feature-level agronomical justifications?

### Feature Schema & Target
- **Input Features (7 continuous features)**:
  1. `N`: Nitrogen ratio in soil (kg/ha)
  2. `P`: Phosphorus ratio in soil (kg/ha)
  3. `K`: Potassium ratio in soil (kg/ha)
  4. `temperature`: Temperature in degrees Celsius (°C)
  5. `humidity`: Relative humidity in percentage (%)
  6. `ph`: Soil pH value (0 to 14)
  7. `rainfall`: Rainfall in millimeters (mm)
- **Target Label (22 agricultural crops)**:
  `apple`, `banana`, `blackgram`, `chickpea`, `coconut`, `coffee`, `cotton`, `grapes`, `jute`, `kidneybeans`, `lentil`, `maize`, `mango`, `mothbeans`, `mungbean`, `muskmelon`, `orange`, `papaya`, `pigeonpeas`, `pomegranate`, `rice`, `watermelon`.

---

## Key Research Findings

All metrics were recomputed and verified from row-level predictions on the 440-sample frozen historical benchmark (`data/REAL_TEST.csv`, 20 samples per crop):

| Model | Training Regime | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 | Model Artifact Format |
|---|---|---|---|---|---|---|
| **TabNet** | Real-Only | 95.91% (422/440) | 0.959091 | 0.956897 | 0.956897 | PyTorch ZIP (`tabnet_real_only.zip`) |
| **TabNet** | Real + Mode A | 99.09% (436/440) | 0.990909 | 0.990886 | 0.990886 | PyTorch ZIP (`tabnet_mode_a.zip`) |
| **TabNet** | Real + Mode B | 99.32% (437/440) | 0.993182 | 0.993086 | 0.993086 | PyTorch ZIP (`tabnet_mode_b.zip`) |
| **TabNet** | Real + Mode C | 99.09% (436/440) | 0.990909 | 0.990861 | 0.990861 | PyTorch ZIP (`tabnet_mode_c.zip`) |
| **Random Forest** | Real-Only | 100.00% (440/440) | 1.000000 | 1.000000 | 1.000000 | Scikit-Learn Joblib (`random_forest_real_only.joblib`) |
| **Random Forest** | Real + Mode A | 92.05% (405/440) | 0.920455 | 0.916056 | 0.916056 | Scikit-Learn Joblib (`random_forest_mode_a.joblib`) |
| **Random Forest** | Real + Mode B | 91.14% (401/440) | 0.911364 | 0.905530 | 0.905530 | Scikit-Learn Joblib (`random_forest_mode_b.joblib`) |
| **Random Forest** | Real + Mode C | 91.82% (404/440) | 0.918182 | 0.913504 | 0.913504 | Scikit-Learn Joblib (`random_forest_mode_c.joblib`) |
| **XGBoost** | Real-Only | 100.00% (440/440) | 1.000000 | 1.000000 | 1.000000 | Native JSON (`xgboost_real_only.json`) |
| **XGBoost** | Real + Mode A | 100.00% (440/440) | 1.000000 | 1.000000 | 1.000000 | Native JSON (`xgboost_mode_a.json`) |
| **XGBoost** | Real + Mode B | 100.00% (440/440) | 1.000000 | 1.000000 | 1.000000 | Native JSON (`xgboost_mode_b.json`) |
| **XGBoost** | Real + Mode C | 100.00% (440/440) | 1.000000 | 1.000000 | 1.000000 | Native JSON (`xgboost_mode_c.json`) |

*Note: Clean-test 100% tree accuracy reflects the wide margin separability of Kaggle crop clusters. Under environmental stress (Heat, Drought, Sensor Jitter), performance degrades significantly (see `results/robustness/`).*

---

## Repository Structure

```text
Robust-Crop-Recommendation/
├── README.md                           # Canonical project documentation
├── requirements.txt                     # Pinned dependencies
├── CLEANUP_MANIFEST.csv                 # Audit log of archived/retained files (433 entries)
├── config/
│   └── project_config.py               # Deterministic paths, random seeds, and hyperparameters
├── data/
│   ├── README.md                       # Data dictionary, lineage, and SHA-256 hashes
│   ├── Crop_recommendation.csv         # Full canonical dataset (2,200 rows)
│   ├── REAL_TRAIN.csv                  # Real training partition (1,760 rows)
│   ├── REAL_TEST.csv                   # Frozen historical benchmark (440 rows)
│   ├── INTERNAL_TRAIN.csv              # Internal model training partition (1,408 rows)
│   ├── INTERNAL_VAL.csv                # Internal model validation partition (352 rows)
│   ├── SYNTHETIC_Mode_A.csv            # Mode A synthetic data (165,000 rows)
│   ├── SYNTHETIC_Mode_B.csv            # Mode B domain-constrained synthetic data (165,000 rows)
│   ├── SYNTHETIC_Mode_C.csv            # Mode C perturbation synthetic data (165,000 rows)
│   └── manifests/
│       └── DATA_VALIDATION_REPORT.json # Automated data integrity and schema validation
├── src/
│   ├── data_splitting.py               # Leak-free deterministic stratified splitting
│   ├── data_validation.py              # Schema, range, null, and duplicate checks
│   ├── training.py                     # Leak-free training pipeline for all 12 configurations
│   ├── evaluation.py                   # Benchmark scoring, prediction CSVs, and confusion matrices
│   ├── robustness.py                   # 5-scenario distribution shift stress testing
│   ├── calibration.py                  # Multiclass Brier score, ECE, MCE computation
│   ├── explainability.py               # TreeSHAP additivity verification & TabNet attention masks
│   └── pipeline.py                     # Master execution orchestrator
├── tests/
│   └── test_audit_suite.py             # 14-point automated test suite
├── artifacts/
│   ├── models/                         # Serialized models (.json, .joblib, .zip)
│   ├── preprocessing/                  # Fitted scalers (.pkl) and label encoder (.pkl)
│   └── manifests/
│       └── FINAL_ARTIFACT_MANIFEST.json# SHA-256 lineage manifest for all artifacts
├── results/
│   ├── metrics/                        # FINAL_MODEL_COMPARISON.csv, VALIDATION_METRICS.csv
│   ├── predictions/                    # 12 row-level prediction CSVs with hashes
│   ├── confusion_matrices/             # 12 22x22 confusion matrix CSVs (totals = 440)
│   ├── robustness/                     # ROBUSTNESS_ADVANTAGE_TIDY.csv (60 shift scenarios)
│   └── explainability/                 # TREESHAP_AUDIT.csv, attributions, calibration metrics
├── reports/
│   ├── FINAL_TECHNICAL_AUDIT.md        # Technical defect audit & code changes
│   ├── FINAL_PUBLICATION_READINESS.md  # Comprehensive research methodology & peer-review readiness
│   └── LIMITATIONS.md                  # Methodological boundaries & disclosures
└── archive/                            # Historical legacy packages and prior audit outputs
    └── README.md                       # Archive inventory and historical context
```

---

## Environment Setup

### Prerequisites
- Python 3.10 or higher
- Windows, Linux, or macOS

### Installation
Clone the repository and install dependencies using `pip`:
```powershell
# Clone the repository
git clone https://github.com/gowtham-garimella/Robust-Crop-Recommendation-using-realistic-synthetic-data-and-tabnet-using-explainable-ai-.git
cd Robust-Crop-Recommendation-using-realistic-synthetic-data-and-tabnet-using-explainable-ai-

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # On Linux/macOS: source venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

---

## Reproducing the Pipeline

To execute the complete end-to-end research pipeline (data validation, splitting, model evaluation, distribution-shift stress testing, calibration, and explainability):

```powershell
python src/pipeline.py
```

To run individual pipeline stages:
```powershell
# 1. Stratified Data Splitting
python src/data_splitting.py

# 2. Data Validation & Schema Audit
python src/data_validation.py

# 3. Model Training (All 12 Configurations)
python src/training.py

# 4. Holdout Evaluation & Metrics Export
python src/evaluation.py

# 5. Distribution Shift Stress Testing
python src/robustness.py

# 6. Multiclass Calibration (Brier, ECE, MCE)
python src/calibration.py

# 7. Explainability & TreeSHAP Additivity Audit
python src/explainability.py
```

---

## Automated Test Suite

A comprehensive 14-point test suite is provided in `tests/test_audit_suite.py` to verify data schemas, train/val disjointness, zero test contamination, model loading, TreeSHAP additivity, and confusion matrix totals:

```powershell
python -m unittest tests/test_audit_suite.py
```

### Test Results
```text
test_01_feature_schema_and_types .................... PASS
test_02_target_label_mapping ........................ PASS
test_03_train_val_disjointness ...................... PASS
test_04_train_test_disjointness ..................... PASS
test_05_synthetic_data_provenance ................... PASS
test_06_model_artifact_loading ...................... PASS
test_07_xgboost_json_prediction_equivalence ......... PASS
test_08_probability_distribution_validity ........... PASS
test_09_metric_recomputation_from_saved_predictions . PASS
test_10_confusion_matrix_consistency ................ PASS
test_11_robustness_missing_value_handling ........... PASS
test_12_treeshap_additivity_in_declared_space ....... PASS
test_13_tabnet_explanation_terminology .............. PASS
test_14_exact_input_reproducibility_and_manifest .... PASS
----------------------------------------------------------------------
Ran 14 tests in ~19s - OK
```

---

## Artifacts & Outputs

- **Models**: Saved in `artifacts/models/`. XGBoost models are saved in native `.json` format to eliminate pickle vulnerabilities. Random Forest models are stored as `.joblib`, and TabNet models as PyTorch `.zip`.
- **Predictions**: Saved in `results/predictions/`. Each row-level CSV contains `sample_id`, `true_label`, `predicted_label`, `correct`, `confidence`, `model_artifact_hash`, and `dataset_hash`.
- **Confusion Matrices**: Saved in `results/confusion_matrices/`. Every matrix sums exactly to 440, with diagonal elements reproducing model accuracy.
- **Explainability**: `results/explainability/TREESHAP_AUDIT.csv` records additivity residuals ($\le 10^{-5}$ tolerance verified). TabNet attention masks are stored in `results/explainability/TABNET_SALIENCY_MASKS.csv`.

---

## Scientific Disclosures & Limitations

1. **Historical Benchmark Reuse**: `data/REAL_TEST.csv` was previously evaluated in early development phases and cannot be claimed as an untouched prospective cohort. It serves as a frozen benchmark.
2. **Synthetic Data Boundaries**: Synthetic observations are statistically generated samples, not biological field trials. Synthetic augmentation acts as a geometric regularizer, improving TabNet and stress robustness while slightly altering tree ensemble boundaries.
3. **External Generalizability**: The model has not been validated against unobserved geographic soil survey stations. Field-level deployment requires prospective multi-season trial validation.
4. **Suitability vs. Yield**: The model predicts physiological suitability for 22 crops; it does not forecast crop yield, market prices, or farming profitability.

For full details, consult [reports/LIMITATIONS.md](reports/LIMITATIONS.md) and [reports/FINAL_PUBLICATION_READINESS.md](reports/FINAL_PUBLICATION_READINESS.md).

---

## Citation & License

This project is licensed under the MIT License. If you use this codebase, methodology, or synthetic datasets in your research, please cite:

```bibtex
@misc{garimella2026croprecommendation,
  author = {Garimella, Gowtham},
  title = {Robust Crop Recommendation Using Realistic Synthetic Data, TabNet, and Explainable AI},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/gowtham-garimella/Robust-Crop-Recommendation-using-realistic-synthetic-data-and-tabnet-using-explainable-ai-}}
}
```
