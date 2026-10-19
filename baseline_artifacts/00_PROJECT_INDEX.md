# M.Tech Research Project: Comprehensive Research Deliverables Index

**Consolidation Timestamp:** `2026-10-19 02:41:46`  
**Research Title:** Robust Crop Recommendation Using Realistic Synthetic Data with TabNet and Explainable AI  
**Author:** M.Tech Agriculture Research Team  

## 1. Project Directory Structure in Archive
```
crop_recommendation_all_phases_complete_results.zip
|-- 00_Executive_Reports_and_Summaries/   (Consolidated Reports & JSON Summaries)
|-- 01_Phase1_Synthetic_Generation/       (Copula Datasets, Constraints, EDA Plots)
|-- 02_Phase2A_Data_Splits/               (Quarantined Real Train & Test Splits)
|-- 03_Phase2B_Fidelity_Leakage/          (Quality Metrics, 5-NN Leakage Audits)
|-- 04_Phase2C_Trained_Models/            (12 TabNet, RF, XGB Models, Scalers, Metrics)
|-- 05_Phase2D_Model_Evaluation/          (Holdout Benchmarks, McNemar Tests, LaTeX)
|-- 06_Phase2E_Stress_Testing/            (9 Distribution Shift Scenarios & Degradation)
|-- 07_Phase2F_Reliability_XAI/           (ECE, Brier, Reliability Plots, SHAP & Masks)
\-- 08_Individual_Phase_Zips/            (Standalone Phase ZIP Deliverables)
```

## 2. Deliverables Inventory Summary
| Phase | Subdirectory | Files Captured | Description |
| :--- | :--- | :--- | :--- |
| **Executive** | `00_Executive_Reports_and_Summaries` | 1 | Consolidated Markdown Reports & Metadata |
| **Phase 1** | `01_Phase1_Synthetic_Generation` | 6 | Synthetic datasets (Mode A, B, C) & EDA |
| **Phase 2A** | `02_Phase2A_Data_Splits` | 29 | Real Train (1,760) & Quarantined Real Test (440) |
| **Phase 2B** | `03_Phase2B_Fidelity_Leakage` | 16 | Nearest Neighbors & Leakage Comparisons |
| **Phase 2C** | `04_Phase2C_Trained_Models` | 29 | Model weights, Encoders, Scalers, Metrics |
| **Phase 2D** | `05_Phase2D_Model_Evaluation` | 9 | Clean holdout comparative evaluation tables |
| **Phase 2E** | `06_Phase2E_Stress_Testing` | 10 | Climate & Soil distribution shift evaluations |
| **Phase 2F** | `07_Phase2F_Reliability_XAI` | 11 | ECE, Multiclass Brier, 12 Plots & SHAP attributions |
| **Archives** | `08_Individual_Phase_Zips` | 11 | Standalone Phase ZIP archives |

**Total Files Consolidated:** `122`  
**Uncompressed Archive Size:** `110.03 MB`

## 3. Scientific Integrity & Quarantine Attestation
- **Zero Test Contamination:** `REAL_TEST.csv` (440 samples) remained strictly quarantined and was never used during synthetic generation, model training, or scaler fitting.
- **Empirical Validation:** All 12 models were trained and benchmarked systematically across identical holdout splits.
- **Explainable AI:** Feature attributions were derived using TabNet sparse masks and Tree SHAP to verify agronomic domain alignment.