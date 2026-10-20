# DATA LINEAGE & TRANSFORMATION SPECIFICATION

**Project**: Robust Crop Recommendation Using Synthetic Agricultural Data  
**Authority**: Senior Data Scientist & MLOps Specialist  
**Date**: October 19, 2026  

---

## 1. Primary Source Provenance
- **Dataset Title**: Agricultural Crop Recommendation Dataset
- **Origin**: Indian Council of Agricultural Research (ICAR) Agro-Climatic Registry / Kaggle Precision Agriculture Benchmark
- **File Name**: `Crop_recommendation.csv`
- **Total Records**: 2,200 rows
- **Schema**: 7 continuous numeric input features, 1 categorical target label
- **Class Balance**: 22 classes, exactly 100 rows per class (100% balanced)
- **Missing / Null / Infinite Values**: 0 (0.00%)

---

## 2. Partitioning Hierarchy & Checksums

| Partition Name | Purpose | Row Count | Class Distribution | SHA-256 Checksum |
| :--- | :--- | :--- | :--- | :--- |
| `Crop_recommendation.csv` | Canonical Master Real Dataset | 2,200 | 22 crops x 100 | `c8300fc7707e77a1e0b57e7eb1135c15e8e81d7f6a73c1d102e3b5e408d16d12` |
| `REAL_TRAIN.csv` | Real Training Partition (80%) | 1,760 | 22 crops x 80 | `9a4b6c0e718c386b6f220d2cc591089af575a0baf88735b787a544eb9b2d7b07` |
| `REAL_TEST.csv` | Quarantined Holdout Test Set (20%) | 440 | 22 crops x 20 | `78aa4cb6b9651625ac84bd7e2ea22329bf10b90b86a759288fe455d92be479ad` |
| `INTERNAL_TRAIN.csv` | Internal Training Split for Tuning | 1,408 | 22 crops x 64 | `9ca6e90355ffea34731770f1bfe4e7e3e23d54cd0f56edad477f47733b96862b` |
| `INTERNAL_VAL.csv` | Internal Validation Split for Model Selection | 352 | 22 crops x 16 | `5dc38207c942d771267b41462b04fc5b0c19e57de633a0a13c74d1b1ec678e8f` |

---

## 3. Synthetic Data Lineage

All synthetic generators were fitted exclusively on `REAL_TRAIN` (or `INTERNAL_TRAIN` during validation):

1. **Mode A (Gaussian Copula)**:
   - Method: Parametric Gaussian Copula with non-parametric kernel density marginals.
   - Output: `SYNTHETIC_Mode_A.csv` (165,000 records, 7,500/class).
   - Fit Scope: `REAL_TRAIN.csv` only.
2. **Mode B (CTGAN)**:
   - Method: Conditional Tabular GAN with variational Gaussian Mixture mode-specific normalization.
   - Output: `SYNTHETIC_Mode_B.csv` (165,000 records, 7,500/class).
   - Fit Scope: `REAL_TRAIN.csv` only.
3. **Mode C (Agronomically-Bounded Hybrid Diffusion)**:
   - Method: Langevin dynamic manifold perturbations with FAO ecocrop physiological bounding gates.
   - Output: `SYNTHETIC_Mode_C.csv` (165,000 records, 7,500/class).
   - Fit Scope: `REAL_TRAIN.csv` only.
4. **Master Consolidation**:
   - File: `SYNTHETIC_CROP_DATA_MASTER.csv` (495,000 records, 165,000 per mode).
   - Metadata: `generation_mode`, `generation_seed`, `constraints_applied`.

---

## 4. Preprocessing & Transformation Rules
- **TabNet**: Requires feature standardization via `StandardScaler(mean=0, std=1)`. Scalers fit strictly on training data.
- **Random Forest**: Requires **RAW, UNSCALED** features. Scalers are bypassed to preserve orthogonal decision boundary precision.
- **XGBoost**: Requires **RAW, UNSCALED** features. Serialized to native JSON (`xgboost_*.json`) alongside joblib to guarantee cross-environment unpickling compatibility.
