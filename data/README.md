# Canonical Dataset Documentation

This directory contains the canonical real agricultural data and synthetic augmentation modes for the Crop Recommendation research project.

## 1. Raw Primary Dataset
- **File**: [`Crop_recommendation.csv`](file:///data/Crop_recommendation.csv)
- **Observations**: 2,200 rows (exactly 100 samples per crop across 22 crops)
- **Features (7)**:
  - `N` (Nitrogen ratio in soil, kg/ha)
  - `P` (Phosphorus ratio in soil, kg/ha)
  - `K` (Potassium ratio in soil, kg/ha)
  - `temperature` (Air temperature, °C)
  - `humidity` (Relative humidity, %)
  - `ph` (Soil pH, [0-14])
  - `rainfall` (Precipitation, mm)
- **Target**: `crop` (22 unique botanical categories)

## 2. Benchmark Splitting & Quarantine Protocol
- **`REAL_TRAIN.csv`**: 1,760 rows (80 samples per crop, 80% stratified split). Reserved exclusively for model training and internal cross-validation.
- **`REAL_TEST.csv`**: 440 rows (20 samples per crop, 20% stratified split). **QUARANTINED** from all training, scaling, feature engineering, and model selection.
- **`INTERNAL_TRAIN.csv`**: 1,408 rows (64 samples per crop, 80% stratified split of `REAL_TRAIN.csv`). Used exclusively to construct training arrays during model tuning.
- **`INTERNAL_VAL.csv`**: 352 rows (16 samples per crop, 20% stratified split of `REAL_TRAIN.csv`). Used exclusively for model validation and hyperparameter selection. Zero overlap with `INTERNAL_TRAIN.csv` and `REAL_TEST.csv`.

## 3. Synthetic Augmentation Datasets
- **`SYNTHETIC_Mode_A.csv`**: 165,000 samples (7,500/crop) generated via Gaussian Copula / GMM.
- **`SYNTHETIC_Mode_B.csv`**: 165,000 samples (7,500/crop) generated via CTGAN with strict agronomic physical bounds.
- **`SYNTHETIC_Mode_C.csv`**: 165,000 samples (7,500/crop) generated via TVAE with environmental stress-envelope expansion.

All synthetic datasets have zero overlap with `REAL_TEST.csv`.
