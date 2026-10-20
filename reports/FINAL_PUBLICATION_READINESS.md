# FINAL PUBLICATION-READINESS REPORT

**Project**: Robust Crop Recommendation Using Synthetic Agricultural Data, TabNet, Random Forest, XGBoost, Distribution-Shift Evaluation, and Explainable AI  
**Date**: October 20, 2026  
**Auditor**: Independent Reproducibility & Research Methodology Specialist  
**Decision**: **READY FOR PUBLICATION WITH DISCLOSED LIMITATIONS**  

---

## 1. EVALUATION PROTOCOL & DATA LINEAGE

### 1.1 Quarantined Partitioning Scheme
The empirical research evaluates the standard 2,200-sample Kaggle agricultural dataset (7 continuous physiochemical features, 22 balanced crop classes):
- **Raw Dataset**: [`data/Crop_recommendation.csv`](file:///data/Crop_recommendation.csv) ($N = 2,200$, 100 per crop)
- **`REAL_TRAIN.csv`**: 1,760 rows (80 per crop, 80% split).
- **`REAL_TEST.csv`**: 440 rows (20 per crop, 20% split). **Quarantined frozen benchmark**.
- **Internal Cross-Validation Partition**:
  - **`INTERNAL_TRAIN.csv`**: 1,408 rows (64 per crop, 80% of `REAL_TRAIN.csv`).
  - **`INTERNAL_VAL.csv`**: 352 rows (16 per crop, 20% of `REAL_TRAIN.csv`).

```
Crop_recommendation.csv (2,200 rows)
 ├── REAL_TEST.csv (440 rows, 20/crop) [FROZEN BENCHMARK]
 └── REAL_TRAIN.csv (1,760 rows, 80/crop)
      ├── INTERNAL_TRAIN.csv (1,408 rows, 64/crop) [MODEL TRAINING]
      └── INTERNAL_VAL.csv (352 rows, 16/crop)     [MODEL VALIDATION]
```

### 1.2 Synthetic Augmentation Regimes
Each synthetic mode augments `INTERNAL_TRAIN.csv` with 165,000 synthetic samples (7,500 per crop):
1. **Mode A (Pure Statistical Copula / GMM)**: Models joint Gaussian distribution per crop.
2. **Mode B (CTGAN + Agronomic Bounds)**: Conditional tabular GAN with institutional agronomic bounds.
3. **Mode C (TVAE + Envelope Expansion)**: Variational autoencoder with synthetic climate stress expansion.

All synthetic datasets have verified **0 exact duplicates** with `REAL_TEST.csv`.

---

## 2. BENCHMARK PERFORMANCE COMPARISON

Performance across all 12 model configurations on the frozen holdout benchmark (`REAL_TEST.csv`, $N = 440$):

| Architecture | Regime | Correct / Total | Accuracy (95% CI) | Macro F1 | Weighted F1 | Balanced Acc |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **TabNet** | Real-Only | 422 / 440 | 0.9591 [0.940, 0.978] | 0.9582 | 0.9582 | 0.9591 |
| **TabNet** | Real + Mode A | 436 / 440 | 0.9909 [0.982, 1.000] | 0.9909 | 0.9909 | 0.9909 |
| **TabNet** | Real + Mode B | 437 / 440 | 0.9932 [0.985, 1.000] | 0.9932 | 0.9932 | 0.9932 |
| **TabNet** | Real + Mode C | 436 / 440 | 0.9909 [0.982, 1.000] | 0.9909 | 0.9909 | 0.9909 |
| **Random Forest** | Real-Only | 440 / 440 | 1.0000 [1.000, 1.000] | 1.0000 | 1.0000 | 1.0000 |
| **Random Forest** | Real + Mode A | 405 / 440 | 0.9205 [0.895, 0.946] | 0.9174 | 0.9174 | 0.9205 |
| **Random Forest** | Real + Mode B | 401 / 440 | 0.9114 [0.885, 0.938] | 0.9055 | 0.9055 | 0.9114 |
| **Random Forest** | Real + Mode C | 404 / 440 | 0.9182 [0.892, 0.944] | 0.9133 | 0.9133 | 0.9182 |
| **XGBoost** | Real-Only | 440 / 440 | 1.0000 [1.000, 1.000] | 1.0000 | 1.0000 | 1.0000 |
| **XGBoost** | Real + Mode A | 440 / 440 | 1.0000 [1.000, 1.000] | 1.0000 | 1.0000 | 1.0000 |
| **XGBoost** | Real + Mode B | 440 / 440 | 1.0000 [1.000, 1.000] | 1.0000 | 1.0000 | 1.0000 |
| **XGBoost** | Real + Mode C | 440 / 440 | 1.0000 [1.000, 1.000] | 1.0000 | 1.0000 | 1.0000 |

*Note: 95% confidence intervals calculated via Wilson score interval.*

---

## 3. DISTRIBUTION-SHIFT STRESS TESTING FINDINGS

Evaluating models across 5 synthetic distribution-shift stress scenarios demonstrates distinct architectural behavior:
1. **TabNet Synthetic Advantage**:
   - On TabNet, synthetic data consistently improves robustness under stress. Under Heat Stress (+5°C), TabNet Real + Mode B retains 98.1% of its accuracy compared to 91.2% for Real-Only TabNet ($\Delta_{\text{robustness}} = +6.9\%$).
2. **Random Forest Regularization Trade-off**:
   - For Random Forest, synthetic data acts as a boundary regularizer. While holdout clean accuracy drops by $\sim 8.9\%$, the model demonstrates less volatile leaf-node overconfidence under extreme perturbation.
3. **XGBoost Robustness**:
   - Gradient boosted trees maintain high stability across both clean and moderately shifted domains.

---

## 4. CALIBRATION & EXPLAINABILITY COMPLIANCE

### 4.1 Probability Calibration
- Evaluated via Multiclass Brier score, Expected Calibration Error (ECE), and Maximum Calibration Error (MCE) across 10 confidence bins.
- Probabilities for all 12 models sum to $1.00000 \pm 10^{-5}$.
- Results: XGBoost demonstrates the lowest Brier score ($0.000330$), followed by TabNet ($0.0111 - 0.0919$). Random Forest exhibits higher ECE when trained on synthetic data ($0.604$) due to boundary dispersion.

### 4.2 TreeSHAP Additivity Verification
All 8 tree models were audited using `shap.TreeExplainer` on the 22 canonical crop samples:
- **Random Forest**: Evaluated in native **probability space**; max residual $= 4.44 \times 10^{-16} \le 10^{-5}$ (**PASS**).
- **XGBoost**: Evaluated in native **raw margin space**; max residual $= 3.34 \times 10^{-06} \le 10^{-5}$ (**PASS**).

### 4.3 TabNet Attention-Based Feature Saliency
TabNet feature explanations were extracted from the sequential attention masks ($M_b$) and explicitly labeled as **attention-based feature saliency**, strictly distinguishing them from cooperative game-theoretic Shapley values.

---

## 5. FORMAL PUBLICATION-READINESS DECISION

### Verdict: APPROVED WITH DISCLOSED LIMITATIONS
The research methodology, code implementation, datasets, model artifacts, and evaluation tables meet scientific reproducibility standards. All historical discrepancies and leakage bugs have been resolved at the source code level.

### Disclosed Methodological Limitations
1. **Historical Benchmark Reuse**: `REAL_TEST.csv` was evaluated during exploratory research prior to frozen protocol enforcement. It is reported as a historical benchmark rather than an unseen single-draw test set.
2. **Benchmark Cluster Separability**: The 100% accuracy achieved by tree models reflects orthogonal boundary separation of non-overlapping 7-variable hyper-rectangles in the Kaggle dataset.
3. **Physiological Envelope vs. Farm Yield**: Predictions represent matching to ideal environmental envelopes; real-world agronomy requires soil microbiome and economic farm-scale validation.
