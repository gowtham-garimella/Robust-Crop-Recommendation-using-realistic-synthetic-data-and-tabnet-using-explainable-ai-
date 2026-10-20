# Methodological Limitations and Boundary Disclosures

**Project:** Robust Crop Recommendation Using Realistic Synthetic Data, TabNet, and Explainable AI  
**Repository:** [gowtham-garimella/Robust-Crop-Recommendation-using-realistic-synthetic-data-and-tabnet-using-explainable-ai-](https://github.com/gowtham-garimella/Robust-Crop-Recommendation-using-realistic-synthetic-data-and-tabnet-using-explainable-ai-)  
**Date of Audit & Canonical Freeze:** October 20, 2026  
**Auditor & Verification Standard:** Senior ML Research Engineer & Reproducibility Auditor  

---

## Executive Summary

Scientific integrity requires transparent disclosure of the epistemic and empirical boundaries of any machine learning study. While this project provides a rigorous, leak-free re-evaluation of TabNet, Random Forest, and XGBoost across real and synthetic training regimes, the findings must be interpreted within the specific methodological boundaries detailed below.

---

## 1. Historical Test-Set Reuse Disclosure

### Limitation
The evaluation set `data/REAL_TEST.csv` (440 samples, 20 per crop across 22 classes) originated from the initial 80/20 train/test split of the Kaggle Crop Recommendation dataset. During prior development cycles and historical audits, this exact 440-row partition was evaluated repeatedly. Historical log files and audit reports explicitly indicate that test-set metrics were inspected and informed subsequent engineering iterations.

### Scientific Consequence
- Under standard statistical learning theory, repeated testing without independent holdout invalidates standard hypothesis testing p-values and confidence intervals due to implicit hyperparameter tuning and model selection on the test partition.
- Therefore, `data/REAL_TEST.csv` cannot honestly be described as an "untouched prospective test set."
- In this canonical repository, `data/REAL_TEST.csv` is formally designated as a **Frozen Historical Benchmark**. All model selection, early stopping, and hyperparameter decisions were strictly confined to `INTERNAL_TRAIN.csv` (1,408 rows) and `INTERNAL_VAL.csv` (352 rows). However, the benchmark metrics reflect performance on a historically reused benchmark partition.

---

## 2. Synthetic-Data Assumptions and Distributional Boundaries

### Limitation
Modes A, B, and C generated 165,000 synthetic observations per mode (7,500 samples per crop class). The generation algorithms use:
- **Mode A (Unconstrained Statistical / Gaussian Copula / KDE)**: Unconditional parameterization modeling multi-modal density.
- **Mode B (Agronomic Domain Constrained)**: Hard boundary clipping enforcing biological physiological envelopes (e.g., pH $\in [3.5, 9.5]$, rainfall $\ge 0$).
- **Mode C (Physics-Informed Perturbation / Mixup-Style)**: Convex combination and noise injection within empirical convex hulls.

### Scientific Consequence
- Synthetic samples are **not real agricultural field trials**. They do not represent newly measured empirical biological phenomena.
- In several cases, augmenting real training data with Mode B or Mode C synthetic data slightly decreased Random Forest test accuracy (e.g., from 100.00% to 91.14% in Mode B) because synthetic data populated the wide margin gaps between discrete crop clusters, acting as a strong regularizer that altered decision boundary geometry.
- Claims that synthetic data universally improves performance are false. As demonstrated in [`results/metrics/FINAL_MODEL_COMPARISON.csv`](file:///results/metrics/FINAL_MODEL_COMPARISON.csv) and [`results/robustness/ROBUSTNESS_ADVANTAGE_TIDY.csv`](file:///results/robustness/ROBUSTNESS_ADVANTAGE_TIDY.csv), synthetic augmentation benefits neural architectures (TabNet improved from 95.91% to 99.32%) and provides distribution-shift resilience under specific climate stress, but can penalize clean-test tree ensemble accuracy.

---

## 3. Lack of Independent External Geographic Validation

### Limitation
The canonical study lacks a second, truly independent external validation dataset collected from different geographical regions, soil survey stations, or cropping seasons (e.g., FAO, ICAR, or USDA agricultural field stations).

### Scientific Consequence
- All real data (`REAL_TRAIN.csv` and `REAL_TEST.csv`) originate from the same underlying 2,200-row benchmark distribution.
- Although zero sample overlap exists between `INTERNAL_TRAIN.csv`, `INTERNAL_VAL.csv`, and `REAL_TEST.csv` (verified by hash checking and record intersection), identically distributed (i.i.d.) partition evaluation cannot prove out-of-distribution generalizability across unobserved soil-climatic zones.
- Independent prospectively collected field data is required before deploying these models in production agronomic advisory tools.

---

## 4. Distinction Between Crop Suitability and Agronomic Yield

### Limitation
The target label represents multiclass crop recommendation (`label` $\in \{0, \dots, 21\}$). The input features consist of seven environmental parameters:
- Nitrogen ($N$, kg/ha)
- Phosphorus ($P$, kg/ha)
- Potassium ($K$, kg/ha)
- Temperature ($^\circ\text{C}$)
- Relative Humidity ($\%$)
- Soil pH
- Rainfall ($\text{mm}$)

### Scientific Consequence
- The model predicts **optimal crop physiological suitability** given environmental parameters; it does **not predict crop yield** (tons/hectare), economic profitability, market demand, water availability constraints, or pest/disease pressure.
- An agronomic recommendation in practice must account for seasonal crop rotation, farmer capital, irrigation infrastructure, and micro-climatic pest cycles, which are outside the feature schema of this dataset.

---

## 5. Dataset Separability and Perfect Metric Scores

### Limitation
On the 440-row historical benchmark, Random Forest (Real-Only) and XGBoost (all 4 regimes) achieve 100.00% accuracy (440/440 correct predictions).

### Scientific Consequence
- The Kaggle Crop Recommendation dataset features distinct, well-separated cluster centroids for several crops (e.g., apple, papaya, chickpea, cotton) within the 7-dimensional space.
- Extensive investigative audits confirmed that this 100% score is not caused by target leakage, label corruption, or duplicate records. Rather, standard tree ensembles construct axis-aligned orthogonal cuts that completely separate these clusters in the benchmark partition.
- A 100% accuracy score on this benchmark partition **must not be misconstrued as real-world perfection**. Under distribution shift (e.g., Heat Stress or Extreme Climate in Phase 2E), XGBoost accuracy drops to 71.36% and TabNet drops to 57.50%, proving the model's sensitivity to real-world environmental degradation.

---

## 6. Unverified Claims and Retracted Assertions

The following claims from preliminary or historical reports are explicitly retracted or corrected:
1. **Retracted:** *"TabNet explanations provide Shapley value attributions."*  
   **Correction:** TabNet explanations are attention-based feature saliency masks ($M_b$) aggregated across sequential decision steps, not cooperative game-theoretic Shapley values.
2. **Retracted:** *"Synthetic data generation was fully leak-free in all historical checkpoints."*  
   **Correction:** While the synthetic datasets themselves contain zero test records, the historical Phase 2C training script mistakenly included validation records into training arrays. This has been remediated in canonical `src/training.py`.
3. **Retracted:** *"The test set is a prospective, untouched evaluation set."*  
   **Correction:** `REAL_TEST.csv` is a frozen historical benchmark whose historical reuse is fully disclosed.
