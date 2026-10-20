# Phase 2B: Leakage and Memorization Analysis Report

**Project Title:** Robust Crop Recommendation Using Realistic Synthetic Data with TabNet and Explainable AI  
**Research Phase:** Phase 2B - Empirical Memorization, Proximity & Information Leakage Audit  
**Status:** Validated & Scientifically Qualified  
**Generated On (UTC):** 2026-10-19 02:16:38  

---

## 1. Objective

The primary objective of Phase 2B is to empirically evaluate whether the empirical Gaussian-Rank Copula generative pipeline memorized or directly replicated individual training instances from `REAL_TRAIN.csv` (1,760 samples) during the generation of the synthetic records.

## 2. Dataset Quarantine & Protocol Verification

- **REAL_TRAIN.csv** (1,760 rows, 22 crops): Used exclusively for scaling parameter estimation, empirical copula bounds, and training nearest-neighbor baselines.
- **REAL_TEST.csv** (440 rows, 22 crops): **STRICTLY QUARANTINED.** Verified for presence and schema integrity, but completely excluded from all feature standardization, nearest-neighbor distance computations, and leakage audits.
- **SYNTHETIC_Mode_C**: Evaluated against `REAL_TRAIN.csv` across all 7 canonical features (`N, P, K, temperature, humidity, ph, rainfall`).

## 3. Exact and Rounded Match Analysis

No exact or tested rounded row-level replication was detected at 3-, 4-, or 5-decimal precision. These tests provide evidence against direct record replication but do not by themselves exclude all forms of statistical memorization.

| Precision Level | Scope | Matching Records | Total Evaluated | Match Rate (%) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Exact Full-Precision** | 7 Features Only | 0 | 165,000 | 0.0000% | `PASSED (0 Matches)` |
| **Exact Full-Precision** | 7 Features + Label | 0 | 165,000 | 0.0000% | `PASSED (0 Matches)` |
| **Rounded to 5 Decimals** | 7 Features Only | 0 | 165,000 | 0.0000% | `PASSED (0 Matches)` |
| **Rounded to 5 Decimals** | 7 Features + Label | 0 | 165,000 | 0.0000% | `PASSED (0 Matches)` |
| **Rounded to 4 Decimals** | 7 Features Only | 0 | 165,000 | 0.0000% | `PASSED (0 Matches)` |
| **Rounded to 4 Decimals** | 7 Features + Label | 0 | 165,000 | 0.0000% | `PASSED (0 Matches)` |
| **Rounded to 3 Decimals** | 7 Features Only | 0 | 165,000 | 0.0000% | `PASSED (0 Matches)` |
| **Rounded to 3 Decimals** | 7 Features + Label | 0 | 165,000 | 0.0000% | `PASSED (0 Matches)` |

---

## 4. Standardization Methodology

To prevent scale artifacts across heterogeneous agricultural features, all distances were evaluated in a 7-dimensional standardized Euclidean space where the scaling transformation (mean $\mu$ and standard deviation $\sigma$) was calibrated **exclusively** on `REAL_TRAIN.csv`.

## 5. Synthetic -> REAL_TRAIN 1-NN & REAL_TRAIN -> REAL_TRAIN Baseline

To contextualize synthetic proximity, a real-to-real baseline was computed using 2-nearest neighbors on `REAL_TRAIN.csv` (where the 1st neighbor is the observation itself with distance $0.0$, and the 2nd neighbor represents the distance to the closest distinct training record).

### Distance Distribution Quantiles:

| Distribution Channel | Count | Min | p05 | Median | Mean | p95 | Max | Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Synthetic -> REAL_TRAIN (1-NN) | 165,000 | 0.0269 | 0.1953 | 0.3667 | 0.3889 | 0.6629 | 1.4010 | 0.1468 |
| REAL_TRAIN -> REAL_TRAIN (Baseline 1-NN) | 1,760 | 0.1156 | 0.2155 | 0.3937 | 0.4189 | 0.7010 | 1.2695 | 0.1544 |

---

## 6. Diagnostic Proximity Thresholds

> **Note on Diagnostic Thresholds:** These thresholds ($d \le 0.05, 0.10, 0.15, 0.20$) are researcher-defined diagnostic reference points used to compare synthetic-to-real proximity with the empirical REAL_TRAIN-to-REAL_TRAIN baseline. They are not universal or validated memorization cutoffs.

| Diagnostic Threshold ($d$) | Synthetic Count Below | Synthetic Rate (%) | Real Baseline Rate (%) | Diagnostic Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| $d \le 0.05$ | 7 | 0.0042% | 0.0000% | Elevated synthetic count in dense empirical clusters. |
| $d \le 0.10$ | 206 | 0.1248% | 0.0000% | Elevated synthetic count in dense empirical clusters. |
| $d \le 0.15$ | 1981 | 1.2006% | 0.1136% | Elevated synthetic count in dense empirical clusters. |
| $d \le 0.20$ | 9238 | 5.5988% | 2.6136% | Elevated synthetic count in dense empirical clusters. |

---

## 7. Closest-Record Inspection

The closest synthetic observation had a standardized Euclidean distance of **0.0268875** from its nearest REAL_TRAIN observation. This represents high local proximity and was therefore included in the closest-record inspection; however, proximity alone does not establish memorization, particularly in a data distribution with overlapping empirical neighborhoods.

### Top 10 Closest Synthetic Records:

| Rank | Synth Index | Real NN Index | Standardized Distance | Synth Label | Real NN Label | Concordance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 112865 | 1234 | 0.0269 | `muskmelon` | `muskmelon` | `True` |
| 2 | 32799 | 346 | 0.0317 | `coconut` | `coconut` | `True` |
| 3 | 18464 | 202 | 0.0352 | `blackgram` | `blackgram` | `True` |
| 4 | 111772 | 1137 | 0.0397 | `mungbean` | `mungbean` | `True` |
| 5 | 105562 | 1188 | 0.0442 | `mungbean` | `mungbean` | `True` |
| 6 | 63934 | 705 | 0.0444 | `jute` | `jute` | `True` |
| 7 | 115241 | 1250 | 0.0468 | `muskmelon` | `muskmelon` | `True` |
| 8 | 1440 | 10 | 0.0509 | `apple` | `apple` | `True` |
| 9 | 6006 | 78 | 0.0517 | `apple` | `apple` | `True` |
| 10 | 3211 | 0 | 0.0528 | `apple` | `apple` | `True` |

*Full 100 closest records inspection table is saved in `tables/top_100_closest_synthetic_records.csv` (Top 100 Label Concordance: **100%**).*

## 8. Per-Crop Proximity & Crop-Neighbor Concordance

Across all evaluated synthetic records, the overall label concordance with the nearest REAL_TRAIN record was **100.00%** among the top 100 closest records, and **97.86%** across the entire synthetic corpus. This confirms that synthetic observations are consistently generated within their corresponding agronomic feature manifolds.

## 9. Descriptive 5-NN Local-Density Analysis

The 5-NN analysis provides descriptive evidence about local synthetic-to-real feature-space density. Because a corresponding real-to-real 5-NN baseline was not included in this phase, these results should not be interpreted as a definitive test for clustering or memorization.

- **Mean 5-NN Distance:** 0.5126795
- **Median 5-NN Distance:** 0.4890706
- **Standard Deviation:** 0.1612782
- **Minimum Mean 5-NN:** 0.1706599
- **5th / 95th Percentile:** 0.2972173 / 0.8140260

## 10. Distance-Ratio Analysis

This is a distribution-level proximity comparison between the synthetic-to-real nearest neighbor distances and the real-to-real baseline nearest neighbor distances (not a record-to-record leakage score):

| Metric | Synthetic Value | Real Baseline Value | Distance Ratio (Synth / Real) | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **min** | 0.0269 | 0.1156 | 0.2325 | Synthetic min is approximately 23.25% of the real-real baseline min. |
| **p01** | 0.1449 | 0.1695 | 0.8549 | Synthetic p01 is approximately 85.49% of the real-real baseline p01. |
| **p05** | 0.1953 | 0.2155 | 0.9062 | Synthetic p05 is approximately 90.62% of the real-real baseline p05. |
| **p10** | 0.2261 | 0.2467 | 0.9164 | Synthetic p10 is approximately 91.64% of the real-real baseline p10. |
| **p25** | 0.2855 | 0.3056 | 0.9343 | Synthetic p25 is approximately 93.43% of the real-real baseline p25. |
| **median** | 0.3667 | 0.3937 | 0.9313 | Synthetic median is approximately 93.13% of the real-real baseline median. |
| **mean** | 0.3889 | 0.4189 | 0.9283 | Synthetic mean is approximately 92.83% of the real-real baseline mean. |
| **p75** | 0.4653 | 0.5014 | 0.9280 | Synthetic p75 is approximately 92.80% of the real-real baseline p75. |
| **p90** | 0.5772 | 0.6144 | 0.9395 | Synthetic p90 is approximately 93.95% of the real-real baseline p90. |
| **p95** | 0.6629 | 0.7010 | 0.9456 | Synthetic p95 is approximately 94.56% of the real-real baseline p95. |
| **p99** | 0.8584 | 0.9231 | 0.9299 | Synthetic p99 is approximately 92.99% of the real-real baseline p99. |
| **max** | 1.4010 | 1.2695 | 1.1036 | Synthetic max is approximately 110.36% of the real-real baseline max. |
| **median_ratio_bootstrap_95ci** | 0.3667 | 0.3937 | 0.9313 | Bootstrap 95% Confidence Interval for Ratio of Medians: [0.9107, 0.9496] (1,000 iterations, seed=42). |

- **Bootstrap 95% Confidence Interval for Ratio of Medians:** `[0.9107, 0.9496]` (1,000 resamples, fixed seed=42).

## 11. Scientific Limitations

1. **Direct vs. Statistical Replication:** Exact and rounded matching tests detect direct row-level replication but cannot detect every possible form of statistical or manifold-level memorization.
2. **Absence of Universal Cutoffs:** Nearest-neighbor Euclidean distance has no universally accepted cutoff that definitively proves or disproves memorization.
3. **Researcher-Defined Diagnostics:** The diagnostic thresholds ($0.05, 0.10, 0.15, 0.20$) are comparative reference points, not biological or legal privacy standards.
4. **Descriptive 5-NN Baseline:** The 5-NN density analysis is descriptive because a full real-to-real 5-NN baseline was not computed.
5. **Empirical Density Clustering:** High statistical proximity naturally occurs in dense empirical regions of agronomic feature space.
6. **Downstream Independence:** Phase 2B establishes data properties but does not evaluate downstream model robustness or predictive generalization.

## 12. Final Conclusion

Phase 2B detected no exact or tested rounded row-level replication between the synthetic dataset and REAL_TRAIN. The overall synthetic-to-real nearest-neighbor distribution was broadly comparable to the empirical REAL_TRAIN-to-REAL_TRAIN baseline, with a slightly lower synthetic median distance. However, a higher proportion of synthetic observations occupied the closest diagnostic proximity regions at selected thresholds. Because these thresholds are researcher-defined and nearest-neighbor proximity can arise naturally from dense empirical regions, these observations do not establish memorization. The results therefore provide evidence against direct row-level replication while supporting cautious interpretation of local proximity and motivating subsequent downstream robustness evaluation.
