# DEFECT REPAIR CHANGELOG: DEF-01 THROUGH DEF-07
**Project**: Robust Crop Recommendation Research Project  
**Date**: October 19, 2026  
**Auditor**: Senior ML Research Engineer & Model Auditor  

---

## Modification Log

### [2026-10-19] - DEF-01: Decouple Preprocessing Routing at Inference
- **File Modified**: `research_repair_output/src/phase2f_calibration_and_xai.py`
- **Reason**: Baseline applied `scaler_mode_a.pkl` to tree ensembles at inference, collapsing Random Forest accuracy to 4.55%.
- **Change**: Added architecture check: standard scaling is applied exclusively to TabNet inputs; raw unscaled inputs are routed to Random Forest and XGBoost.
- **Effect**: Restored tree model inference accuracy from 4.55% to 100.00% on test holdout.

### [2026-10-19] - DEF-02: Implement Genuine TreeSHAP and TabNet Attention Masks
- **File Modified**: `research_repair_output/src/phase2f_calibration_and_xai.py`
- **Reason**: Baseline substituted $(x - \mu)/\sigma$ heuristic z-scores when SHAP exceptions occurred, violating Shapley additivity.
- **Change**: Integrated exact `shap.TreeExplainer` on tree models and TabNet sequential attention masks ($M_{\text{explain}}$).
- **Effect**: Verified exact mathematical additivity (residual $< 10^{-6}$); stored 528 genuine attribution records in `PHASE2F_SHAP_ATTRIBUTIONS.csv`.

### [2026-10-19] - DEF-03: Eliminate Internal Validation Contamination
- **File Modified**: `research_repair_output/src/phase2c_model_training_and_tuning.py`
- **Reason**: Baseline passed all 1,760 rows of `REAL_TRAIN` to model fitting while using a 352-row subset as validation.
- **Change**: Enforced strict partition separation: generator and models train on `INTERNAL_TRAIN` (1,408 rows) without access to `INTERNAL_VAL` (352 rows). Final models are refit on `REAL_TRAIN` with frozen hyperparameters.
- **Effect**: Eliminated train/val leakage; verified 0 duplicate rows between partitions.

### [2026-10-19] - DEF-04: Remove Asymmetric Hyperparameter Throttling
- **File Modified**: `research_repair_output/src/phase2c_model_training_and_tuning.py`
- **Reason**: Baseline throttled Random Forest to `max_depth=4` and XGBoost to `max_depth=1` to artificially depress their performance relative to TabNet.
- **Change**: Replaced throttled parameters with fair validation-based search (RF depth=12, XGB depth=6, n_est=100).
- **Effect**: All models evaluated fairly; revealed that Random Forest and XGBoost achieve 100.00% on real data, while demonstrating synthetic distribution shift degradation.

### [2026-10-19] - DEF-05: Implement Dual XGBoost Serialization
- **File Modified**: `research_repair_output/src/phase2c_model_training_and_tuning.py`
- **Reason**: XGBoost `.joblib` artifacts corrupted C++ memory buffers across runtime versions (`input stream corrupted`).
- **Change**: Implemented dual-serialization function saving native `.json` booster schema alongside `.joblib` wrapper.
- **Effect**: Guaranteed forward runtime loading stability; verified 100.00% prediction agreement between JSON and joblib models.

### [2026-10-19] - DEF-06: Preserve Exact Unrounded Explanation Inputs
- **File Modified**: `research_repair_output/src/phase2f_calibration_and_xai.py`
- **Reason**: Baseline saved feature inputs as JSON strings rounded to 2 decimal places, preventing exact floating-point reproduction.
- **Change**: Stored full 64-bit IEEE float vectors into `PHASE2F_EXPLANATION_INPUTS.csv`.
- **Effect**: Enabled exact inference reproduction within numerical precision ($< 10^{-7}$).

### [2026-10-19] - DEF-07: Format Tidy Robustness Advantage Schema
- **File Modified**: `research_repair_output/src/phase2e_distribution_shift_stress_testing.py`
- **Reason**: Baseline robustness advantage table contained misaligned commas, empty cells, and silent zero replacement.
- **Change**: Standardized schema into 10 explicit columns: `model, synthetic_mode, scenario, metric, baseline_value, shifted_value, absolute_change, relative_change, n_samples, random_seed`.
- **Effect**: Removed all unexplained NaNs and aligned unshifted baselines with Phase 2D holdout metrics.
