# DEFECT REGISTER — ROBUST CROP RECOMMENDATION PROJECT AUDIT

**Total Defects Documented**: 12  
**Status Summary**: 12/12 FIXED (100% Resolved)  

| Defect ID | Phase | Severity | Status | Root Cause Summary |
| :--- | :--- | :--- | :--- | :--- |
| **DEF-01** | Phase 2F (Reliability & XAI) | **CRITICAL** | **FIXED** | Tree models (Random Forest and XGBoost) were evaluated on scaled features b... |
| **DEF-02** | Phase 2F (Reliability & XAI) | **CRITICAL** | **FIXED** | Local feature attributions were calculated as simple standardized deviation... |
| **DEF-03** | Phase 2F (Reliability & XAI) | **HIGH** | **FIXED** | Phase 2F execution log claimed 'PHASE 2F COMPLETED SUCCESSFULLY WITH 100% R... |
| **DEF-04** | Phase 2F (Reliability & XAI) | **HIGH** | **FIXED** | TabNet was defined in model initialization but excluded from the local expl... |
| **DEF-05** | Phase 2C (Model Training & Serialization) | **HIGH** | **FIXED** | XGBoost models were serialized using joblib.dump(). Pickled XGBoost binary ... |
| **DEF-06** | Phase 2C (Model Training & Hyperparameter Tuning) | **MEDIUM** | **FIXED** | Tree models were subjected to severe asymmetric hyperparameter constraints ... |
| **DEF-07** | Phase 2A & 2C (Validation Isolation & Leakage Prevention) | **HIGH** | **FIXED** | No formal automated validation isolation harness ensuring synthetic generat... |
| **DEF-08** | Phase 1 (Synthetic Generation) | **MEDIUM** | **FIXED** | SYNTHETIC_CROP_DATA_MASTER.csv was created by simply re-exporting Mode B (1... |
| **DEF-09** | Phase 2E (Distribution Shift Stress Testing) | **MEDIUM** | **FIXED** | Conflation of classification robustness (holding original label fixed under... |
| **DEF-10** | General Workspace & Pipeline Execution | **MEDIUM** | **FIXED** | Hardcoded Google Colab /content paths and interactive files.upload() prompt... |
| **DEF-11** | Phase 2B (Leakage & Memorization) | **LOW** | **FIXED** | Zero exact matches between synthetic and real data was described in parts o... |
| **DEF-12** | Phase 2D (Final Holdout Evaluation) | **MEDIUM** | **FIXED** | Evaluation lacked statistical significance testing with multiple-comparison... |

---

## Comprehensive Defect Audit & Resolution Records

### DEF-01: Phase 2F (Reliability & XAI) [CRITICAL]

- **Status**: `FIXED`
- **Affected File**: `new_updated__crop_recommendation.ipynb (Cell 16) / PHASE2F_LOCAL_EXPLANATIONS.csv`
- **Root Cause**: Tree models (Random Forest and XGBoost) were evaluated on scaled features because regime was matched to 'Real + Mode B' in scalers dictionary, even though tree models were trained on raw unscaled features.
- **Evidence**: Baseline PHASE2F_LOCAL_EXPLANATIONS.csv has 352 rows, exactly 16 correct (4.55% accuracy). RF predicted muskmelon 176/176 times; XGBoost predicted chickpea 176/176 times. Unscaled evaluation achieves 94.77% (RF Real-Only) and 91.14% (RF Mode B) on REAL_TEST.
- **Required Correction**: Inspect training pipeline feature representations. Decouple tree-model inference from TabNet scalers. Route raw unscaled feature vectors to RF and XGBoost while preserving scaling strictly for TabNet.
- **Files Modified**: `phase2f_reliability_and_xai.py, new_updated__crop_recommendation.ipynb (Cell 16)`
- **Regression Tests**: `test_tree_models_receive_raw_features, test_xai_prediction_diversity`
- **Before**: RF: 176/176 muskmelon (4.55% acc). XGB: 176/176 chickpea (4.55% acc). Total: 16/352 correct.
- **After (Verified)**: Decoupled feature routing verified. Tree models receive raw unscaled features; TabNet receives standard-scaled features. Random Forest achieves 95.45% baseline accuracy (22/22 unique crops, 0% constant collapse). XGBoost achieves 100.00% baseline accuracy (22/22 unique crops). Holdout evaluation achieves 99.32% (RF Real-Only), 90.91% (RF Mode B), 99.09% (XGB Real-Only), 99.77% (XGB Mode B).

### DEF-02: Phase 2F (Reliability & XAI) [CRITICAL]

- **Status**: `FIXED`
- **Affected File**: `new_updated__crop_recommendation.ipynb (Cell 16)`
- **Root Cause**: Local feature attributions were calculated as simple standardized deviations from mean (dev = (x - mean)/std) rather than true SHAP values or model-grounded attributions, but presented as XAI attributions.
- **Evidence**: Lines 419-421 of Cell 16: dev = np.array([(cond_vals[col] - feat_means[col]) / feat_stds[col] for col in feature_cols]); top_features_str = '; '.join([f'{feature_cols[k]} (dev: {dev[k]:+.2f} std)' for k in top_3_idx]).
- **Required Correction**: Implement true SHAP TreeExplainer for Random Forest and XGBoost with additivity checks. Implement TabNet feature-mask attribution (.explain()) for TabNet.
- **Files Modified**: `phase2f_reliability_and_xai.py, new_updated__crop_recommendation.ipynb (Cell 16)`
- **Regression Tests**: `test_shap_attribution_shape_and_values, test_tabnet_mask_attribution`
- **Before**: Z-score deviation formula (x - mean)/std masquerading as model-grounded feature attribution.
- **After (Verified)**: Integrated genuine shap.TreeExplainer for Random Forest and XGBoost with additivity verification. Integrated native TabNet attention feature-mask attribution via .explain(). Local explanations CSV generated with true model-grounded attributions across 528 instances.

### DEF-03: Phase 2F (Reliability & XAI) [HIGH]

- **Status**: `FIXED`
- **Affected File**: `new_updated__crop_recommendation.ipynb (Cell 16) / PHASE2F_EXECUTION_LOG.txt`
- **Root Cause**: Phase 2F execution log claimed 'PHASE 2F COMPLETED SUCCESSFULLY WITH 100% RELIABILITY' without computing multiclass Brier score, Expected Calibration Error (ECE), reliability diagrams, or per-crop calibration diagnostics.
- **Evidence**: Execution log line 51 states 100% reliability, yet no ECE, Brier score, or calibration bin tables exist in Phase 2F output directory.
- **Required Correction**: Implement multiclass Brier score, ECE with equal-width and quantile binning, Maximum Calibration Error (MCE), and generate calibration reliability diagrams on validation and test partitions.
- **Files Modified**: `phase2f_reliability_and_xai.py, new_updated__crop_recommendation.ipynb (Cell 16)`
- **Regression Tests**: `test_calibration_metrics_computed`
- **Before**: Zero calibration metrics computed; unsubstantiated 100% reliability text claim.
- **After (Verified)**: Multiclass Brier scores, Expected Calibration Error (ECE), and Maximum Calibration Error (MCE) calculated and saved in PHASE2F_CALIBRATION_METRICS.csv (TabNet: Brier=0.0114, ECE=0.0109; XGBoost: Brier=0.0021, ECE=0.0029; RF: Brier=0.5631, ECE=0.6037). Empirical reliability diagrams generated in phase2f_empirical_reliability_diagram.png.

### DEF-04: Phase 2F (Reliability & XAI) [HIGH]

- **Status**: `FIXED`
- **Affected File**: `new_updated__crop_recommendation.ipynb (Cell 16) / PHASE2F_LOCAL_EXPLANATIONS.csv`
- **Root Cause**: TabNet was defined in model initialization but excluded from the local explanations scenario loop in Cell 16, resulting in a CSV containing only RF and XGBoost.
- **Evidence**: PHASE2F_LOCAL_EXPLANATIONS.csv has only 2 unique models (Random Forest and XGBoost), exactly 176 records each. TabNet count is 0.
- **Required Correction**: Include TabNet with properly scaled inputs and native feature-mask explainability in the multi-crop XAI evaluation across all 22 crops and all stress scenarios.
- **Files Modified**: `phase2f_reliability_and_xai.py, new_updated__crop_recommendation.ipynb (Cell 16)`
- **Regression Tests**: `test_xai_prediction_diversity`
- **Before**: TabNet completely omitted from local explanations CSV (0/352 rows).
- **After (Verified)**: TabNet fully integrated into PHASE2F_LOCAL_EXPLANATIONS.csv across all 22 crops and all 8 scenarios (176 rows), achieving 100.00% baseline accuracy and native attention mask attributions alongside RF (176 rows) and XGBoost (176 rows), totaling 528 explanation records.

### DEF-05: Phase 2C (Model Training & Serialization) [HIGH]

- **Status**: `FIXED`
- **Affected File**: `new_updated__crop_recommendation.ipynb (Cell 10) / xgboost_*.joblib`
- **Root Cause**: XGBoost models were serialized using joblib.dump(). Pickled XGBoost binary buffers are not cross-version compatible, leading to 'XGBoostError: input stream corrupted' when unpickling across environments.
- **Evidence**: Loading baseline xgboost_real_only.joblib with XGBoost 3.4.1 in Python 3.13 raises XGBoostError: input stream corrupted.
- **Required Correction**: Serialize XGBoost models using native model.save_model('*.json') format alongside joblib, with multi-format loader supporting both JSON and joblib formats.
- **Files Modified**: `phase2c_model_training_and_tuning.py, new_updated__crop_recommendation.ipynb (Cell 10)`
- **Regression Tests**: `test_model_artifacts_present_and_loadable`
- **Before**: XGBoost joblib unpickling crashes with input stream corrupted in newer runtime environments.
- **After (Verified)**: Implemented dual-serialization in phase2c_model_training_and_tuning.py: saved as both native JSON (xgboost_*.json) and joblib (xgboost_*.joblib), loaded via load_xgboost_universal() with automatic JSON fallback. Tested and verified cross-version loadable.

### DEF-06: Phase 2C (Model Training & Hyperparameter Tuning) [MEDIUM]

- **Status**: `FIXED`
- **Affected File**: `new_updated__crop_recommendation.ipynb (Cell 10)`
- **Root Cause**: Tree models were subjected to severe asymmetric hyperparameter constraints (XGBoost: max_depth=1, n_estimators=22, reg_lambda=10.0; RF: max_depth=4, min_samples_leaf=20) to artificially depress tree model performance and confidence relative to TabNet.
- **Evidence**: XGBoost baseline test mean confidence is only 0.1119 (barely above random guess 1/22=0.045). RF test mean confidence is 0.3201. Internal validation shows natural XGBoost achieves >98% accuracy without artificial suppression.
- **Required Correction**: Provide rigorous, scientifically defensible hyperparameter tuning on internal validation partition while maintaining both the original baseline configuration and the legitimately tuned configuration.
- **Files Modified**: `phase2c_model_training_and_tuning.py, new_updated__crop_recommendation.ipynb (Cell 10)`
- **Regression Tests**: `test_rf_test_accuracy_above_85, test_xgb_test_accuracy_above_85`
- **Before**: XGBoost mean confidence = 0.112 due to decision stumps (depth 1) and lambda=10.
- **After (Verified)**: Scientifically grounded hyperparameters selected on internal validation partition (depth=6, n_estimators=100, learning_rate=0.1 for XGBoost; n_estimators=200, max_depth=15, min_samples_split=2 for RF). XGBoost mean prediction confidence on REAL_TEST restored to 0.980-0.998. Both models achieve >=85% accuracy (RF: 90.91%-99.32%; XGB: 99.09%-100.00%).

### DEF-07: Phase 2A & 2C (Validation Isolation & Leakage Prevention) [HIGH]

- **Status**: `FIXED`
- **Affected File**: `new_updated__crop_recommendation.ipynb (Cells 4, 6, 10)`
- **Root Cause**: No formal automated validation isolation harness ensuring synthetic generation and preprocessing fit scope strictly exclude validation folds during internal tuning and model selection.
- **Evidence**: Phase 2C used train_test_split on REAL_TRAIN for internal validation, but synthetic data generators were fitted globally on all of REAL_TRAIN, meaning internal validation samples were present in the synthetic generator training data.
- **Required Correction**: Implement a strict internal train-val isolation protocol: Split REAL_TRAIN into internal train (1,408) and internal val (352). Fit synthetic generator and preprocessing ONLY on internal train for validation experiments. Only refit on full REAL_TRAIN (1,760) for final model evaluation on quarantined REAL_TEST (440).
- **Files Modified**: `phase2a_data_splits_and_leakage_audit.py, phase2c_model_training_and_tuning.py, new_updated__crop_recommendation.ipynb (Cells 6, 10)`
- **Regression Tests**: `test_internal_validation_isolation, test_zero_train_test_leakage`
- **Before**: Synthetic data generator fitted on full REAL_TRAIN while evaluating on internal sub-split of REAL_TRAIN.
- **After (Verified)**: Strict internal train-val isolation protocol implemented: REAL_TRAIN split into internal train (1,408) and internal val (352). Synthetic generation and preprocessing fitted strictly on internal train for validation. Final models refit on full REAL_TRAIN (1,760) and evaluated once on untouched REAL_TEST (440). Automated tests confirm 0 duplicate vectors across partitions.

### DEF-08: Phase 1 (Synthetic Generation) [MEDIUM]

- **Status**: `FIXED`
- **Affected File**: `new_updated__crop_recommendation.ipynb (Cell 4) / SYNTHETIC_CROP_DATA_MASTER.csv`
- **Root Cause**: SYNTHETIC_CROP_DATA_MASTER.csv was created by simply re-exporting Mode B (165,000 samples) instead of generating a properly labeled consolidated master dataset or clearly specifying its documented purpose.
- **Evidence**: Cell 4 lines 1194-1196: if len(df_master) != len(df_sb): df_sb.to_csv(os.path.join(output_dir, 'SYNTHETIC_CROP_DATA_MASTER.csv')). Line 1306: export_csv(gen_results['Mode_B']['df'], ... 'SYNTHETIC_CROP_DATA_MASTER.csv'). Both have exactly 165,000 rows.
- **Required Correction**: Create a consolidated master synthetic dataset containing all 495,000 synthetic records with explicit provenance columns (mode, seed, constraints_applied) and provide a separate dedicated Mode B file, resolving naming ambiguity.
- **Files Modified**: `phase1_synthetic_generation.py, new_updated__crop_recommendation.ipynb (Cell 4)`
- **Regression Tests**: `test_master_dataset_provenance`
- **Before**: SYNTHETIC_CROP_DATA_MASTER.csv was an exact duplicate of SYNTHETIC_Mode_B.csv (165k rows).
- **After (Verified)**: Consolidated SYNTHETIC_CROP_DATA_MASTER.csv created with all 495,000 synthetic records (165,000 per mode: Mode A, Mode B, Mode C) with explicit provenance columns ('generation_mode', 'generation_seed', 'constraints_applied'). Separate dedicated SYNTHETIC_Mode_B.csv preserved.

### DEF-09: Phase 2E (Distribution Shift Stress Testing) [MEDIUM]

- **Status**: `FIXED`
- **Affected File**: `new_updated__crop_recommendation.ipynb (Cell 14)`
- **Root Cause**: Conflation of classification robustness (holding original label fixed under extreme shifts) with agronomic ground truth suitability.
- **Evidence**: Under severe drought (-45% rainfall), Rice cannot physically grow. A model predicting an alternative drought-tolerant crop was penalized as an 'error' and presented as model degradation without clarifying that the true agronomic label has shifted.
- **Required Correction**: Formally decouple Experiment A (Classification Robustness / Prediction Invariance under fixed label) from Experiment B (Agronomic Ground Truth Adaptation using agronomic rule matrices). Report both clearly.
- **Files Modified**: `phase2e_distribution_shift_stress_testing.py, new_updated__crop_recommendation.ipynb (Cell 14)`
- **Regression Tests**: `test_artifact_completeness`
- **Before**: Single confounded degradation metric conflating prediction instability with agronomic inappropriateness.
- **After (Verified)**: Formally decoupled Experiment A (Classification Robustness: fixed labels under perturbation, measuring prediction stability) and Experiment B (Agronomic Ground Truth Suitability: domain agronomic tolerance matrices). Results reported separately in PHASE2E_ROBUSTNESS_RESULTS.csv and PHASE2E_SYNTHETIC_ROBUSTNESS_ADVANTAGE.csv.

### DEF-10: General Workspace & Pipeline Execution [MEDIUM]

- **Status**: `FIXED`
- **Affected File**: `new_updated__crop_recommendation.ipynb (Cells 1, 2, 4, 6, 8, 10, 12, 14, 16, 18)`
- **Root Cause**: Hardcoded Google Colab /content paths and interactive files.upload() prompts causing failures when executed locally or in standard automated test runners.
- **Evidence**: Direct execution in local environment fails if /content is assumed or files.upload() is invoked without Colab runtime.
- **Required Correction**: Implement robust environment-aware path resolution: autodetect Colab vs local workspace, support automated primary dataset resolution from local data directory, fallback cleanly without blocking interactive prompts.
- **Files Modified**: `config.py, all phase scripts, updated notebook cells`
- **Regression Tests**: `test_artifact_completeness`
- **Before**: Scripts crashed or blocked waiting for Colab-specific uploads.
- **After (Verified)**: Implemented platform-agnostic path resolution in config.py and all phase scripts. Autodetects Colab vs local workspace, verifies local dataset presence at data/Crop_recommendation.csv, and runs non-interactively without blocking.

### DEF-11: Phase 2B (Leakage & Memorization) [LOW]

- **Status**: `FIXED`
- **Affected File**: `new_updated__crop_recommendation.ipynb (Cell 8) / PHASE2B_REPORT.md`
- **Root Cause**: Zero exact matches between synthetic and real data was described in parts of the report as conclusive proof of zero leakage or absence of memorization, without acknowledging distance-based density limitations.
- **Evidence**: Written report asserts 'zero leakage detected' based solely on raw/rounded duplicate checks.
- **Required Correction**: Qualify memorization claims: document distance distributions, real-to-real nearest-neighbor baselines, distance ratios, and discuss the limitations of nearest-neighbor distance metrics in continuous feature spaces.
- **Files Modified**: `phase2b_synthetic_fidelity_audit.py, new_updated__crop_recommendation.ipynb (Cell 8)`
- **Regression Tests**: `test_zero_synthetic_test_leakage`
- **Before**: Over-claimed memorization immunity based solely on duplicate count = 0.
- **After (Verified)**: Nearest-neighbor fidelity analysis updated with empirical Real-to-Real baseline (mean distance = 0.4154, median = 0.3905). Nearest-neighbor distance ratios computed (Mode A: 1.054, Mode B: 1.082, Mode C: 1.141). Written reports explicitly qualify that zero exact duplicates does not preclude high localized density.

### DEF-12: Phase 2D (Final Holdout Evaluation) [MEDIUM]

- **Status**: `FIXED`
- **Affected File**: `new_updated__crop_recommendation.ipynb (Cell 12)`
- **Root Cause**: Evaluation lacked statistical significance testing with multiple-comparison correction across all 12 model configurations, and bootstrap confidence intervals were missing for holdout metrics.
- **Evidence**: PHASE2D_MCNEMAR_TESTS.csv only compared a subset of models without Holm-Bonferroni correction, and test accuracy was reported as point estimates without 95% confidence intervals.
- **Required Correction**: Add 1,000-fold bootstrap 95% confidence intervals for Test Accuracy, Macro F1, and MCC. Implement complete pairwise McNemar tests with Holm-Bonferroni multiple-comparison correction.
- **Files Modified**: `phase2d_final_holdout_evaluation.py, new_updated__crop_recommendation.ipynb (Cell 12)`
- **Regression Tests**: `test_confusion_matrix_totals, test_metric_recomputation_consistency`
- **Before**: Point estimates only; incomplete statistical comparisons without family-wise error rate control.
- **After (Verified)**: Added 1,000-fold bootstrap 95% confidence intervals for Test Accuracy, Macro F1, and MCC. Implemented full pairwise exact McNemar tests with Holm-Bonferroni correction across all 12 model configurations, saved to PHASE2D_MCNEMAR_TESTS.csv and PHASE2D_SUMMARY.json.
