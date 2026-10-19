# PHASE 2F VALIDATION REPORT: RELIABILITY, CALIBRATION & XAI

## Verification Check Status
- `inference_reproduction_passed`: True
- `shap_additivity_passed`: True (Exact additivity verified with residual error < 1e-6 across 352 TreeSHAP records)
- `tabnet_attention_masks_extracted`: True (Native .explain() extracted across 176 records)
- `calibration_metrics_generated`: True (Multiclass Brier score and ECE computed across 10 bins)
- `missing_probability_count`: 0 (All 528 explanation records have non-null probabilities)
- `explanation_record_count`: 528 records (22 crops x 8 conditions x 3 models)
- `unsupported_reliability_claims_excised`: True ('100% reliability' text claim replaced with empirical calibration)
- `decoupled_feature_routing_verified`: True (Random Forest and XGBoost receive raw unscaled features; TabNet receives standard-scaled features)

## Calibration Metrics Summary (REAL_TEST, N=440)
| Model | Preprocessing | Multiclass Brier Score | Expected Calibration Error (ECE) | Maximum Calibration Error (MCE) |
| :--- | :--- | :--- | :--- | :--- |
| **TabNet (Real + Mode B)** | StandardScaler | 0.01227 | 0.01134 (1.13%) | 0.47964 |
| **XGBoost (Real + Mode B)** | RAW Unscaled | 0.00150 | 0.00306 (0.31%) | 0.49711 |
| **Random Forest (Real + Mode B)** | RAW Unscaled | 0.56054 | 0.60459 (60.46%) | 0.90569 |

## Attributions & Botanical Alignment
- **Rice**: Positive drivers are Rainfall (>180 mm) and Humidity (>80%).
- **Cotton**: Positive drivers are Potassium (K > 40) and Temperature (>25 C).
- **Chickpea**: Governed by low rainfall (arid tolerance) and high phosphorus.
- **Apple & Grapes**: Strictly gated by high potassium threshold (K > 190).
