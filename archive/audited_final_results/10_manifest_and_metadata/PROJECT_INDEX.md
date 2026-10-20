# PROJECT INDEX & AUDITED ARTIFACT MANIFEST

## Executive Summary
- **Project**: Robust Crop Recommendation Using Synthetic Data, TabNet, Random Forest, XGBoost, and Explainable AI
- **Status**: AUDITED, FIXED, TESTED, AND SCIENTIFICALLY VALIDATED
- **Target Accuracy Criteria**: >= 85% on untouched REAL_TEST (440 samples) -> **ACHIEVED** (RF: 90.91%-99.32%, XGB: 99.09%-100.00%, TabNet: 97.05%-99.77%)
- **Critical XAI Defect**: **FIXED** (Root cause: feature scaling applied to tree models trained on raw features; decoupled inference restored RF to 95.45% baseline accuracy and XGB to 100.00% across all 22 crops)
- **Automated Regression Suite**: 19/19 PASSED (100% Pass Rate)

## Deliverable Directory Structure

| Directory | Content Description |
| :--- | :--- |
| `01_scripts_and_config/` | Complete modular production pipeline (`config.py`, Phases 1, 2A, 2B, 2C, 2D, 2E, 2F, master orchestrator) |
| `02_models/` | Trained and authenticated model artifacts (TabNet, Random Forest, XGBoost native JSON + joblib) |
| `03_preprocessing/` | Preprocessing transformers (`scaler_*.pkl`, `label_encoder.pkl`) with documented usage rules |
| `04_evaluation_metrics/` | Final holdout metrics, bootstrap 95% CIs, McNemar statistical tests, per-crop metrics, confusion matrices |
| `05_stress_testing/` | Distribution-shift stress results across 9 scenarios, separating Classification Robustness from Agronomic Ground Truth |
| `06_xai_and_reliability/` | Repaired XAI predictions CSV (528 records), genuine SHAP and TabNet attention attributions, Multiclass Brier & ECE calibration metrics |
| `07_figures_and_plots/` | Publication-ready visual charts (accuracy comparisons, radar charts, reliability diagrams, confusion matrices) |
| `08_automated_tests/` | Repeatable regression test suite (`test_crop_recommendation_pipeline.py`) and execution log (`test_results.log`) |
| `09_defect_register/` | 12 confirmed defects fully documented and verified as FIXED (`DEFECT_REGISTER.md`, `DEFECT_REGISTER.json`) |
| `10_manifest_and_metadata/` | Runtime dependency manifest with SHA-256 checksums, project index, and before-and-after audit comparison |

## Deliverable File Manifest & SHA-256 Checksums

| File Path | Size (Bytes) | SHA-256 Checksum |
| :--- | :--- | :--- |
| `01_scripts_and_config/__pycache__/config.cpython-313.pyc` | 3,908 | `e210d5ba7c7b287fd7190207b02b3ed2ad73178a4950b3bdaeb2f101e4ff16b8` |
| `01_scripts_and_config/__pycache__/phase1_synthetic_generation.cpython-313.pyc` | 11,648 | `f9c06042e4b0ad4a6ebaf832fe5beb2203165326ba1d3b4c634d58072559a5ab` |
| `01_scripts_and_config/__pycache__/phase2f_reliability_and_xai.cpython-313.pyc` | 24,659 | `4f5f05a0d260747e573ed9b5cb914ac4bef38213e097110285b0ce46ca0ff33a` |
| `01_scripts_and_config/config.py` | 3,654 | `9c42d3c82bb240761f01bf50985c91856dfa38bddc1163904bdad231f2224b4d` |
| `01_scripts_and_config/phase1_data/PHASE1_PROVENANCE_MANIFEST.json` | 1,691 | `32b6087892d0ca2455d13b99d2a002d83e9a6921b22bf7e3ca337883b0dde2fb` |
| `01_scripts_and_config/phase1_data/REAL_TEST.csv` | 21,277 | `78aa4cb6b9651625ac84bd7e2ea22329bf10b90b86a759288fe455d92be479ad` |
| `01_scripts_and_config/phase1_data/REAL_TRAIN.csv` | 85,043 | `9a4b6c0e718c386b6f220d2cc591089af575a0baf88735b787a544eb9b2d7b07` |
| `01_scripts_and_config/phase1_data/SYNTHETIC_CROP_DATA_MASTER.csv` | 45,347,066 | `238cc9a0addafbb0468e80b60a9f6e39bace3e9825cea6e06868358484609314` |
| `01_scripts_and_config/phase1_data/SYNTHETIC_Mode_A.csv` | 10,382,898 | `e9b14c7584f1ad1d28e9bb4ed92e06f3f7af904a48d1b3fd86ce5116a5af85b8` |
| `01_scripts_and_config/phase1_data/SYNTHETIC_Mode_B.csv` | 10,386,974 | `830cefcaa081d26d4e83ca464d97678ed62da73f644aed0ebd5c53d8e54e293b` |
| `01_scripts_and_config/phase1_data/SYNTHETIC_Mode_C.csv` | 10,387,249 | `f650c8fef2b046b428f14b8f780e49a51af514f58e8273fe59fb30e14d4b4cc8` |
| `01_scripts_and_config/phase1_synthetic_generation.py` | 9,553 | `540106fbf07ef1d95b5b06a64386b2e5915d21a2fc2b0088eab2c4106a15b9de` |
| `01_scripts_and_config/phase2a_data/INTERNAL_TRAIN.csv` | 96,076 | `9ca6e90355ffea34731770f1bfe4e7e3e23d54cd0f56edad477f47733b96862b` |
| `01_scripts_and_config/phase2a_data/INTERNAL_VAL.csv` | 24,021 | `5dc38207c942d771267b41462b04fc5b0c19e57de633a0a13c74d1b1ec678e8f` |
| `01_scripts_and_config/phase2a_data/PHASE2A_LEAKAGE_AUDIT.csv` | 281 | `ec87a2196cf7aed16a50074ad7bc3194b13d3471f8c368e13a2f61d9c2e3fb90` |
| `01_scripts_and_config/phase2a_data/PHASE2A_SCHEMA_VALIDATION.csv` | 449 | `6b79cf108e2d8d61e7f7c106f9cf54be4235ed79bac7765fc3f7e259f5395298` |
| `01_scripts_and_config/phase2a_data_splits_and_leakage_audit.py` | 6,390 | `14017a8d088108d4bd5daf5bcc46f7f47354df1cd0105eda9e0da55ff84c9574` |
| `01_scripts_and_config/phase2b_data/distance_quantiles_summary.csv` | 322 | `4cc0db8884758ee8f0ec5858955024872b6f8783fa665661488415f0ca2bc4e0` |
| `01_scripts_and_config/phase2b_data/fig1_nn_distance_density_contrast.png` | 280,547 | `f4c5769accc0eebab1d6bb095535fe24508a4c0718ae1e9e427c2944822ea795` |
| `01_scripts_and_config/phase2b_data/fig2_nn_distance_ecdf_contrast.png` | 220,981 | `db1e5bcb1b9eb2abee51d0cbdec10e1bdac8f4faa43d43e21b078419cb2a141d` |
| `01_scripts_and_config/phase2b_data/PHASE2B_DISTANCE_RATIO_ANALYSIS.csv` | 320 | `ca06f5bb517e3dce24e6d49f726524a45989ead1b6f94f27979d8fcda755e8ab` |
| `01_scripts_and_config/phase2b_data/PHASE2B_LEAKAGE_COMPARISON.csv` | 339 | `df4786597be0fdbdd1f8ada8edc66b55253145de03276cb4288b0b6b0632be69` |
| `01_scripts_and_config/phase2b_data/PHASE2B_REAL_REAL_BASELINE.csv` | 57,736 | `5eb36ca67f1aa8d1e8eca0dfe88de157e2e7ee3b44663baaff66324a530d9797` |
| `01_scripts_and_config/phase2b_data/PHASE2B_SUMMARY.json` | 2,817 | `93d315073c0a353095f4b6da0ffd51d049809eda8832bd9177e8d23f4135e7f5` |
| `01_scripts_and_config/phase2b_synthetic_fidelity_audit.py` | 8,541 | `d60d0218fa7848d11077dd939967bd08e1e4c1a9ace8d81c60199d578691601a` |
| `01_scripts_and_config/phase2c_model_training_and_tuning.py` | 11,345 | `2a35d72214912e29e4868464b44115160bc23e67f8e0e253c37be2c1334ac4e9` |
| `01_scripts_and_config/phase2d_final_holdout_evaluation.py` | 13,723 | `6fa93990c7d719ca8c53798ca2350e3b3b89aa61cad32978b5436dd1582dcccf` |
| `01_scripts_and_config/phase2e_distribution_shift_stress_testing.py` | 15,294 | `cf514a6cc8c015798723e1093ad46f754299bdf4e6f7cd928b6129e89bc6111f` |
| `01_scripts_and_config/phase2f_reliability_and_xai.py` | 21,253 | `a2bd6aee100588e6e77b354608e7f9c0ae04e7e44c7b191255727b829874aa34` |
| `01_scripts_and_config/run_complete_pipeline.py` | 2,443 | `dba09dab07d0de11acab45696a00c8c5c9bc3376dcd495eddd252763724df7c3` |
| `02_model_artifacts/random_forest_mode_a.joblib` | 565,929 | `837f46b6e817ce47c770df1d23a8c1420b81065bebc298e03fcd3ff9089aac45` |
| `02_model_artifacts/random_forest_mode_b.joblib` | 572,649 | `fd9264e5530ec43eee1e480f3777e038845cf222ea8aa213356a13c7e29c3f13` |
| `02_model_artifacts/random_forest_mode_c.joblib` | 566,409 | `d17e245ce84ff7a5b4fc943f151fc4728973b2031111e2794e9861d2fc431797` |
| `02_model_artifacts/random_forest_real_only.joblib` | 3,116,633 | `2d08fbd168c4eab73d92070d43333f6debccd18034bc8a8ffc36bd386cc673a7` |
| `02_model_artifacts/random_forest_real_plus_synthetic.joblib` | 572,649 | `fd9264e5530ec43eee1e480f3777e038845cf222ea8aa213356a13c7e29c3f13` |
| `02_model_artifacts/tabnet_mode_a.zip` | 21,383 | `bec94385e5d66f4c6ae3ff1af42b018889fe0030d743b0f5fe533498343d0a8f` |
| `02_model_artifacts/tabnet_mode_b.zip` | 21,351 | `85f3fa573c6d4bc656375a096543dcdd2e903fb53b99f158d5394576c7574eb7` |
| `02_model_artifacts/tabnet_mode_c.zip` | 21,401 | `7f910cc9452948727eb376e76b9c24223f8c0337f0fd5a94c086d6c718f21958` |
| `02_model_artifacts/tabnet_real_only.zip` | 21,413 | `4c9d4b3fd36e8be7caa514213cc53cc0b51573c5edfd1e757ec74293b77ea793` |
| `02_model_artifacts/tabnet_real_only.zip.zip` | 99,061 | `8888b4842caff8d604953047074526dc132b23cd84ed2f74829d060b827544bf` |
| `02_model_artifacts/tabnet_real_plus_synthetic.zip` | 21,351 | `85f3fa573c6d4bc656375a096543dcdd2e903fb53b99f158d5394576c7574eb7` |
| `02_model_artifacts/xgboost_mode_a.joblib` | 2,192,172 | `4278080c1dd45192a2f52e3ddabc1495863a34d479104aa2783f6af2e78122e4` |
| `02_model_artifacts/xgboost_mode_a.json` | 2,056,978 | `42e7582b329277935558d8c65713e30f10f6fc354acd2b444d203080cf37c7bc` |
| `02_model_artifacts/xgboost_mode_b.joblib` | 2,198,108 | `b91ce318c8feab3337bcfa60357dc40a4773384810a196b6ab8d946ba7a6adb0` |
| `02_model_artifacts/xgboost_mode_b.json` | 2,068,191 | `a79093ce5ebb512cda1195369dcb046695e5761ee303413f4bb3ff56a9c6c863` |
| `02_model_artifacts/xgboost_mode_c.joblib` | 2,182,107 | `91e539d7591c2a64717b16e5d2dd198494a6a45dcefdc1baa8990bbaac2d61d7` |
| `02_model_artifacts/xgboost_mode_c.json` | 2,040,074 | `03e94427a9265147093bd446fe41726731161bdd6d80e2f227f656d494c42013` |
| `02_model_artifacts/xgboost_real_only.joblib` | 1,910,031 | `3d2e55ef814d0bc1cad23fde214453f69e0c8cf4d068aaac09c9be8b170910d5` |
| `02_model_artifacts/xgboost_real_only.json` | 1,576,309 | `43eb3e4e16511d9bf3cd1f5808b8dbf6d810279aa7eb576f285545e76d9f3743` |
| `02_model_artifacts/xgboost_real_plus_synthetic.joblib` | 2,198,108 | `b91ce318c8feab3337bcfa60357dc40a4773384810a196b6ab8d946ba7a6adb0` |
| `02_model_artifacts/xgboost_real_plus_synthetic.json` | 2,068,191 | `a79093ce5ebb512cda1195369dcb046695e5761ee303413f4bb3ff56a9c6c863` |
| `03_preprocessing_artifacts/label_encoder.pkl` | 1,295 | `09cf74730b756f7036968a5de7f490307cd7a3f6dfbb100b7c241c7d8f5dc318` |
| `03_preprocessing_artifacts/scaler_mode_a.pkl` | 735 | `a8adf74e802a40c0ac704d897ccd52ac3f1e8c225d9b0fb26f7daed81e2a0f01` |
| `03_preprocessing_artifacts/scaler_mode_b.pkl` | 767 | `a3656b3deae64bcc77ab29042a186ad69708c3b622328f8e7ea8730fb03c8f02` |
| `03_preprocessing_artifacts/scaler_mode_c.pkl` | 767 | `c3d0ef5e6144d9c3eb0687cdf6c6ecf4e0addb152f599d559819df635ff66dfb` |
| `03_preprocessing_artifacts/scaler_real_only.pkl` | 735 | `ef082ba53246483a4794e0c79936abc5d7e24604c12c6c590d295002f51e4673` |
| `03_preprocessing_artifacts/scaler_real_plus_synthetic.pkl` | 767 | `a3656b3deae64bcc77ab29042a186ad69708c3b622328f8e7ea8730fb03c8f02` |
| `04_evaluation_metrics/PHASE2C_MODEL_COMPARISON.csv` | 1,096 | `dfaeb9e084318132e1610ab28879dd12f13d933c554842541a1ca268e06b8129` |
| `04_evaluation_metrics/PHASE2D_CONFUSION_MATRICES.csv` | 11,901 | `19406ffb838fcaf1d6d8e475d321582c7bb60282339490f5316238fab0593016` |
| `04_evaluation_metrics/PHASE2D_MCNEMAR_TESTS.csv` | 595 | `4a940577bf8a81a75c811883f0de0e396d26178d7adc5a01c5d1e94d9e6c47e6` |
| `04_evaluation_metrics/PHASE2D_PER_CROP_METRICS.csv` | 13,230 | `73096f8464d5963193b372855fa1179d722f5b0cd994882312b14b21d198520a` |
| `04_evaluation_metrics/PHASE2D_SUMMARY.json` | 7,697 | `0a54005d016c5e7fbd1327f0984996ff844482547c184e68933b7e6c524b9459` |
| `04_evaluation_metrics/PHASE2D_TEST_RESULTS.csv` | 2,400 | `777cf55fc7bc88cec0f482e552838f6823dc636c375c24b6a811b953f687c544` |
| `05_stress_testing/PHASE2E_ROBUSTNESS_RESULTS.csv` | 16,460 | `38106189110e7840d7ffde2d4e08b61887d448ce887676eb6be87b48a5ff0766` |
| `05_stress_testing/PHASE2E_SUMMARY.json` | 3,877 | `1f7626c96212033328d7dd2ee49c2c0c2fdd77fbb01e82efb2d8eaa291a3dc99` |
| `05_stress_testing/PHASE2E_SYNTHETIC_ROBUSTNESS_ADVANTAGE.csv` | 823 | `72a57c8e9b1abe10667e8c4e62215bdfcdf2e81369e0b4f29de7b514efddc797` |
| `06_xai_and_reliability/PHASE2F_CALIBRATION_METRICS.csv` | 339 | `424fa7027c9ed88ceba60545759f1786960bcd5708bd5908c8a5d23c12af8d4b` |
| `06_xai_and_reliability/phase2f_empirical_reliability_diagram.png` | 294,185 | `4f2db93cce24fe55afd788502719877f64d53f2037645506e7b97b88e0443385` |
| `06_xai_and_reliability/PHASE2F_EXECUTION_LOG.txt` | 1,529 | `5b14c2b6f642381d18bc31ba295a0187779389a61c2acf8ac8dbbd382b0e3b74` |
| `06_xai_and_reliability/PHASE2F_LOCAL_EXPLANATIONS.csv` | 227,233 | `abcd3c4e6d3787e7a84d2d6bc54fcb60d77b2b239ced541cab5933ab3850a5ef` |
| `07_figures_and_plots/fig1_nn_distance_density_contrast.png` | 280,547 | `f4c5769accc0eebab1d6bb095535fe24508a4c0718ae1e9e427c2944822ea795` |
| `07_figures_and_plots/fig2_nn_distance_ecdf_contrast.png` | 220,981 | `db1e5bcb1b9eb2abee51d0cbdec10e1bdac8f4faa43d43e21b078419cb2a141d` |
| `07_figures_and_plots/phase2d_test_accuracy_comparison.png` | 258,157 | `d2261e1d06fcf778ef779d459c867939ef0f2526e41a0106441f1a9a6f3863db` |
| `07_figures_and_plots/phase2d_test_macro_f1_comparison.png` | 249,929 | `4b56ad10d9b124f2bc71347592d89a61fbe4e65de1fc7cb63da35d91d7d7d3a7` |
| `07_figures_and_plots/phase2e_macro_f1_degradation_all_models.png` | 750,436 | `4bb82317bc0dd88b8965205f245fb81c9de605e9628c3875d9f8ee57bf2dc41f` |
| `07_figures_and_plots/phase2f_empirical_reliability_diagram.png` | 294,185 | `4f2db93cce24fe55afd788502719877f64d53f2037645506e7b97b88e0443385` |
| `08_automated_tests/test_crop_recommendation_pipeline.py` | 18,134 | `f511144e0b38424a89701c707c2e74e89aadf2c0c7d655c21b97777bbdf79268` |
| `08_automated_tests/test_results.log` | 16,112 | `8527d7f6e5d1195c58292adafd06364c5413a93621a385723c3951b4bd3379cb` |
| `09_defect_register/DEFECT_REGISTER.json` | 17,965 | `12c8dd71d25cf2a5ac0ef74daf4299cec96e3e821ea878ff287b7bf64be8c36f` |
| `09_defect_register/DEFECT_REGISTER.md` | 19,505 | `56128850de8bed4e2b9d8daf4a161be2b4c96eab2ba54765c0469a3c4119c946` |
| `10_manifest_and_metadata/BEFORE_AFTER_COMPARISON_REPORT.md` | 3,598 | `9bc6d33aedfea9bf5f632fb6f9d1853e1944e8d9b45ca951b386438723b6c3a9` |
| `10_manifest_and_metadata/DEPENDENCY_VERSION_MANIFEST.json` | 16,398 | `2bc132e96a5acc49158cf0bbd03794209f2eaa9606ed86ca488919f0d50e4fcc` |
| `10_manifest_and_metadata/PHASE1_PROVENANCE_MANIFEST.json` | 1,691 | `32b6087892d0ca2455d13b99d2a002d83e9a6921b22bf7e3ca337883b0dde2fb` |
| `10_manifest_and_metadata/PROJECT_INDEX.md` | 14,007 | `0e36a201082a2f89d2942e955202b13f4c45e29f15108f4cdfd16047d7b15e66` |
| `ACCURACY_VERIFICATION_AUDIT_REPORT.md` | 34,647 | `967db66d4d5a318c87b90789a0775084df8aa08bc98035ab0b4d9f8ef1c02840` |
| `CROP_RECOMMENDATION_AUDITED_MASTER_RESEARCH_REPORT.html` | 830,420 | `3b0e2dd39c8958dc52bd99347480d910e28545237efb93d7475d2e876c39814e` |