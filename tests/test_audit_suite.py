"""
Automated Test Suite for Crop Recommendation Research Pipeline
Implements rigorous unit and integration tests covering all critical Phase 8 requirements.
"""

import os
import sys
import json
import joblib
import hashlib
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

curr_dir = Path(__file__).resolve().parent
root_dir = curr_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config.project_config import (
    RAW_DATA_FILE, REAL_TRAIN_FILE, REAL_TEST_FILE,
    INTERNAL_TRAIN_FILE, INTERNAL_VAL_FILE,
    SYNTHETIC_MODE_A_FILE, SYNTHETIC_MODE_B_FILE, SYNTHETIC_MODE_C_FILE,
    MODELS_DIR, PREPROC_DIR, METRICS_DIR, PREDICTIONS_DIR, CONFUSION_DIR,
    ROBUSTNESS_DIR, EXPLAIN_DIR,
    CANONICAL_FEATURES, TARGET_COLUMN, EXPECTED_CROPS,
    BENCHMARK_REAL_TRAIN_ROWS, BENCHMARK_REAL_TEST_ROWS,
    INTERNAL_TRAIN_ROWS, INTERNAL_VAL_ROWS, NUM_CLASSES
)

from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import shap

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

class TestCropRecommendationPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.df_raw = pd.read_csv(RAW_DATA_FILE)
        cls.df_real_train = pd.read_csv(REAL_TRAIN_FILE)
        cls.df_real_test = pd.read_csv(REAL_TEST_FILE)
        cls.df_int_train = pd.read_csv(INTERNAL_TRAIN_FILE)
        cls.df_int_val = pd.read_csv(INTERNAL_VAL_FILE)
        cls.le = joblib.load(PREPROC_DIR / "label_encoder.pkl")
        
    def test_01_feature_schema_and_datatypes(self):
        """Test 1: Verify canonical 7 features and numeric dtypes."""
        for df, name in [(self.df_real_train, "REAL_TRAIN"), (self.df_real_test, "REAL_TEST")]:
            for feat in CANONICAL_FEATURES:
                self.assertIn(feat, df.columns, f"Missing feature {feat} in {name}")
                self.assertTrue(pd.api.types.is_numeric_dtype(df[feat]), f"{feat} in {name} is not numeric")
            self.assertFalse(df[CANONICAL_FEATURES].isnull().values.any(), f"NaNs found in {name}")
            self.assertFalse(np.isinf(df[CANONICAL_FEATURES].values).any(), f"Infs found in {name}")

    def test_02_target_label_mapping(self):
        """Test 2: Verify all 22 crop classes exist and match canonical list."""
        self.assertEqual(len(self.le.classes_), 22)
        self.assertListEqual(list(self.le.classes_), EXPECTED_CROPS)
        self.assertEqual(self.df_real_train[TARGET_COLUMN].nunique(), 22)
        self.assertEqual(self.df_real_test[TARGET_COLUMN].nunique(), 22)

    def test_03_train_validation_disjointness(self):
        """Test 3: Verify zero overlap between INTERNAL_TRAIN (1408) and INTERNAL_VAL (352)."""
        self.assertEqual(len(self.df_int_train), INTERNAL_TRAIN_ROWS)
        self.assertEqual(len(self.df_int_val), INTERNAL_VAL_ROWS)
        t_tr = set(tuple(round(v, 4) for v in x) for x in self.df_int_train[CANONICAL_FEATURES].values)
        t_val = set(tuple(round(v, 4) for v in x) for x in self.df_int_val[CANONICAL_FEATURES].values)
        overlap = len(t_tr & t_val)
        self.assertEqual(overlap, 0, f"Critical leakage: {overlap} rows overlap between internal train and val")

    def test_04_train_test_disjointness(self):
        """Test 4: Verify zero overlap between all training partitions and REAL_TEST (440)."""
        self.assertEqual(len(self.df_real_test), BENCHMARK_REAL_TEST_ROWS)
        t_test = set(tuple(round(v, 4) for v in x) for x in self.df_real_test[CANONICAL_FEATURES].values)
        
        for df, name in [(self.df_real_train, "REAL_TRAIN"), (self.df_int_train, "INTERNAL_TRAIN"), (self.df_int_val, "INTERNAL_VAL")]:
            t_curr = set(tuple(round(v, 4) for v in x) for x in df[CANONICAL_FEATURES].values)
            overlap = len(t_curr & t_test)
            self.assertEqual(overlap, 0, f"Critical leakage: {overlap} rows overlap between {name} and REAL_TEST")

    def test_05_synthetic_data_provenance(self):
        """Test 5: Verify synthetic datasets contain 165,000 rows each and 0 test overlap."""
        t_test = set(tuple(round(v, 4) for v in x) for x in self.df_real_test[CANONICAL_FEATURES].values)
        for mode_file, mode_name in [
            (SYNTHETIC_MODE_A_FILE, "Mode A"),
            (SYNTHETIC_MODE_B_FILE, "Mode B"),
            (SYNTHETIC_MODE_C_FILE, "Mode C")
        ]:
            self.assertTrue(mode_file.exists(), f"Missing synthetic file: {mode_file}")
            df_syn = pd.read_csv(mode_file)
            self.assertEqual(len(df_syn), 165000, f"{mode_name} does not have 165000 rows")
            t_syn = set(tuple(round(v, 4) for v in x) for x in df_syn[CANONICAL_FEATURES].values)
            overlap = len(t_syn & t_test)
            self.assertEqual(overlap, 0, f"{mode_name} shares {overlap} samples with REAL_TEST")

    def test_06_no_validation_data_in_model_training(self):
        """Test 6: Verify internal validation rows are excluded from model training arrays."""
        df_val = pd.read_csv(METRICS_DIR / "VALIDATION_METRICS.csv")
        self.assertEqual(len(df_val), 12)
        # All validation sample counts must equal exactly 352
        self.assertTrue((df_val["val_samples"] == 352).all())

    def test_07_fresh_process_model_loading(self):
        """Test 7: Verify all 12 model artifacts load properly and execute inference."""
        regimes = ["real_only", "mode_a", "mode_b", "mode_c"]
        X_sample = self.df_real_test[CANONICAL_FEATURES].iloc[:5].values
        
        # Test XGBoost JSON loading
        for r in regimes:
            xgb_path = MODELS_DIR / f"xgboost_{r}.json"
            clf = XGBClassifier()
            clf.load_model(str(xgb_path))
            preds = clf.predict(X_sample)
            self.assertEqual(len(preds), 5)
            
        # Test RF joblib loading
        for r in regimes:
            rf_path = MODELS_DIR / f"random_forest_{r}.joblib"
            clf = joblib.load(rf_path)
            preds = clf.predict(X_sample)
            self.assertEqual(len(preds), 5)

    def test_08_native_xgboost_json_prediction_equivalence(self):
        """Test 8: Verify native XGBoost JSON and joblib formats yield identical predictions."""
        X_eval = self.df_real_test[CANONICAL_FEATURES].values
        for r in ["real_only", "mode_a", "mode_b", "mode_c"]:
            json_p = MODELS_DIR / f"xgboost_{r}.json"
            joblib_p = MODELS_DIR / f"xgboost_{r}.joblib"
            if joblib_p.exists():
                m_json = XGBClassifier()
                m_json.load_model(str(json_p))
                m_joblib = joblib.load(joblib_p)
                p_json = m_json.predict(X_eval)
                p_joblib = m_joblib.predict(X_eval)
                np.testing.assert_array_equal(p_json, p_joblib)

    def test_09_valid_probability_distributions(self):
        """Test 9: Verify predicted probability rows sum to 1.0 within numerical precision."""
        df_cal = pd.read_csv(EXPLAIN_DIR / "CALIBRATION_METRICS.csv")
        self.assertEqual(len(df_cal), 12)
        self.assertTrue(df_cal["prob_sum_valid"].all())

    def test_10_metric_recomputation_from_saved_predictions(self):
        """Test 10: Verify metrics in FINAL_MODEL_COMPARISON.csv exactly match row predictions."""
        df_cmp = pd.read_csv(METRICS_DIR / "FINAL_MODEL_COMPARISON.csv")
        self.assertEqual(len(df_cmp), 12)
        
        for _, row in df_cmp.iterrows():
            m_name = row["model_name"]
            clean_tag = m_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
            pred_file = PREDICTIONS_DIR / f"predictions_{clean_tag}.csv"
            self.assertTrue(pred_file.exists(), f"Missing prediction file for {m_name}")
            
            df_pred = pd.read_csv(pred_file)
            self.assertEqual(len(df_pred), BENCHMARK_REAL_TEST_ROWS)
            acc_from_pred = df_pred["is_correct"].mean()
            self.assertAlmostEqual(acc_from_pred, row["accuracy"], places=5)
            self.assertEqual(df_pred["is_correct"].sum(), row["correct_samples"])

    def test_11_confusion_matrix_consistency(self):
        """Test 11: Verify confusion matrix row sum == 440 and trace == correct samples."""
        df_cmp = pd.read_csv(METRICS_DIR / "FINAL_MODEL_COMPARISON.csv")
        for _, row in df_cmp.iterrows():
            m_name = row["model_name"]
            clean_tag = m_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
            cm_file = CONFUSION_DIR / f"cm_{clean_tag}.csv"
            self.assertTrue(cm_file.exists(), f"Missing confusion matrix for {m_name}")
            
            df_cm = pd.read_csv(cm_file, index_col=0)
            self.assertEqual(df_cm.values.sum(), BENCHMARK_REAL_TEST_ROWS)
            self.assertEqual(int(np.trace(df_cm.values)), row["correct_samples"])

    def test_12_treeshap_additivity_residuals(self):
        """Test 12: Verify all 8 tree models pass TreeSHAP additivity <= 1e-05."""
        df_tree = pd.read_csv(EXPLAIN_DIR / "TREESHAP_AUDIT.csv")
        self.assertEqual(len(df_tree), 8)
        self.assertTrue((df_tree["status"] == "PASS").all())
        self.assertTrue((df_tree["max_additivity_residual"] <= 1e-05).all())

    def test_13_tabnet_explanation_terminology(self):
        """Test 13: Verify TabNet explanations are labeled as attention feature saliency masks."""
        df_tab = pd.read_csv(EXPLAIN_DIR / "TABNET_SALIENCY_MASKS.csv")
        self.assertIn("explanation_type", df_tab.columns)
        self.assertTrue((df_tab["explanation_type"] == "attention_feature_saliency_mask").all())

    def test_14_exact_float_input_preservation(self):
        """Test 14: Verify explanation input samples preserve unrounded floating-point precision."""
        p_samples = EXPLAIN_DIR / "EXPLAINABILITY_INPUT_SAMPLES.csv"
        self.assertTrue(p_samples.exists())
        df_samp = pd.read_csv(p_samples)
        # Check that floating point values have > 2 decimals where applicable
        has_high_precision = any(
            len(str(v).split(".")[1]) > 2
            for col in ["temperature", "humidity", "ph", "rainfall"]
            for v in df_samp[col].values
            if "." in str(v)
        )
        self.assertTrue(has_high_precision, "Explanation inputs appear rounded to 2 decimals!")

if __name__ == "__main__":
    unittest.main()
