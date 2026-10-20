"""
Module: pipeline.py
Master End-to-End Scientific Evaluation Pipeline Runner.
Orchestrates:
1. data_splitting (deterministic quarantine & split)
2. data_validation (physical ranges, schema, zero leakage)
3. training (leak-free training on INTERNAL_TRAIN)
4. evaluation (frozen benchmark evaluation on REAL_TEST)
5. robustness (Phase 2E distribution-shift stress testing)
6. calibration (Brier score, ECE, MCE)
7. explainability (TreeSHAP additivity & TabNet feature saliency)
"""

import sys
import time
from pathlib import Path

curr_dir = Path(__file__).resolve().parent
root_dir = curr_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.data_splitting import partition_and_quarantine_datasets
from src.data_validation import run_comprehensive_validation
from src.training import run_training_pipeline
from src.evaluation import run_evaluation_pipeline
from src.robustness import run_robustness_pipeline
from src.calibration import run_calibration_pipeline
from src.explainability import run_explainability_pipeline

def run_master_pipeline():
    start_time = time.time()
    print("=" * 80)
    print("STARTING CANONICAL MASTER RESEARCH PIPELINE")
    print("=" * 80)
    
    # Step 1: Split & Quarantine
    print("\n>>> STEP 1: DATA SPLITTING & BENCHMARK QUARANTINE")
    partition_and_quarantine_datasets()
    
    # Step 2: Validation & Leakage Audit
    print("\n>>> STEP 2: DATA VALIDATION & PHYSICAL INVARIANTS")
    run_comprehensive_validation()
    
    # Step 3: Leak-Free Training
    print("\n>>> STEP 3: 12-MODEL ARCHITECTURE TRAINING")
    run_training_pipeline()
    
    # Step 4: Benchmark Evaluation
    print("\n>>> STEP 4: FROZEN BENCHMARK EVALUATION & ROW PREDICTIONS")
    run_evaluation_pipeline()
    
    # Step 5: Distribution Shift Stress Testing
    print("\n>>> STEP 5: DISTRIBUTION SHIFT EVALUATION")
    run_robustness_pipeline()
    
    # Step 6: Calibration Audit
    print("\n>>> STEP 6: CALIBRATION AUDIT (BRIER, ECE, MCE)")
    run_calibration_pipeline()
    
    # Step 7: Explainability & TreeSHAP Additivity
    print("\n>>> STEP 7: EXPLAINABILITY & TREESHAP AUDIT")
    run_explainability_pipeline()
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"CANONICAL RESEARCH PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f}s")
    print("=" * 80)

if __name__ == "__main__":
    run_master_pipeline()
