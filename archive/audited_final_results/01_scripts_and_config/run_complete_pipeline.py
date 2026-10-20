"""
Master Pipeline Orchestrator: Complete Crop Recommendation Project Audit & Execution
Project: Robust Crop Recommendation Using Synthetic Data, TabNet, and Explainable AI
"""

import os
import sys
import time
import json
from pathlib import Path

curr_dir = Path(__file__).resolve().parent
if str(curr_dir) not in sys.path:
    sys.path.insert(0, str(curr_dir))

from config import (
    AUDITED_RESULTS_DIR, MANIFEST_DIR, DEFECTS_DIR
)

from phase1_synthetic_generation import run_phase1_pipeline
from phase2a_data_splits_and_leakage_audit import run_phase2a_audit
from phase2b_synthetic_fidelity_audit import run_phase2b_fidelity_audit
from phase2c_model_training_and_tuning import run_phase2c_training
from phase2d_final_holdout_evaluation import run_phase2d_evaluation
from phase2e_distribution_shift_stress_testing import run_phase2e_stress_testing
from phase2f_reliability_and_xai import run_phase2f_reliability_and_xai

def execute_all_phases():
    """Runs all project phases end-to-end and authenticates results."""
    t_start = time.time()
    print("=" * 80)
    print("STARTING END-TO-END AUDITED CROP RECOMMENDATION PIPELINE")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 1. Phase 1
    t0 = time.time()
    p1_out = run_phase1_pipeline()
    print(f"[Phase 1 Complete in {time.time()-t0:.1f}s]")
    
    # 2. Phase 2A
    t0 = time.time()
    p2a_out = run_phase2a_audit()
    print(f"[Phase 2A Complete in {time.time()-t0:.1f}s]")
    
    # 3. Phase 2B
    t0 = time.time()
    p2b_out = run_phase2b_fidelity_audit()
    print(f"[Phase 2B Complete in {time.time()-t0:.1f}s]")
    
    # 4. Phase 2C
    t0 = time.time()
    p2c_out = run_phase2c_training()
    print(f"[Phase 2C Complete in {time.time()-t0:.1f}s]")
    
    # 5. Phase 2D
    t0 = time.time()
    p2d_out = run_phase2d_evaluation()
    print(f"[Phase 2D Complete in {time.time()-t0:.1f}s]")
    
    # 6. Phase 2E
    t0 = time.time()
    p2e_out = run_phase2e_stress_testing()
    print(f"[Phase 2E Complete in {time.time()-t0:.1f}s]")
    
    # 7. Phase 2F
    t0 = time.time()
    p2f_out = run_phase2f_reliability_and_xai()
    print(f"[Phase 2F Complete in {time.time()-t0:.1f}s]")
    
    total_elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"ALL PHASES SUCCESSFULLY COMPLETED IN {total_elapsed:.1f}s")
    print("=" * 80)

if __name__ == "__main__":
    execute_all_phases()
