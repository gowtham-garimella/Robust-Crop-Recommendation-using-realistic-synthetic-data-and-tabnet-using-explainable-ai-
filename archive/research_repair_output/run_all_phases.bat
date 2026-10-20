@echo off
REM ==============================================================================
REM MASTER EXECUTION SCRIPT: CROP RECOMMENDATION AUDITED RESEARCH PIPELINE
REM Windows-compatible runner executing all phases sequentially with error trapping
REM ==============================================================================

echo [INFO] Starting Crop Recommendation Audited Research Pipeline...
set SCRIPT_DIR=%~dp0src
set DATA_DIR=%~dp0data

REM Verify Python environment
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH. Exiting.
    exit /b 1
)

echo [STEP 1/7] Running Phase 1: Synthetic Data Generation & Audit...
python "%SCRIPT_DIR%\phase1_synthetic_generation.py"
if errorlevel 1 (
    echo [ERROR] Phase 1 failed!
    exit /b 1
)

echo [STEP 2/7] Running Phase 2A: Data Splitting and Leakage Audit...
python "%SCRIPT_DIR%\phase2a_data_splits_and_leakage_audit.py"
if errorlevel 1 (
    echo [ERROR] Phase 2A failed!
    exit /b 1
)

echo [STEP 3/7] Running Phase 2B: Synthetic Fidelity & Leakage Evaluation...
python "%SCRIPT_DIR%\phase2b_synthetic_fidelity_audit.py"
if errorlevel 1 (
    echo [ERROR] Phase 2B failed!
    exit /b 1
)

echo [STEP 4/7] Running Phase 2C: Model Training & Validation...
python "%SCRIPT_DIR%\phase2c_model_training_and_tuning.py"
if errorlevel 1 (
    echo [ERROR] Phase 2C failed!
    exit /b 1
)

echo [STEP 5/7] Running Phase 2D: Final Holdout Evaluation...
python "%SCRIPT_DIR%\phase2d_final_holdout_evaluation.py"
if errorlevel 1 (
    echo [ERROR] Phase 2D failed!
    exit /b 1
)

echo [STEP 6/7] Running Phase 2E: Distribution-Shift Stress Testing...
python "%SCRIPT_DIR%\phase2e_distribution_shift_stress_testing.py"
if errorlevel 1 (
    echo [ERROR] Phase 2E failed!
    exit /b 1
)

echo [STEP 7/7] Running Phase 2F: Reliability, Calibration and XAI...
python "%SCRIPT_DIR%\phase2f_reliability_and_xai.py"
if errorlevel 1 (
    echo [ERROR] Phase 2F failed!
    exit /b 1
)

echo [VERIFICATION] Executing Automated Regression Test Suite...
python "%~dp0tests\test_crop_recommendation_pipeline.py"
if errorlevel 1 (
    echo [ERROR] Automated regression tests failed!
    exit /b 1
)

echo ==============================================================================
echo [SUCCESS] ALL 7 PHASES AND AUTOMATED TESTS COMPLETED WITH 100%% INTEGRITY!
echo ==============================================================================
