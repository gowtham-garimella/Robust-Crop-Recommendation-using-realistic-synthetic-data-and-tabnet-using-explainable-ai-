# Robust Crop Recommendation Using Realistic Synthetic Data, TabNet, and Explainable AI

A reproducible machine-learning research project evaluating deep tabular models and tree ensembles with agriculturally constrained synthetic data.

---

## 1. Project Title
**Robust Crop Recommendation Using Realistic Synthetic Data, TabNet, and Explainable AI**

This project provides a clean, rigorous, and fully reproducible machine-learning framework to recommend the most suitable crop for a given plot of land based on environmental and soil measurements. It explores whether biologically and physically constrained synthetic data can improve model robustness under standard conditions and controlled distribution shifts.

---

## 2. Project Overview
Choosing the right crop is one of the most critical decisions a farmer makes. If a farmer plants a crop that is not well suited to their local soil chemistry or climate, crop growth suffers, fertilizers are wasted, and financial losses follow. Machine-learning models can analyze historical agricultural conditions and recommend optimal crops.

### The Seven Input Features
The model takes exactly seven numerical measurements:
1. **Nitrogen (N):** Ratio of Nitrogen content in soil (kg/ha).
2. **Phosphorus (P):** Ratio of Phosphorus content in soil (kg/ha).
3. **Potassium (K):** Ratio of Potassium content in soil (kg/ha).
4. **Temperature:** Ambient air temperature in degrees Celsius (°C).
5. **Humidity:** Relative atmospheric humidity percentage (%).
6. **Soil pH:** Acidity or alkalinity of the soil (measured on a 0 to 14 standard pH scale).
7. **Rainfall:** Total rainfall precipitation volume in millimeters (mm).

### What the Model Predicts
The model predicts a **crop category** (one of 22 supported crops: rice, maize, chickpea, kidneybeans, pigeonpeas, mothbeans, mungbean, blackgram, lentil, pomegranate, banana, mango, grapes, watermelon, muskmelon, apple, orange, papaya, coconut, cotton, jute, or coffee).

> **Important Clarification:** The model performs categorical **crop suitability recommendation**. It does **not** predict numerical crop yield (kg/hectare) or guarantee financial profit.

---

## 3. Project Objectives
1. **Build a multiclass crop recommendation classifier** across 22 crop categories.
2. **Compare three distinct machine-learning architectures:** Attentive deep learning (TabNet) versus tree-based ensembles (Random Forest, XGBoost).
3. **Investigate realistic synthetic data augmentation:** Evaluate whether generating synthetic samples using empirical copulas and physiological constraints helps neural and tree models.
4. **Evaluate predictive accuracy:** Rigorously test models using both quarantined internal validation and a frozen holdout benchmark.
5. **Examine robustness under simulated distribution shifts:** Subject models to climate stress (heat waves, drought, combined volatility), soil degradation, and sensor noise.
6. **Apply Explainable AI (XAI):** Audit feature contributions using exact TreeSHAP additivity and inspect TabNet sparse attention masks.

---

## 4. Models Used

### Models Compared
| Model | Simple Explanation |
|---|---|
| **TabNet** | A neural-network architecture designed for tabular data that uses sequential attention mechanisms to select features at each decision step. |
| **Random Forest** | A collection of decision trees whose predictions are combined (averaged) to reduce variance and improve generalization. |
| **XGBoost** | A boosted-tree algorithm that builds trees sequentially to improve predictions, with each tree correcting errors from earlier iterations. |

### Four Training Regimes
Every model architecture is trained and compared across four distinct data regimes (12 configurations in total):
1. **Real-Only:** Trained strictly on 1,408 internal real training records (64 records per crop).
2. **Real + Synthetic Mode A (Pure Empirical Copula):** Real training data combined with synthetic samples generated via a class-conditional Empirical Gaussian Rank Copula, preserving nonlinear cross-feature correlations without parametric assumptions.
3. **Real + Synthetic Mode B (Agronomically Constrained Copula):** Mode A samples subjected to institutional agronomic and physiological bounding rules (derived from Tamil Nadu Agricultural University and ICAR crop guides).
4. **Real + Synthetic Mode C (Empirical Quantile Envelopes):** Synthetic samples generated within empirical 5th to 95th percentile envelopes derived per crop from the internal training data.

---

## 5. Explainable AI
In agricultural machine learning, farmers and agronomists need to understand *why* a model makes a specific crop recommendation.

- **TreeSHAP for Random Forest and XGBoost:**  
  SHAP (SHapley Additive exPlanations) computes cooperative game-theoretic feature contributions. For Random Forest, TreeSHAP is evaluated in **probability space**; for XGBoost, it is evaluated in **raw margin space**. In both cases, exact local additivity is audited:
  $$\sum_{j=1}^M \phi_j(x) + E[f(x)] = f(x)$$
  The numerical residual between sum of SHAP values plus base value and model prediction is verified to be $\le 10^{-5}$.
- **TabNet Attention Feature Saliency:**  
  TabNet uses internal sparse attention masks ($M_b$) at each decision step. These masks indicate which features the neural network focused on for each sample. **These attention masks are strictly identified as attention-based feature saliency weights, not cooperative game-theoretic Shapley values.**
- **Correlation vs. Causation:** Feature importance identifies which measurements the model relied on to separate classes in this dataset; it does not prove that changing a feature in real soil will cause higher yield.

---

## 6. Project Files
The clean final deliverable folder contains **only three files**:

| File Name | Purpose |
|---|---|
| `crop_recommendation_modified.ipynb` | **The Final Modified Notebook:** Streamlined, clean, modular Google Colab research pipeline implementing the 8 core research phases with step-by-step documentation. |
| `crop_recommendation_updated.ipynb` | **The Final Updated Notebook:** Comprehensive, high-throughput academic edition featuring expanded theory, automated 10-point dataset validation, memorization audits, and master archive export. |
| `README.md` | **Documentation:** This comprehensive guide explaining project concepts, execution instructions, metrics, limitations, and troubleshooting. |

---

## 7. Requirements

### Standard Python Libraries (Built-in)
`os`, `sys`, `json`, `time`, `random`, `pathlib`, `shutil`, `urllib`, `importlib`, `subprocess`

### External Third-Party Packages
| Package | Version Tested | Purpose |
|---|---|---|
| `numpy` | `>= 1.24, < 3.0` | Numerical matrix and vector operations |
| `pandas` | `>= 2.0, < 4.0` | Tabular data manipulation and storage |
| `scipy` | `>= 1.10` | Rank statistics and Gaussian copula transformations |
| `scikit-learn` | `>= 1.2` | Data splitting, metrics, and Random Forest classifier |
| `pytorch-tabnet`| `>= 4.0` | TabNet attentive tabular deep neural network |
| `torch` | `>= 2.0` | PyTorch backend for TabNet (CPU or CUDA) |
| `xgboost` | `>= 1.7` | Extreme gradient boosting classifier |
| `shap` | `>= 0.42` | TreeSHAP exact attribution explainer |
| `matplotlib` | `>= 3.7` | Plotting and visualization |
| `seaborn` | `>= 0.12` | Statistical data plotting |

---

## 8. How to Run the Project in Google Colab

Follow these simple step-by-step instructions:

1. **Open Google Colab:**  
   Navigate to [colab.research.google.com](https://colab.research.google.com/) in your web browser.
2. **Upload the Notebook:**  
   Click `File` > `Upload notebook`, then select either:
   - `crop_recommendation_modified.ipynb` (recommended for standard demonstration, ~5-10 minutes).
   - `crop_recommendation_updated.ipynb` (recommended for full academic paper reproduction).
3. **Select Runtime:**  
   Click `Runtime` > `Change runtime type`. Standard CPU works well; GPU (T4) accelerates TabNet training.
4. **Execute Cell 1 (Dependencies):**  
   Run the first code cell to install and verify all necessary packages.
5. **Execute Cell 2 & 3 (Data Loading & Splitting):**  
   The notebook will automatically download `Crop_recommendation.csv` from GitHub raw or you can upload your own copy to `/content/`. The notebook partitions the data into 1,408 internal training, 352 internal validation, and 440 benchmark rows.
6. **Run Remaining Cells in Sequence:**  
   Click `Runtime` > `Run after` or press `Shift + Enter` on each cell in order.
7. **Inspect Outputs:**  
   View generated metric tables, confusion matrices, distribution-shift retention plots, and TreeSHAP waterfall charts directly in the notebook.

---

## 9. Dataset Information
- **Canonical Dataset:** Kaggle Crop Recommendation Dataset (`Crop_recommendation.csv`).
- **Total Records:** 2,200 rows across 22 crops (exactly 100 rows per crop).
- **Schema:** 7 continuous numerical features (`N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`) and 1 string label (`crop`).
- **Are Datasets Bundled?** No. To keep this deliverable clean and lightweight, CSV datasets are not bundled in this folder. The notebooks automatically download `Crop_recommendation.csv` at runtime from the GitHub repository.

---

## 10. Data Splitting and Leakage Prevention
To guarantee complete scientific integrity and prevent data snooping:
- **`INTERNAL_TRAIN.csv` (1,408 rows, 64 per crop):** The **only** dataset used to fit models, train scalers, and estimate copula distributions.
- **`INTERNAL_VAL.csv` (352 rows, 16 per crop):** Quarantined exclusively to evaluate models after training. It is never seen during model fitting or synthetic generation.
- **`REAL_TEST.csv` (440 rows, 20 per crop):** Evaluated strictly as a **frozen historical benchmark**. It has been queried during earlier research and is therefore disclosed as a baseline comparison set, not an untouched prospective field test.

---

## 11. Evaluation Metrics
- **Accuracy:** Proportion of total predictions that were exactly correct.
- **Balanced Accuracy:** Average recall across all 22 classes, preventing dominant classes from masking minority errors.
- **Macro Precision:** Unweighted average of precision across all classes (how many predicted instances of a crop were truly that crop).
- **Macro Recall:** Unweighted average of recall across all classes (how many true instances of a crop were correctly detected).
- **Macro F1-Score:** Harmonic mean of macro precision and macro recall.
- **Matthews Correlation Coefficient (MCC):** A balanced correlation metric between observed and predicted multiclass classifications (ranges from -1 to +1).
- **Brier Score:** Mean squared difference between predicted probability vectors and one-hot true class vectors (lower is better calibrated).
- **Expected Calibration Error (ECE):** Weighted absolute difference between model confidence and actual accuracy across probability bins.

> **Evaluation Disclosures:** High accuracy on this curated benchmark does not prove identical performance across diverse real-world farms, unmeasured microclimates, or varying soil conditions.

---

## 12. Results
When executed in Google Colab, results will be printed and exported to `./crop_rec_colab_workspace/results/`:
- `VALIDATION_METRICS.csv`: Performance on the 352 quarantined validation records.
- `BENCHMARK_METRICS.csv`: Performance on the 440 benchmark records.
- `ROBUSTNESS_ADVANTAGE_TIDY.csv`: Retention under 5 distribution shift scenarios.
- `CALIBRATION_METRICS.csv`: Brier scores and ECE values.
- `TREESHAP_AUDIT.csv`: Exact additivity residuals verifying explainability math.

Readers running the notebook will generate fresh, verified metrics directly in their session.

---

## 13. Limitations
1. **Synthetic Data is Not Real Soil:** Synthetic generation expands coverage of covariance space, but cannot invent unobserved biological phenomena.
2. **Benchmark Familiarity:** The historical benchmark dataset was evaluated during iterative research.
3. **No External Farm Data:** All records originate from the single historical 2,200-row repository. Real-world agricultural deployment requires prospective multi-farm field testing across distinct geographic regions.
4. **Agronomic Advice:** Recommendations from this model must be validated by agricultural extension officers or certified agronomists before planting.

---

## 14. Reproducibility
- **Global Seed:** Fixed to `42` across Python `random`, NumPy `RandomState`, and PyTorch `manual_seed`.
- **Deterministic Copula:** Sampling with a fixed seed produces identical synthetic numbers down to floating-point precision.
- **Manifests:** Generated models and datasets record sample counts and cryptographic SHA-256 hashes.

---

## 15. Troubleshooting
| Issue | Cause | Practical Fix |
|---|---|---|
| `FileNotFoundError: Crop_recommendation.csv` | Dataset download blocked by network policy | Upload `Crop_recommendation.csv` manually to `/content/` in Colab. |
| `OutOfMemoryError` on TabNet | Large batch size on low-memory runtime | Reduce `batch_size` in the training cell from 1024 to 256. |
| TabNet Training Takes Too Long | Running on CPU with 165k synthetic rows | In Colab, go to `Runtime` > `Change runtime type` and select `T4 GPU`. |
| `ImportError: pytorch_tabnet` | Dependency installation cell was skipped | Run Cell 1 (`pip install pytorch-tabnet xgboost shap`) before running model cells. |
| TreeSHAP Additivity Warning | Different output margin space | TreeSHAP for XGBoost is computed in `raw_margin` space; for RF in `probability` space. |

---

## 16. Research Disclaimer
This software, notebooks, and models are published solely for academic research, education, and algorithmic reproducibility. They are not certified for commercial farming, financial lending, or automated agricultural decision-making without qualified expert supervision.
