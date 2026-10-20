"""
Module: synthetic_generation.py
Phase 2: Reproducible, Leak-Free High-Throughput Synthetic Agricultural Data Generation.

SCIENTIFIC GUARANTEE:
Generator parameters (Empirical Gaussian Rank Copulas and Empirical Envelopes)
are fitted EXCLUSIVELY on data/INTERNAL_TRAIN.csv (1,408 rows, 64/crop).
Neither data/INTERNAL_VAL.csv nor data/REAL_TEST.csv is ever seen or used.
"""

import sys
import os
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import scipy.stats as stats

curr_dir = Path(__file__).resolve().parent
root_dir = curr_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config.project_config import (
    INTERNAL_TRAIN_FILE,
    DATA_DIR,
    CANONICAL_FEATURES,
    TARGET_COLUMN,
    EXPECTED_CROPS,
    INTERNAL_TRAIN_ROWS,
    SYNTHETIC_PER_CROP,
    SYNTHETIC_TOTAL_PER_MODE,
    RANDOM_SEED
)

GENERATED_V1_DIR = DATA_DIR / "generated_v1"

# Universal Physical Thermodynamic / Soil Laws
UNIVERSAL_PHYSICAL_RULES = {
    "N": {"hard_min": 0.0, "hard_max": 300.0, "note": "Soil nitrogen concentration (kg/ha)."},
    "P": {"hard_min": 0.0, "hard_max": 300.0, "note": "Soil phosphorus concentration (kg/ha)."},
    "K": {"hard_min": 0.0, "hard_max": 300.0, "note": "Soil potassium concentration (kg/ha)."},
    "temperature": {"hard_min": -10.0, "hard_max": 60.0, "note": "Terrestrial ambient air temperature (°C)."},
    "humidity": {"hard_min": 0.0, "hard_max": 100.0, "note": "Relative humidity percentage (%)."},
    "ph": {"hard_min": 0.0, "hard_max": 14.0, "note": "Aqueous soil chemical pH scale."},
    "rainfall": {"hard_min": 0.0, "hard_max": 1500.0, "note": "Precipitation volume (mm)."}
}

# Source-Traceable Institutional Agronomic Rules (TNAU / ICAR)
# Documented from Tamil Nadu Agricultural University Agritech Portal & ICAR Crop Guidelines
AGRICULTURAL_RULES_REGISTRY = {
    "apple": {
        "ph": (5.5, 6.5),
        "temperature": (5.0, 28.0),
        "source": "TNAU Commercial Fruit / Apple Guidance"
    },
    "banana": {
        "temperature": (10.0, 40.0),
        "rainfall": (100.0, 350.0),
        "source": "TNAU Banana Expert System"
    },
    "blackgram": {
        "ph": (6.0, 8.5),
        "temperature": (20.0, 40.0),
        "source": "ICAR/TNAU Blackgram Guidance"
    },
    "chickpea": {
        "temperature": (10.0, 35.0),
        "humidity": (10.0, 65.0),
        "ph": (5.5, 8.5),
        "source": "ICAR Pulse Production Guidance"
    },
    "coconut": {
        "temperature": (20.0, 38.0),
        "humidity": (60.0, 100.0),
        "ph": (5.0, 8.0),
        "source": "TNAU/NABARD Coconut Technical Guidance"
    },
    "coffee": {
        "ph": (4.5, 6.5),
        "temperature": (15.0, 32.0),
        "humidity": (50.0, 95.0),
        "source": "TNAU Coffee Production Guidance"
    },
    "cotton": {
        "ph": (5.5, 8.5),
        "temperature": (18.0, 40.0),
        "source": "ICAR AICRP Agrometeorology Cotton Guidance"
    },
    "grapes": {
        "temperature": (12.0, 40.0),
        "ph": (5.5, 7.5),
        "source": "TNAU Grape Cultivation Guidance"
    },
    "jute": {
        "humidity": (65.0, 100.0),
        "temperature": (20.0, 42.0),
        "ph": (5.5, 8.5),
        "source": "TNAU Jute Crop Production Guidance"
    },
    "kidneybeans": {
        "temperature": (12.0, 32.0),
        "ph": (5.0, 7.5),
        "source": "ICAR Pulse Production Guidelines"
    },
    "lentil": {
        "temperature": (10.0, 32.0),
        "ph": (5.5, 8.2),
        "source": "ICAR Pulse Production Guidelines"
    },
    "maize": {
        "temperature": (15.0, 35.0),
        "ph": (5.0, 8.0),
        "source": "ICAR Agronomy Compendium"
    },
    "mango": {
        "ph": (5.5, 8.0),
        "temperature": (20.0, 42.0),
        "source": "TNAU Mango Cultivation Guidance"
    },
    "mothbeans": {
        "temperature": (20.0, 42.0),
        "ph": (5.0, 8.5),
        "source": "ICAR Arid-Region Crop Guidance (CAZRI)"
    },
    "mungbean": {
        "temperature": (20.0, 40.0),
        "ph": (5.5, 8.0),
        "source": "ICAR Crop/Weather Guidance"
    },
    "muskmelon": {
        "ph": (6.0, 7.5),
        "temperature": (20.0, 40.0),
        "source": "TNAU Muskmelon Guidance"
    },
    "orange": {
        "ph": (5.5, 7.5),
        "temperature": (10.0, 38.0),
        "source": "TNAU Sweet Orange Guidance"
    },
    "papaya": {
        "temperature": (18.0, 38.0),
        "ph": (5.5, 7.5),
        "source": "ICAR-MGIFRI Papaya Cultivation Guidance"
    },
    "pigeonpeas": {
        "ph": (5.5, 7.5),
        "temperature": (18.0, 38.0),
        "source": "ICAR AICRP Agrometeorology Pigeonpea Guidance"
    },
    "pomegranate": {
        "temperature": (15.0, 42.0),
        "ph": (5.5, 8.5),
        "source": "TNAU Tropical Fruit Production Guidance"
    },
    "rice": {
        "temperature": (15.0, 40.0),
        "humidity": (50.0, 100.0),
        "ph": (4.5, 8.5),
        "source": "TNAU Rice Expert System"
    },
    "watermelon": {
        "temperature": (20.0, 35.0),
        "ph": (5.5, 7.5),
        "source": "TNAU Watermelon Guidance"
    }
}

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

class EmpiricalGaussianRankCopula:
    """Class-conditional non-parametric empirical copula fitted on internal training data."""
    def __init__(self, crop: str, train_df: pd.DataFrame):
        self.crop = crop
        self.features = CANONICAL_FEATURES
        self.d = len(self.features)
        self.n_samples = len(train_df)

        self.constant_features: Dict[str, float] = {}
        self.varying_features: List[str] = []
        self.marginal_data: Dict[str, np.ndarray] = {}
        self.marginal_iqr: Dict[str, float] = {}

        for feat in self.features:
            vals = train_df[feat].values.astype(float)
            if np.isclose(np.var(vals), 0.0) or np.all(vals == vals[0]):
                self.constant_features[feat] = float(vals[0])
            else:
                self.varying_features.append(feat)
                sorted_vals = np.sort(vals)
                self.marginal_data[feat] = sorted_vals
                iqr = float(np.percentile(sorted_vals, 75) - np.percentile(sorted_vals, 25))
                self.marginal_iqr[feat] = max(1e-4, iqr)

        n_var = len(self.varying_features)
        if n_var == 0:
            raise RuntimeError(f"All features for crop '{crop}' have zero variance.")

        # Transform empirical marginals to standard normal latent space via empirical rank CDF
        z_matrix = np.zeros((self.n_samples, n_var))
        for j, feat in enumerate(self.varying_features):
            raw_vals = train_df[feat].values.astype(float)
            ranks = stats.rankdata(raw_vals, method="average")
            u = ranks / (self.n_samples + 1.0)
            z_matrix[:, j] = stats.norm.ppf(u)

        if n_var == 1:
            self.cholesky_l = np.array([[1.0]])
        else:
            cov_z = np.corrcoef(z_matrix, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov_z)
            eigvals = np.maximum(eigvals, 1e-6)
            clean_cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
            diag_sqrt = np.sqrt(np.diag(clean_cov))
            corr_matrix = clean_cov / np.outer(diag_sqrt, diag_sqrt)
            try:
                self.cholesky_l = np.linalg.cholesky(corr_matrix)
            except np.linalg.LinAlgError:
                self.cholesky_l = np.linalg.cholesky(corr_matrix + 1e-5 * np.eye(n_var))

    def sample(self, n: int, rng: np.random.RandomState, apply_micro_jitter: bool = True) -> np.ndarray:
        n_var = len(self.varying_features)
        candidates = np.zeros((n, self.d))

        if n_var > 0:
            z_ind = rng.normal(size=(n, n_var))
            z_corr = z_ind @ self.cholesky_l.T
            u_samples = stats.norm.cdf(z_corr)

            for j, feat in enumerate(self.varying_features):
                sorted_m = self.marginal_data[feat]
                u_col = np.clip(u_samples[:, j], 1e-6, 1.0 - 1e-6)

                # Piecewise linear quantile interpolation
                q_idx = u_col * (len(sorted_m) - 1)
                lower_idx = np.floor(q_idx).astype(int)
                upper_idx = np.ceil(q_idx).astype(int)
                frac = q_idx - lower_idx
                interp_vals = sorted_m[lower_idx] + frac * (sorted_m[upper_idx] - sorted_m[lower_idx])

                if apply_micro_jitter:
                    jitter_sigma = 0.002 * self.marginal_iqr[feat]
                    interp_vals += rng.normal(0.0, jitter_sigma, size=n)

                u_rule = UNIVERSAL_PHYSICAL_RULES.get(feat, {})
                interp_vals = np.clip(
                    interp_vals,
                    u_rule.get("hard_min", -np.inf),
                    u_rule.get("hard_max", np.inf)
                )

                feat_idx = self.features.index(feat)
                candidates[:, feat_idx] = interp_vals

        for feat, const_val in self.constant_features.items():
            feat_idx = self.features.index(feat)
            candidates[:, feat_idx] = const_val

        return candidates

def compute_empirical_envelopes(train_df: pd.DataFrame) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Computes 1st and 99th empirical percentiles strictly from internal training data."""
    envelopes = {}
    for crop, group in train_df.groupby(TARGET_COLUMN):
        envelopes[crop] = {}
        for feat in CANONICAL_FEATURES:
            vals = group[feat].values.astype(float)
            envelopes[crop][feat] = (float(np.percentile(vals, 1.0)), float(np.percentile(vals, 99.0)))
    return envelopes

def generate_reproducible_synthetic_datasets(
    train_file: Path = INTERNAL_TRAIN_FILE,
    output_dir: Path = GENERATED_V1_DIR,
    quota_per_crop: int = SYNTHETIC_PER_CROP,
    seed: int = RANDOM_SEED
) -> Dict[str, Any]:
    """Generates Modes A, B, and C strictly fitted on internal training data."""
    t0 = time.time()
    print("=" * 80)
    print("PHASE 2: LEAK-FREE SYNTHETIC AGRICULTURAL DATA GENERATION")
    print("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Strict Data Quarantine Validation
    assert train_file.exists(), f"Internal train file not found at {train_file}"
    df_train = pd.read_csv(train_file)
    assert len(df_train) == INTERNAL_TRAIN_ROWS, f"Expected {INTERNAL_TRAIN_ROWS} rows, got {len(df_train)}"
    
    crops_in_train = sorted(df_train[TARGET_COLUMN].unique())
    assert crops_in_train == EXPECTED_CROPS, "Training crops do not match canonical 22 crops"

    input_hash = compute_sha256(train_file)
    print(f"[+] Loaded strictly quarantined training set: {train_file.name}")
    print(f"    - Rows: {len(df_train)} (64/crop across 22 crops)")
    print(f"    - SHA-256: {input_hash}")

    # 2. Fit Copulas strictly on internal training records
    print("\n[*] Fitting Empirical Gaussian Rank Copulas across 22 crops...")
    copulas = {c: EmpiricalGaussianRankCopula(c, df_train[df_train[TARGET_COLUMN] == c]) for c in EXPECTED_CROPS}
    mode_c_envelopes = compute_empirical_envelopes(df_train)
    print("[+] All 22 crop copulas and Mode C empirical envelopes successfully fitted on INTERNAL_TRAIN.")

    feat_to_idx = {f: i for i, f in enumerate(CANONICAL_FEATURES)}
    modes = ["Mode_A", "Mode_B", "Mode_C"]
    generated_files = {}

    for mode in modes:
        mode_t0 = time.time()
        print(f"\n[*] Generating {mode} ({quota_per_crop:,} samples/crop -> {quota_per_crop*22:,} total)...")
        rng = np.random.RandomState(seed + (11 if mode == "Mode_A" else 22 if mode == "Mode_B" else 33))
        
        mode_records = []
        for crop in EXPECTED_CROPS:
            copula = copulas[crop]
            collected = []
            n_collected = 0
            attempts = 0
            max_attempts = quota_per_crop * 100

            # Establish bounds
            bounds_low = np.full(len(CANONICAL_FEATURES), -np.inf)
            bounds_high = np.full(len(CANONICAL_FEATURES), np.inf)

            # Universal limits
            for f in CANONICAL_FEATURES:
                u_rule = UNIVERSAL_PHYSICAL_RULES.get(f, {})
                idx = feat_to_idx[f]
                bounds_low[idx] = max(bounds_low[idx], u_rule.get("hard_min", -np.inf))
                bounds_high[idx] = min(bounds_high[idx], u_rule.get("hard_max", np.inf))

            # Mode B & C: TNAU/ICAR Agronomic Constraints
            if mode in ("Mode_B", "Mode_C"):
                agri_rules = AGRICULTURAL_RULES_REGISTRY.get(crop, {})
                for f, rule_range in agri_rules.items():
                    if f in feat_to_idx and isinstance(rule_range, tuple):
                        idx = feat_to_idx[f]
                        bounds_low[idx] = max(bounds_low[idx], rule_range[0])
                        bounds_high[idx] = min(bounds_high[idx], rule_range[1])

            # Mode C: Empirical 1st-99th percentile envelopes
            if mode == "Mode_C":
                c_env = mode_c_envelopes.get(crop, {})
                for f, (q01, q99) in c_env.items():
                    idx = feat_to_idx[f]
                    bounds_low[idx] = max(bounds_low[idx], q01)
                    bounds_high[idx] = min(bounds_high[idx], q99)

            while n_collected < quota_per_crop and attempts < max_attempts:
                needed = quota_per_crop - n_collected
                batch_size = max(1000, needed * 2)
                raw_cands = copula.sample(batch_size, rng, apply_micro_jitter=True)
                attempts += batch_size

                valid_mask = np.all(
                    (raw_cands >= bounds_low) & (raw_cands <= bounds_high) & np.isfinite(raw_cands),
                    axis=1
                )
                valid_cands = raw_cands[valid_mask]
                if len(valid_cands) > 0:
                    take = min(needed, len(valid_cands))
                    collected.append(valid_cands[:take])
                    n_collected += take

            crop_samples = np.vstack(collected)
            crop_df = pd.DataFrame(crop_samples, columns=CANONICAL_FEATURES)
            crop_df[TARGET_COLUMN] = crop
            mode_records.append(crop_df)

        mode_df = pd.concat(mode_records, ignore_index=True)
        assert len(mode_df) == SYNTHETIC_TOTAL_PER_MODE, f"Expected {SYNTHETIC_TOTAL_PER_MODE}, got {len(mode_df)}"
        assert mode_df[TARGET_COLUMN].value_counts().min() == quota_per_crop
        assert not mode_df[CANONICAL_FEATURES].isnull().values.any()
        assert not np.isinf(mode_df[CANONICAL_FEATURES].values).any()

        mode_file = output_dir / f"SYNTHETIC_{mode}.csv"
        mode_df.to_csv(mode_file, index=False)
        sha256 = compute_sha256(mode_file)
        generated_files[mode] = {
            "file": mode_file.name,
            "rows": len(mode_df),
            "sha256": sha256,
            "duration_s": round(time.time() - mode_t0, 2)
        }
        print(f"[+] {mode} generated in {time.time()-mode_t0:.2f}s | SHA-256: {sha256[:16]}...")

    # 3. Create Consolidated Master Synthetic Dataset
    print("\n[*] Creating consolidated master synthetic dataset (495,000 rows)...")
    df_a = pd.read_csv(output_dir / "SYNTHETIC_Mode_A.csv")
    df_a["generation_mode"] = "Mode_A_Pure_Copula"
    df_a["generation_seed"] = seed

    df_b = pd.read_csv(output_dir / "SYNTHETIC_Mode_B.csv")
    df_b["generation_mode"] = "Mode_B_Agronomic_Constraints"
    df_b["generation_seed"] = seed

    df_c = pd.read_csv(output_dir / "SYNTHETIC_Mode_C.csv")
    df_c["generation_mode"] = "Mode_C_Empirical_Envelopes"
    df_c["generation_seed"] = seed

    master_df = pd.concat([df_a, df_b, df_c], ignore_index=True)
    master_file = output_dir / "SYNTHETIC_CROP_DATA_MASTER.csv"
    master_df.to_csv(master_file, index=False)
    master_sha256 = compute_sha256(master_file)
    print(f"[+] Master synthetic saved: {master_file.name} ({len(master_df):,} rows) | SHA-256: {master_sha256[:16]}...")

    manifest = {
        "generator_version": "2.0_leak_free",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "random_seed": seed,
        "source_training_file": train_file.name,
        "source_training_rows": len(df_train),
        "source_training_sha256": input_hash,
        "quarantined_files_excluded": ["INTERNAL_VAL.csv", "REAL_TEST.csv"],
        "modes_generated": generated_files,
        "master_synthetic_file": master_file.name,
        "master_synthetic_rows": len(master_df),
        "master_synthetic_sha256": master_sha256,
        "total_runtime_s": round(time.time() - t0, 2)
    }

    manifest_path = output_dir / "SYNTHETIC_GENERATION_MANIFEST.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[+] Generation manifest written to: {manifest_path}")

    return manifest

if __name__ == "__main__":
    generate_reproducible_synthetic_datasets()
