"""
RiskScope AI - Data Preprocessing Pipeline
============================================
Author: Data Preprocessing Lead
Purpose: Generate, clean, validate, and export the final training.csv
         that is fully compatible with train.py and feature_engineering.py.

Usage:
    python preprocess.py                          # default 50,000 samples
    python preprocess.py --samples 100000         # custom sample count
    python preprocess.py --input existing.csv     # preprocess an existing CSV
"""

import numpy as np
import pandas as pd
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    VITAL_FEATURES, SYMPTOM_FEATURES, DEMOGRAPHIC_FEATURES,
    VITAL_NORMAL_RANGES, CRITICAL_THRESHOLDS, ESI_LEVELS
)
from generate_synthetic_data import generate_synthetic_dataset


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  PHYSIOLOGICAL LIMITS  â€“ used to clip impossible outliers
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
PHYSIOLOGICAL_LIMITS = {
    'heart_rate':       (20, 250),
    'bp_systolic':      (40, 260),
    'bp_diastolic':     (20, 160),
    'spo2':             (50, 100),
    'temperature':      (30.0, 43.0),
    'respiratory_rate': (4, 60),
}

# Expected columns in the final training.csv  (30 features + 1 target)
REQUIRED_COLUMNS = (
    DEMOGRAPHIC_FEATURES       # age, gender
    + VITAL_FEATURES           # 6 vitals
    + SYMPTOM_FEATURES         # 21 symptom flags
    + ['symptom_duration_hours']
)
TARGET_COLUMN = 'esi_level'


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  STEP 1 â€“ GENERATE OR LOAD RAW DATA
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def load_or_generate(input_path: str = None, n_samples: int = 50_000) -> pd.DataFrame:
    """Load an existing CSV or generate fresh synthetic data."""

    if input_path and os.path.isfile(input_path):
        print(f"  ðŸ“‚ Loading existing data from {input_path}")
        df = pd.read_csv(input_path)
        print(f"     Loaded {len(df):,} rows Ã— {len(df.columns)} columns")
        return df

    # Fresh generation via teammate's script
    print(f"  ðŸ”§ Generating {n_samples:,} synthetic patients â€¦")
    X, y = generate_synthetic_dataset(n_samples=n_samples)
    X[TARGET_COLUMN] = y
    print(f"     Generated {len(X):,} rows Ã— {len(X.columns)} columns")
    return X


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  STEP 2 â€“ CLEAN THE DATA
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, clip outliers, fix types."""

    df = df.copy()

    # -- 2a. Missing values -------------------------------------------------
    missing_total = df.isna().sum().sum()
    if missing_total > 0:
        # Fill symptom flags with 0 (not present)
        for col in SYMPTOM_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)
        # Fill vitals with column median
        for col in VITAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        # Fill demographics
        if 'age' in df.columns:
            df['age'] = df['age'].fillna(df['age'].median()).astype(int)
        if 'gender' in df.columns:
            df['gender'] = df['gender'].fillna(0).astype(int)
        if 'symptom_duration_hours' in df.columns:
            df['symptom_duration_hours'] = df['symptom_duration_hours'].fillna(
                df['symptom_duration_hours'].median()
            )
        print(f"  ðŸ”§ Handled {missing_total:,} missing values")
    else:
        print(f"  âœ… No missing values found")

    # -- 2b. Clip vital-sign outliers to physiological range ----------------
    outliers_clipped = 0
    for vital, (lo, hi) in PHYSIOLOGICAL_LIMITS.items():
        if vital in df.columns:
            mask = (df[vital] < lo) | (df[vital] > hi)
            outliers_clipped += mask.sum()
            df[vital] = np.clip(df[vital], lo, hi)
    print(f"  ðŸ”§ Clipped {outliers_clipped:,} outlier vital-sign values")

    # -- 2c. Ensure symptom flags are strictly 0 or 1 ----------------------
    for col in SYMPTOM_FEATURES:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 1 if x else 0).astype(int)

    # -- 2d. Remove rows with invalid / missing ESI level -------------------
    before = len(df)
    df = df[df[TARGET_COLUMN].isin([1, 2, 3, 4, 5])].copy()
    removed = before - len(df)
    if removed > 0:
        print(f"  ðŸ”§ Removed {removed:,} rows with invalid ESI levels")
    else:
        print(f"  âœ… All ESI levels valid (1-5)")

    # -- 2e. Remove exact duplicate rows ------------------------------------
    dupes = df.duplicated().sum()
    if dupes > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"  ðŸ”§ Removed {dupes:,} duplicate rows")
    else:
        print(f"  âœ… No duplicate rows")

    return df


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  STEP 3 â€“ VALIDATE SCHEMA (ensure all required columns exist)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure every column train.py and feature_engineering.py expect exists."""

    df = df.copy()

    # Add any missing symptom columns as 0
    missing_symptoms = [s for s in SYMPTOM_FEATURES if s not in df.columns]
    for col in missing_symptoms:
        df[col] = 0
    if missing_symptoms:
        print(f"  ðŸ”§ Added {len(missing_symptoms)} missing symptom columns: {missing_symptoms}")

    # Add symptom_duration_hours if missing
    if 'symptom_duration_hours' not in df.columns:
        df['symptom_duration_hours'] = np.random.exponential(12, size=len(df))
        print(f"  ðŸ”§ Added missing symptom_duration_hours column")

    # Verify all required columns
    all_needed = REQUIRED_COLUMNS + [TARGET_COLUMN]
    present = [c for c in all_needed if c in df.columns]
    missing = [c for c in all_needed if c not in df.columns]

    print(f"  âœ… {len(present)}/{len(all_needed)} required columns present")
    if missing:
        print(f"  âŒ MISSING columns: {missing}")
    else:
        print(f"  âœ… Schema validation passed â€” fully compatible with train.py")

    return df


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  STEP 4 â€“ ENRICH DATA (add clinically-derived features)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def enrich_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns that boost model performance."""

    df = df.copy()

    # -- 4a. Symptom count --------------------------------------------------
    symptom_cols = [c for c in SYMPTOM_FEATURES if c in df.columns]
    df['symptom_count'] = df[symptom_cols].sum(axis=1)
    print(f"  âž• symptom_count  (range: {df['symptom_count'].min()}-{df['symptom_count'].max()})")

    # -- 4b. Vital abnormality count ----------------------------------------
    abnormal_count = pd.Series(0, index=df.index)
    for vital, ranges in VITAL_NORMAL_RANGES.items():
        if vital in df.columns:
            lo, hi = ranges['low'], ranges['high']
            abnormal_count += ((df[vital] < lo) | (df[vital] > hi)).astype(int)
    df['vital_abnormality_count'] = abnormal_count
    print(f"  âž• vital_abnormality_count  (range: {abnormal_count.min()}-{abnormal_count.max()})")

    # -- 4c. Critical flag (any critical threshold breached?) ---------------
    critical = pd.Series(0, index=df.index)
    if 'spo2' in df.columns:
        critical |= (df['spo2'] < CRITICAL_THRESHOLDS['spo2_critical']).astype(int)
    if 'bp_systolic' in df.columns:
        critical |= (df['bp_systolic'] < CRITICAL_THRESHOLDS['bp_systolic_low']).astype(int)
        critical |= (df['bp_systolic'] > CRITICAL_THRESHOLDS['bp_systolic_high']).astype(int)
    if 'heart_rate' in df.columns:
        critical |= (df['heart_rate'] < CRITICAL_THRESHOLDS['heart_rate_low']).astype(int)
        critical |= (df['heart_rate'] > CRITICAL_THRESHOLDS['heart_rate_high']).astype(int)
    if 'temperature' in df.columns:
        critical |= (df['temperature'] > CRITICAL_THRESHOLDS['temperature_high']).astype(int)
    if 'respiratory_rate' in df.columns:
        critical |= (df['respiratory_rate'] > CRITICAL_THRESHOLDS['respiratory_rate_high']).astype(int)
    df['has_critical_vital'] = critical
    print(f"  âž• has_critical_vital  ({critical.sum():,} patients flagged)")

    # -- 4d. Age bucket (encode clinical risk tiers) -----------------------
    df['age_bucket'] = pd.cut(
        df['age'],
        bins=[0, 18, 40, 65, 120],
        labels=[0, 1, 2, 3]  # pediatric, young-adult, middle-age, elderly
    ).astype(int)
    print(f"  âž• age_bucket  (0=pediatric, 1=young, 2=middle, 3=elderly)")

    # -- 4e. MAP (Mean Arterial Pressure) -----------------------------------
    if 'bp_systolic' in df.columns and 'bp_diastolic' in df.columns:
        df['map_pressure'] = (
            df['bp_diastolic'] + (df['bp_systolic'] - df['bp_diastolic']) / 3
        ).round(1)
        print(f"  âž• map_pressure  (range: {df['map_pressure'].min():.1f}-{df['map_pressure'].max():.1f})")

    # -- 4f. Shock index (HR / SBP) â€“ key predictor of hemodynamic instability
    if 'heart_rate' in df.columns and 'bp_systolic' in df.columns:
        df['shock_index'] = (df['heart_rate'] / df['bp_systolic'].replace(0, 1)).round(3)
        print(f"  âž• shock_index  (range: {df['shock_index'].min():.2f}-{df['shock_index'].max():.2f})")

    return df


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  STEP 5 â€“ PRINT FULL REPORT
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def print_report(df: pd.DataFrame):
    """Print a comprehensive dataset summary."""

    total = len(df)

    print("\n  ðŸ“Š CLASS DISTRIBUTION")
    for esi in range(1, 6):
        count = (df[TARGET_COLUMN] == esi).sum()
        pct = count / total * 100
        bar = "â–ˆ" * int(pct)
        label = ESI_LEVELS[esi]
        print(f"     ESI {esi}: {count:>7,} ({pct:>5.1f}%)  {bar}  {label}")

    print(f"\n  ðŸ“ DATASET SIZE")
    print(f"     Rows:    {total:>10,}")
    print(f"     Columns: {len(df.columns):>10}")

    print(f"\n  ðŸ¥ VITAL SIGN RANGES")
    for vital in VITAL_FEATURES:
        if vital in df.columns:
            vmin, vmax, vmean = df[vital].min(), df[vital].max(), df[vital].mean()
            print(f"     {vital:<20s}  min={vmin:>6.1f}  max={vmax:>6.1f}  mean={vmean:>6.1f}")

    print(f"\n  ðŸ©º SYMPTOM PREVALENCE (top 10)")
    symptom_prev = {}
    for sym in SYMPTOM_FEATURES:
        if sym in df.columns:
            symptom_prev[sym] = df[sym].sum()
    for sym, count in sorted(symptom_prev.items(), key=lambda x: -x[1])[:10]:
        pct = count / total * 100
        print(f"     {sym:<25s}  {count:>6,} ({pct:>5.1f}%)")

    remaining_missing = df.isna().sum().sum()
    print(f"\n  ðŸ” FINAL QUALITY CHECK")
    print(f"     Remaining NaN values: {remaining_missing}")
    print(f"     Duplicate rows:       {df.duplicated().sum()}")
    print(f"     {'âœ… DATASET IS CLEAN AND READY' if remaining_missing == 0 else 'âŒ STILL HAS ISSUES'}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  MAIN PIPELINE
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def run_pipeline(input_path: str = None,
                 output_path: str = "data/training.csv",
                 n_samples: int = 50_000):
    """Execute the full preprocessing pipeline."""

    print("=" * 64)
    print("  RiskScope AI â€” Data Preprocessing Pipeline")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 64)

    # â”€â”€ Step 1 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[1/5] LOADING DATA")
    df = load_or_generate(input_path, n_samples)

    # â”€â”€ Step 2 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[2/5] CLEANING DATA")
    df = clean_data(df)

    # â”€â”€ Step 3 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[3/5] VALIDATING SCHEMA")
    df = validate_schema(df)

    # â”€â”€ Step 4 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[4/5] ENRICHING DATA")
    df = enrich_data(df)

    # â”€â”€ Step 5 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[5/5] FINAL REPORT")
    print_report(df)

    # â”€â”€ Save â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 64)
    print(f"  âœ… Saved {len(df):,} rows â†’ {output_path}")
    print(f"  ðŸ“¦ File size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
    print(f"  ðŸš€ Next step:  python train.py --data {output_path} --output models/")
    print("=" * 64)

    return df


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  CLI ENTRY POINT
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RiskScope AI â€” Data Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python preprocess.py                              # generate 50K synthetic + preprocess
  python preprocess.py --samples 100000             # generate 100K samples
  python preprocess.py --input raw_data.csv         # preprocess an existing CSV
  python preprocess.py --output data/training.csv   # custom output path
        """
    )
    parser.add_argument("--input",   type=str, default=None,
                        help="Path to existing raw CSV (skips generation)")
    parser.add_argument("--output",  type=str, default="data/training.csv",
                        help="Output path for cleaned training CSV")
    parser.add_argument("--samples", type=int, default=50000,
                        help="Number of synthetic samples to generate (default: 50000)")

    args = parser.parse_args()
    run_pipeline(args.input, args.output, args.samples)

