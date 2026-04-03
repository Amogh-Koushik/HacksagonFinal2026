"""
RiskScope AI — Synthea-to-Training Data Parser
================================================
Translates raw Synthea CSV exports into the exact column format
required by train.py and feature_engineering.py.

Synthea outputs:
  patients.csv, encounters.csv, observations.csv, conditions.csv
  (with LOINC codes for vitals, SNOMED codes for conditions)

This script outputs:
  A single CSV with 30 feature columns + esi_level target,
  identical in schema to what generate_synthetic_data.py produces.

Usage:
  python synthea_to_training.py --synthea-dir ../synthea/output/csv/
  python synthea_to_training.py --synthea-dir ../synthea/output/csv/ --output data/synthea_processed.csv
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from config import SYMPTOM_FEATURES, VITAL_FEATURES, CRITICAL_THRESHOLDS


# ═══════════════════════════════════════════════════════════════════════════════
#  LOINC CODE → VITAL SIGN MAPPING
#  These are the standard LOINC codes Synthea uses in observations.csv
# ═══════════════════════════════════════════════════════════════════════════════
LOINC_TO_VITAL = {
    # Heart Rate
    '8867-4':  'heart_rate',
    # Systolic Blood Pressure
    '8480-6':  'bp_systolic',
    # Diastolic Blood Pressure
    '8462-4':  'bp_diastolic',
    # Oxygen Saturation (SpO2)
    '2708-6':  'spo2',
    '59408-5': 'spo2',          # Pulse oximetry variant
    # Body Temperature
    '8310-5':  'temperature',
    # Respiratory Rate
    '9279-1':  'respiratory_rate',
}

# ═══════════════════════════════════════════════════════════════════════════════
#  SNOMED CODE → SYMPTOM FLAG MAPPING
#  Maps Synthea condition SNOMED codes to our 21 binary symptom columns
# ═══════════════════════════════════════════════════════════════════════════════
SNOMED_TO_SYMPTOM = {
    # ── Chest Pain / Cardiac ─────────────────────────────────────────────
    '29857009':  'chest_pain',          # Chest pain
    '413838009': 'chest_pain',          # Acute coronary syndrome
    '22298006':  'chest_pain',          # Myocardial infarction
    '53741008':  'chest_pain',          # Coronary heart disease
    '59621000':  'chest_pain',          # Essential hypertension (may present w/ chest pain)
    '10509002':  'chest_pain',          # Acute bronchitis (chest tightness)
    '427314002': 'chest_pain',          # Angina
    '194828000': 'chest_pain',          # Angina pectoris

    # ── Arm Pain Left / Jaw Pain (MI radiation) ──────────────────────────
    # These are less directly coded; we infer from MI diagnosis
    '22298006_arm':  'arm_pain_left',   # handled specially for MI cases
    '22298006_jaw':  'jaw_pain',        # handled specially for MI cases

    # ── Dyspnea / Shortness of Breath ────────────────────────────────────
    '267036007': 'dyspnea',             # Dyspnea
    '11833005':  'dyspnea',             # Dyspnea on exertion
    '49727002':  'shortness_of_breath', # Cough (often w/ SOB)
    '195662009': 'shortness_of_breath', # Acute viral pharyngitis (resp)
    '233604007': 'shortness_of_breath', # Pneumonia
    '13645005':  'shortness_of_breath', # COPD
    '195967001': 'dyspnea',            # Asthma
    '19829001':  'dyspnea',            # Disorder of lung

    # ── Stroke (FAST symptoms) ───────────────────────────────────────────
    '230690007': 'facial_droop',        # Stroke / CVA
    '230691006': 'facial_droop',        # TIA
    # For stroke, we set multiple flags in post-processing

    # ── Neurological ─────────────────────────────────────────────────────
    '91175000':  'seizure',             # Seizure disorder
    '84757009':  'seizure',             # Epilepsy
    '25064002':  'headache',            # Headache
    '37796009':  'headache',            # Migraine
    '230461009': 'headache',            # Cluster headache
    '271594007': 'syncope',             # Syncope
    '404640003': 'dizziness',           # Dizziness
    '40917007':  'confusion',           # Confusion / clouded consciousness
    '419219000': 'altered_mental_status', # AMS

    # ── Abdominal ────────────────────────────────────────────────────────
    '21522001':  'abdominal_pain',      # Abdominal pain
    '396275006': 'abdominal_pain',      # Osteoarthritis (sometimes coded)
    '65966004':  'abdominal_pain',      # Fracture (abdominal area)
    '43878008':  'rigid_abdomen',       # Peritonitis → rigid abdomen
    '197480006': 'abdominal_pain',      # Acute appendicitis
    '444814009': 'abdominal_pain',      # Viral sinusitis (mismap guard)
    '235595009': 'abdominal_pain',      # Gastroesophageal reflux

    # ── GI Symptoms ──────────────────────────────────────────────────────
    '422587007': 'nausea',              # Nausea
    '16932000':  'nausea',              # Nausea and vomiting
    '422400008': 'vomiting',            # Vomiting

    # ── Fever / Infection ────────────────────────────────────────────────
    '386661006': 'fever',               # Fever
    '36971009':  'fever',               # Sinusitis (often w/ fever)
    '444814009': 'fever',               # Viral sinusitis
    '195662009': 'fever',               # Acute viral pharyngitis
    '40055000':  'fever',               # Chronic sinusitis
    '43878008_f': 'fever',              # Peritonitis (fever)

    # ── Bleeding ─────────────────────────────────────────────────────────
    '110030002': 'uncontrolled_bleeding', # Hemorrhage
    '12063002':  'uncontrolled_bleeding', # Rectal hemorrhage
    '74474003':  'uncontrolled_bleeding', # GI hemorrhage

    # ── Pain ─────────────────────────────────────────────────────────────
    '279039007': 'severe_pain',         # Severe pain / low back pain
    '239873007': 'severe_pain',         # Osteoarthritis of knee (pain)
}

# SNOMED codes that are stroke-related → set multiple FAST flags
STROKE_SNOMED_CODES = {'230690007', '230691006'}

# SNOMED codes that are MI-related → set radiation flags
MI_SNOMED_CODES = {'22298006', '413838009'}


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1: LOAD SYNTHEA CSVs
# ═══════════════════════════════════════════════════════════════════════════════
def load_synthea_csvs(synthea_dir: str) -> dict:
    """Load the 4 key Synthea CSV files."""

    required_files = ['patients.csv', 'encounters.csv', 'observations.csv', 'conditions.csv']
    csvs = {}

    print(f"\n  📂 Loading Synthea CSVs from: {synthea_dir}")

    for fname in required_files:
        fpath = os.path.join(synthea_dir, fname)
        if not os.path.exists(fpath):
            print(f"  ❌ Missing: {fpath}")
            sys.exit(1)
        df = pd.read_csv(fpath)
        key = fname.replace('.csv', '')
        csvs[key] = df
        print(f"     {fname:<20s}  {len(df):>8,} rows × {len(df.columns)} cols")

    return csvs


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2: FILTER TO EMERGENCY ENCOUNTERS ONLY
# ═══════════════════════════════════════════════════════════════════════════════
def filter_emergency_encounters(csvs: dict) -> pd.DataFrame:
    """Keep only emergency and urgent-care encounters."""

    encounters = csvs['encounters'].copy()

    # Synthea uses ENCOUNTERCLASS column
    class_col = None
    for col_name in ['ENCOUNTERCLASS', 'encounterclass', 'EncounterClass']:
        if col_name in encounters.columns:
            class_col = col_name
            break

    if class_col is None:
        print("  ⚠️  No ENCOUNTERCLASS column found. Using all encounters.")
        return encounters

    # Filter for emergency / urgentcare / inpatient (they all go through triage)
    ed_mask = encounters[class_col].str.lower().isin(['emergency', 'urgentcare', 'inpatient'])
    ed_encounters = encounters[ed_mask].copy()

    print(f"\n  🏥 Encounters: {len(encounters):,} total → {len(ed_encounters):,} emergency/urgent/inpatient")

    if len(ed_encounters) == 0:
        print("  ⚠️  No emergency encounters found! Falling back to all encounters.")
        return encounters

    return ed_encounters


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3: EXTRACT DEMOGRAPHICS (age, gender)
# ═══════════════════════════════════════════════════════════════════════════════
def extract_demographics(csvs: dict, encounters: pd.DataFrame) -> pd.DataFrame:
    """Join patients table to get age and gender per encounter."""

    patients = csvs['patients'].copy()

    # Find the right column names (Synthea capitalizes them)
    id_col = next((c for c in patients.columns if c.upper() == 'ID'), patients.columns[0])
    birth_col = next((c for c in patients.columns if 'BIRTH' in c.upper()), None)
    gender_col = next((c for c in patients.columns if 'GENDER' in c.upper()), None)

    # Find encounter's patient reference column
    enc_patient_col = next((c for c in encounters.columns if 'PATIENT' in c.upper()), None)
    enc_start_col = next((c for c in encounters.columns if c.upper() == 'START'), None)
    enc_id_col = next((c for c in encounters.columns if c.upper() == 'ID'), encounters.columns[0])

    # Merge
    merged = encounters.merge(
        patients[[id_col, birth_col, gender_col]],
        left_on=enc_patient_col,
        right_on=id_col,
        how='left'
    )

    # Calculate age at encounter
    if birth_col and enc_start_col:
        merged[birth_col] = pd.to_datetime(merged[birth_col], errors='coerce')
        merged[enc_start_col] = pd.to_datetime(merged[enc_start_col], errors='coerce')
        merged['age'] = ((merged[enc_start_col] - merged[birth_col]).dt.days / 365.25).astype(int)
        merged['age'] = merged['age'].clip(0, 120)
    else:
        merged['age'] = 50  # fallback

    # Gender: 1 = male, 0 = female
    if gender_col:
        merged['gender'] = merged[gender_col].str.upper().map({'M': 1, 'F': 0}).fillna(0).astype(int)
    else:
        merged['gender'] = np.random.choice([0, 1], size=len(merged))

    # Keep encounter ID for joining
    merged['encounter_id'] = merged[enc_id_col]

    print(f"  👤 Demographics: age range {merged['age'].min()}-{merged['age'].max()}, "
          f"{(merged['gender']==1).sum()} male / {(merged['gender']==0).sum()} female")

    return merged


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 4: EXTRACT VITALS FROM OBSERVATIONS
# ═══════════════════════════════════════════════════════════════════════════════
def extract_vitals(csvs: dict, encounter_ids: set) -> pd.DataFrame:
    """Pivot observations.csv to get one row per encounter with vital columns."""

    obs = csvs['observations'].copy()

    # Find column names
    enc_col = next((c for c in obs.columns if 'ENCOUNTER' in c.upper()), None)
    code_col = next((c for c in obs.columns if c.upper() == 'CODE'), None)
    value_col = next((c for c in obs.columns if c.upper() == 'VALUE'), None)

    if not all([enc_col, code_col, value_col]):
        print("  ❌ Cannot find required columns in observations.csv")
        return pd.DataFrame()

    # Filter to vital-sign LOINC codes only
    vital_loinc_codes = set(LOINC_TO_VITAL.keys())
    obs_vitals = obs[obs[code_col].astype(str).isin(vital_loinc_codes)].copy()

    # Filter to our encounters
    obs_vitals = obs_vitals[obs_vitals[enc_col].isin(encounter_ids)]

    # Map LOINC code to our vital column name
    obs_vitals['vital_name'] = obs_vitals[code_col].astype(str).map(LOINC_TO_VITAL)

    # Convert value to numeric
    obs_vitals[value_col] = pd.to_numeric(obs_vitals[value_col], errors='coerce')

    # Pivot: one row per encounter, one column per vital
    vitals_pivot = obs_vitals.pivot_table(
        index=enc_col,
        columns='vital_name',
        values=value_col,
        aggfunc='first'  # take first reading per encounter
    ).reset_index()

    vitals_pivot = vitals_pivot.rename(columns={enc_col: 'encounter_id'})

    # Report coverage
    for vital in VITAL_FEATURES:
        if vital in vitals_pivot.columns:
            coverage = vitals_pivot[vital].notna().sum()
            pct = coverage / len(vitals_pivot) * 100
            print(f"     {vital:<20s}  {coverage:>6,} values ({pct:.0f}%)")
        else:
            print(f"     {vital:<20s}  MISSING")

    return vitals_pivot


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 5: EXTRACT SYMPTOMS FROM CONDITIONS
# ═══════════════════════════════════════════════════════════════════════════════
def extract_symptoms(csvs: dict, encounter_ids: set) -> pd.DataFrame:
    """Map conditions.csv SNOMED codes to binary symptom flags."""

    conds = csvs['conditions'].copy()

    # Find column names
    enc_col = next((c for c in conds.columns if 'ENCOUNTER' in c.upper()), None)
    code_col = next((c for c in conds.columns if c.upper() == 'CODE'), None)

    if not all([enc_col, code_col]):
        print("  ❌ Cannot find required columns in conditions.csv")
        return pd.DataFrame()

    # Filter to our encounters
    conds = conds[conds[enc_col].isin(encounter_ids)].copy()
    conds[code_col] = conds[code_col].astype(str)

    # Initialize symptom DataFrame
    symptom_records = []

    for enc_id, group in conds.groupby(enc_col):
        row = {'encounter_id': enc_id}

        # Start with all symptoms = 0
        for sym in SYMPTOM_FEATURES:
            row[sym] = 0

        codes_in_encounter = set(group[code_col].values)

        # Map SNOMED codes to symptoms
        for code in codes_in_encounter:
            if code in SNOMED_TO_SYMPTOM:
                symptom_name = SNOMED_TO_SYMPTOM[code]
                if symptom_name in SYMPTOM_FEATURES:
                    row[symptom_name] = 1

            # Special: stroke → set multiple FAST flags
            if code in STROKE_SNOMED_CODES:
                row['facial_droop'] = 1
                row['arm_weakness'] = 1
                row['speech_difficulty'] = 1
                row['altered_mental_status'] = np.random.choice([0, 1], p=[0.3, 0.7])

            # Special: MI → set radiation flags
            if code in MI_SNOMED_CODES:
                row['chest_pain'] = 1
                row['arm_pain_left'] = np.random.choice([0, 1], p=[0.3, 0.7])
                row['jaw_pain'] = np.random.choice([0, 1], p=[0.6, 0.4])
                row['dyspnea'] = np.random.choice([0, 1], p=[0.4, 0.6])
                row['severe_pain'] = 1

        symptom_records.append(row)

    symptoms_df = pd.DataFrame(symptom_records)

    # Count how many encounters had at least one symptom
    symptom_cols = [c for c in symptoms_df.columns if c in SYMPTOM_FEATURES]
    has_symptom = (symptoms_df[symptom_cols].sum(axis=1) > 0).sum()
    print(f"  🩺 Symptoms: {has_symptom:,}/{len(symptoms_df):,} encounters have ≥1 symptom mapped")

    return symptoms_df


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 6: DERIVE ESI LEVELS
#  Since Synthea doesn't assign ESI, we infer from vitals + conditions
#  using clinical rules similar to safety_engine.py
# ═══════════════════════════════════════════════════════════════════════════════
def derive_esi_level(row: pd.Series) -> int:
    """Assign ESI 1-5 based on vitals and symptoms using clinical heuristics."""

    # ── ESI 1: Immediate life-threat ──────────────────────────────────────
    # Cardiac arrest / unresponsive
    if row.get('altered_mental_status', 0) == 1 and row.get('bp_systolic', 120) < 80:
        return 1

    # Severe hypoxia
    spo2 = row.get('spo2', 98)
    if pd.notna(spo2) and spo2 < 90:
        return 1

    # Shock: hypotension + tachycardia
    sbp = row.get('bp_systolic', 120)
    hr = row.get('heart_rate', 80)
    if pd.notna(sbp) and pd.notna(hr) and sbp < 90 and hr > 100:
        return 1

    # Stroke (FAST positive)
    fast_count = sum([
        row.get('facial_droop', 0),
        row.get('arm_weakness', 0),
        row.get('speech_difficulty', 0)
    ])
    if fast_count >= 2:
        return 1

    # MI with radiation
    if row.get('chest_pain', 0) == 1 and (row.get('arm_pain_left', 0) == 1 or row.get('jaw_pain', 0) == 1):
        return 1

    # Seizure with AMS
    if row.get('seizure', 0) == 1 and row.get('altered_mental_status', 0) == 1:
        return 1

    # Uncontrolled bleeding
    if row.get('uncontrolled_bleeding', 0) == 1:
        return 1

    # Severe respiratory distress
    rr = row.get('respiratory_rate', 16)
    if pd.notna(rr) and rr > 30 and (row.get('dyspnea', 0) == 1 or row.get('shortness_of_breath', 0) == 1):
        return 1

    # ── ESI 2: Emergent / high risk ───────────────────────────────────────
    # Chest pain with stable vitals
    if row.get('chest_pain', 0) == 1:
        return 2

    # Moderate hypoxia
    if pd.notna(spo2) and 90 <= spo2 <= 94:
        return 2

    # High fever with confusion
    temp = row.get('temperature', 37.0)
    if pd.notna(temp) and temp >= 39.5:
        return 2

    # Single stroke symptom
    if fast_count == 1:
        return 2

    # Syncope
    if row.get('syncope', 0) == 1:
        return 2

    # Severe pain
    if row.get('severe_pain', 0) == 1:
        return 2

    # Tachycardia > 130
    if pd.notna(hr) and hr > 130:
        return 2

    # Hypertensive urgency
    if pd.notna(sbp) and sbp > 180:
        return 2

    # AMS alone
    if row.get('altered_mental_status', 0) == 1:
        return 2

    # ── ESI 3: Urgent ─────────────────────────────────────────────────────
    # Abdominal pain with vomiting or fever
    if row.get('abdominal_pain', 0) == 1 and (row.get('vomiting', 0) == 1 or row.get('fever', 0) == 1):
        return 3

    # Fever with other symptoms
    if row.get('fever', 0) == 1:
        return 3

    # Moderate vital abnormalities
    if pd.notna(hr) and (hr > 100 or hr < 50):
        return 3

    if pd.notna(sbp) and (sbp < 100 or sbp > 160):
        return 3

    if pd.notna(rr) and (rr > 22 or rr < 10):
        return 3

    # Abdominal pain alone
    if row.get('abdominal_pain', 0) == 1:
        return 3

    # Dyspnea without distress
    if row.get('dyspnea', 0) == 1 or row.get('shortness_of_breath', 0) == 1:
        return 3

    # Seizure (resolved)
    if row.get('seizure', 0) == 1:
        return 3

    # ── ESI 4: Less urgent ────────────────────────────────────────────────
    # Nausea / vomiting without other symptoms
    if row.get('nausea', 0) == 1 or row.get('vomiting', 0) == 1:
        return 4

    # Headache without red flags
    if row.get('headache', 0) == 1:
        return 4

    # Dizziness alone
    if row.get('dizziness', 0) == 1:
        return 4

    # Confusion alone
    if row.get('confusion', 0) == 1:
        return 4

    # Any remaining symptom
    symptom_count = sum(row.get(s, 0) for s in SYMPTOM_FEATURES)
    if symptom_count > 0:
        return 4

    # ── ESI 5: Non-urgent ─────────────────────────────────────────────────
    return 5


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 7: GENERATE DURATION
# ═══════════════════════════════════════════════════════════════════════════════
def generate_symptom_duration(esi_level: int) -> float:
    """Generate realistic symptom duration based on ESI level."""
    durations = {
        1: lambda: float(np.clip(np.random.exponential(1), 0.1, 6)),
        2: lambda: float(np.clip(np.random.exponential(3), 0.5, 24)),
        3: lambda: float(np.clip(np.random.exponential(12), 2, 72)),
        4: lambda: float(np.clip(np.random.exponential(24), 6, 168)),
        5: lambda: float(np.clip(np.random.exponential(48), 24, 336)),
    }
    return durations.get(esi_level, durations[3])()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
def parse_synthea(synthea_dir: str, output_path: str = "data/synthea_processed.csv"):
    """Full Synthea → training format conversion pipeline."""

    print("=" * 64)
    print("  RiskScope AI — Synthea Data Parser")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 64)

    # ── Load CSVs ──────────────────────────────────────────────────────────
    csvs = load_synthea_csvs(synthea_dir)

    # ── Filter to ED encounters ────────────────────────────────────────────
    ed_encounters = filter_emergency_encounters(csvs)

    # ── Get encounter IDs ──────────────────────────────────────────────────
    enc_id_col = next((c for c in ed_encounters.columns if c.upper() == 'ID'), ed_encounters.columns[0])
    encounter_ids = set(ed_encounters[enc_id_col].values)
    print(f"  📋 Processing {len(encounter_ids):,} encounters")

    # ── Extract demographics ───────────────────────────────────────────────
    print("\n[1/4] EXTRACTING DEMOGRAPHICS")
    demo_df = extract_demographics(csvs, ed_encounters)

    # ── Extract vitals ─────────────────────────────────────────────────────
    print("\n[2/4] EXTRACTING VITALS (LOINC → columns)")
    vitals_df = extract_vitals(csvs, encounter_ids)

    # ── Extract symptoms ───────────────────────────────────────────────────
    print("\n[3/4] EXTRACTING SYMPTOMS (SNOMED → flags)")
    symptoms_df = extract_symptoms(csvs, encounter_ids)

    # ── Merge everything ───────────────────────────────────────────────────
    print("\n[4/4] MERGING & DERIVING ESI LEVELS")

    # Start with demographics
    final = demo_df[['encounter_id', 'age', 'gender']].copy()

    # Merge vitals
    if len(vitals_df) > 0:
        final = final.merge(vitals_df, on='encounter_id', how='left')

    # Merge symptoms
    if len(symptoms_df) > 0:
        final = final.merge(symptoms_df, on='encounter_id', how='left')

    # Fill missing vitals with reasonable defaults
    vital_defaults = {
        'heart_rate': 80,
        'bp_systolic': 120,
        'bp_diastolic': 80,
        'spo2': 98,
        'temperature': 37.0,
        'respiratory_rate': 16,
    }
    for vital, default in vital_defaults.items():
        if vital not in final.columns:
            final[vital] = default
        else:
            final[vital] = final[vital].fillna(default)

    # Fill missing symptoms with 0
    for sym in SYMPTOM_FEATURES:
        if sym not in final.columns:
            final[sym] = 0
        else:
            final[sym] = final[sym].fillna(0).astype(int)

    # ── Derive ESI levels ──────────────────────────────────────────────────
    print("  🏷️  Deriving ESI levels from vitals + symptoms …")
    final['esi_level'] = final.apply(derive_esi_level, axis=1)

    # ── Add symptom duration ───────────────────────────────────────────────
    final['symptom_duration_hours'] = final['esi_level'].apply(generate_symptom_duration)

    # ── Select only the columns train.py expects ───────────────────────────
    output_columns = (
        ['age', 'gender']
        + VITAL_FEATURES
        + SYMPTOM_FEATURES
        + ['symptom_duration_hours', 'esi_level']
    )

    # Ensure all output columns exist
    for col in output_columns:
        if col not in final.columns:
            final[col] = 0

    final = final[output_columns].copy()

    # ── Drop encounter_id (train.py would auto-drop it anyway) ─────────────
    # Already excluded from output_columns

    # ── Print distribution ─────────────────────────────────────────────────
    print(f"\n  📊 ESI DISTRIBUTION")
    total = len(final)
    for esi in range(1, 6):
        count = (final['esi_level'] == esi).sum()
        pct = count / total * 100
        bar = "█" * int(pct)
        print(f"     ESI {esi}: {count:>7,} ({pct:>5.1f}%)  {bar}")

    # ── Save ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    final.to_csv(output_path, index=False)

    print(f"\n  ✅ Saved {len(final):,} rows × {len(final.columns)} cols → {output_path}")
    print(f"  📦 File size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
    print("=" * 64)

    return final


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RiskScope AI — Synthea to Training Data Parser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python synthea_to_training.py --synthea-dir ../synthea/output/csv/
  python synthea_to_training.py --synthea-dir ../synthea/output/csv/ --output data/synthea_processed.csv
        """
    )
    parser.add_argument("--synthea-dir", type=str, required=True,
                        help="Path to Synthea's output/csv/ directory")
    parser.add_argument("--output", type=str, default="data/synthea_processed.csv",
                        help="Output CSV path (default: data/synthea_processed.csv)")

    args = parser.parse_args()
    parse_synthea(args.synthea_dir, args.output)
