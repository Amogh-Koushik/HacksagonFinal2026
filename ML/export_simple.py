"""
Simple script to export backend model - runs standalone.
"""
import sys
import os

# Add ML directory to path
ml_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ml_dir)
os.chdir(ml_dir)

import json
import math
from pathlib import Path
from types import SimpleNamespace
import cloudpickle

from esi_predictor import ESITriagePredictor
from generate_synthetic_data import generate_synthetic_dataset

BACKEND_FEATURE_COLUMNS = [
    "age", "gender_encoded", "heart_rate", "bp_systolic", "bp_diastolic",
    "spo2", "temperature", "resp_rate", "complaint_encoded", "sirs_score",
    "qsofa_score", "shock_index", "age_group", "critical_spo2", "tachycardia",
    "hypotension", "high_fever", "tachypnea",
]

def derive_complaint_code(patient):
    symptoms = {key for key, value in patient.items() if isinstance(value, (int, float)) and value == 1}
    if {"chest_pain", "arm_pain_left", "jaw_pain"} & symptoms: return 1
    if "shortness_of_breath" in symptoms or "dyspnea" in symptoms: return 2
    if "abdominal_pain" in symptoms: return 3
    if "headache" in symptoms: return 4
    if {"facial_droop", "arm_weakness", "speech_difficulty"} & symptoms: return 5
    if "dizziness" in symptoms: return 6
    if "fever" in symptoms: return 7
    if "cough" in symptoms: return 8
    if {"nausea", "vomiting"} & symptoms: return 9
    if "back_pain" in symptoms: return 10
    if "extremity_pain" in symptoms: return 11
    if {"laceration", "uncontrolled_bleeding"} & symptoms: return 12
    if "fall" in symptoms: return 13
    if {"altered_mental_status", "confusion"} & symptoms: return 14
    return 0

def backend_row_from_patient(patient):
    age = int(patient.get("age", 0) or 0)
    gender = patient.get("gender", 0)
    heart_rate = float(patient.get("heart_rate", 0) or 0)
    bp_systolic = float(patient.get("bp_systolic", 0) or 0)
    bp_diastolic = float(patient.get("bp_diastolic", 0) or 0)
    spo2 = float(patient.get("spo2", 0) or 0)
    temperature = float(patient.get("temperature", 0) or 0)
    resp_rate = float(patient.get("respiratory_rate", 0) or 0)

    complaint_encoded = derive_complaint_code(patient)
    sirs_score = int((temperature > 38.0 or temperature < 36.0)) + int(heart_rate > 90) + int(resp_rate > 20)
    qsofa_score = int(resp_rate >= 22) + int(bp_systolic <= 100) + int(bool(patient.get("altered_mental_status", 0)))
    shock_index = heart_rate / bp_systolic if bp_systolic > 0 else 0.0

    if age < 18: age_group = 1
    elif age < 45: age_group = 2
    elif age < 65: age_group = 3
    else: age_group = 4

    return [
        age, int(gender), heart_rate, bp_systolic, bp_diastolic, spo2, temperature, resp_rate,
        complaint_encoded, sirs_score, qsofa_score, round(shock_index, 4), age_group,
        1 if spo2 < 90 else 0,
        1 if heart_rate > 100 else 0,
        1 if bp_systolic < 90 else 0,
        1 if temperature >= 38.5 else 0,
        1 if resp_rate > 20 else 0,
    ]

def euclidean_distance(a, b, stds):
    total = 0.0
    for idx, (left, right) in enumerate(zip(a, b)):
        scale = stds[idx] if stds[idx] > 1e-9 else 1.0
        diff = (left - right) / scale
        total += diff * diff
    return math.sqrt(total)

def build_portable_model(reference_rows, reference_labels):
    feature_count = len(reference_rows[0])
    means = [sum(row[i] for row in reference_rows) / len(reference_rows) for i in range(feature_count)]
    stds = []
    for i in range(feature_count):
        variance = sum((row[i] - means[i]) ** 2 for row in reference_rows) / max(len(reference_rows) - 1, 1)
        stds.append(math.sqrt(variance) or 1.0)

    classes = [1, 2, 3, 4, 5]
    k = 25

    def predict(samples):
        predictions = []
        for sample in samples:
            sample_row = [float(v) for v in sample]
            distances = [(euclidean_distance(sample_row, ref, stds), label) for ref, label in zip(reference_rows, reference_labels)]
            distances.sort(key=lambda x: x[0])
            votes = {label: 0.0 for label in classes}
            for dist, label in distances[:k]:
                votes[label] += 1.0 / (dist + 1e-6)
            predictions.append(max(votes, key=votes.get))
        return predictions

    def predict_proba(samples):
        probabilities = []
        for sample in samples:
            sample_row = [float(v) for v in sample]
            distances = [(euclidean_distance(sample_row, ref, stds), label) for ref, label in zip(reference_rows, reference_labels)]
            distances.sort(key=lambda x: x[0])
            votes = {label: 0.0 for label in classes}
            for dist, label in distances[:k]:
                votes[label] += 1.0 / (dist + 1e-6)
            total = sum(votes.values()) or 1.0
            probabilities.append([votes[label] / total for label in classes])
        return probabilities

    return SimpleNamespace(
        predict=predict, predict_proba=predict_proba,
        feature_columns=BACKEND_FEATURE_COLUMNS, class_labels=classes,
        means=means, stds=stds, reference_rows=reference_rows, reference_labels=reference_labels,
    )

def main():
    model_path = Path(ml_dir) / "models" / "esi_ensemble_model.pkl"
    backend_dir = Path(ml_dir).parent / "backend" / "artifacts"
    backend_dir.mkdir(parents=True, exist_ok=True)

    print("Loading trained model...")
    predictor = ESITriagePredictor.load(str(model_path))

    print("Generating 1000 reference samples...")
    X, _ = generate_synthetic_dataset(n_samples=1000)

    reference_rows = []
    reference_labels = []

    print("Building reference dataset...")
    for idx, (_, row) in enumerate(X.iterrows()):
        patient = row.to_dict()
        backend_row = backend_row_from_patient(patient)
        label = int(predictor.predict(row.to_frame().T)[0])
        reference_rows.append(backend_row)
        reference_labels.append(label)
        if (idx + 1) % 250 == 0:
            print(f"  Processed {idx + 1}/1000")

    print("Building portable model...")
    portable_model = build_portable_model(reference_rows, reference_labels)

    output_path = backend_dir / "ml.pkl"
    with output_path.open("wb") as f:
        cloudpickle.dump(portable_model, f)

    feature_path = backend_dir / "feature_columns.json"
    feature_path.write_text(json.dumps(BACKEND_FEATURE_COLUMNS, indent=2), encoding="utf-8")

    print(f"\n✓ Exported to {output_path}")
    print(f"✓ Feature columns: {feature_path}")
    print(f"  Reference samples: {len(reference_rows)}")
    print(f"  Feature columns: {len(BACKEND_FEATURE_COLUMNS)}")

if __name__ == "__main__":
    main()
