"""
Export a backend-safe model artifact from the ML training output.

The backend currently loads `ml.pkl` and calls `.predict([row])` and
`.predict_proba([row])` on 18-column patient rows. The raw ML training
artifact is not directly compatible with that interface, so this script
distills the trained predictor into a pure-Python portable artifact that
uses only standard-library types at runtime.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import cloudpickle

from esi_predictor import ESITriagePredictor
from generate_synthetic_data import generate_synthetic_dataset


BACKEND_FEATURE_COLUMNS = [
    "age",
    "gender_encoded",
    "heart_rate",
    "bp_systolic",
    "bp_diastolic",
    "spo2",
    "temperature",
    "resp_rate",
    "complaint_encoded",
    "sirs_score",
    "qsofa_score",
    "shock_index",
    "age_group",
    "critical_spo2",
    "tachycardia",
    "hypotension",
    "high_fever",
    "tachypnea",
]


def _derive_complaint_code(patient: dict[str, Any]) -> int:
    symptoms = {key for key, value in patient.items() if isinstance(value, (int, float)) and value == 1}

    if {"chest_pain", "arm_pain_left", "jaw_pain"} & symptoms:
        return 1
    if "shortness_of_breath" in symptoms or "dyspnea" in symptoms:
        return 2
    if "abdominal_pain" in symptoms:
        return 3
    if "headache" in symptoms:
        return 4
    if {"facial_droop", "arm_weakness", "speech_difficulty"} & symptoms:
        return 5
    if "dizziness" in symptoms:
        return 6
    if "fever" in symptoms:
        return 7
    if "cough" in symptoms:
        return 8
    if {"nausea", "vomiting"} & symptoms:
        return 9
    if "back_pain" in symptoms:
        return 10
    if "extremity_pain" in symptoms:
        return 11
    if {"laceration", "uncontrolled_bleeding"} & symptoms:
        return 12
    if "fall" in symptoms:
        return 13
    if {"altered_mental_status", "confusion"} & symptoms:
        return 14
    return 0


def _backend_row_from_patient(patient: dict[str, Any]) -> list[float]:
    age = int(patient.get("age", 0) or 0)
    gender = patient.get("gender", 0)
    heart_rate = float(patient.get("heart_rate", 0) or 0)
    bp_systolic = float(patient.get("bp_systolic", 0) or 0)
    bp_diastolic = float(patient.get("bp_diastolic", 0) or 0)
    spo2 = float(patient.get("spo2", 0) or 0)
    temperature = float(patient.get("temperature", 0) or 0)
    resp_rate = float(patient.get("respiratory_rate", 0) or 0)

    complaint_encoded = _derive_complaint_code(patient)
    sirs_score = int((temperature > 38.0 or temperature < 36.0)) + int(heart_rate > 90) + int(resp_rate > 20)
    qsofa_score = int(resp_rate >= 22) + int(bp_systolic <= 100) + int(bool(patient.get("altered_mental_status", 0)))
    shock_index = heart_rate / bp_systolic if bp_systolic > 0 else 0.0

    if age < 18:
        age_group = 1
    elif age < 45:
        age_group = 2
    elif age < 65:
        age_group = 3
    else:
        age_group = 4

    critical_spo2 = 1 if spo2 < 90 else 0
    tachycardia = 1 if heart_rate > 100 else 0
    hypotension = 1 if bp_systolic < 90 else 0
    high_fever = 1 if temperature >= 38.5 else 0
    tachypnea = 1 if resp_rate > 20 else 0

    return [
        age,
        int(gender),
        heart_rate,
        bp_systolic,
        bp_diastolic,
        spo2,
        temperature,
        resp_rate,
        complaint_encoded,
        sirs_score,
        qsofa_score,
        round(shock_index, 4),
        age_group,
        critical_spo2,
        tachycardia,
        hypotension,
        high_fever,
        tachypnea,
    ]


def _euclidean_distance(a: list[float], b: list[float], stds: list[float]) -> float:
    total = 0.0
    for index, (left, right) in enumerate(zip(a, b)):
        scale = stds[index] if stds[index] > 1e-9 else 1.0
        diff = (left - right) / scale
        total += diff * diff
    return math.sqrt(total)


def _build_portable_model(reference_rows: list[list[float]], reference_labels: list[int]) -> SimpleNamespace:
    feature_count = len(reference_rows[0])
    means = [
        sum(row[index] for row in reference_rows) / len(reference_rows)
        for index in range(feature_count)
    ]
    stds = []
    for index in range(feature_count):
        variance = sum((row[index] - means[index]) ** 2 for row in reference_rows) / max(len(reference_rows) - 1, 1)
        stds.append(math.sqrt(variance) or 1.0)

    classes = [1, 2, 3, 4, 5]
    k = 25

    def predict(samples: Iterable[Iterable[float]]) -> list[int]:
        sample_rows = list(samples)
        predictions: list[int] = []
        for sample in sample_rows:
            sample_row = [float(value) for value in sample]
            distances = [
                (_euclidean_distance(sample_row, reference_row, stds), label)
                for reference_row, label in zip(reference_rows, reference_labels)
            ]
            distances.sort(key=lambda item: item[0])
            votes: dict[int, float] = {label: 0.0 for label in classes}
            for distance, label in distances[:k]:
                weight = 1.0 / (distance + 1e-6)
                votes[label] += weight
            predictions.append(max(votes, key=votes.get))
        return predictions

    def predict_proba(samples: Iterable[Iterable[float]]) -> list[list[float]]:
        sample_rows = list(samples)
        probabilities: list[list[float]] = []
        for sample in sample_rows:
            sample_row = [float(value) for value in sample]
            distances = [
                (_euclidean_distance(sample_row, reference_row, stds), label)
                for reference_row, label in zip(reference_rows, reference_labels)
            ]
            distances.sort(key=lambda item: item[0])
            votes: dict[int, float] = {label: 0.0 for label in classes}
            for distance, label in distances[:k]:
                weight = 1.0 / (distance + 1e-6)
                votes[label] += weight
            total_weight = sum(votes.values()) or 1.0
            probabilities.append([votes[label] / total_weight for label in classes])
        return probabilities

    return SimpleNamespace(
        predict=predict,
        predict_proba=predict_proba,
        feature_columns=BACKEND_FEATURE_COLUMNS,
        class_labels=classes,
        means=means,
        stds=stds,
        reference_rows=reference_rows,
        reference_labels=reference_labels,
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    trained_model_path = repo_root / "models" / "esi_ensemble_model.pkl"
    backend_artifacts_dir = repo_root.parent / "backend" / "artifacts"
    backend_artifacts_dir.mkdir(parents=True, exist_ok=True)

    predictor = ESITriagePredictor.load(str(trained_model_path))

    X, _ = generate_synthetic_dataset(n_samples=5000)
    reference_rows: list[list[float]] = []
    reference_labels: list[int] = []

    for _, row in X.iterrows():
        patient = row.to_dict()
        backend_row = _backend_row_from_patient(patient)
        label = int(predictor.predict(row.to_frame().T)[0])
        reference_rows.append(backend_row)
        reference_labels.append(label)

    portable_model = _build_portable_model(reference_rows, reference_labels)

    model_path = backend_artifacts_dir / "ml.pkl"
    feature_columns_path = backend_artifacts_dir / "feature_columns.json"

    with model_path.open("wb") as model_file:
        cloudpickle.dump(portable_model, model_file)
    feature_columns_path.write_text(json.dumps(BACKEND_FEATURE_COLUMNS, indent=2), encoding="utf-8")

    print(f"Exported backend model to {model_path}")
    print(f"Exported backend feature columns to {feature_columns_path}")


if __name__ == "__main__":
    main()