"""
RiskScope AI - Backend Model Service
=====================================
Loads the portable ml.pkl model and provides predict/predict_proba.
Includes a fallback rule-based predictor if the model cannot be loaded
or its predict() method fails (e.g. cloudpickle incompatibility).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib

from app.core.config import get_settings


# =====================================================================
#  FALLBACK PREDICTOR
# =====================================================================
# If the cloudpickle model segfaults (common with Python 3.13), this
# fallback implements the same weighted-KNN logic in pure Python,
# reading the reference data from the loaded model object.
# =====================================================================

def _euclidean_distance(a: list[float], b: list[float], stds: list[float]) -> float:
    total = 0.0
    for idx, (left, right) in enumerate(zip(a, b)):
        scale = stds[idx] if stds[idx] > 1e-9 else 1.0
        diff = (left - right) / scale
        total += diff * diff
    return math.sqrt(total)


def _knn_predict(
    sample_row: list[float],
    reference_rows: list[list[float]],
    reference_labels: list[int],
    stds: list[float],
    k: int = 25,
) -> tuple[int, list[float]]:
    """Pure-Python KNN prediction. Returns (label, class_probabilities)."""
    classes = [1, 2, 3, 4, 5]
    distances = [
        (_euclidean_distance(sample_row, ref_row, stds), label)
        for ref_row, label in zip(reference_rows, reference_labels)
    ]
    distances.sort(key=lambda item: item[0])

    votes: dict[int, float] = {label: 0.0 for label in classes}
    for distance, label in distances[:k]:
        weight = 1.0 / (distance + 1e-6)
        votes[label] += weight

    predicted = max(votes, key=votes.get)  # type: ignore[arg-type]
    total_weight = sum(votes.values()) or 1.0
    proba = [votes[label] / total_weight for label in classes]

    return predicted, proba


class ModelService:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._model = None
        self._reference_rows: list[list[float]] | None = None
        self._reference_labels: list[int] | None = None
        self._stds: list[float] | None = None
        self._feature_columns: list[str] | None = None
        self.loaded = False
        self.model_path = Path(self._settings.model_path)
        self._load_model()

    def _load_model(self) -> None:
        if not self.model_path.exists():
            self.loaded = False
            return

        try:
            self._model = joblib.load(self.model_path)
        except Exception:
            try:
                import cloudpickle
                with open(self.model_path, "rb") as f:
                    self._model = cloudpickle.load(f)
            except Exception:
                self.loaded = False
                return

        # Extract reference data for fallback predictor
        if hasattr(self._model, "reference_rows"):
            self._reference_rows = self._model.reference_rows
        if hasattr(self._model, "reference_labels"):
            self._reference_labels = self._model.reference_labels
        if hasattr(self._model, "stds"):
            self._stds = self._model.stds
        if hasattr(self._model, "feature_columns"):
            self._feature_columns = self._model.feature_columns

        self.loaded = True

    def predict(self, features: dict[str, Any]) -> tuple[int, float | None]:
        if not self.loaded or self._model is None:
            raise RuntimeError(
                f"Model is not loaded. Place your file at: {self.model_path}"
            )

        # Build the feature row
        model_feature_columns = self._feature_columns or getattr(
            self._model, "feature_columns", None
        )
        if isinstance(model_feature_columns, list) and model_feature_columns:
            missing = [col for col in model_feature_columns if col not in features]
            if missing:
                raise RuntimeError(
                    "Model input is missing required features: " + ", ".join(missing)
                )
            row = [float(features[col]) for col in model_feature_columns]
        else:
            row = [float(features[key]) for key in features.keys()]

        # Use the pure-Python KNN when reference data is available.
        # The cloudpickle closures in ml.pkl crash on Python 3.13 (segfault),
        # so we avoid calling the deserialized predict() and use our own
        # implementation of the same KNN algorithm directly.
        if (
            self._reference_rows is not None
            and self._reference_labels is not None
            and self._stds is not None
        ):
            label, proba = _knn_predict(
                row, self._reference_rows, self._reference_labels, self._stds
            )
            confidence = float(max(proba))
            return label, confidence

        # Last resort: try the native model (may crash on some Python versions)
        try:
            y_pred = self._model.predict([row])
            label = int(y_pred[0])

            confidence_val: float | None = None
            if hasattr(self._model, "predict_proba"):
                probabilities = self._model.predict_proba([row])
                confidence_val = float(max(probabilities[0]))

            return label, confidence_val
        except Exception as exc:
            raise RuntimeError(f"Model prediction failed: {exc}") from exc

