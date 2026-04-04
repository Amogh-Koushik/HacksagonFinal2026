from pathlib import Path
from typing import Any

import joblib

from app.core.config import get_settings


class ModelService:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._model = None
        self.loaded = False
        self.model_path = Path(self._settings.model_path)
        self._load_model()

    def _load_model(self) -> None:
        if not self.model_path.exists():
            self.loaded = False
            return
        self._model = joblib.load(self.model_path)
        self.loaded = True

    def predict(self, features: dict[str, Any]) -> tuple[int, float | None]:
        if not self.loaded or self._model is None:
            raise RuntimeError(
                f"Model is not loaded. Place your file at: {self.model_path}"
            )

        row = [features[key] for key in features.keys()]
        y_pred = self._model.predict([row])
        label = int(y_pred[0])

        confidence: float | None = None
        if hasattr(self._model, "predict_proba"):
            probabilities = self._model.predict_proba([row])
            confidence = float(max(probabilities[0]))

        return label, confidence
