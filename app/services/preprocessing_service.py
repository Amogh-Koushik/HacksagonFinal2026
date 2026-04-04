import csv
import io
import json
from pathlib import Path
from typing import Any

from app.core.config import Settings


DEFAULT_FEATURE_COLUMNS = [
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


class PreprocessingService:
    def __init__(self, settings: Settings):
        self._settings = settings
        self.feature_columns = self._load_feature_columns()

    def _load_feature_columns(self) -> list[str]:
        path = Path(self._settings.feature_columns_path)
        if not path.exists():
            return DEFAULT_FEATURE_COLUMNS

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and all(isinstance(item, str) for item in data):
                return data
        except Exception:
            pass
        return DEFAULT_FEATURE_COLUMNS

    def _encode_complaint(self, payload: dict[str, Any]) -> int:
        explicit = payload.get("complaint_encoded")
        if explicit not in (None, ""):
            try:
                return int(explicit)
            except (TypeError, ValueError):
                return 0

        complaint_text = str(payload.get("complaint") or "").strip().lower()
        if not complaint_text:
            return 0

        keyword_to_code = {
            1: ["chest", "cardiac", "heart attack", "mi"],
            2: ["shortness", "breath", "dyspnea", "wheeze", "asthma"],
            3: ["abdominal", "stomach", "abdomen"],
            4: ["headache", "migraine"],
            5: ["stroke", "facial", "weakness", "speech"],
            6: ["dizzy", "dizziness", "vertigo"],
            7: ["fever", "infection"],
            8: ["cough", "cold", "sore throat"],
            9: ["nausea", "vomit"],
            10: ["back pain", "back"],
            11: ["extremity", "arm pain", "leg pain"],
            12: ["laceration", "cut", "bleeding", "wound"],
            13: ["fall", "trauma"],
            14: ["confusion", "mental", "altered"],
        }

        for code, keywords in keyword_to_code.items():
            if any(keyword in complaint_text for keyword in keywords):
                return code
        return 0

    @staticmethod
    def _coerce_numeric(value: Any) -> float | int:
        if value in (None, ""):
            return 0
        try:
            num = float(value)
        except (TypeError, ValueError):
            return 0
        return int(num) if num.is_integer() else num

    def build_model_input_row(self, payload: dict[str, Any]) -> dict[str, Any]:
        row: dict[str, Any] = {}
        for col in self.feature_columns:
            if col == "complaint_encoded":
                row[col] = self._encode_complaint(payload)
                continue
            row[col] = self._coerce_numeric(payload.get(col))
        return row

    def row_to_csv(self, row: dict[str, Any]) -> str:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=self.feature_columns)
        writer.writeheader()
        writer.writerow(row)
        return output.getvalue().strip()
