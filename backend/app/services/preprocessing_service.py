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

    @staticmethod
    def _coerce_bool_flag(value: Any) -> int:
        if value in (None, "", False, 0, "0"):
            return 0
        return 1

    def _derive_value(self, column: str, payload: dict[str, Any]) -> float | int:
        heart_rate = self._coerce_numeric(payload.get("heart_rate"))
        bp_systolic = self._coerce_numeric(payload.get("bp_systolic"))
        temperature = self._coerce_numeric(payload.get("temperature"))
        resp_rate = self._coerce_numeric(payload.get("resp_rate") or payload.get("respiratory_rate"))

        if column == "gender_encoded":
            explicit = payload.get("gender_encoded")
            if explicit not in (None, ""):
                return self._coerce_numeric(explicit)
            gender = str(payload.get("gender") or "").strip().upper()
            if gender == "M":
                return 1
            if gender == "F":
                return 0
            return 0

        if column == "resp_rate":
            return resp_rate

        if column == "age_group":
            explicit = payload.get("age_group")
            if explicit not in (None, ""):
                return self._coerce_numeric(explicit)
            age = self._coerce_numeric(payload.get("age"))
            if age <= 17:
                return 1
            if age <= 44:
                return 2
            if age <= 64:
                return 3
            return 4

        if column == "critical_spo2":
            explicit = payload.get("critical_spo2")
            if explicit not in (None, ""):
                return self._coerce_numeric(explicit)
            spo2 = self._coerce_numeric(payload.get("spo2"))
            return 1 if spo2 > 0 and spo2 < 90 else 0

        if column == "tachycardia":
            explicit = payload.get("tachycardia")
            if explicit not in (None, ""):
                return self._coerce_numeric(explicit)
            return 1 if heart_rate > 100 else 0

        if column == "hypotension":
            explicit = payload.get("hypotension")
            if explicit not in (None, ""):
                return self._coerce_numeric(explicit)
            return 1 if bp_systolic > 0 and bp_systolic < 90 else 0

        if column == "high_fever":
            explicit = payload.get("high_fever")
            if explicit not in (None, ""):
                return self._coerce_numeric(explicit)
            fever_flag = self._coerce_bool_flag(payload.get("fever"))
            return 1 if temperature >= 38.5 or fever_flag == 1 else 0

        if column == "tachypnea":
            explicit = payload.get("tachypnea")
            if explicit not in (None, ""):
                return self._coerce_numeric(explicit)
            return 1 if resp_rate > 20 else 0

        if column == "sirs_score":
            explicit = payload.get("sirs_score")
            if explicit not in (None, ""):
                return self._coerce_numeric(explicit)
            score = 0
            if heart_rate > 90:
                score += 1
            if resp_rate > 20:
                score += 1
            if temperature >= 38 or (temperature > 0 and temperature < 36):
                score += 1
            return score

        if column == "qsofa_score":
            explicit = payload.get("qsofa_score")
            if explicit not in (None, ""):
                return self._coerce_numeric(explicit)
            score = 0
            if resp_rate >= 22:
                score += 1
            if bp_systolic > 0 and bp_systolic <= 100:
                score += 1
            if self._coerce_bool_flag(payload.get("altered_mental_status")) == 1:
                score += 1
            return score

        if column == "shock_index":
            explicit = payload.get("shock_index")
            if explicit not in (None, ""):
                return self._coerce_numeric(explicit)
            if heart_rate > 0 and bp_systolic > 0:
                return round(float(heart_rate) / float(bp_systolic), 2)
            return 0

        return self._coerce_numeric(payload.get(column))

    def build_model_input_row(self, payload: dict[str, Any]) -> dict[str, Any]:
        row: dict[str, Any] = {}
        for col in self.feature_columns:
            if col == "complaint_encoded":
                row[col] = self._encode_complaint(payload)
                continue
            row[col] = self._derive_value(col, payload)
        return row

    def row_to_csv(self, row: dict[str, Any]) -> str:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=self.feature_columns)
        writer.writeheader()
        writer.writerow(row)
        return output.getvalue().strip()
