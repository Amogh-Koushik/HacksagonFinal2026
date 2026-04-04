from datetime import datetime, timezone
import logging
import re
from typing import Any
from uuid import uuid4

from supabase import Client, create_client

from app.core.config import Settings


logger = logging.getLogger(__name__)


MISSING_COLUMN_PATTERN = re.compile(r"Could not find the '([^']+)' column")


class StorageService:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._supabase: Client | None = None
        self._memory_records: list[dict[str, Any]] = []

        if settings.supabase_url and settings.supabase_service_role_key:
            try:
                self._supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)
            except Exception:
                logger.exception("Failed to initialize Supabase client; using in-memory storage instead")
                self._supabase = None

    @property
    def mode(self) -> str:
        return "supabase" if self._supabase else "memory"

    def _to_frontend_shape(self, record: dict[str, Any]) -> dict[str, Any]:
        created_at = record.get("created_at") or datetime.now(timezone.utc).isoformat()
        added_at = record.get("added_at") or created_at

        return {
            "id": str(record.get("id")),
            "created_at": created_at,
            "addedAt": added_at,
            "patient_id": record.get("patient_id"),
            "name": record.get("name"),
            "age": record.get("age"),
            "gender": record.get("gender"),
            "gender_encoded": record.get("gender_encoded"),
            "heart_rate": record.get("heart_rate"),
            "bp_systolic": record.get("bp_systolic"),
            "bp_diastolic": record.get("bp_diastolic"),
            "spo2": record.get("spo2"),
            "temperature": record.get("temperature"),
            "respiratory_rate": record.get("respiratory_rate"),
            "resp_rate": record.get("resp_rate"),
            "complaint": record.get("complaint"),
            "complaint_encoded": record.get("complaint_encoded"),
            "sirs_score": record.get("sirs_score"),
            "qsofa_score": record.get("qsofa_score"),
            "shock_index": record.get("shock_index"),
            "age_group": record.get("age_group"),
            "critical_spo2": record.get("critical_spo2", 0),
            "tachycardia": record.get("tachycardia", 0),
            "hypotension": record.get("hypotension", 0),
            "high_fever": record.get("high_fever", 0),
            "tachypnea": record.get("tachypnea", 0),
            "chest_pain": record.get("chest_pain", 0),
            "arm_pain_left": record.get("arm_pain_left", 0),
            "jaw_pain": record.get("jaw_pain", 0),
            "dyspnea": record.get("dyspnea", 0),
            "shortness_of_breath": record.get("shortness_of_breath", 0),
            "facial_droop": record.get("facial_droop", 0),
            "arm_weakness": record.get("arm_weakness", 0),
            "speech_difficulty": record.get("speech_difficulty", 0),
            "abdominal_pain": record.get("abdominal_pain", 0),
            "rigid_abdomen": record.get("rigid_abdomen", 0),
            "altered_mental_status": record.get("altered_mental_status", 0),
            "confusion": record.get("confusion", 0),
            "fever": record.get("fever", 0),
            "nausea": record.get("nausea", 0),
            "vomiting": record.get("vomiting", 0),
            "dizziness": record.get("dizziness", 0),
            "syncope": record.get("syncope", 0),
            "headache": record.get("headache", 0),
            "seizure": record.get("seizure", 0),
            "uncontrolled_bleeding": record.get("uncontrolled_bleeding", 0),
            "severe_pain": record.get("severe_pain", 0),
            "symptom_duration_hours": record.get("symptom_duration_hours"),
            "notes": record.get("notes"),
            "symptoms": record.get("symptoms"),
            "esi": record.get("esi_level") or record.get("esi"),
            "confidence": record.get("confidence"),
            "status": record.get("status", "waiting"),
            "model_version": record.get("model_version"),
            "method": record.get("method"),
            "protocol": record.get("protocol"),
            "action": record.get("action"),
            "rule_triggered": record.get("rule_triggered"),
            "escalated": record.get("escalated", False),
            "recommendation": record.get("recommendation"),
            "explanation": record.get("explanation"),
        }

    def _build_storage_record(
        self,
        payload: dict[str, Any],
        esi_level: int,
        confidence: float,
        model_version: str,
        model_input_row: dict[str, Any],
        model_input_csv: str,
        method: str | None = None,
        protocol: str | None = None,
        action: str | None = None,
        rule_triggered: str | None = None,
        escalated: bool = False,
        recommendation: str | None = None,
        explanation: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        def from_payload_or_model(field: str, model_field: str | None = None) -> Any:
            payload_val = payload.get(field)
            if payload_val not in (None, ""):
                return payload_val
            key = model_field or field
            return model_input_row.get(key)

        return {
            "id": str(uuid4()),
            "created_at": now,
            "added_at": now,
            "patient_id": payload.get("patient_id"),
            "name": payload.get("name"),
            "age": from_payload_or_model("age"),
            "gender": payload.get("gender"),
            "gender_encoded": from_payload_or_model("gender_encoded"),
            "heart_rate": from_payload_or_model("heart_rate"),
            "bp_systolic": from_payload_or_model("bp_systolic"),
            "bp_diastolic": from_payload_or_model("bp_diastolic"),
            "spo2": from_payload_or_model("spo2"),
            "temperature": from_payload_or_model("temperature"),
            "respiratory_rate": from_payload_or_model("respiratory_rate", "resp_rate"),
            "resp_rate": from_payload_or_model("resp_rate"),
            "complaint": payload.get("complaint"),
            "complaint_encoded": from_payload_or_model("complaint_encoded"),
            "sirs_score": from_payload_or_model("sirs_score"),
            "qsofa_score": from_payload_or_model("qsofa_score"),
            "shock_index": from_payload_or_model("shock_index"),
            "age_group": from_payload_or_model("age_group"),
            "critical_spo2": from_payload_or_model("critical_spo2"),
            "tachycardia": from_payload_or_model("tachycardia"),
            "hypotension": from_payload_or_model("hypotension"),
            "high_fever": from_payload_or_model("high_fever"),
            "tachypnea": from_payload_or_model("tachypnea"),
            "chest_pain": payload.get("chest_pain", 0),
            "arm_pain_left": payload.get("arm_pain_left", 0),
            "jaw_pain": payload.get("jaw_pain", 0),
            "dyspnea": payload.get("dyspnea", 0),
            "shortness_of_breath": payload.get("shortness_of_breath", 0),
            "facial_droop": payload.get("facial_droop", 0),
            "arm_weakness": payload.get("arm_weakness", 0),
            "speech_difficulty": payload.get("speech_difficulty", 0),
            "abdominal_pain": payload.get("abdominal_pain", 0),
            "rigid_abdomen": payload.get("rigid_abdomen", 0),
            "altered_mental_status": payload.get("altered_mental_status", 0),
            "confusion": payload.get("confusion", 0),
            "fever": payload.get("fever", 0),
            "nausea": payload.get("nausea", 0),
            "vomiting": payload.get("vomiting", 0),
            "dizziness": payload.get("dizziness", 0),
            "syncope": payload.get("syncope", 0),
            "headache": payload.get("headache", 0),
            "seizure": payload.get("seizure", 0),
            "uncontrolled_bleeding": payload.get("uncontrolled_bleeding", 0),
            "severe_pain": payload.get("severe_pain", 0),
            "symptom_duration_hours": payload.get("symptom_duration_hours"),
            "notes": payload.get("notes"),
            "symptoms": payload.get("symptoms"),
            "esi_level": esi_level,
            "confidence": confidence,
            "model_version": model_version,
            "status": "waiting",
            "model_input_json": model_input_row,
            "vitals_json": {
                "heart_rate": payload.get("heart_rate"),
                "bp_systolic": payload.get("bp_systolic"),
                "bp_diastolic": payload.get("bp_diastolic"),
                "spo2": payload.get("spo2"),
                "temperature": payload.get("temperature"),
                "respiratory_rate": payload.get("respiratory_rate"),
                "resp_rate": payload.get("resp_rate"),
            },
            "symptoms_json": {
                "complaint": payload.get("complaint"),
                "complaint_encoded": payload.get("complaint_encoded"),
                "notes": payload.get("notes"),
                "symptoms": payload.get("symptoms"),
                "chest_pain": payload.get("chest_pain", 0),
                "arm_pain_left": payload.get("arm_pain_left", 0),
                "jaw_pain": payload.get("jaw_pain", 0),
                "dyspnea": payload.get("dyspnea", 0),
                "shortness_of_breath": payload.get("shortness_of_breath", 0),
                "facial_droop": payload.get("facial_droop", 0),
                "arm_weakness": payload.get("arm_weakness", 0),
                "speech_difficulty": payload.get("speech_difficulty", 0),
                "abdominal_pain": payload.get("abdominal_pain", 0),
                "rigid_abdomen": payload.get("rigid_abdomen", 0),
                "altered_mental_status": payload.get("altered_mental_status", 0),
                "confusion": payload.get("confusion", 0),
                "fever": payload.get("fever", 0),
                "nausea": payload.get("nausea", 0),
                "vomiting": payload.get("vomiting", 0),
                "dizziness": payload.get("dizziness", 0),
                "syncope": payload.get("syncope", 0),
                "headache": payload.get("headache", 0),
                "seizure": payload.get("seizure", 0),
                "uncontrolled_bleeding": payload.get("uncontrolled_bleeding", 0),
                "severe_pain": payload.get("severe_pain", 0),
                "symptom_duration_hours": payload.get("symptom_duration_hours"),
            },
            "model_input_csv": model_input_csv,
            "method": method,
            "protocol": protocol,
            "action": action,
            "rule_triggered": rule_triggered,
            "escalated": escalated,
            "recommendation": recommendation,
            "explanation": explanation,
        }

    def create_patient(
        self,
        payload: dict[str, Any],
        esi_level: int,
        confidence: float,
        model_version: str,
        model_input_row: dict[str, Any],
        model_input_csv: str,
        method: str | None = None,
        protocol: str | None = None,
        action: str | None = None,
        rule_triggered: str | None = None,
        escalated: bool = False,
        recommendation: str | None = None,
        explanation: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        record = self._build_storage_record(
            payload,
            esi_level,
            confidence,
            model_version,
            model_input_row,
            model_input_csv,
            method=method,
            protocol=protocol,
            action=action,
            rule_triggered=rule_triggered,
            escalated=escalated,
            recommendation=recommendation,
            explanation=explanation,
        )

        if self._supabase:
            try:
                insert_data = record.copy()
                insert_data.pop("id", None)
                insert_data.pop("model_input_csv", None)
                result = self._insert_with_schema_fallback(insert_data)
                if result.data:
                    return self._to_frontend_shape(result.data[0])
            except Exception:
                logger.exception("Supabase insert failed; falling back to in-memory storage")

        self._memory_records.append(record)
        return self._to_frontend_shape(record)

    def _insert_with_schema_fallback(self, insert_data: dict[str, Any]) -> Any:
        if not self._supabase:
            raise RuntimeError("Supabase client is not initialized")

        sanitized = insert_data.copy()
        while True:
            try:
                return (
                    self._supabase
                    .table(self._settings.supabase_patients_table)
                    .insert(sanitized)
                    .execute()
                )
            except Exception as exc:
                message = str(exc)
                match = MISSING_COLUMN_PATTERN.search(message)
                if not match:
                    raise

                missing_col = match.group(1)
                if missing_col not in sanitized:
                    raise

                logger.warning(
                    "Supabase table '%s' missing column '%s'; retrying insert without it",
                    self._settings.supabase_patients_table,
                    missing_col,
                )
                sanitized.pop(missing_col, None)

    def list_patients(self) -> list[dict[str, Any]]:
        if self._supabase:
            try:
                result = (
                    self._supabase
                    .table(self._settings.supabase_patients_table)
                    .select("*")
                    .order("esi_level", desc=False)
                    .order("created_at", desc=False)
                    .execute()
                )
                return [self._to_frontend_shape(item) for item in (result.data or [])]
            except Exception:
                logger.exception("Supabase list failed; falling back to in-memory storage")

        records = sorted(
            self._memory_records,
            key=lambda item: (item.get("esi_level", 5), item.get("created_at", "")),
        )
        return [self._to_frontend_shape(item) for item in records]

    def update_status(self, patient_id: str, status: str) -> dict[str, Any] | None:
        if self._supabase:
            try:
                result = (
                    self._supabase
                    .table(self._settings.supabase_patients_table)
                    .update({"status": status})
                    .eq("id", patient_id)
                    .execute()
                )
                if result.data:
                    return self._to_frontend_shape(result.data[0])
                return None
            except Exception:
                logger.exception("Supabase status update failed; falling back to in-memory storage")

        for record in self._memory_records:
            if str(record.get("id")) == str(patient_id):
                record["status"] = status
                return self._to_frontend_shape(record)
        return None

    def delete_patient(self, patient_id: str) -> bool:
        if self._supabase:
            try:
                result = (
                    self._supabase
                    .table(self._settings.supabase_patients_table)
                    .delete()
                    .eq("id", patient_id)
                    .execute()
                )
                return bool(result.data)
            except Exception:
                logger.exception("Supabase delete failed; falling back to in-memory storage")

        before = len(self._memory_records)
        self._memory_records = [r for r in self._memory_records if str(r.get("id")) != str(patient_id)]
        return len(self._memory_records) < before
