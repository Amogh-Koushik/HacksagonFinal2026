from datetime import datetime, timezone
import logging
from typing import Any
from uuid import uuid4

from supabase import Client, create_client

from app.core.config import Settings


logger = logging.getLogger(__name__)


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
            "gender_encoded": record.get("gender_encoded"),
            "heart_rate": record.get("heart_rate"),
            "bp_systolic": record.get("bp_systolic"),
            "bp_diastolic": record.get("bp_diastolic"),
            "spo2": record.get("spo2"),
            "temperature": record.get("temperature"),
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
            "notes": record.get("notes"),
            "symptoms": record.get("symptoms"),
            "esi": record.get("esi_level") or record.get("esi"),
            "confidence": record.get("confidence"),
            "status": record.get("status", "waiting"),
            "model_version": record.get("model_version"),
        }

    def _build_storage_record(
        self,
        payload: dict[str, Any],
        esi_level: int,
        confidence: float,
        model_version: str,
        model_input_row: dict[str, Any],
        model_input_csv: str,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        return {
            "id": str(uuid4()),
            "created_at": now,
            "added_at": now,
            "patient_id": payload.get("patient_id"),
            "name": payload.get("name"),
            "age": payload.get("age"),
            "gender_encoded": payload.get("gender_encoded"),
            "heart_rate": payload.get("heart_rate"),
            "bp_systolic": payload.get("bp_systolic"),
            "bp_diastolic": payload.get("bp_diastolic"),
            "spo2": payload.get("spo2"),
            "temperature": payload.get("temperature"),
            "resp_rate": payload.get("resp_rate"),
            "complaint": payload.get("complaint"),
            "complaint_encoded": payload.get("complaint_encoded"),
            "sirs_score": payload.get("sirs_score"),
            "qsofa_score": payload.get("qsofa_score"),
            "shock_index": payload.get("shock_index"),
            "age_group": payload.get("age_group"),
            "critical_spo2": payload.get("critical_spo2", 0),
            "tachycardia": payload.get("tachycardia", 0),
            "hypotension": payload.get("hypotension", 0),
            "high_fever": payload.get("high_fever", 0),
            "tachypnea": payload.get("tachypnea", 0),
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
                "resp_rate": payload.get("resp_rate"),
            },
            "symptoms_json": {
                "complaint": payload.get("complaint"),
                "complaint_encoded": payload.get("complaint_encoded"),
                "notes": payload.get("notes"),
                "symptoms": payload.get("symptoms"),
            },
            "model_input_csv": model_input_csv,
        }

    def create_patient(
        self,
        payload: dict[str, Any],
        esi_level: int,
        confidence: float,
        model_version: str,
        model_input_row: dict[str, Any],
        model_input_csv: str,
    ) -> dict[str, Any]:
        record = self._build_storage_record(
            payload,
            esi_level,
            confidence,
            model_version,
            model_input_row,
            model_input_csv,
        )

        if self._supabase:
            try:
                insert_data = record.copy()
                insert_data.pop("id", None)
                insert_data.pop("model_input_csv", None)
                result = (
                    self._supabase
                    .table(self._settings.supabase_patients_table)
                    .insert(insert_data)
                    .execute()
                )
                if result.data:
                    return self._to_frontend_shape(result.data[0])
            except Exception:
                logger.exception("Supabase insert failed; falling back to in-memory storage")

        self._memory_records.append(record)
        return self._to_frontend_shape(record)

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
