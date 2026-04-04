from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class TriageRequest(BaseModel):
    patient_id: str = Field(..., min_length=1)
    name: str | None = None

    age: int | None = None
    gender_encoded: int | None = None
    heart_rate: int | None = None
    bp_systolic: int | None = None
    bp_diastolic: int | None = None
    spo2: float | None = None
    temperature: float | None = None
    resp_rate: int | None = None
    complaint: str | None = None
    complaint_encoded: int | None = None
    sirs_score: int | None = None
    qsofa_score: int | None = None
    shock_index: float | None = None
    age_group: int | None = None

    critical_spo2: int = 0
    tachycardia: int = 0
    hypotension: int = 0
    high_fever: int = 0
    tachypnea: int = 0

    notes: str | None = None
    symptoms: str | None = None


class PatientOut(BaseModel):
    id: str
    created_at: datetime
    addedAt: datetime

    patient_id: str
    name: str | None = None

    age: int | None = None
    gender_encoded: int | None = None
    heart_rate: int | None = None
    bp_systolic: int | None = None
    bp_diastolic: int | None = None
    spo2: float | None = None
    temperature: float | None = None
    resp_rate: int | None = None
    complaint: str | None = None
    complaint_encoded: int | None = None
    sirs_score: int | None = None
    qsofa_score: int | None = None
    shock_index: float | None = None
    age_group: int | None = None

    critical_spo2: int = 0
    tachycardia: int = 0
    hypotension: int = 0
    high_fever: int = 0
    tachypnea: int = 0

    notes: str | None = None
    symptoms: str | None = None

    esi: int
    confidence: float | None = None
    status: Literal["waiting", "treating", "discharged"] = "waiting"
    model_version: str | None = None


class TriageResponse(BaseModel):
    esi_level: int
    confidence: float | None
    model_version: str
    prediction_id: str
    created_at: datetime
    patient: PatientOut
    model_input_csv: str
    model_input_row: dict[str, Any]


class StatusUpdateRequest(BaseModel):
    status: Literal["waiting", "treating", "discharged"]


class HealthResponse(BaseModel):
    status: str
    app_name: str
    app_env: str
    model_loaded: bool
    model_version: str
    storage_mode: Literal["supabase", "memory"]


class CSVPrepareResponse(BaseModel):
    patient_id: str
    model_input_row: dict[str, Any]
    model_input_csv: str
    feature_columns: list[str]
