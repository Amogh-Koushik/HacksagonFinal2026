from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class TriageRequest(BaseModel):
    patient_id: str = Field(..., min_length=1)
    name: str | None = None

    age: int | None = None
    gender: Literal["M", "F"] | None = None
    gender_encoded: int | None = None
    heart_rate: int | None = None
    bp_systolic: int | None = None
    bp_diastolic: int | None = None
    spo2: float | None = None
    temperature: float | None = None
    respiratory_rate: int | None = None
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

    chest_pain: int = 0
    arm_pain_left: int = 0
    jaw_pain: int = 0
    dyspnea: int = 0
    shortness_of_breath: int = 0
    facial_droop: int = 0
    arm_weakness: int = 0
    speech_difficulty: int = 0
    abdominal_pain: int = 0
    rigid_abdomen: int = 0
    altered_mental_status: int = 0
    confusion: int = 0
    fever: int = 0
    nausea: int = 0
    vomiting: int = 0
    dizziness: int = 0
    syncope: int = 0
    headache: int = 0
    seizure: int = 0
    uncontrolled_bleeding: int = 0
    severe_pain: int = 0
    symptom_duration_hours: float | None = None

    notes: str | None = None
    symptoms: str | None = None


class PatientOut(BaseModel):
    id: str
    created_at: datetime
    addedAt: datetime

    patient_id: str
    name: str | None = None

    age: int | None = None
    gender: Literal["M", "F"] | None = None
    gender_encoded: int | None = None
    heart_rate: int | None = None
    bp_systolic: int | None = None
    bp_diastolic: int | None = None
    spo2: float | None = None
    temperature: float | None = None
    respiratory_rate: int | None = None
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

    chest_pain: int = 0
    arm_pain_left: int = 0
    jaw_pain: int = 0
    dyspnea: int = 0
    shortness_of_breath: int = 0
    facial_droop: int = 0
    arm_weakness: int = 0
    speech_difficulty: int = 0
    abdominal_pain: int = 0
    rigid_abdomen: int = 0
    altered_mental_status: int = 0
    confusion: int = 0
    fever: int = 0
    nausea: int = 0
    vomiting: int = 0
    dizziness: int = 0
    syncope: int = 0
    headache: int = 0
    seizure: int = 0
    uncontrolled_bleeding: int = 0
    severe_pain: int = 0
    symptom_duration_hours: float | None = None

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
