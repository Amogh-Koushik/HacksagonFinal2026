from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.models.schemas import CSVPrepareResponse, TriageRequest, TriageResponse
from app.services.runtime import (
    get_model_service,
    get_preprocessing_service,
    get_storage_service,
)
from app.services.safety_service import SafetyService

router = APIRouter(tags=["predict"])

_safety = SafetyService()


@router.post("/predict", response_model=TriageResponse)
def predict_and_store(payload: TriageRequest) -> TriageResponse:
    settings = get_settings()
    preprocessing = get_preprocessing_service()
    model_service = get_model_service()
    storage = get_storage_service()

    payload_dict = payload.model_dump()

    # ── Layer 1: Emergency rule check (runs BEFORE ML) ──────────────
    safety_result = _safety.check_emergency_rules(payload_dict)

    if safety_result.triggered:
        # Build model input anyway so it can be stored / inspected
        model_input_row = preprocessing.build_model_input_row(payload_dict)
        model_input_csv = preprocessing.row_to_csv(model_input_row)

        esi_level = safety_result.esi_level
        confidence = 1.0  # rule-based -> 100% confidence

        patient = storage.create_patient(
            payload=payload_dict,
            esi_level=esi_level,
            confidence=confidence,
            model_version=settings.model_version,
            model_input_row=model_input_row,
            model_input_csv=model_input_csv,
            method=safety_result.method,
            protocol=safety_result.protocol,
            action=safety_result.action,
            rule_triggered=safety_result.rule_name,
            recommendation=_safety.generate_recommendation(esi_level),
            explanation=safety_result.explanation,
        )

        return TriageResponse(
            esi_level=esi_level,
            confidence=confidence,
            model_version=settings.model_version,
            prediction_id=patient["id"],
            created_at=patient["created_at"],
            patient=patient,
            model_input_csv=model_input_csv,
            model_input_row=model_input_row,
            method=safety_result.method,
            protocol=safety_result.protocol,
            action=safety_result.action,
            rule_triggered=safety_result.rule_name,
            escalated=False,
            recommendation=_safety.generate_recommendation(esi_level),
            explanation=safety_result.explanation,
        )

    # ── Layer 2: ML prediction ──────────────────────────────────────
    model_input_row = preprocessing.build_model_input_row(payload_dict)
    model_input_csv = preprocessing.row_to_csv(model_input_row)

    try:
        esi_level, confidence = model_service.predict(model_input_row)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # ── Layer 3: Confidence calibration ─────────────────────────────
    esi_level, confidence, escalated, escalation_reason = (
        _safety.apply_confidence_calibration(esi_level, confidence)
    )

    recommendation = _safety.generate_recommendation(esi_level)
    if escalated and escalation_reason:
        recommendation += f" ({escalation_reason})"

    patient = storage.create_patient(
        payload=payload_dict,
        esi_level=esi_level,
        confidence=confidence,
        model_version=settings.model_version,
        model_input_row=model_input_row,
        model_input_csv=model_input_csv,
        method="ML_PREDICTION",
        protocol=None,
        action=None,
        rule_triggered=None,
        recommendation=recommendation,
        explanation=None,
    )

    return TriageResponse(
        esi_level=esi_level,
        confidence=confidence,
        model_version=settings.model_version,
        prediction_id=patient["id"],
        created_at=patient["created_at"],
        patient=patient,
        model_input_csv=model_input_csv,
        model_input_row=model_input_row,
        method="ML_PREDICTION",
        protocol=None,
        action=None,
        rule_triggered=None,
        escalated=escalated,
        recommendation=recommendation,
        explanation=None,
    )


@router.post("/predict/prepare-csv", response_model=CSVPrepareResponse)
def prepare_csv(payload: TriageRequest) -> CSVPrepareResponse:
    preprocessing = get_preprocessing_service()
    payload_dict = payload.model_dump()
    model_input_row = preprocessing.build_model_input_row(payload_dict)
    model_input_csv = preprocessing.row_to_csv(model_input_row)

    return CSVPrepareResponse(
        patient_id=payload.patient_id,
        model_input_row=model_input_row,
        model_input_csv=model_input_csv,
        feature_columns=preprocessing.feature_columns,
    )
