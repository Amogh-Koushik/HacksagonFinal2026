from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.models.schemas import CSVPrepareResponse, TriageRequest, TriageResponse
from app.services.runtime import (
    get_model_service,
    get_preprocessing_service,
    get_storage_service,
)

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=TriageResponse)
def predict_and_store(payload: TriageRequest) -> TriageResponse:
    settings = get_settings()
    preprocessing = get_preprocessing_service()
    model_service = get_model_service()
    storage = get_storage_service()

    payload_dict = payload.model_dump()
    model_input_row = preprocessing.build_model_input_row(payload_dict)
    model_input_csv = preprocessing.row_to_csv(model_input_row)

    try:
        esi_level, confidence = model_service.predict(model_input_row)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    patient = storage.create_patient(
        payload=payload_dict,
        esi_level=esi_level,
        confidence=confidence,
        model_version=settings.model_version,
        model_input_row=model_input_row,
        model_input_csv=model_input_csv,
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
