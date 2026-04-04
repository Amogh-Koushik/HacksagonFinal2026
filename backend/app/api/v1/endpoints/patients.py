from fastapi import APIRouter, HTTPException

from app.models.schemas import PatientOut, StatusUpdateRequest
from app.services.runtime import get_storage_service

router = APIRouter(tags=["patients"])


@router.get("/patients", response_model=list[PatientOut])
def get_patients() -> list[PatientOut]:
    storage = get_storage_service()
    return storage.list_patients()


@router.patch("/patients/{patient_id}/status", response_model=PatientOut)
def update_patient_status(patient_id: str, request: StatusUpdateRequest) -> PatientOut:
    storage = get_storage_service()
    updated = storage.update_status(patient_id, request.status)
    if not updated:
        raise HTTPException(status_code=404, detail="Patient not found")
    return updated


@router.delete("/patients/{patient_id}")
def delete_patient(patient_id: str) -> dict[str, bool]:
    storage = get_storage_service()
    ok = storage.delete_patient(patient_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Patient not found")
    return {"ok": True}
