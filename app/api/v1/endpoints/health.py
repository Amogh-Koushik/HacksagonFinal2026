from fastapi import APIRouter

from app.core.config import get_settings
from app.models.schemas import HealthResponse
from app.services.runtime import get_model_service, get_storage_service

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    settings = get_settings()
    model_service = get_model_service()
    storage_service = get_storage_service()
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        app_env=settings.app_env,
        model_loaded=model_service.loaded,
        model_version=settings.model_version,
        storage_mode=storage_service.mode,
    )
