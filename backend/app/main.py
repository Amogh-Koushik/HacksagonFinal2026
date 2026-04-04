from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.patients import router as patients_router
from app.api.v1.endpoints.predict import router as predict_router
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    docs_url="/docs" if settings.enable_swagger else None,
    redoc_url="/redoc" if settings.enable_swagger else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api = settings.api_prefix
app.include_router(health_router, prefix=api)
app.include_router(predict_router, prefix=api)
app.include_router(patients_router, prefix=api)


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}
