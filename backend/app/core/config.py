from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parents[2] / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "RiskScope Backend"
    app_env: str = "development"
    app_port: int = 8000
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"

    cors_allowed_origins: List[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    model_path: str = str(Path(__file__).resolve().parents[2] / "artifacts" / "ml.pkl")
    preprocessor_path: str = str(Path(__file__).resolve().parents[2] / "artifacts" / "preprocessor.pkl")
    feature_columns_path: str = str(Path(__file__).resolve().parents[2] / "artifacts" / "feature_columns.json")
    model_version: str = "v0.0.0-local"

    supabase_url: str | None = None
    supabase_service_role_key: str | None = None
    supabase_patients_table: str = "patients"

    request_timeout_seconds: int = 10
    enable_swagger: bool = True

    @field_validator("cors_allowed_origins", mode="before")
    @classmethod
    def parse_cors_allowed_origins(cls, value: str | List[str]) -> List[str]:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
