from functools import lru_cache

from app.core.config import get_settings
from app.services.model_service import ModelService
from app.services.preprocessing_service import PreprocessingService
from app.services.storage_service import StorageService


@lru_cache
def get_preprocessing_service() -> PreprocessingService:
    return PreprocessingService(get_settings())


@lru_cache
def get_model_service() -> ModelService:
    return ModelService()


@lru_cache
def get_storage_service() -> StorageService:
    return StorageService(get_settings())
