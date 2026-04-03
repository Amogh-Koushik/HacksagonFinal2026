# RiskScope AI - Machine Learning Module
# Core ML pipeline for Emergency Severity Index (ESI) prediction

__version__ = "1.0.0"
__author__ = "RiskScope AI Team"

from .esi_predictor import ESITriagePredictor
from .safety_engine import ClinicalSafetyEngine, EmergencyRuleEngine, patient_to_raw_features
from .feature_engineering import FeatureEngineer, engineer_features
from .ood_detector import OutOfDistributionDetector, detect_unusual_vitals
from .evaluate import ESIEvaluator, run_evaluation
from .config import (
    ESI_LEVELS, ESI_COLORS, TARGET_METRICS,
    LIGHTGBM_PARAMS, RANDOM_FOREST_PARAMS, ENSEMBLE_WEIGHTS
)

__all__ = [
    # Core classes
    'ESITriagePredictor',
    'ClinicalSafetyEngine',
    'EmergencyRuleEngine',
    'FeatureEngineer',
    'OutOfDistributionDetector',
    'ESIEvaluator',
    
    # Functions
    'engineer_features',
    'detect_unusual_vitals',
    'run_evaluation',
    'patient_to_raw_features',
    
    # Config
    'ESI_LEVELS',
    'ESI_COLORS',
    'TARGET_METRICS',
]

