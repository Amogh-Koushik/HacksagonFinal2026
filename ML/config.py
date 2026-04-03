"""
RiskScope AI - Configuration Settings
ML model hyperparameters, feature definitions, and constants
"""

# =============================================================================
# ESI LEVEL DEFINITIONS
# =============================================================================
ESI_LEVELS = {
    1: "Immediate - Life Threatening",
    2: "Emergent - High Risk",
    3: "Urgent - Moderate Risk", 
    4: "Less Urgent - Low Risk",
    5: "Non-Urgent - Minimal Risk"
}

ESI_COLORS = {
    1: "#dc2626",  # Red
    2: "#ea580c",  # Orange
    3: "#eab308",  # Yellow
    4: "#22c55e",  # Green
    5: "#3b82f6"   # Blue
}

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
LIGHTGBM_PARAMS = {
    'objective': 'multiclass',
    'num_class': 5,
    'max_depth': 6,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'n_estimators': 200,
    'class_weight': 'balanced',
    'random_state': 42,
    'verbose': -1,
    'force_col_wise': True,
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 8,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1,
}

# Ensemble weights (LightGBM 60%, Random Forest 40%)
ENSEMBLE_WEIGHTS = {
    'lightgbm': 0.6,
    'random_forest': 0.4
}

# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================
DEMOGRAPHIC_FEATURES = ['age', 'gender']

VITAL_FEATURES = [
    'heart_rate',
    'bp_systolic', 
    'bp_diastolic',
    'spo2',
    'temperature',
    'respiratory_rate'
]

SYMPTOM_FEATURES = [
    'chest_pain',
    'arm_pain_left',
    'jaw_pain',
    'dyspnea',
    'shortness_of_breath',
    'facial_droop',
    'arm_weakness',
    'speech_difficulty',
    'abdominal_pain',
    'rigid_abdomen',
    'altered_mental_status',
    'confusion',
    'fever',
    'nausea',
    'vomiting',
    'dizziness',
    'syncope',
    'headache',
    'seizure',
    'uncontrolled_bleeding',
    'severe_pain'
]

CHIEF_COMPLAINTS = [
    'chest_pain',
    'shortness_of_breath',
    'abdominal_pain',
    'headache',
    'weakness',
    'dizziness',
    'fever',
    'cough',
    'nausea_vomiting',
    'back_pain',
    'extremity_pain',
    'laceration',
    'fall',
    'altered_mental_status',
    'other'
]

# =============================================================================
# RISK SCORE THRESHOLDS
# =============================================================================
VITAL_NORMAL_RANGES = {
    'heart_rate': {'low': 60, 'high': 100},
    'bp_systolic': {'low': 90, 'high': 140},
    'bp_diastolic': {'low': 60, 'high': 90},
    'spo2': {'low': 95, 'high': 100},
    'temperature': {'low': 36.1, 'high': 37.2},
    'respiratory_rate': {'low': 12, 'high': 20}
}

# Critical thresholds that trigger immediate escalation
CRITICAL_THRESHOLDS = {
    'spo2_critical': 90,
    'bp_systolic_low': 90,
    'bp_systolic_high': 180,
    'heart_rate_low': 40,
    'heart_rate_high': 150,
    'temperature_high': 39.0,
    'respiratory_rate_high': 30
}

# =============================================================================
# CONFIDENCE SETTINGS
# =============================================================================
CONFIDENCE_THRESHOLD = 0.6  # Below this, escalate one level
OOD_THRESHOLD = 0.8  # Out-of-distribution score threshold

# =============================================================================
# MODEL PATHS
# =============================================================================
MODEL_SAVE_PATH = "models/"
LIGHTGBM_MODEL_FILE = "lightgbm_model.pkl"
RF_MODEL_FILE = "random_forest_model.pkl"
SCALER_FILE = "scaler.pkl"
ENCODER_FILE = "label_encoder.pkl"
ENSEMBLE_MODEL_FILE = "esi_ensemble_model.pkl"

# =============================================================================
# VALIDATION TARGETS
# =============================================================================
TARGET_METRICS = {
    'cohens_kappa': 0.75,  # Excellent agreement with nurses
    'esi_12_sensitivity': 0.95,  # Must catch 95% of emergencies
    'esi_12_specificity': 0.75,  # Avoid over-triage
    'response_time_ms': 500  # <500ms inference
}
