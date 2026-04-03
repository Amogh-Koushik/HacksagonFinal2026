"""
RiskScope AI - Feature Engineering Module
Transforms raw patient data into ML-ready features including:
- Temporal features (symptom duration, onset)
- Clinical interaction terms (red flag combinations)
- Risk scores (SIRS, qSOFA, NEWS)
- Vital sign derangement indicators
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

from config import (
    VITAL_FEATURES, SYMPTOM_FEATURES, DEMOGRAPHIC_FEATURES,
    CHIEF_COMPLAINTS, VITAL_NORMAL_RANGES, CRITICAL_THRESHOLDS
)


class FeatureEngineer:
    """
    Feature engineering pipeline for ESI triage prediction.
    Transforms raw patient data into 73+ ML features.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'FeatureEngineer':
        """Fit scalers and encoders on training data."""
        # Fit scaler on numeric features
        numeric_cols = [c for c in VITAL_FEATURES if c in df.columns]
        if numeric_cols:
            self.scaler.fit(df[numeric_cols].fillna(df[numeric_cols].median()))
        
        # Fit label encoders for categorical features
        if 'chief_complaint' in df.columns:
            self.label_encoders['chief_complaint'] = LabelEncoder()
            self.label_encoders['chief_complaint'].fit(CHIEF_COMPLAINTS)
        
        if 'gender' in df.columns:
            self.label_encoders['gender'] = LabelEncoder()
            self.label_encoders['gender'].fit(['M', 'F', 'Other'])
        
        self.is_fitted = True
        return self
    
    def transform(self, data: Union[pd.DataFrame, Dict]) -> np.ndarray:
        """Transform patient data into feature vector."""
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        features = []
        self.feature_names = []
        
        # 1. Demographic features
        features.extend(self._extract_demographic_features(data))
        
        # 2. Vital sign features (raw + normalized)
        features.extend(self._extract_vital_features(data))
        
        # 3. Symptom flags (binary)
        features.extend(self._extract_symptom_features(data))
        
        # 4. Clinical interaction terms
        features.extend(self._extract_interaction_features(data))
        
        # 5. Risk scores (SIRS, qSOFA, NEWS)
        features.extend(self._extract_risk_scores(data))
        
        # 6. Vital derangement indicators
        features.extend(self._extract_derangement_features(data))
        
        # 7. Temporal features
        features.extend(self._extract_temporal_features(data))
        
        return np.array(features).reshape(1, -1)
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(df)
        return np.vstack([self.transform(row) for _, row in df.iterrows()])
    
    # =========================================================================
    # FEATURE EXTRACTION METHODS
    # =========================================================================
    
    def _extract_demographic_features(self, data) -> List[float]:
        """Extract age and gender features."""
        features = []
        
        # Helper to get value from dict or DataFrame
        def get_val(key, default):
            if isinstance(data, pd.DataFrame):
                return data[key].iloc[0] if key in data.columns else default
            elif isinstance(data, dict):
                return data.get(key, default)
            return default
        
        # Age
        age = float(get_val('age', 45))
        features.append(age)
        self.feature_names.append('age')
        
        # Age groups (one-hot)
        features.append(1.0 if age < 18 else 0.0)
        self.feature_names.append('age_pediatric')
        features.append(1.0 if 18 <= age < 65 else 0.0)
        self.feature_names.append('age_adult')
        features.append(1.0 if age >= 65 else 0.0)
        self.feature_names.append('age_elderly')
        
        # Gender encoding
        gender = get_val('gender', 'M')
        features.append(1.0 if gender == 'M' or gender == 1 else 0.0)
        self.feature_names.append('gender_male')
        
        return features
    
    def _extract_vital_features(self, data) -> List[float]:
        """Extract vital sign features (raw and normalized)."""
        features = []
        
        vital_defaults = {
            'heart_rate': 80,
            'bp_systolic': 120,
            'bp_diastolic': 80,
            'spo2': 98,
            'temperature': 37.0,
            'respiratory_rate': 16
        }
        
        # Helper to get value
        def get_val(key, default):
            if isinstance(data, pd.DataFrame):
                return data[key].iloc[0] if key in data.columns else default
            elif isinstance(data, dict):
                return data.get(key, default)
            return default
        
        for vital in VITAL_FEATURES:
            # Raw value
            val = float(get_val(vital, vital_defaults.get(vital, 0)))
            features.append(val)
            self.feature_names.append(vital)
            
            # Normalized value (z-score style using known ranges)
            ranges = VITAL_NORMAL_RANGES.get(vital, {'low': 0, 'high': 100})
            mid = (ranges['high'] + ranges['low']) / 2
            spread = (ranges['high'] - ranges['low']) / 2
            normalized = (val - mid) / spread if spread > 0 else 0
            features.append(normalized)
            self.feature_names.append(f'{vital}_normalized')
        
        # Derived vitals
        bp_sys = features[VITAL_FEATURES.index('bp_systolic') * 2]
        bp_dia = features[VITAL_FEATURES.index('bp_diastolic') * 2]
        
        # Mean Arterial Pressure
        map_val = (bp_sys + 2 * bp_dia) / 3
        features.append(map_val)
        self.feature_names.append('mean_arterial_pressure')
        
        # Pulse Pressure
        pulse_pressure = bp_sys - bp_dia
        features.append(pulse_pressure)
        self.feature_names.append('pulse_pressure')
        
        # Shock Index (HR / SBP)
        hr = features[VITAL_FEATURES.index('heart_rate') * 2]
        shock_index = hr / bp_sys if bp_sys > 0 else 0
        features.append(shock_index)
        self.feature_names.append('shock_index')
        
        return features
    
    def _extract_symptom_features(self, data) -> List[float]:
        """Extract binary symptom flags."""
        features = []
        
        # Helper to get value
        def get_val(key, default=False):
            if isinstance(data, pd.DataFrame):
                return data[key].iloc[0] if key in data.columns else default
            elif isinstance(data, dict):
                return data.get(key, default)
            return default
        
        for symptom in SYMPTOM_FEATURES:
            val = get_val(symptom, False)
            
            # Handle nested symptoms dict
            if isinstance(val, dict):
                val = val.get(symptom, False)
            
            features.append(1.0 if val else 0.0)
            self.feature_names.append(f'symptom_{symptom}')
        
        # Total symptom count
        symptom_count = sum(features[-len(SYMPTOM_FEATURES):])
        features.append(symptom_count)
        self.feature_names.append('total_symptom_count')
        
        return features
    
    def _extract_interaction_features(self, data) -> List[float]:
        """Extract clinical red flag combinations."""
        features = []
        
        def get_val(key, default=False):
            if isinstance(data, pd.DataFrame):
                return data[key].iloc[0] if key in data.columns else default
            elif isinstance(data, dict):
                return data.get(key, default)
            return default
        
        # Cardiac red flags
        chest_pain = get_val('chest_pain')
        arm_pain = get_val('arm_pain_left')
        jaw_pain = get_val('jaw_pain')
        dyspnea = get_val('dyspnea') or get_val('shortness_of_breath')
        
        features.append(1.0 if chest_pain and arm_pain else 0.0)
        self.feature_names.append('cardiac_chest_arm')
        
        features.append(1.0 if chest_pain and dyspnea else 0.0)
        self.feature_names.append('cardiac_chest_dyspnea')
        
        features.append(1.0 if chest_pain and (arm_pain or jaw_pain) else 0.0)
        self.feature_names.append('cardiac_mi_pattern')
        
        # Stroke red flags (FAST)
        facial_droop = get_val('facial_droop')
        arm_weakness = get_val('arm_weakness')
        speech_diff = get_val('speech_difficulty')
        
        fast_count = sum([facial_droop, arm_weakness, speech_diff])
        features.append(float(fast_count))
        self.feature_names.append('stroke_fast_count')
        
        features.append(1.0 if fast_count >= 2 else 0.0)
        self.feature_names.append('stroke_fast_positive')
        
        # Sepsis red flags
        fever = get_val('fever')
        confusion = get_val('altered_mental_status') or get_val('confusion')
        
        features.append(1.0 if fever and confusion else 0.0)
        self.feature_names.append('sepsis_fever_confusion')
        
        # Abdominal emergency
        abd_pain = get_val('abdominal_pain')
        rigid_abd = get_val('rigid_abdomen')
        
        features.append(1.0 if abd_pain and rigid_abd else 0.0)
        self.feature_names.append('acute_abdomen')
        
        return features
    
    def _extract_risk_scores(self, data) -> List[float]:
        """Calculate clinical risk scores: SIRS, qSOFA, Shock Index."""
        features = []
        
        def get_vital(key, default):
            if isinstance(data, pd.DataFrame):
                return float(data[key].iloc[0]) if key in data.columns else float(default)
            return float(data.get(key, default))
        
        def get_val(key, default=False):
            if isinstance(data, pd.DataFrame):
                return data[key].iloc[0] if key in data.columns else default
            return data.get(key, default)
        
        hr = get_vital('heart_rate', 80)
        temp = get_vital('temperature', 37.0)
        rr = get_vital('respiratory_rate', 16)
        sbp = get_vital('bp_systolic', 120)
        
        # SIRS Score (0-4)
        sirs_score = 0
        sirs_score += 1 if temp > 38.0 or temp < 36.0 else 0
        sirs_score += 1 if hr > 90 else 0
        sirs_score += 1 if rr > 20 else 0
        # WBC not available, skip
        
        features.append(float(sirs_score))
        self.feature_names.append('sirs_score')
        features.append(1.0 if sirs_score >= 2 else 0.0)
        self.feature_names.append('sirs_positive')
        
        # qSOFA Score (0-3)
        qsofa = 0
        qsofa += 1 if rr >= 22 else 0
        qsofa += 1 if sbp <= 100 else 0
        # GCS not available, using altered mental status
        ams = get_val('altered_mental_status', False)
        qsofa += 1 if ams else 0
        
        features.append(float(qsofa))
        self.feature_names.append('qsofa_score')
        features.append(1.0 if qsofa >= 2 else 0.0)
        self.feature_names.append('qsofa_positive')
        
        # Modified Early Warning Score (simplified NEWS)
        news = 0
        # Respiratory rate
        if rr <= 8 or rr >= 25:
            news += 3
        elif rr <= 11 or rr >= 21:
            news += 2
        elif rr <= 12 or rr >= 20:
            news += 1
        
        # SpO2
        spo2 = get_vital('spo2', 98)
        if spo2 <= 91:
            news += 3
        elif spo2 <= 93:
            news += 2
        elif spo2 <= 95:
            news += 1
        
        # Systolic BP
        if sbp <= 90 or sbp >= 220:
            news += 3
        elif sbp <= 100 or sbp >= 200:
            news += 2
        elif sbp <= 110:
            news += 1
        
        # Heart rate
        if hr <= 40 or hr >= 131:
            news += 3
        elif hr <= 50 or hr >= 111:
            news += 2
        elif hr <= 60 or hr >= 91:
            news += 1
        
        # Temperature
        if temp <= 35.0 or temp >= 39.1:
            news += 2
        elif temp <= 36.0 or temp >= 38.1:
            news += 1
        
        features.append(float(news))
        self.feature_names.append('news_score')
        features.append(1.0 if news >= 5 else 0.0)
        self.feature_names.append('news_high_risk')
        
        return features
    
    def _extract_derangement_features(self, data) -> List[float]:
        """Extract vital sign derangement indicators."""
        features = []
        
        def get_vital(key, default):
            if isinstance(data, pd.DataFrame):
                return float(data[key].iloc[0]) if key in data.columns else float(default)
            return float(data.get(key, default))
        
        hr = get_vital('heart_rate', 80)
        sbp = get_vital('bp_systolic', 120)
        dbp = get_vital('bp_diastolic', 80)
        spo2 = get_vital('spo2', 98)
        temp = get_vital('temperature', 37.0)
        rr = get_vital('respiratory_rate', 16)
        
        # Critical derangements
        features.append(1.0 if spo2 < CRITICAL_THRESHOLDS['spo2_critical'] else 0.0)
        self.feature_names.append('critical_hypoxia')
        
        features.append(1.0 if sbp < CRITICAL_THRESHOLDS['bp_systolic_low'] else 0.0)
        self.feature_names.append('critical_hypotension')
        
        features.append(1.0 if sbp > CRITICAL_THRESHOLDS['bp_systolic_high'] else 0.0)
        self.feature_names.append('critical_hypertension')
        
        features.append(1.0 if hr < CRITICAL_THRESHOLDS['heart_rate_low'] else 0.0)
        self.feature_names.append('critical_bradycardia')
        
        features.append(1.0 if hr > CRITICAL_THRESHOLDS['heart_rate_high'] else 0.0)
        self.feature_names.append('critical_tachycardia')
        
        features.append(1.0 if temp > CRITICAL_THRESHOLDS['temperature_high'] else 0.0)
        self.feature_names.append('critical_fever')
        
        features.append(1.0 if rr > CRITICAL_THRESHOLDS['respiratory_rate_high'] else 0.0)
        self.feature_names.append('critical_tachypnea')
        
        # Tachycardia with hypotension (shock sign)
        features.append(1.0 if hr > 100 and sbp < 90 else 0.0)
        self.feature_names.append('shock_pattern')
        
        # Count of critical derangements
        critical_count = sum(features[-8:])
        features.append(critical_count)
        self.feature_names.append('critical_derangement_count')
        
        return features
    
    def _extract_temporal_features(self, data) -> List[float]:
        """Extract temporal features (symptom duration, onset)."""
        features = []
        
        def get_val(key, default):
            if isinstance(data, pd.DataFrame):
                return data[key].iloc[0] if key in data.columns else default
            return data.get(key, default)
        
        # Symptom duration in hours
        duration = float(get_val('symptom_duration_hours', 2.0))
        features.append(duration)
        self.feature_names.append('symptom_duration_hours')
        
        # Acute vs chronic
        features.append(1.0 if duration < 24 else 0.0)
        self.feature_names.append('is_acute')
        
        features.append(1.0 if duration < 2 else 0.0)
        self.feature_names.append('is_rapid_onset')
        
        features.append(1.0 if duration > 168 else 0.0)  # >1 week
        self.feature_names.append('is_chronic')
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names.copy()
    
    def save(self, path: str):
        """Save fitted transformers."""
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'FeatureEngineer':
        """Load fitted transformers."""
        data = joblib.load(path)
        fe = cls()
        fe.scaler = data['scaler']
        fe.label_encoders = data['label_encoders']
        fe.feature_names = data['feature_names']
        fe.is_fitted = data['is_fitted']
        return fe


def engineer_features(patient_data: Dict[str, Any]) -> np.ndarray:
    """
    Convenience function to transform a single patient's data.
    Used by the API endpoint.
    """
    fe = FeatureEngineer()
    return fe.transform(patient_data)
