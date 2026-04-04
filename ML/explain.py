"""
RiskScope AI - Explainability Engine
=====================================
Wraps SHAP TreeExplainer to produce human-readable feature explanations
with clinical context for every ESI prediction.

Used by:
  - ClinicalSafetyEngine  (in safety_engine.py)
  - Flask API             (in backend/app.py)
  - Doctor Dashboard      (via /api/predict response)
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union

# ---------------------------------------------------------------------------
# Human-readable feature name mapping
# ---------------------------------------------------------------------------
FEATURE_DISPLAY_NAMES = {
    # Demographics
    'age': 'Patient Age',
    'gender': 'Gender',
    'age_bucket': 'Age Risk Group',
    'age_pediatric': 'Pediatric Patient',
    'age_adult': 'Adult Patient',
    'age_elderly': 'Elderly Patient',
    'gender_male': 'Male Patient',

    # Vitals
    'heart_rate': 'Heart Rate',
    'bp_systolic': 'Systolic Blood Pressure',
    'bp_diastolic': 'Diastolic Blood Pressure',
    'spo2': 'Oxygen Saturation (SpO2)',
    'temperature': 'Body Temperature',
    'respiratory_rate': 'Respiratory Rate',
    'map_pressure': 'Mean Arterial Pressure',
    'mean_arterial_pressure': 'Mean Arterial Pressure',
    'pulse_pressure': 'Pulse Pressure',
    'shock_index': 'Shock Index (HR/SBP)',

    # Normalized vitals
    'heart_rate_normalized': 'Heart Rate Deviation',
    'bp_systolic_normalized': 'BP Systolic Deviation',
    'bp_diastolic_normalized': 'BP Diastolic Deviation',
    'spo2_normalized': 'SpO2 Deviation',
    'temperature_normalized': 'Temperature Deviation',
    'respiratory_rate_normalized': 'Respiratory Rate Deviation',

    # Symptoms
    'chest_pain': 'Chest Pain',
    'arm_pain_left': 'Left Arm Pain',
    'jaw_pain': 'Jaw Pain',
    'dyspnea': 'Difficulty Breathing',
    'shortness_of_breath': 'Shortness of Breath',
    'facial_droop': 'Facial Droop',
    'arm_weakness': 'Arm Weakness',
    'speech_difficulty': 'Speech Difficulty',
    'abdominal_pain': 'Abdominal Pain',
    'rigid_abdomen': 'Rigid Abdomen',
    'altered_mental_status': 'Altered Mental Status',
    'confusion': 'Confusion',
    'fever': 'Fever',
    'nausea': 'Nausea',
    'vomiting': 'Vomiting',
    'dizziness': 'Dizziness',
    'syncope': 'Fainting/Syncope',
    'headache': 'Headache',
    'seizure': 'Seizure',
    'uncontrolled_bleeding': 'Uncontrolled Bleeding',
    'severe_pain': 'Severe Pain',
    'cough': 'Cough',
    'fatigue': 'Fatigue',

    # Symptom prefixed variants
    'symptom_chest_pain': 'Chest Pain',
    'symptom_arm_pain_left': 'Left Arm Pain',
    'symptom_jaw_pain': 'Jaw Pain',
    'symptom_dyspnea': 'Difficulty Breathing',
    'symptom_shortness_of_breath': 'Shortness of Breath',
    'symptom_facial_droop': 'Facial Droop',
    'symptom_arm_weakness': 'Arm Weakness',
    'symptom_speech_difficulty': 'Speech Difficulty',
    'symptom_abdominal_pain': 'Abdominal Pain',
    'symptom_rigid_abdomen': 'Rigid Abdomen',
    'symptom_altered_mental_status': 'Altered Mental Status',
    'symptom_confusion': 'Confusion',
    'symptom_fever': 'Fever',
    'symptom_nausea': 'Nausea',
    'symptom_vomiting': 'Vomiting',
    'symptom_dizziness': 'Dizziness',
    'symptom_syncope': 'Fainting/Syncope',
    'symptom_headache': 'Headache',
    'symptom_seizure': 'Seizure',
    'symptom_uncontrolled_bleeding': 'Uncontrolled Bleeding',
    'symptom_severe_pain': 'Severe Pain',
    'symptom_cough': 'Cough',
    'symptom_fatigue': 'Fatigue',

    # Derived / engineered
    'symptom_count': 'Total Symptom Count',
    'total_symptom_count': 'Total Symptom Count',
    'vital_abnormality_count': 'Number of Abnormal Vitals',
    'has_critical_vital': 'Has Critical Vital Sign',
    'symptom_duration_hours': 'Symptom Duration',

    # Interaction terms
    'cardiac_chest_arm': 'Cardiac Pattern (Chest + Arm Pain)',
    'cardiac_chest_dyspnea': 'Cardiac Pattern (Chest Pain + Dyspnea)',
    'cardiac_mi_pattern': 'MI Pattern (Chest + Arm/Jaw Pain)',
    'stroke_fast_count': 'Stroke FAST Score',
    'stroke_fast_positive': 'Stroke FAST Positive',
    'sepsis_fever_confusion': 'Sepsis Pattern (Fever + Confusion)',
    'acute_abdomen': 'Acute Abdomen (Pain + Guarding)',

    # Risk scores
    'sirs_score': 'SIRS Score',
    'sirs_positive': 'SIRS Criteria Met',
    'qsofa_score': 'qSOFA Score',
    'qsofa_positive': 'qSOFA Criteria Met',
    'news_score': 'NEWS Score',
    'news_high_risk': 'NEWS High Risk',

    # Derangement indicators
    'critical_hypoxia': 'Critical Low Oxygen',
    'critical_hypotension': 'Critical Low Blood Pressure',
    'critical_hypertension': 'Critical High Blood Pressure',
    'critical_bradycardia': 'Critical Slow Heart Rate',
    'critical_tachycardia': 'Critical Fast Heart Rate',
    'critical_fever': 'Critical High Temperature',
    'critical_tachypnea': 'Critical Fast Breathing',
    'shock_pattern': 'Shock Pattern Detected',
    'critical_derangement_count': 'Number of Critical Derangements',

    # Temporal
    'is_acute': 'Acute Onset (<24h)',
    'is_rapid_onset': 'Rapid Onset (<2h)',
    'is_chronic': 'Chronic (>1 week)',
}

# Clinical context for key features (what the feature means clinically)
CLINICAL_CONTEXT = {
    'heart_rate': {
        'high': 'Tachycardia (HR >{val}) may indicate cardiac distress, pain, dehydration, or shock',
        'low': 'Bradycardia (HR <{val}) may indicate heart block or medication effect',
        'threshold_high': 100,
        'threshold_low': 60,
        'unit': 'bpm',
    },
    'bp_systolic': {
        'high': 'Hypertension (SBP >{val}) increases risk of stroke and cardiac events',
        'low': 'Hypotension (SBP <{val}) may indicate shock or hemorrhage',
        'threshold_high': 140,
        'threshold_low': 90,
        'unit': 'mmHg',
    },
    'spo2': {
        'low': 'Hypoxemia (SpO2 <{val}%) requires immediate oxygen supplementation',
        'threshold_low': 94,
        'unit': '%',
    },
    'temperature': {
        'high': 'Fever (>{val}°C) suggests infection or inflammatory response',
        'low': 'Hypothermia (<{val}°C) may indicate exposure or sepsis',
        'threshold_high': 38.0,
        'threshold_low': 36.0,
        'unit': '°C',
    },
    'respiratory_rate': {
        'high': 'Tachypnea (RR >{val}) indicates respiratory distress',
        'low': 'Bradypnea (RR <{val}) may indicate CNS depression',
        'threshold_high': 20,
        'threshold_low': 12,
        'unit': '/min',
    },
    'shock_index': {
        'high': 'Elevated shock index (>{val}) indicates hemodynamic instability',
        'threshold_high': 0.9,
        'unit': '',
    },
    'chest_pain': {
        'present': 'Chest pain requires cardiac evaluation (EKG, troponin)',
    },
    'facial_droop': {
        'present': 'Facial droop is a key FAST stroke criterion',
    },
    'arm_weakness': {
        'present': 'Arm weakness is a key FAST stroke criterion',
    },
    'speech_difficulty': {
        'present': 'Speech difficulty is a key FAST stroke criterion',
    },
    'altered_mental_status': {
        'present': 'Altered mental status suggests serious neurological or metabolic compromise',
    },
    'uncontrolled_bleeding': {
        'present': 'Active hemorrhage requires immediate hemostasis',
    },
}


class ExplainabilityEngine:
    """
    Produces human-readable explanations for ESI predictions.

    Uses SHAP values from the trained LightGBM model, with a fallback
    to feature importance when SHAP is unavailable.
    """

    def __init__(self, predictor=None):
        """
        Parameters
        ----------
        predictor : ESITriagePredictor, optional
            Trained predictor with SHAP explainer. If None, only
            feature-importance-based explanations are available.
        """
        self.predictor = predictor
        self.feature_names = predictor.feature_names if predictor else []

    def explain(self, features: np.ndarray,
                patient_data: Dict[str, Any] = None,
                predicted_esi: int = None,
                top_n: int = 5) -> Dict[str, Any]:
        """
        Generate a complete explanation for a prediction.

        Parameters
        ----------
        features : np.ndarray
            Feature vector (1 x n_features) as passed to the model.
        patient_data : dict, optional
            Original patient data dict (for clinical context).
        predicted_esi : int, optional
            The predicted ESI level (1-5).
        top_n : int
            Number of top contributing features to return.

        Returns
        -------
        explanation : dict with keys:
            - top_features : List[dict]  — top contributing features
            - risk_summary : str         — one-line risk summary
            - clinical_notes : List[str] — clinical context notes
            - confidence_factors : dict  — prediction confidence breakdown
        """
        # Get SHAP-based or fallback feature contributions
        contributions = self._get_feature_contributions(features, top_n)

        # Add human-readable names and clinical context
        top_features = []
        clinical_notes = []

        for contrib in contributions:
            feature_name = contrib['feature']
            display_name = FEATURE_DISPLAY_NAMES.get(
                feature_name, feature_name.replace('_', ' ').title()
            )

            # Get actual value from patient data if available
            actual_value = None
            if patient_data:
                actual_value = patient_data.get(feature_name)

            # Format the value string
            value_str = self._format_feature_value(
                feature_name, actual_value, contrib['impact']
            )

            entry = {
                'feature': feature_name,
                'display_name': display_name,
                'impact': contrib['impact'],
                'impact_direction': 'increases urgency' if contrib['impact'] == '+' else 'decreases urgency',
                'shap_value': contrib['value'],
                'actual_value': actual_value,
                'value_display': value_str,
            }
            top_features.append(entry)

            # Add clinical context note if available
            note = self._get_clinical_note(feature_name, actual_value)
            if note:
                clinical_notes.append(note)

        # Build risk summary
        risk_summary = self._build_risk_summary(
            top_features, predicted_esi, patient_data
        )

        return {
            'top_features': top_features,
            'risk_summary': risk_summary,
            'clinical_notes': clinical_notes,
            'predicted_esi': predicted_esi,
        }

    def _get_feature_contributions(self, features: np.ndarray,
                                   top_n: int) -> List[Dict]:
        """Get feature contributions via SHAP or fallback."""
        if self.predictor and self.predictor.shap_explainer is not None:
            return self._shap_contributions(features, top_n)
        elif self.predictor and self.predictor.is_fitted:
            return self._importance_contributions(features, top_n)
        else:
            return self._basic_contributions(features, top_n)

    def _shap_contributions(self, features: np.ndarray,
                            top_n: int) -> List[Dict]:
        """Get contributions from SHAP TreeExplainer."""
        try:
            import shap

            X = features.reshape(1, -1) if features.ndim == 1 else features
            X_scaled = self.predictor.scaler.transform(X)

            shap_values = self.predictor.shap_explainer.shap_values(X_scaled)

            # Get predicted class
            proba = self.predictor.predict_proba(X_scaled, already_scaled=True)[0]
            pred_class = np.argmax(proba)

            # Get SHAP values for predicted class
            if isinstance(shap_values, list):
                class_shap = shap_values[pred_class][0]
            else:
                class_shap = shap_values[0]

            # Sort by absolute SHAP value
            abs_shap = np.abs(class_shap)
            top_indices = np.argsort(abs_shap)[-top_n:][::-1]

            contributions = []
            for idx in top_indices:
                name = (self.feature_names[idx]
                        if idx < len(self.feature_names)
                        else f'feature_{idx}')
                contributions.append({
                    'feature': name,
                    'impact': '+' if class_shap[idx] > 0 else '-',
                    'value': round(float(abs(class_shap[idx])), 4),
                })

            return contributions

        except Exception as e:
            print(f"  SHAP explanation failed ({e}), using fallback")
            return self._importance_contributions(features, top_n)

    def _importance_contributions(self, features: np.ndarray,
                                  top_n: int) -> List[Dict]:
        """Fallback: use LightGBM feature importance."""
        importances = self.predictor.lgb_model.feature_importances_
        top_indices = np.argsort(importances)[-top_n:][::-1]

        contributions = []
        for idx in top_indices:
            name = (self.feature_names[idx]
                    if idx < len(self.feature_names)
                    else f'feature_{idx}')

            # Use feature value to determine impact direction
            flat = features.flatten()
            val = flat[idx] if idx < len(flat) else 0

            contributions.append({
                'feature': name,
                'impact': '+' if val > 0 else '-',
                'value': round(float(importances[idx] / importances.max()), 4),
            })

        return contributions

    def _basic_contributions(self, features: np.ndarray,
                             top_n: int) -> List[Dict]:
        """Last resort: return features with highest absolute values."""
        flat = features.flatten()
        top_indices = np.argsort(np.abs(flat))[-top_n:][::-1]

        return [
            {
                'feature': f'feature_{idx}',
                'impact': '+' if flat[idx] > 0 else '-',
                'value': round(float(abs(flat[idx])), 4),
            }
            for idx in top_indices
        ]

    def _format_feature_value(self, feature_name: str,
                              actual_value, impact: str) -> str:
        """Format a feature value for display."""
        context = CLINICAL_CONTEXT.get(feature_name)
        if actual_value is not None and context:
            unit = context.get('unit', '')
            return f"{actual_value} {unit}".strip()

        if actual_value is not None:
            if isinstance(actual_value, (int, float)):
                if actual_value == 0 or actual_value == 1:
                    return "Present" if actual_value == 1 else "Absent"
                return str(round(actual_value, 2))
            return str(actual_value)

        return "Contributing factor" if impact == '+' else "Protective factor"

    def _get_clinical_note(self, feature_name: str,
                           actual_value) -> Optional[str]:
        """Get clinical context note for a feature."""
        context = CLINICAL_CONTEXT.get(feature_name)
        if not context:
            return None

        # Binary symptoms
        if 'present' in context and actual_value:
            return context['present']

        # Continuous vitals
        if actual_value is not None and isinstance(actual_value, (int, float)):
            high_thresh = context.get('threshold_high')
            low_thresh = context.get('threshold_low')

            if high_thresh and actual_value > high_thresh and 'high' in context:
                return context['high'].format(val=high_thresh)
            if low_thresh and actual_value < low_thresh and 'low' in context:
                return context['low'].format(val=low_thresh)

        return None

    def _build_risk_summary(self, top_features: List[Dict],
                            predicted_esi: int,
                            patient_data: Dict = None) -> str:
        """Build a one-line risk summary for the prediction."""
        if not top_features:
            return "Insufficient data for risk assessment"

        # Get top increasing-urgency features
        risk_factors = [f['display_name'] for f in top_features
                        if f['impact'] == '+']

        if predicted_esi and predicted_esi <= 2:
            severity = "HIGH RISK"
        elif predicted_esi and predicted_esi == 3:
            severity = "MODERATE RISK"
        else:
            severity = "LOW RISK"

        if risk_factors:
            factors_str = ', '.join(risk_factors[:3])
            return f"{severity}: Primary factors — {factors_str}"
        else:
            return f"{severity}: No major risk factors identified"

    def get_waterfall_data(self, features: np.ndarray) -> Optional[Dict]:
        """
        Get SHAP waterfall chart data for visualization.

        Returns dict with:
          - base_value : float (expected value)
          - feature_names : list
          - shap_values : list
          - feature_values : list
        """
        if not self.predictor or self.predictor.shap_explainer is None:
            return None

        try:
            import shap

            X = features.reshape(1, -1) if features.ndim == 1 else features
            X_scaled = self.predictor.scaler.transform(X)

            shap_values = self.predictor.shap_explainer.shap_values(X_scaled)

            proba = self.predictor.predict_proba(X_scaled, already_scaled=True)[0]
            pred_class = np.argmax(proba)

            if isinstance(shap_values, list):
                class_shap = shap_values[pred_class][0]
            else:
                class_shap = shap_values[0]

            names = self.feature_names if self.feature_names else [
                f'feature_{i}' for i in range(len(class_shap))
            ]
            display_names = [
                FEATURE_DISPLAY_NAMES.get(n, n.replace('_', ' ').title())
                for n in names
            ]

            return {
                'base_value': float(self.predictor.shap_explainer.expected_value[pred_class]
                                    if isinstance(self.predictor.shap_explainer.expected_value, (list, np.ndarray))
                                    else self.predictor.shap_explainer.expected_value),
                'feature_names': display_names,
                'shap_values': [round(float(v), 4) for v in class_shap],
                'feature_values': [round(float(v), 4) for v in X_scaled.flatten()],
                'predicted_class': int(pred_class),
            }

        except Exception:
            return None
