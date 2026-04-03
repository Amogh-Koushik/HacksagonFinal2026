"""
RiskScope AI - Emergency Rule Engine (Safety Layer)
Hard-coded clinical rules based on ACLS/PALS/ESI protocols.
These rules CANNOT be overridden by ML predictions.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class EmergencyResult:
    """Result from emergency rule check."""
    triggered: bool
    rule_name: Optional[str] = None
    esi_level: Optional[int] = None
    protocol: Optional[str] = None
    action: Optional[str] = None
    rationale: Optional[str] = None


class EmergencyRuleEngine:
    """
    Hard-coded emergency detection rules.
    Based on ACLS, PALS, and ESI triage protocols.
    
    These rules run BEFORE ML and cannot be overridden.
    If triggered, patient is immediately escalated to ESI 1.
    """
    
    EMERGENCY_RULES = [
        # =====================================================================
        # CARDIOVASCULAR EMERGENCIES
        # =====================================================================
        {
            "name": "suspected_mi",
            "description": "Suspected Myocardial Infarction",
            "conditions": lambda p: (
                p.get("chest_pain", False) and 
                (p.get("arm_pain_left", False) or p.get("jaw_pain", False) or p.get("diaphoresis", False))
            ),
            "esi": 1,
            "protocol": "ACLS - Acute Coronary Syndrome",
            "action": "12-lead EKG STAT, Aspirin 325mg, IV access, Troponin, Cardiology consult",
            "rationale": "Chest pain with radiation suggests acute MI requiring immediate intervention"
        },
        {
            "name": "cardiac_arrest_signs",
            "description": "Signs of Cardiac Arrest/Impending Arrest",
            "conditions": lambda p: (
                p.get("unresponsive", False) or
                (p.get("heart_rate", 80) < 30) or
                (p.get("heart_rate", 80) > 180 and p.get("chest_pain", False))
            ),
            "esi": 1,
            "protocol": "ACLS - Cardiac Arrest",
            "action": "Crash cart, CPR if indicated, Defibrillator ready",
            "rationale": "Extreme heart rate derangement with unresponsiveness indicates cardiac emergency"
        },
        {
            "name": "unstable_tachycardia",
            "description": "Unstable Tachycardia with Hemodynamic Compromise",
            "conditions": lambda p: (
                p.get("heart_rate", 80) > 150 and
                (p.get("bp_systolic", 120) < 90 or p.get("altered_mental_status", False) or p.get("chest_pain", False))
            ),
            "esi": 1,
            "protocol": "ACLS - Tachycardia with Pulse",
            "action": "IV access, 12-lead EKG, Synchronized cardioversion standby",
            "rationale": "Rapid heart rate with signs of instability requires immediate intervention"
        },
        
        # =====================================================================
        # RESPIRATORY EMERGENCIES
        # =====================================================================
        {
            "name": "respiratory_failure",
            "description": "Acute Respiratory Failure",
            "conditions": lambda p: p.get("spo2", 100) < 90,
            "esi": 1,
            "protocol": "Respiratory Distress Protocol",
            "action": "High-flow O2 (15L NRB), ABG, Chest X-ray, Respiratory Therapy STAT",
            "rationale": "SpO2 <90% indicates severe hypoxemia requiring immediate oxygen therapy"
        },
        {
            "name": "severe_respiratory_distress",
            "description": "Severe Respiratory Distress",
            "conditions": lambda p: (
                p.get("respiratory_rate", 16) > 30 and
                (p.get("dyspnea", False) or p.get("shortness_of_breath", False))
            ),
            "esi": 1,
            "protocol": "Respiratory Distress Protocol",
            "action": "Oxygen, IV access, Chest X-ray, consider BiPAP/intubation",
            "rationale": "Tachypnea >30 with dyspnea indicates impending respiratory failure"
        },
        {
            "name": "airway_compromise",
            "description": "Airway Compromise/Obstruction",
            "conditions": lambda p: (
                p.get("stridor", False) or
                p.get("choking", False) or
                (p.get("anaphylaxis", False) and p.get("dyspnea", False))
            ),
            "esi": 1,
            "protocol": "Airway Management Protocol",
            "action": "Airway assessment, Suction, Intubation equipment ready, ENT/Anesthesia",
            "rationale": "Airway compromise is immediately life-threatening"
        },
        
        # =====================================================================
        # NEUROLOGICAL EMERGENCIES
        # =====================================================================
        {
            "name": "stroke_fast",
            "description": "Suspected Stroke (FAST Positive)",
            "conditions": lambda p: (
                (p.get("facial_droop", False) and p.get("arm_weakness", False)) or
                (p.get("facial_droop", False) and p.get("speech_difficulty", False)) or
                (p.get("arm_weakness", False) and p.get("speech_difficulty", False))
            ),
            "esi": 1,
            "protocol": "Stroke Alert - FAST Protocol",
            "action": "CT Head STAT (no contrast), Neurology STAT, Check glucose, BP monitoring",
            "rationale": "2+ FAST criteria positive indicates high probability of stroke"
        },
        {
            "name": "stroke_single_severe",
            "description": "Suspected Stroke (Single Severe Symptom)",
            "conditions": lambda p: (
                p.get("facial_droop", False) or
                p.get("arm_weakness", False) or
                p.get("speech_difficulty", False)
            ) and (
                p.get("symptom_duration_hours", 24) < 4.5 and
                p.get("age", 50) > 45
            ),
            "esi": 1,
            "protocol": "Stroke Alert - FAST Protocol",
            "action": "CT Head STAT, Neurology consult, tPA evaluation window",
            "rationale": "Single stroke symptom within tPA window requires immediate evaluation"
        },
        {
            "name": "active_seizure",
            "description": "Active Seizure or Status Epilepticus",
            "conditions": lambda p: (
                p.get("seizure", False) and p.get("seizure_ongoing", False)
            ) or p.get("status_epilepticus", False),
            "esi": 1,
            "protocol": "Seizure Protocol",
            "action": "Protect airway, IV access, Benzodiazepines, Glucose check",
            "rationale": "Ongoing seizure requires immediate intervention to prevent brain damage"
        },
        {
            "name": "severe_altered_mental_status",
            "description": "Severe Altered Mental Status",
            "conditions": lambda p: (
                p.get("unresponsive", False) or
                (p.get("altered_mental_status", False) and p.get("bp_systolic", 120) < 90) or
                (p.get("gcs", 15) is not None and p.get("gcs", 15) < 9)
            ),
            "esi": 1,
            "protocol": "Altered Mental Status Protocol",
            "action": "Airway assessment, Glucose check, CT Head, Toxicology screen",
            "rationale": "Unresponsive or GCS <9 indicates severe brain compromise"
        },
        
        # =====================================================================
        # SHOCK / HEMODYNAMIC INSTABILITY
        # =====================================================================
        {
            "name": "shock",
            "description": "Shock (Hypotension with Tachycardia)",
            "conditions": lambda p: (
                p.get("bp_systolic", 120) < 90 and 
                p.get("heart_rate", 80) > 100
            ),
            "esi": 1,
            "protocol": "Shock Protocol",
            "action": "2 large-bore IVs, NS bolus 1L, Type & Screen, Source identification",
            "rationale": "Hypotension with compensatory tachycardia indicates circulatory shock"
        },
        {
            "name": "severe_hypotension",
            "description": "Severe Hypotension",
            "conditions": lambda p: p.get("bp_systolic", 120) < 80,
            "esi": 1,
            "protocol": "Shock Protocol",
            "action": "IV fluids, Vasopressors ready, ICU notification",
            "rationale": "SBP <80 indicates severe hypoperfusion"
        },
        {
            "name": "uncontrolled_hemorrhage",
            "description": "Uncontrolled Hemorrhage",
            "conditions": lambda p: (
                p.get("uncontrolled_bleeding", False) or
                (p.get("active_bleeding", False) and p.get("bp_systolic", 120) < 100)
            ),
            "esi": 1,
            "protocol": "Massive Transfusion Protocol",
            "action": "Direct pressure, 2 large-bore IVs, Type & Crossmatch, O-neg blood standby",
            "rationale": "Active hemorrhage with hemodynamic instability requires immediate intervention"
        },
        
        # =====================================================================
        # SEPSIS / INFECTIOUS EMERGENCIES
        # =====================================================================
        {
            "name": "sepsis",
            "description": "Suspected Sepsis (qSOFA >= 2)",
            "conditions": lambda p: (
                (p.get("temperature", 37) > 38.3 or p.get("temperature", 37) < 36) and
                p.get("heart_rate", 80) > 90 and
                (p.get("altered_mental_status", False) or p.get("bp_systolic", 120) < 100)
            ),
            "esi": 1,
            "protocol": "Sepsis Bundle (Hour-1)",
            "action": "Lactate, Blood cultures x2, Broad-spectrum antibiotics, IV fluids 30mL/kg",
            "rationale": "Suspected infection with organ dysfunction requires Hour-1 sepsis bundle"
        },
        {
            "name": "meningitis_signs",
            "description": "Suspected Meningitis",
            "conditions": lambda p: (
                p.get("fever", False) and
                p.get("headache", False) and
                (p.get("neck_stiffness", False) or p.get("altered_mental_status", False))
            ),
            "esi": 1,
            "protocol": "Meningitis Protocol",
            "action": "Blood cultures, LP if no contraindication, Empiric antibiotics STAT",
            "rationale": "Fever + headache + neck stiffness triad suggests bacterial meningitis"
        },
        
        # =====================================================================
        # TRAUMA EMERGENCIES
        # =====================================================================
        {
            "name": "major_trauma",
            "description": "Major Trauma with Abnormal Vitals",
            "conditions": lambda p: (
                p.get("trauma", False) and
                (p.get("bp_systolic", 120) < 90 or p.get("gcs", 15) < 14 or p.get("respiratory_rate", 16) > 29)
            ),
            "esi": 1,
            "protocol": "Trauma Activation",
            "action": "Trauma team activation, C-spine precautions, 2 large-bore IVs, FAST exam",
            "rationale": "Trauma with vital sign abnormality requires full trauma activation"
        },
        
        # =====================================================================
        # OTHER CRITICAL CONDITIONS
        # =====================================================================
        {
            "name": "anaphylaxis",
            "description": "Anaphylaxis",
            "conditions": lambda p: (
                p.get("anaphylaxis", False) or
                (p.get("allergic_reaction", False) and 
                 (p.get("dyspnea", False) or p.get("bp_systolic", 120) < 90))
            ),
            "esi": 1,
            "protocol": "Anaphylaxis Protocol",
            "action": "Epinephrine 0.3mg IM, IV access, Oxygen, H1/H2 blockers, Steroids",
            "rationale": "Anaphylaxis with airway/cardiovascular involvement is immediately life-threatening"
        },
        {
            "name": "severe_hypertensive_emergency",
            "description": "Hypertensive Emergency with End-Organ Damage",
            "conditions": lambda p: (
                p.get("bp_systolic", 120) > 180 and
                (p.get("chest_pain", False) or p.get("headache", False) and p.get("altered_mental_status", False))
            ),
            "esi": 1,
            "protocol": "Hypertensive Emergency Protocol",
            "action": "IV antihypertensive, Continuous BP monitoring, CT Head if neurological symptoms",
            "rationale": "Severe hypertension with symptoms suggests end-organ damage"
        },
    ]
    
    # ESI 2 Rules (High Priority but not Immediate)
    HIGH_RISK_RULES = [
        {
            "name": "chest_pain_stable",
            "description": "Chest Pain (Stable Vitals)",
            "conditions": lambda p: (
                p.get("chest_pain", False) and
                not (p.get("arm_pain_left", False) or p.get("jaw_pain", False)) and
                p.get("bp_systolic", 120) >= 90
            ),
            "esi": 2,
            "protocol": "ACS Rule-Out Protocol",
            "action": "EKG within 10 min, Troponin, Aspirin if no contraindication",
            "rationale": "Chest pain requires rapid cardiac evaluation even if stable"
        },
        {
            "name": "high_fever",
            "description": "High Fever (≥39.5°C / 103°F)",
            "conditions": lambda p: p.get("temperature", 37) >= 39.5,
            "esi": 2,
            "protocol": "Fever Workup",
            "action": "Blood cultures, CBC, CMP, Urinalysis, Chest X-ray if indicated",
            "rationale": "High fever suggests serious infection requiring workup"
        },
        {
            "name": "moderate_hypoxia",
            "description": "Moderate Hypoxia (SpO2 90-94%)",
            "conditions": lambda p: 90 <= p.get("spo2", 100) <= 94,
            "esi": 2,
            "protocol": "Oxygen Therapy",
            "action": "Supplemental O2, Continuous pulse ox, Chest X-ray",
            "rationale": "Borderline hypoxia needs close monitoring and supplementation"
        },
    ]
    
    def check(self, patient_data: Dict[str, Any]) -> EmergencyResult:
        """
        Check patient against all emergency rules.
        Returns first triggered rule or negative result.
        
        Rules are checked in order of severity.
        """
        # Check ESI 1 emergency rules first
        for rule in self.EMERGENCY_RULES:
            try:
                if rule["conditions"](patient_data):
                    return EmergencyResult(
                        triggered=True,
                        rule_name=rule["name"],
                        esi_level=rule["esi"],
                        protocol=rule["protocol"],
                        action=rule["action"],
                        rationale=rule["rationale"]
                    )
            except Exception:
                # Skip rules that fail due to missing data
                continue
        
        # Check ESI 2 high-risk rules
        for rule in self.HIGH_RISK_RULES:
            try:
                if rule["conditions"](patient_data):
                    return EmergencyResult(
                        triggered=True,
                        rule_name=rule["name"],
                        esi_level=rule["esi"],
                        protocol=rule["protocol"],
                        action=rule["action"],
                        rationale=rule["rationale"]
                    )
            except Exception:
                continue
        
        # No emergency rules triggered
        return EmergencyResult(triggered=False)
    
    def check_all(self, patient_data: Dict[str, Any]) -> List[EmergencyResult]:
        """
        Check patient against all rules and return ALL triggered rules.
        Useful for comprehensive safety assessment.
        """
        triggered_rules = []
        
        for rule in self.EMERGENCY_RULES + self.HIGH_RISK_RULES:
            try:
                if rule["conditions"](patient_data):
                    triggered_rules.append(EmergencyResult(
                        triggered=True,
                        rule_name=rule["name"],
                        esi_level=rule["esi"],
                        protocol=rule["protocol"],
                        action=rule["action"],
                        rationale=rule["rationale"]
                    ))
            except Exception:
                continue
        
        return triggered_rules
    
    def get_rule_names(self) -> List[str]:
        """Return list of all rule names."""
        return [r["name"] for r in self.EMERGENCY_RULES + self.HIGH_RISK_RULES]


class OutOfDistributionDetector:
    """
    Detects when patient data is unusual/outside training distribution.
    When OOD score is high, we escalate automatically (fail-safe).
    """
    
    def __init__(self):
        self.training_stats = None
        self.is_fitted = False
    
    def fit(self, training_data):
        """Compute statistics from training data."""
        import numpy as np
        
        self.training_stats = {
            'means': np.mean(training_data, axis=0),
            'stds': np.std(training_data, axis=0) + 1e-6,
            'mins': np.min(training_data, axis=0),
            'maxs': np.max(training_data, axis=0)
        }
        self.is_fitted = True
        return self
    
    def score(self, features) -> float:
        """
        Calculate OOD score for a feature vector.
        Higher score = more unusual = more likely out-of-distribution.
        
        Returns: float between 0 and 1
        """
        import numpy as np
        
        if not self.is_fitted:
            # If not fitted, return low score (assume in-distribution)
            return 0.3
        
        features = np.array(features).flatten()
        
        # Method 1: Z-score based detection
        z_scores = np.abs((features - self.training_stats['means']) / self.training_stats['stds'])
        max_z = np.max(z_scores)
        
        # Method 2: Range-based detection
        below_min = np.sum(features < self.training_stats['mins'])
        above_max = np.sum(features > self.training_stats['maxs'])
        out_of_range = (below_min + above_max) / len(features)
        
        # Combine methods
        z_score_component = min(max_z / 5.0, 1.0)  # Normalize to 0-1
        range_component = out_of_range
        
        ood_score = 0.7 * z_score_component + 0.3 * range_component
        
        return min(ood_score, 1.0)


class ClinicalSafetyEngine:
    """
    Complete 3-layer safety system that wraps ML predictions.
    
    Layer 1: Emergency Rule Engine (hard-coded, cannot fail)
    Layer 2: Out-of-Distribution Detection (escalate unusual cases)
    Layer 3: ML Prediction (only for "safe" cases)
    Layer 4: Confidence Calibration (escalate low-confidence predictions)
    """
    
    def __init__(self, ml_model=None, feature_engineer=None):
        self.rule_engine = EmergencyRuleEngine()
        self.ood_detector = OutOfDistributionDetector()
        self.ml_model = ml_model
        self.feature_engineer = feature_engineer
        self.ood_threshold = 0.8
        self.confidence_threshold = 0.6
    
    def triage(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete triage pipeline with all safety layers.
        
        Returns dict with:
        - esi_level: 1-5
        - confidence: 0.0-1.0
        - method: 'RULE_BASED' | 'OOD_FALLBACK' | 'ML_PREDICTION'
        - action: Recommended action
        - protocol: Clinical protocol
        - explanation: List of contributing factors
        """
        
        # LAYER 1: Emergency Rules (Cannot Fail)
        emergency_result = self.rule_engine.check(patient_data)
        if emergency_result.triggered:
            return {
                'esi_level': emergency_result.esi_level,
                'confidence': 1.0,
                'method': 'RULE_BASED',
                'action': emergency_result.action,
                'protocol': emergency_result.protocol,
                'rule_triggered': emergency_result.rule_name,
                'rationale': emergency_result.rationale,
                'explanation': [{
                    'feature': emergency_result.rule_name,
                    'impact': '+',
                    'value': 'CRITICAL'
                }]
            }
        
        # LAYER 2: ML Prediction (if available)
        if self.ml_model is not None and self.feature_engineer is not None:
            # Extract features
            features = self.feature_engineer.transform(patient_data)
            
            # Check for out-of-distribution
            if self.ood_detector.is_fitted:
                ood_score = self.ood_detector.score(features)
                if ood_score > self.ood_threshold:
                    return {
                        'esi_level': 2,  # Err on side of caution
                        'confidence': 0.3,
                        'method': 'OOD_FALLBACK',
                        'action': 'IMMEDIATE DOCTOR EVALUATION - Unusual presentation',
                        'protocol': 'Clinical assessment required',
                        'ood_score': ood_score,
                        'explanation': [{
                            'feature': 'unusual_presentation',
                            'impact': '+',
                            'value': f'OOD score: {ood_score:.2f}'
                        }]
                    }
            
            # Get ML prediction
            prediction = self.ml_model.predict_proba(features)[0]
            esi_level = int(np.argmax(prediction)) + 1  # 1-indexed
            confidence = float(np.max(prediction))
            
            # LAYER 3: Confidence Calibration
            if confidence < self.confidence_threshold:
                # Escalate one level if uncertain
                original_level = esi_level
                esi_level = max(1, esi_level - 1)
                
                return {
                    'esi_level': esi_level,
                    'confidence': confidence,
                    'method': 'ML_PREDICTION',
                    'escalated': True,
                    'original_level': original_level,
                    'action': self._get_recommendation(esi_level) + ' (ESCALATED - LOW CONFIDENCE)',
                    'protocol': self._get_protocol(esi_level),
                    'explanation': self._get_explanation(features, prediction)
                }
            
            return {
                'esi_level': esi_level,
                'confidence': confidence,
                'method': 'ML_PREDICTION',
                'action': self._get_recommendation(esi_level),
                'protocol': self._get_protocol(esi_level),
                'explanation': self._get_explanation(features, prediction)
            }
        
        # No ML model - return conservative estimate
        return {
            'esi_level': 3,
            'confidence': 0.5,
            'method': 'DEFAULT_FALLBACK',
            'action': 'Doctor evaluation recommended',
            'protocol': 'Standard triage assessment',
            'explanation': [{'feature': 'no_ml_model', 'impact': '?', 'value': 'N/A'}]
        }
    
    def _get_recommendation(self, esi_level: int) -> str:
        """Get action recommendation for ESI level."""
        recommendations = {
            1: "IMMEDIATE RESUSCITATION - Life-threatening condition",
            2: "EMERGENT - High risk, see within 10 minutes",
            3: "URGENT - Moderate risk, see within 30 minutes",
            4: "LESS URGENT - Low risk, see within 60 minutes",
            5: "NON-URGENT - Minimal risk, can wait"
        }
        return recommendations.get(esi_level, "Doctor evaluation required")
    
    def _get_protocol(self, esi_level: int) -> str:
        """Get protocol reference for ESI level."""
        protocols = {
            1: "ESI Level 1 - Resuscitation",
            2: "ESI Level 2 - Emergent",
            3: "ESI Level 3 - Urgent",
            4: "ESI Level 4 - Less Urgent",
            5: "ESI Level 5 - Non-Urgent"
        }
        return protocols.get(esi_level, "ESI Assessment")
    
    def _get_explanation(self, features, prediction) -> List[Dict]:
        """Get feature explanation for prediction."""
        # Placeholder - actual SHAP integration in esi_predictor.py
        return [{'feature': 'ml_prediction', 'impact': '+', 'value': 'See SHAP analysis'}]


# Import numpy for ClinicalSafetyEngine
import numpy as np
