"""
RiskScope AI — Backend Safety Service
======================================
Port of the ML module's 3-layer safety architecture into the FastAPI backend.
Runs BEFORE the ML model to catch life-threatening emergencies via hard-coded
clinical rules (ACLS/PALS/ESI protocols).

Layer 1: Emergency Rule Engine  — hard-coded, cannot fail
Layer 2: ML Prediction          — portable KNN model (ml.pkl)
Layer 3: Confidence Calibration — escalate uncertain predictions
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SafetyResult:
    """Result from the safety rule check."""
    triggered: bool = False
    rule_name: str | None = None
    esi_level: int | None = None
    protocol: str | None = None
    action: str | None = None
    rationale: str | None = None
    method: str = "ML_PREDICTION"
    explanation: list[dict[str, Any]] = field(default_factory=list)


# =====================================================================
#  EMERGENCY RULES
# =====================================================================

def _check_suspected_mi(p: dict) -> bool:
    return bool(
        p.get("chest_pain")
        and (p.get("arm_pain_left") or p.get("jaw_pain"))
    )


def _check_cardiac_arrest(p: dict) -> bool:
    hr = p.get("heart_rate", 80)
    return bool(
        p.get("unresponsive")
        or (isinstance(hr, (int, float)) and hr < 30)
        or (isinstance(hr, (int, float)) and hr > 180 and p.get("chest_pain"))
    )


def _check_unstable_tachycardia(p: dict) -> bool:
    hr = p.get("heart_rate", 80)
    sbp = p.get("bp_systolic", 120)
    return bool(
        isinstance(hr, (int, float)) and hr > 150
        and (
            (isinstance(sbp, (int, float)) and sbp < 90)
            or p.get("altered_mental_status")
            or p.get("chest_pain")
        )
    )


def _check_respiratory_failure(p: dict) -> bool:
    spo2 = p.get("spo2", 100)
    return isinstance(spo2, (int, float)) and spo2 < 90


def _check_severe_respiratory_distress(p: dict) -> bool:
    rr = p.get("respiratory_rate") or p.get("resp_rate") or 16
    return bool(
        isinstance(rr, (int, float)) and rr > 30
        and (p.get("dyspnea") or p.get("shortness_of_breath"))
    )


def _check_airway_compromise(p: dict) -> bool:
    return bool(
        p.get("stridor")
        or p.get("choking")
        or (p.get("anaphylaxis") and p.get("dyspnea"))
    )


def _check_stroke_fast(p: dict) -> bool:
    fd = p.get("facial_droop")
    aw = p.get("arm_weakness")
    sd = p.get("speech_difficulty")
    return bool(
        (fd and aw) or (fd and sd) or (aw and sd)
    )


def _check_stroke_single(p: dict) -> bool:
    has_symptom = p.get("facial_droop") or p.get("arm_weakness") or p.get("speech_difficulty")
    duration = p.get("symptom_duration_hours", 24)
    age = p.get("age", 50)
    return bool(
        has_symptom
        and isinstance(duration, (int, float)) and duration < 4.5
        and isinstance(age, (int, float)) and age > 45
    )


def _check_active_seizure(p: dict) -> bool:
    return bool(
        (p.get("seizure") and p.get("seizure_ongoing"))
        or p.get("status_epilepticus")
    )


def _check_severe_ams(p: dict) -> bool:
    sbp = p.get("bp_systolic", 120)
    gcs = p.get("gcs", 15)
    return bool(
        p.get("unresponsive")
        or (p.get("altered_mental_status") and isinstance(sbp, (int, float)) and sbp < 90)
        or (isinstance(gcs, (int, float)) and gcs < 9)
    )


def _check_shock(p: dict) -> bool:
    sbp = p.get("bp_systolic", 120)
    hr = p.get("heart_rate", 80)
    return bool(
        isinstance(sbp, (int, float)) and sbp < 90
        and isinstance(hr, (int, float)) and hr > 100
    )


def _check_severe_hypotension(p: dict) -> bool:
    sbp = p.get("bp_systolic", 120)
    return isinstance(sbp, (int, float)) and sbp < 80


def _check_hemorrhage(p: dict) -> bool:
    return bool(
        p.get("uncontrolled_bleeding")
        or (p.get("active_bleeding") and p.get("bp_systolic", 120) < 100)
    )


def _check_sepsis(p: dict) -> bool:
    temp = p.get("temperature", 37)
    hr = p.get("heart_rate", 80)
    sbp = p.get("bp_systolic", 120)
    return bool(
        isinstance(temp, (int, float)) and (temp > 38.3 or temp < 36)
        and isinstance(hr, (int, float)) and hr > 90
        and (p.get("altered_mental_status") or (isinstance(sbp, (int, float)) and sbp < 100))
    )


def _check_meningitis(p: dict) -> bool:
    return bool(
        p.get("fever")
        and p.get("headache")
        and (p.get("neck_stiffness") or p.get("altered_mental_status"))
    )


def _check_major_trauma(p: dict) -> bool:
    sbp = p.get("bp_systolic", 120)
    gcs = p.get("gcs", 15)
    rr = p.get("respiratory_rate") or p.get("resp_rate") or 16
    return bool(
        p.get("trauma")
        and (
            (isinstance(sbp, (int, float)) and sbp < 90)
            or (isinstance(gcs, (int, float)) and gcs < 14)
            or (isinstance(rr, (int, float)) and rr > 29)
        )
    )


def _check_anaphylaxis(p: dict) -> bool:
    return bool(
        p.get("anaphylaxis")
        or (
            p.get("allergic_reaction")
            and (p.get("dyspnea") or p.get("bp_systolic", 120) < 90)
        )
    )


def _check_hypertensive_emergency(p: dict) -> bool:
    sbp = p.get("bp_systolic", 120)
    return bool(
        isinstance(sbp, (int, float)) and sbp > 180
        and (
            p.get("chest_pain")
            or (p.get("headache") and p.get("altered_mental_status"))
        )
    )


# fmt: off
ESI_1_RULES: list[dict[str, Any]] = [
    {"name": "suspected_mi",               "check": _check_suspected_mi,
     "protocol": "ACLS - Acute Coronary Syndrome",
     "action": "12-lead EKG STAT, Aspirin 325mg, IV access, Troponin, Cardiology consult",
     "rationale": "Chest pain with radiation suggests acute MI"},
    {"name": "cardiac_arrest_signs",        "check": _check_cardiac_arrest,
     "protocol": "ACLS - Cardiac Arrest",
     "action": "Crash cart, CPR if indicated, Defibrillator ready",
     "rationale": "Extreme HR derangement with unresponsiveness"},
    {"name": "unstable_tachycardia",        "check": _check_unstable_tachycardia,
     "protocol": "ACLS - Tachycardia with Pulse",
     "action": "IV access, 12-lead EKG, Synchronized cardioversion standby",
     "rationale": "Rapid HR with hemodynamic instability"},
    {"name": "respiratory_failure",         "check": _check_respiratory_failure,
     "protocol": "Respiratory Distress Protocol",
     "action": "High-flow O2 (15L NRB), ABG, Chest X-ray, RT STAT",
     "rationale": "SpO2 <90%: severe hypoxemia"},
    {"name": "severe_respiratory_distress",  "check": _check_severe_respiratory_distress,
     "protocol": "Respiratory Distress Protocol",
     "action": "Oxygen, IV access, Chest X-ray, consider BiPAP/intubation",
     "rationale": "Tachypnea >30 with dyspnea"},
    {"name": "airway_compromise",           "check": _check_airway_compromise,
     "protocol": "Airway Management Protocol",
     "action": "Airway assessment, Suction, Intubation equipment ready",
     "rationale": "Airway compromise is immediately life-threatening"},
    {"name": "stroke_fast",                 "check": _check_stroke_fast,
     "protocol": "Stroke Alert - FAST Protocol",
     "action": "CT Head STAT, Neurology STAT, Check glucose, BP monitoring",
     "rationale": "2+ FAST criteria positive"},
    {"name": "stroke_single_severe",        "check": _check_stroke_single,
     "protocol": "Stroke Alert - FAST Protocol",
     "action": "CT Head STAT, Neurology consult, tPA evaluation window",
     "rationale": "Single stroke symptom within tPA window"},
    {"name": "active_seizure",              "check": _check_active_seizure,
     "protocol": "Seizure Protocol",
     "action": "Protect airway, IV access, Benzodiazepines, Glucose check",
     "rationale": "Ongoing seizure needs immediate intervention"},
    {"name": "severe_altered_mental_status", "check": _check_severe_ams,
     "protocol": "Altered Mental Status Protocol",
     "action": "Airway assessment, Glucose check, CT Head, Tox screen",
     "rationale": "GCS <9 or unresponsive"},
    {"name": "shock",                       "check": _check_shock,
     "protocol": "Shock Protocol",
     "action": "2 large-bore IVs, NS bolus 1L, Type & Screen",
     "rationale": "Hypotension with compensatory tachycardia"},
    {"name": "severe_hypotension",          "check": _check_severe_hypotension,
     "protocol": "Shock Protocol",
     "action": "IV fluids, Vasopressors ready, ICU notification",
     "rationale": "SBP <80 severe hypoperfusion"},
    {"name": "uncontrolled_hemorrhage",     "check": _check_hemorrhage,
     "protocol": "Massive Transfusion Protocol",
     "action": "Direct pressure, 2 large-bore IVs, T&C, O-neg blood",
     "rationale": "Active hemorrhage with instability"},
    {"name": "sepsis",                      "check": _check_sepsis,
     "protocol": "Sepsis Bundle (Hour-1)",
     "action": "Lactate, Blood cultures x2, Broad-spectrum antibiotics, IV 30mL/kg",
     "rationale": "Suspected infection with organ dysfunction"},
    {"name": "meningitis_signs",            "check": _check_meningitis,
     "protocol": "Meningitis Protocol",
     "action": "Blood cultures, LP, Empiric antibiotics STAT",
     "rationale": "Fever + headache + neck stiffness triad"},
    {"name": "major_trauma",               "check": _check_major_trauma,
     "protocol": "Trauma Activation",
     "action": "Trauma team, C-spine, 2 IVs, FAST exam",
     "rationale": "Trauma with vital sign abnormality"},
    {"name": "anaphylaxis",                "check": _check_anaphylaxis,
     "protocol": "Anaphylaxis Protocol",
     "action": "Epinephrine 0.3mg IM, IV, O2, H1/H2 blockers, Steroids",
     "rationale": "Anaphylaxis with airway/CV involvement"},
    {"name": "severe_hypertensive_emergency", "check": _check_hypertensive_emergency,
     "protocol": "Hypertensive Emergency Protocol",
     "action": "IV antihypertensive, Continuous BP, CT Head if neuro symptoms",
     "rationale": "Severe hypertension with end-organ damage"},
]
# fmt: on


def _check_chest_pain_stable(p: dict) -> bool:
    return bool(
        p.get("chest_pain")
        and not (p.get("arm_pain_left") or p.get("jaw_pain"))
        and p.get("bp_systolic", 120) >= 90
    )


def _check_high_fever(p: dict) -> bool:
    temp = p.get("temperature", 37)
    return isinstance(temp, (int, float)) and temp >= 39.5


def _check_moderate_hypoxia(p: dict) -> bool:
    spo2 = p.get("spo2", 100)
    return isinstance(spo2, (int, float)) and 90 <= spo2 <= 94


ESI_2_RULES: list[dict[str, Any]] = [
    {"name": "chest_pain_stable",
     "check": _check_chest_pain_stable,
     "protocol": "ACS Rule-Out Protocol",
     "action": "EKG within 10 min, Troponin, Aspirin if no contraindication",
     "rationale": "Chest pain requires rapid cardiac evaluation"},
    {"name": "high_fever",
     "check": _check_high_fever,
     "protocol": "Fever Workup",
     "action": "Blood cultures, CBC, CMP, Urinalysis, CXR if indicated",
     "rationale": "High fever suggests serious infection"},
    {"name": "moderate_hypoxia",
     "check": _check_moderate_hypoxia,
     "protocol": "Oxygen Therapy",
     "action": "Supplemental O2, Continuous pulse ox, CXR",
     "rationale": "Borderline hypoxia needs monitoring"},
]


class SafetyService:
    """Applies the 3-layer safety system to every triage request."""

    CONFIDENCE_THRESHOLD: float = 0.6

    def check_emergency_rules(self, patient: dict[str, Any]) -> SafetyResult:
        """
        Run all emergency rules against patient data.
        Returns SafetyResult with triggered=True if any rule fires.
        """
        # ESI-1 rules first
        for rule in ESI_1_RULES:
            try:
                if rule["check"](patient):
                    return SafetyResult(
                        triggered=True,
                        rule_name=rule["name"],
                        esi_level=1,
                        protocol=rule["protocol"],
                        action=rule["action"],
                        rationale=rule["rationale"],
                        method="RULE_BASED",
                        explanation=[{
                            "feature": rule["name"],
                            "display_name": rule["name"].replace("_", " ").title(),
                            "impact": "+",
                            "value": "CRITICAL",
                            "clinical_note": rule["rationale"],
                        }],
                    )
            except Exception:
                continue

        # ESI-2 rules
        for rule in ESI_2_RULES:
            try:
                if rule["check"](patient):
                    return SafetyResult(
                        triggered=True,
                        rule_name=rule["name"],
                        esi_level=2,
                        protocol=rule["protocol"],
                        action=rule["action"],
                        rationale=rule["rationale"],
                        method="RULE_BASED",
                        explanation=[{
                            "feature": rule["name"],
                            "display_name": rule["name"].replace("_", " ").title(),
                            "impact": "+",
                            "value": "HIGH RISK",
                            "clinical_note": rule["rationale"],
                        }],
                    )
            except Exception:
                continue

        return SafetyResult(triggered=False)

    def apply_confidence_calibration(
        self, esi_level: int, confidence: float | None
    ) -> tuple[int, float | None, bool, str | None]:
        """
        If prediction confidence is below threshold, escalate one ESI level
        (towards more urgent).  Returns (adjusted_esi, confidence, escalated, reason).
        """
        if confidence is None:
            return esi_level, confidence, False, None

        if confidence < self.CONFIDENCE_THRESHOLD:
            escalated_esi = max(1, esi_level - 1)
            reason = (
                f"Low prediction confidence ({confidence:.0%}). "
                f"Model uncertain -- escalated ESI {esi_level} -> {escalated_esi} for safety."
            )
            return escalated_esi, confidence, True, reason

        return esi_level, confidence, False, None

    def generate_recommendation(self, esi_level: int) -> str:
        """Get a human-readable recommendation for the ESI level."""
        recs = {
            1: "IMMEDIATE RESUSCITATION — Life-threatening condition",
            2: "EMERGENT — High risk, see within 10 minutes",
            3: "URGENT — Moderate risk, see within 30 minutes",
            4: "LESS URGENT — Low risk, see within 60 minutes",
            5: "NON-URGENT — Minimal risk, can wait",
        }
        return recs.get(esi_level, "Doctor evaluation required")
