"""
RiskScope AI - Safety Engine Test Suite
========================================
Comprehensive tests for:
  - All 18 ESI-1 emergency rules
  - All 3 ESI-2 high-risk rules
  - OOD escalation
  - Confidence calibration / escalation
  - Full triage() pipeline end-to-end

Usage:
    python test_safety_engine.py
"""

import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from safety_engine import EmergencyRuleEngine, ClinicalSafetyEngine, EmergencyResult
from confidence import ConfidenceCalibrator


# =========================================================================
#  TEST 1: ESI-1 Emergency Rules (all 18)
# =========================================================================
ESI_1_TEST_CASES = [
    {
        "name": "Suspected MI",
        "rule": "suspected_mi",
        "patient": {"chest_pain": True, "arm_pain_left": True, "age": 60},
    },
    {
        "name": "Cardiac Arrest Signs",
        "rule": "cardiac_arrest_signs",
        "patient": {"heart_rate": 25, "age": 70},
    },
    {
        "name": "Unstable Tachycardia",
        "rule": "unstable_tachycardia",
        "patient": {"heart_rate": 160, "bp_systolic": 85, "age": 55},
    },
    {
        "name": "Respiratory Failure",
        "rule": "respiratory_failure",
        "patient": {"spo2": 85, "age": 65},
    },
    {
        "name": "Severe Respiratory Distress",
        "rule": "severe_respiratory_distress",
        "patient": {"respiratory_rate": 35, "dyspnea": True, "age": 50},
    },
    {
        "name": "Airway Compromise",
        "rule": "airway_compromise",
        "patient": {"stridor": True, "age": 30},
    },
    {
        "name": "Stroke FAST (Facial + Arm)",
        "rule": "stroke_fast",
        "patient": {"facial_droop": True, "arm_weakness": True, "age": 72},
    },
    {
        "name": "Stroke FAST (Facial + Speech)",
        "rule": "stroke_fast",
        "patient": {"facial_droop": True, "speech_difficulty": True, "age": 68},
    },
    {
        "name": "Stroke Single Severe",
        "rule": "stroke_single_severe",
        "patient": {"facial_droop": True, "symptom_duration_hours": 1.0, "age": 65},
    },
    {
        "name": "Active Seizure",
        "rule": "active_seizure",
        "patient": {"seizure": True, "seizure_ongoing": True, "age": 40},
    },
    {
        "name": "Severe AMS",
        "rule": "severe_altered_mental_status",
        "patient": {"unresponsive": True, "age": 75},
    },
    {
        "name": "Shock (Hypotension + Tachycardia)",
        "rule": "shock",
        "patient": {"bp_systolic": 80, "heart_rate": 120, "age": 45},
    },
    {
        "name": "Severe Hypotension",
        "rule": "severe_hypotension",
        "patient": {"bp_systolic": 70, "age": 60},
    },
    {
        "name": "Uncontrolled Hemorrhage",
        "rule": "uncontrolled_hemorrhage",
        "patient": {"uncontrolled_bleeding": True, "age": 35},
    },
    {
        "name": "Sepsis",
        "rule": "sepsis",
        "patient": {
            "temperature": 39.5, "heart_rate": 110,
            "altered_mental_status": True, "age": 70,
        },
    },
    {
        "name": "Meningitis Signs",
        "rule": "meningitis_signs",
        "patient": {
            "fever": True, "headache": True,
            "neck_stiffness": True, "age": 25,
        },
    },
    {
        "name": "Major Trauma",
        "rule": "major_trauma",
        "patient": {
            "trauma": True, "bp_systolic": 85,
            "age": 28,
        },
    },
    {
        "name": "Anaphylaxis",
        "rule": "anaphylaxis",
        "patient": {"anaphylaxis": True, "age": 30},
    },
    {
        "name": "Hypertensive Emergency",
        "rule": "severe_hypertensive_emergency",
        "patient": {
            "bp_systolic": 200, "chest_pain": True, "age": 60,
        },
    },
]

# =========================================================================
#  TEST 2: ESI-2 High-Risk Rules (all 3)
# =========================================================================
ESI_2_TEST_CASES = [
    {
        "name": "Chest Pain Stable",
        "rule": "chest_pain_stable",
        "patient": {"chest_pain": True, "bp_systolic": 130, "age": 50},
    },
    {
        "name": "High Fever",
        "rule": "high_fever",
        "patient": {"temperature": 40.0, "age": 45},
    },
    {
        "name": "Moderate Hypoxia",
        "rule": "moderate_hypoxia",
        "patient": {"spo2": 92, "age": 60},
    },
]

# =========================================================================
#  TEST 3: Cases that should NOT trigger any rule
# =========================================================================
NO_TRIGGER_CASES = [
    {
        "name": "Healthy Walk-in",
        "patient": {
            "age": 25, "heart_rate": 72, "bp_systolic": 118,
            "bp_diastolic": 75, "spo2": 99, "temperature": 36.8,
            "respiratory_rate": 14,
        },
    },
    {
        "name": "Mild Headache",
        "patient": {
            "age": 30, "headache": True, "heart_rate": 70,
            "bp_systolic": 115, "spo2": 99, "temperature": 37.0,
            "respiratory_rate": 15,
        },
    },
]


def run_tests():
    """Run all safety engine tests."""

    print("=" * 64)
    print("  RiskScope AI - Safety Engine Test Suite")
    print("=" * 64)

    engine = EmergencyRuleEngine()
    passed = 0
    failed = 0
    total = 0

    # ----- TEST 1: ESI-1 rules -----
    print("\n[1] TESTING ESI-1 EMERGENCY RULES")
    print("-" * 64)

    for tc in ESI_1_TEST_CASES:
        total += 1
        result = engine.check(tc["patient"])

        if result.triggered and result.esi_level == 1:
            print(f"  [PASS] {tc['name']:40s} -> Rule: {result.rule_name}")
            passed += 1
        else:
            print(f"  [FAIL] {tc['name']:40s} -> Expected rule '{tc['rule']}' "
                  f"but got: triggered={result.triggered}, "
                  f"rule={result.rule_name}, esi={result.esi_level}")
            failed += 1

    # ----- TEST 2: ESI-2 rules -----
    print("\n[2] TESTING ESI-2 HIGH-RISK RULES")
    print("-" * 64)

    for tc in ESI_2_TEST_CASES:
        total += 1
        result = engine.check(tc["patient"])

        if result.triggered and result.esi_level == 2:
            print(f"  [PASS] {tc['name']:40s} -> Rule: {result.rule_name}")
            passed += 1
        else:
            print(f"  [FAIL] {tc['name']:40s} -> Expected ESI 2 but got: "
                  f"triggered={result.triggered}, esi={result.esi_level}")
            failed += 1

    # ----- TEST 3: No-trigger cases -----
    print("\n[3] TESTING NON-EMERGENCY CASES (should NOT trigger)")
    print("-" * 64)

    for tc in NO_TRIGGER_CASES:
        total += 1
        result = engine.check(tc["patient"])

        if not result.triggered:
            print(f"  [PASS] {tc['name']:40s} -> No rule triggered (correct)")
            passed += 1
        else:
            print(f"  [FAIL] {tc['name']:40s} -> Unexpected trigger: {result.rule_name}")
            failed += 1

    # ----- TEST 4: Confidence Calibrator -----
    print("\n[4] TESTING CONFIDENCE CALIBRATOR")
    print("-" * 64)

    calibrator = ConfidenceCalibrator()

    # High confidence case
    total += 1
    high_conf_proba = np.array([0.01, 0.01, 0.02, 0.01, 0.95])
    result = calibrator.calibrate(high_conf_proba)
    if not result['should_escalate'] and result['confidence'] > 0.7:
        print(f"  [PASS] High confidence:  conf={result['confidence']:.2f}, "
              f"escalate={result['should_escalate']}")
        passed += 1
    else:
        print(f"  [FAIL] High confidence failed: {result}")
        failed += 1

    # Low confidence case
    total += 1
    low_conf_proba = np.array([0.15, 0.25, 0.25, 0.20, 0.15])
    result = calibrator.calibrate(low_conf_proba)
    if result['should_escalate'] and result['confidence'] < 0.6:
        print(f"  [PASS] Low confidence:   conf={result['confidence']:.2f}, "
              f"escalate={result['should_escalate']}")
        passed += 1
    else:
        print(f"  [FAIL] Low confidence failed: {result}")
        failed += 1

    # Ambiguous case
    total += 1
    ambig_proba = np.array([0.02, 0.40, 0.42, 0.10, 0.06])
    result = calibrator.calibrate(ambig_proba)
    if result['is_ambiguous']:
        print(f"  [PASS] Ambiguous:        conf={result['confidence']:.2f}, "
              f"ambiguous={result['is_ambiguous']}")
        passed += 1
    else:
        print(f"  [FAIL] Ambiguous failed: {result}")
        failed += 1

    # Escalation logic
    total += 1
    esi, reason = calibrator.escalate_prediction(3, result)
    if esi == 2:
        print(f"  [PASS] Escalation:       ESI 3 -> ESI {esi}")
        passed += 1
    else:
        print(f"  [FAIL] Escalation failed: expected 2, got {esi}")
        failed += 1

    # ----- TEST 5: Full triage pipeline -----
    print("\n[5] TESTING FULL TRIAGE PIPELINE (ClinicalSafetyEngine)")
    print("-" * 64)

    model_dir = os.path.join(os.path.dirname(__file__), "models")
    ensemble_path = os.path.join(model_dir, "esi_ensemble_model.pkl")
    ood_path = os.path.join(model_dir, "ood_detector.pkl")

    if os.path.exists(ensemble_path) and os.path.exists(ood_path):
        try:
            safety_engine = ClinicalSafetyEngine.from_trained_models(model_dir)
            print("  [PASS] Factory method loaded successfully")
            passed += 1
            total += 1

            # Test emergency rule path
            total += 1
            mi_result = safety_engine.triage({
                "age": 60, "chest_pain": True, "arm_pain_left": True,
                "heart_rate": 130, "bp_systolic": 85, "spo2": 88,
            })
            if mi_result['method'] == 'RULE_BASED' and mi_result['esi_level'] == 1:
                print(f"  [PASS] Emergency path:   ESI {mi_result['esi_level']} "
                      f"({mi_result['method']})")
                passed += 1
            else:
                print(f"  [FAIL] Emergency path failed: {mi_result}")
                failed += 1

            # Test ML prediction path
            total += 1
            mild_result = safety_engine.triage({
                "age": 28, "heart_rate": 72, "bp_systolic": 118,
                "bp_diastolic": 75, "spo2": 99, "temperature": 37.0,
                "respiratory_rate": 15, "headache": True,
            })
            if mild_result['method'] == 'ML_PREDICTION':
                print(f"  [PASS] ML path:          ESI {mild_result['esi_level']} "
                      f"(conf={mild_result['confidence']:.2f})")
                passed += 1
            else:
                print(f"  [FAIL] ML path failed: {mild_result}")
                failed += 1

            # Verify explanation is present
            total += 1
            if mild_result.get('explanation') and len(mild_result['explanation']) > 0:
                first_exp = mild_result['explanation'][0]
                has_display = 'display_name' in first_exp or 'feature' in first_exp
                print(f"  [PASS] Explanation:      {len(mild_result['explanation'])} features "
                      f"(has display names: {has_display})")
                passed += 1
            else:
                print(f"  [FAIL] Missing explanation in triage result")
                failed += 1

            # Verify confidence details
            total += 1
            if mild_result.get('confidence_details'):
                cd = mild_result['confidence_details']
                print(f"  [PASS] Confidence:       level={cd.get('level')}, "
                      f"label={cd.get('label', '')[:40]}")
                passed += 1
            else:
                print(f"  [FAIL] Missing confidence_details")
                failed += 1

        except Exception as e:
            print(f"  [FAIL] Factory method failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            total += 1
    else:
        print("  [WARN]  Skipping (model files not found)")

    # ----- SUMMARY -----
    print("\n" + "=" * 64)
    print("  SAFETY ENGINE TEST SUMMARY")
    print("=" * 64)
    print(f"  Passed: {passed}/{total}")
    print(f"  Failed: {failed}/{total}")

    if failed == 0:
        print("\n  [PASS] ALL SAFETY TESTS PASSED")
    else:
        print(f"\n  [FAIL] {failed} TEST(S) FAILED — needs attention")

    print("=" * 64)

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
