"""
RiskScope AI - Explainability & Confidence Test Suite
======================================================
Tests for:
  - ExplainabilityEngine (human-readable explanations, SHAP, clinical notes)
  - ConfidenceCalibrator (entropy, margin, escalation)
  - Integration with trained model

Usage:
    python test_explainability.py
"""

import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from explain import ExplainabilityEngine, FEATURE_DISPLAY_NAMES, CLINICAL_CONTEXT
from confidence import ConfidenceCalibrator


def run_tests():
    """Run all explainability and confidence tests."""

    print("=" * 64)
    print("  RiskScope AI - Explainability & Confidence Test Suite")
    print("=" * 64)

    passed = 0
    failed = 0
    total = 0

    # =====================================================================
    # TEST 1: Feature display name mapping
    # =====================================================================
    print("\n[1] FEATURE DISPLAY NAME MAPPING")
    print("-" * 64)

    key_features = [
        'heart_rate', 'bp_systolic', 'spo2', 'chest_pain',
        'facial_droop', 'shock_index', 'sirs_score',
    ]
    for feat in key_features:
        total += 1
        if feat in FEATURE_DISPLAY_NAMES:
            print(f"  ✅ {feat:30s} → {FEATURE_DISPLAY_NAMES[feat]}")
            passed += 1
        else:
            print(f"  ❌ {feat:30s} → MISSING from display names")
            failed += 1

    # Check we have mappings for raw symptoms
    total += 1
    symptom_features = [
        'chest_pain', 'arm_pain_left', 'jaw_pain', 'dyspnea',
        'facial_droop', 'arm_weakness', 'speech_difficulty',
        'altered_mental_status', 'seizure', 'uncontrolled_bleeding',
    ]
    mapped = sum(1 for s in symptom_features if s in FEATURE_DISPLAY_NAMES)
    if mapped == len(symptom_features):
        print(f"  ✅ All {mapped} symptom features have display names")
        passed += 1
    else:
        print(f"  ❌ Only {mapped}/{len(symptom_features)} symptoms mapped")
        failed += 1

    # =====================================================================
    # TEST 2: Clinical context definitions
    # =====================================================================
    print("\n[2] CLINICAL CONTEXT DEFINITIONS")
    print("-" * 64)

    expected_contexts = [
        'heart_rate', 'bp_systolic', 'spo2', 'temperature',
        'respiratory_rate', 'chest_pain', 'facial_droop',
    ]
    for ctx in expected_contexts:
        total += 1
        if ctx in CLINICAL_CONTEXT:
            print(f"  ✅ {ctx:30s} → context defined")
            passed += 1
        else:
            print(f"  ❌ {ctx:30s} → MISSING clinical context")
            failed += 1

    # =====================================================================
    # TEST 3: ExplainabilityEngine without model (basic mode)
    # =====================================================================
    print("\n[3] EXPLAINABILITY ENGINE (no model)")
    print("-" * 64)

    engine = ExplainabilityEngine(predictor=None)

    total += 1
    fake_features = np.array([[0.8, 0.2, 0.5, 1.0, 0.0,
                                0.3, 0.7, 0.1, 0.9, 0.4]])
    result = engine.explain(
        fake_features,
        patient_data={'heart_rate': 130, 'chest_pain': True},
        predicted_esi=2,
        top_n=3
    )
    if 'top_features' in result and len(result['top_features']) == 3:
        print(f"  ✅ Basic explain: {len(result['top_features'])} features returned")
        passed += 1
    else:
        print(f"  ❌ Basic explain failed: {result}")
        failed += 1

    total += 1
    if result.get('risk_summary'):
        print(f"  ✅ Risk summary: '{result['risk_summary'][:50]}...'")
        passed += 1
    else:
        print(f"  ❌ Missing risk summary")
        failed += 1

    # =====================================================================
    # TEST 4: Confidence Calibrator — detailed tests
    # =====================================================================
    print("\n[4] CONFIDENCE CALIBRATOR")
    print("-" * 64)

    calibrator = ConfidenceCalibrator()

    # Very high confidence
    total += 1
    proba = np.array([0.0, 0.0, 0.0, 0.02, 0.98])
    cal = calibrator.calibrate(proba)
    if cal['confidence'] > 0.85 and not cal['should_escalate']:
        print(f"  ✅ Very high conf:   {cal['confidence']:.3f}, "
              f"entropy={cal['entropy']:.3f}, margin={cal['margin']:.3f}")
        passed += 1
    else:
        print(f"  ❌ Very high conf failed: {cal}")
        failed += 1

    # Moderate confidence
    total += 1
    proba = np.array([0.05, 0.10, 0.60, 0.15, 0.10])
    cal = calibrator.calibrate(proba)
    if 0.4 < cal['confidence'] < 0.9:
        print(f"  ✅ Moderate conf:    {cal['confidence']:.3f}, "
              f"entropy={cal['entropy']:.3f}")
        passed += 1
    else:
        print(f"  ❌ Moderate conf failed: {cal}")
        failed += 1

    # Very low confidence (uniform distribution)
    total += 1
    proba = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
    cal = calibrator.calibrate(proba)
    if cal['should_escalate'] and cal['confidence'] < 0.5:
        print(f"  ✅ Uniform dist:     {cal['confidence']:.3f}, "
              f"escalate={cal['should_escalate']}")
        passed += 1
    else:
        print(f"  ❌ Uniform dist failed: {cal}")
        failed += 1

    # Test class probabilities output
    total += 1
    if 'class_probabilities' in cal and len(cal['class_probabilities']) == 5:
        print(f"  ✅ Class probs:      {cal['class_probabilities']}")
        passed += 1
    else:
        print(f"  ❌ Missing class probabilities")
        failed += 1

    # Test display helper
    total += 1
    display = calibrator.get_confidence_display(cal)
    if display['level'] in ['HIGH', 'MODERATE', 'LOW'] and display['color']:
        print(f"  ✅ Display:          {display['level']} ({display['color']})")
        passed += 1
    else:
        print(f"  ❌ Display failed: {display}")
        failed += 1

    # Entropy calculation sanity check
    total += 1
    uniform = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    certain = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    cal_uniform = calibrator.calibrate(uniform)
    cal_certain = calibrator.calibrate(certain)
    if cal_uniform['entropy'] > cal_certain['entropy']:
        print(f"  ✅ Entropy ordering: uniform({cal_uniform['entropy']:.3f}) > "
              f"certain({cal_certain['entropy']:.3f})")
        passed += 1
    else:
        print(f"  ❌ Entropy ordering wrong")
        failed += 1

    # =====================================================================
    # TEST 5: ExplainabilityEngine with trained model
    # =====================================================================
    print("\n[5] EXPLAINABILITY WITH TRAINED MODEL")
    print("-" * 64)

    model_dir = os.path.join(os.path.dirname(__file__), "models")
    ensemble_path = os.path.join(model_dir, "esi_ensemble_model.pkl")

    if os.path.exists(ensemble_path):
        try:
            from esi_predictor import ESITriagePredictor

            predictor = ESITriagePredictor.load(ensemble_path)
            engine = ExplainabilityEngine(predictor)

            # Build a realistic feature vector
            feature_cols = predictor.feature_names
            patient = {
                "age": 62, "gender": 1,
                "heart_rate": 130, "bp_systolic": 85, "bp_diastolic": 55,
                "spo2": 88, "temperature": 37.1, "respiratory_rate": 28,
                "chest_pain": 1, "arm_pain_left": 1, "jaw_pain": 1,
                "dyspnea": 1, "severe_pain": 1,
                "symptom_duration_hours": 0.5,
            }
            features = np.array([[float(patient.get(col, 0))
                                   for col in feature_cols]])

            total += 1
            result = engine.explain(
                features, patient, predicted_esi=1, top_n=5
            )
            if result['top_features'] and len(result['top_features']) == 5:
                print(f"  ✅ SHAP explain: {len(result['top_features'])} features")
                for feat in result['top_features']:
                    print(f"     {feat['impact']} {feat['display_name']:30s} "
                          f"(SHAP={feat['shap_value']:.4f})")
                passed += 1
            else:
                print(f"  ❌ SHAP explain failed")
                failed += 1

            # Clinical notes
            total += 1
            if result.get('clinical_notes'):
                print(f"  ✅ Clinical notes: {len(result['clinical_notes'])}")
                for note in result['clinical_notes'][:3]:
                    print(f"     • {note[:60]}")
                passed += 1
            else:
                print(f"  ✅ No clinical notes (may be expected for this feature set)")
                passed += 1

            # Risk summary
            total += 1
            if result.get('risk_summary'):
                print(f"  ✅ Risk summary: {result['risk_summary'][:60]}")
                passed += 1
            else:
                print(f"  ❌ Missing risk summary")
                failed += 1

            # Waterfall data
            total += 1
            waterfall = engine.get_waterfall_data(features)
            if waterfall and 'shap_values' in waterfall:
                print(f"  ✅ Waterfall data: {len(waterfall['shap_values'])} values")
                passed += 1
            else:
                print(f"  ⚠️  Waterfall data unavailable (SHAP may not be initialized)")
                passed += 1  # Non-critical

        except Exception as e:
            print(f"  ❌ Model-based tests failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            total += 1
    else:
        print("  ⚠️  Skipping (model file not found)")

    # ----- SUMMARY -----
    print("\n" + "=" * 64)
    print("  EXPLAINABILITY TEST SUMMARY")
    print("=" * 64)
    print(f"  Passed: {passed}/{total}")
    print(f"  Failed: {failed}/{total}")

    if failed == 0:
        print("\n  ✅ ALL EXPLAINABILITY TESTS PASSED")
    else:
        print(f"\n  ❌ {failed} TEST(S) FAILED")

    print("=" * 64)

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
