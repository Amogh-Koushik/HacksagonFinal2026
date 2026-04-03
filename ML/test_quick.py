"""
RiskScope AI - Quick Test Script
Test all components work correctly before full training
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test all imports work."""
    print("Testing imports...")
    try:
        from config import LIGHTGBM_PARAMS, ESI_LEVELS
        print("  ✅ config.py")
        
        from feature_engineering import FeatureEngineer, engineer_features
        print("  ✅ feature_engineering.py")
        
        from safety_engine import EmergencyRuleEngine, ClinicalSafetyEngine
        print("  ✅ safety_engine.py")
        
        from ood_detector import OutOfDistributionDetector
        print("  ✅ ood_detector.py")
        
        from esi_predictor import ESITriagePredictor
        print("  ✅ esi_predictor.py")
        
        from generate_synthetic_data import generate_synthetic_dataset
        print("  ✅ generate_synthetic_data.py")
        
        from evaluate import ESIEvaluator
        print("  ✅ evaluate.py")
        
        return True
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False


def test_feature_engineering():
    """Test feature engineering pipeline."""
    print("\nTesting Feature Engineering...")
    
    from feature_engineering import FeatureEngineer
    
    # Sample patient
    patient = {
        'age': 58,
        'gender': 'M',
        'heart_rate': 105,
        'bp_systolic': 160,
        'bp_diastolic': 95,
        'spo2': 94,
        'temperature': 37.0,
        'respiratory_rate': 20,
        'chest_pain': True,
        'arm_pain_left': True,
        'dyspnea': True,
    }
    
    fe = FeatureEngineer()
    features = fe.transform(patient)
    
    print(f"  Input: {len(patient)} fields")
    print(f"  Output: {features.shape[1]} features")
    print(f"  Feature names: {len(fe.get_feature_names())}")
    
    if features.shape[1] > 50:
        print("  ✅ Feature engineering working")
        return True
    else:
        print("  ❌ Feature count too low")
        return False


def test_emergency_rules():
    """Test emergency rule engine."""
    print("\nTesting Emergency Rules...")
    
    from safety_engine import EmergencyRuleEngine
    
    rule_engine = EmergencyRuleEngine()
    
    # Test cases
    test_cases = [
        {
            'name': 'MI Case',
            'data': {'chest_pain': True, 'arm_pain_left': True, 'spo2': 94},
            'expected_triggered': True,
            'expected_esi': 1
        },
        {
            'name': 'Stroke Case',
            'data': {'facial_droop': True, 'arm_weakness': True, 'speech_difficulty': True},
            'expected_triggered': True,
            'expected_esi': 1
        },
        {
            'name': 'Hypoxia',
            'data': {'spo2': 85},
            'expected_triggered': True,
            'expected_esi': 1
        },
        {
            'name': 'Shock',
            'data': {'bp_systolic': 85, 'heart_rate': 120},
            'expected_triggered': True,
            'expected_esi': 1
        },
        {
            'name': 'Normal Case',
            'data': {'spo2': 98, 'heart_rate': 75, 'bp_systolic': 120},
            'expected_triggered': False,
            'expected_esi': None
        },
    ]
    
    passed = 0
    for case in test_cases:
        result = rule_engine.check(case['data'])
        
        triggered_match = result.triggered == case['expected_triggered']
        esi_match = (not result.triggered) or (result.esi_level == case['expected_esi'])
        
        if triggered_match and esi_match:
            print(f"  ✅ {case['name']}: {'Triggered' if result.triggered else 'Not triggered'}")
            passed += 1
        else:
            print(f"  ❌ {case['name']}: Expected triggered={case['expected_triggered']}, "
                  f"got triggered={result.triggered}")
    
    print(f"  Passed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_synthetic_data():
    """Test synthetic data generation."""
    print("\nTesting Synthetic Data Generation...")
    
    from generate_synthetic_data import generate_synthetic_dataset
    
    X, y = generate_synthetic_dataset(n_samples=1000)
    
    print(f"  Generated: {len(X)} samples")
    print(f"  Features: {X.shape[1]}")
    
    # Check class distribution
    for i in range(1, 6):
        count = (y == i).sum()
        print(f"    ESI {i}: {count} ({count/len(y)*100:.1f}%)")
    
    if len(X) == 1000 and X.shape[1] > 20:
        print("  ✅ Synthetic data generation working")
        return True
    else:
        print("  ❌ Synthetic data generation failed")
        return False


def test_model_training():
    """Test model training on small dataset."""
    print("\nTesting Model Training (small dataset)...")
    
    from generate_synthetic_data import generate_synthetic_dataset
    from esi_predictor import ESITriagePredictor
    
    # Generate small dataset
    X, y = generate_synthetic_dataset(n_samples=500)
    
    # Train model
    model = ESITriagePredictor(use_smote=True)
    
    try:
        model.fit(X, y, validate=False, verbose=False)
        
        # Test prediction
        test_sample = X.iloc[[0]]
        pred = model.predict(test_sample)
        proba = model.predict_proba(test_sample)
        
        print(f"  Prediction: ESI {pred[0]}")
        print(f"  Confidence: {proba[0].max():.1%}")
        print("  ✅ Model training working")
        return True
        
    except Exception as e:
        print(f"  ❌ Training failed: {e}")
        return False


def test_full_pipeline():
    """Test complete triage pipeline."""
    print("\nTesting Full Triage Pipeline...")
    
    from generate_synthetic_data import generate_synthetic_dataset
    from esi_predictor import ESITriagePredictor
    from feature_engineering import FeatureEngineer
    from safety_engine import ClinicalSafetyEngine
    
    # Generate data and train
    X, y = generate_synthetic_dataset(n_samples=500)
    model = ESITriagePredictor(use_smote=True)
    model.fit(X, y, validate=False, verbose=False)
    
    # Initialize safety engine
    fe = FeatureEngineer()
    safety_engine = ClinicalSafetyEngine(ml_model=model, feature_engineer=fe)
    
    # Test patient
    patient = {
        'age': 58,
        'gender': 'M',
        'heart_rate': 105,
        'bp_systolic': 160,
        'bp_diastolic': 95,
        'spo2': 94,
        'temperature': 37.0,
        'respiratory_rate': 20,
        'chest_pain': True,
        'arm_pain_left': True,
    }
    
    result = safety_engine.triage(patient)
    
    print(f"  ESI Level: {result['esi_level']}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Method: {result['method']}")
    print(f"  Action: {result['action'][:50]}...")
    
    if result['esi_level'] in [1, 2, 3, 4, 5]:
        print("  ✅ Full pipeline working")
        return True
    else:
        print("  ❌ Invalid ESI level")
        return False


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("RiskScope AI - Quick Test Suite")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Feature Engineering", test_feature_engineering),
        ("Emergency Rules", test_emergency_rules),
        ("Synthetic Data", test_synthetic_data),
        ("Model Training", test_model_training),
        ("Full Pipeline", test_full_pipeline),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"  ❌ {name} failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        icon = "✅" if result else "❌"
        print(f"  {icon} {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready for full training.")
    else:
        print("\n⚠️  Some tests failed. Please fix issues before training.")
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
