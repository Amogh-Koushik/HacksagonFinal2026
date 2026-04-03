"""
RiskScope AI - Training Script
Main training pipeline for ESI Triage Predictor
"""

import numpy as np
import pandas as pd
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from esi_predictor import ESITriagePredictor
from feature_engineering import FeatureEngineer
from safety_engine import EmergencyRuleEngine, ClinicalSafetyEngine
from ood_detector import OutOfDistributionDetector
from config import MODEL_SAVE_PATH, TARGET_METRICS


def load_data(data_path: str) -> tuple:
    """
    Load and prepare training data.
    
    Expected CSV columns:
    - Demographics: age, gender
    - Vitals: heart_rate, bp_systolic, bp_diastolic, spo2, temperature, respiratory_rate
    - Symptoms: chest_pain, arm_pain_left, etc. (binary)
    - Target: esi_level (1-5)
    """
    print(f"Loading data from {data_path}...")
    
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} records")
    
    # Separate features and target
    if 'esi_level' not in df.columns:
        raise ValueError("Data must contain 'esi_level' column")
    
    y = df['esi_level'].values
    X = df.drop(columns=['esi_level'])
    
    # Handle any ID columns
    id_cols = [c for c in X.columns if 'id' in c.lower()]
    if id_cols:
        X = X.drop(columns=id_cols)
    
    print(f"  Features: {X.shape[1]}")
    print(f"  Target distribution:")
    for i in range(1, 6):
        count = (y == i).sum()
        pct = count / len(y) * 100
        print(f"    ESI {i}: {count} ({pct:.1f}%)")
    
    return X, y


def train_model(X, y, output_dir: str, verbose: bool = True):
    """Train the ESI predictor and save model."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    if verbose:
        print(f"\nTrain set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
    
    # Initialize and train model
    predictor = ESITriagePredictor(use_smote=True)
    
    start_time = time.time()
    predictor.fit(X_train, y_train, validate=True, verbose=verbose)
    train_time = time.time() - start_time
    
    if verbose:
        print(f"\nTraining time: {train_time:.1f} seconds")
    
    # Evaluate on test set
    metrics = predictor.evaluate(X_test, y_test, verbose=verbose)
    
    # Train OOD detector
    if verbose:
        print("\nTraining Out-of-Distribution detector...")
    ood_detector = OutOfDistributionDetector()
    ood_detector.fit(X_train.values if hasattr(X_train, 'values') else X_train)
    
    # Save models
    model_path = os.path.join(output_dir, 'esi_ensemble_model.pkl')
    predictor.save(model_path)
    
    ood_path = os.path.join(output_dir, 'ood_detector.pkl')
    ood_detector.save(ood_path)
    
    # Save training report
    report = {
        'timestamp': datetime.now().isoformat(),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'training_time_seconds': train_time,
        'metrics': {k: v for k, v in metrics.items() if k != 'confusion_matrix'},
        'targets_met': {
            'cohens_kappa': metrics['cohens_kappa'] >= TARGET_METRICS['cohens_kappa'],
            'esi_12_sensitivity': metrics.get('esi_12_sensitivity', 0) >= TARGET_METRICS['esi_12_sensitivity'],
        }
    }
    
    report_path = os.path.join(output_dir, 'training_report.txt')
    with open(report_path, 'w') as f:
        f.write("RiskScope AI - Training Report\n")
        f.write("=" * 50 + "\n\n")
        for key, value in report.items():
            if isinstance(value, dict):
                f.write(f"{key}:\n")
                for k, v in value.items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    if verbose:
        print(f"\nModel saved to: {model_path}")
        print(f"OOD detector saved to: {ood_path}")
        print(f"Report saved to: {report_path}")
    
    return predictor, ood_detector, metrics


def test_inference(predictor, sample_cases: list = None):
    """Test inference on sample cases."""
    
    if sample_cases is None:
        # Use demo cases
        sample_cases = [
            {
                "name": "Suspected MI",
                "age": 58, "gender": "M",
                "heart_rate": 105, "bp_systolic": 160, "bp_diastolic": 95, "spo2": 94,
                "temperature": 37.0, "respiratory_rate": 20,
                "chest_pain": True, "arm_pain_left": True, "dyspnea": True,
                "expected_esi": 1
            },
            {
                "name": "Stroke FAST+",
                "age": 72, "gender": "F",
                "heart_rate": 88, "bp_systolic": 180, "bp_diastolic": 110, "spo2": 96,
                "temperature": 36.8, "respiratory_rate": 18,
                "facial_droop": True, "arm_weakness": True, "speech_difficulty": True,
                "expected_esi": 1
            },
            {
                "name": "Abdominal Pain",
                "age": 35, "gender": "F",
                "heart_rate": 85, "bp_systolic": 120, "bp_diastolic": 80, "spo2": 98,
                "temperature": 37.2, "respiratory_rate": 16,
                "abdominal_pain": True, "nausea": True,
                "expected_esi": 3
            },
            {
                "name": "Common Cold",
                "age": 25, "gender": "F",
                "heart_rate": 70, "bp_systolic": 110, "bp_diastolic": 70, "spo2": 99,
                "temperature": 37.5, "respiratory_rate": 14,
                "cough": True, "fever": False,
                "expected_esi": 5
            },
        ]
    
    print("\n" + "=" * 60)
    print("Testing Inference on Sample Cases")
    print("=" * 60)
    
    # Initialize safety engine
    safety_engine = EmergencyRuleEngine()
    fe = FeatureEngineer()
    
    for case in sample_cases:
        name = case.pop('name', 'Unknown')
        expected = case.pop('expected_esi', None)
        
        print(f"\n📋 Case: {name}")
        
        # Check emergency rules first
        emergency_result = safety_engine.check(case)
        if emergency_result.triggered:
            print(f"  🚨 EMERGENCY RULE TRIGGERED: {emergency_result.rule_name}")
            print(f"  ESI Level: {emergency_result.esi_level} (Rule-Based)")
            print(f"  Protocol: {emergency_result.protocol}")
            predicted = emergency_result.esi_level
        else:
            # ML prediction - use the same column order as training data
            # Get feature columns from the predictor's scaler
            feature_cols = ['age', 'gender', 'heart_rate', 'bp_systolic', 'bp_diastolic', 
                           'spo2', 'temperature', 'respiratory_rate', 'symptom_duration_hours',
                           'chest_pain', 'arm_pain_left', 'jaw_pain', 'dyspnea', 'severe_pain',
                           'facial_droop', 'arm_weakness', 'speech_difficulty', 'altered_mental_status',
                           'shortness_of_breath', 'dizziness', 'confusion', 'uncontrolled_bleeding',
                           'fever', 'abdominal_pain', 'nausea', 'vomiting', 'headache', 'syncope',
                           'cough', 'fatigue']
            
            # Convert gender string to int if needed
            case_copy = case.copy()
            if isinstance(case_copy.get('gender'), str):
                case_copy['gender'] = 1 if case_copy['gender'] == 'M' else 0
            
            features = np.array([[float(case_copy.get(col, 0)) for col in feature_cols]])
            
            predicted = predictor.predict(features)[0]
            proba = predictor.predict_proba(features)[0]
            confidence = np.max(proba)
            
            print(f"  ESI Level: {predicted} (ML Prediction)")
            print(f"  Confidence: {confidence:.1%}")
        
        if expected:
            match = "✅" if predicted == expected else "⚠️"
            print(f"  Expected: {expected} {match}")
        
        # Restore for next iteration
        case['name'] = name
        case['expected_esi'] = expected


def main():
    parser = argparse.ArgumentParser(description='Train RiskScope AI ESI Predictor')
    parser.add_argument('--data', type=str, help='Path to training CSV')
    parser.add_argument('--output', type=str, default='models/', help='Output directory')
    parser.add_argument('--synthetic', action='store_true', help='Generate synthetic data for testing')
    parser.add_argument('--test-only', action='store_true', help='Only test inference with existing model')
    
    args = parser.parse_args()
    
    if args.test_only:
        # Load existing model and test
        model_path = os.path.join(args.output, 'esi_ensemble_model.pkl')
        if os.path.exists(model_path):
            predictor = ESITriagePredictor.load(model_path)
            test_inference(predictor)
        else:
            print(f"No model found at {model_path}")
        return
    
    if args.synthetic:
        # Generate synthetic data for testing
        from generate_synthetic_data import generate_synthetic_dataset
        X, y = generate_synthetic_dataset(n_samples=5000)
    elif args.data:
        X, y = load_data(args.data)
    else:
        print("Please provide --data PATH or use --synthetic for testing")
        print("Usage: python train.py --data training_data.csv")
        print("       python train.py --synthetic")
        return
    
    # Train model
    predictor, ood_detector, metrics = train_model(X, y, args.output)
    
    # Test inference
    test_inference(predictor)
    
    # Summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Cohen's Kappa: {metrics['cohens_kappa']:.3f}")
    print(f"ESI 1-2 Sensitivity: {metrics.get('esi_12_sensitivity', 0):.1%}")
    print(f"Critical Errors: {metrics['critical_errors']}")
    
    # Check targets
    kappa_met = metrics['cohens_kappa'] >= TARGET_METRICS['cohens_kappa']
    sens_met = metrics.get('esi_12_sensitivity', 0) >= TARGET_METRICS['esi_12_sensitivity']
    
    print(f"\nTarget Metrics:")
    print(f"  Kappa >= {TARGET_METRICS['cohens_kappa']}: {'✅ MET' if kappa_met else '❌ NOT MET'}")
    print(f"  Sensitivity >= {TARGET_METRICS['esi_12_sensitivity']:.0%}: {'✅ MET' if sens_met else '❌ NOT MET'}")


if __name__ == '__main__':
    main()
