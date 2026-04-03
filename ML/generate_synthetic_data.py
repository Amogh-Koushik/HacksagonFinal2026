"""
RiskScope AI - Synthetic Data Generator
Generates realistic synthetic patient data for model training and testing.
Use this when MIMIC-IV is not available.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
import random


def generate_synthetic_dataset(n_samples: int = 10000, 
                               random_state: int = 42) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate synthetic patient data for ESI triage prediction.
    
    The data is designed to be realistic:
    - ESI 1 patients have critical vital signs and emergency symptoms
    - ESI 5 patients have normal vitals and minor symptoms
    - Class distribution follows typical ED distribution
    
    Parameters:
    -----------
    n_samples : Number of samples to generate
    random_state : Random seed for reproducibility
    
    Returns:
    --------
    X : DataFrame with features
    y : Array with ESI levels (1-5)
    """
    np.random.seed(random_state)
    random.seed(random_state)
    
    # ESI distribution (typical ED: more level 3-4)
    # ESI 1: 2%, ESI 2: 10%, ESI 3: 35%, ESI 4: 35%, ESI 5: 18%
    esi_distribution = {
        1: int(n_samples * 0.02),
        2: int(n_samples * 0.10),
        3: int(n_samples * 0.35),
        4: int(n_samples * 0.35),
        5: n_samples - int(n_samples * 0.02) - int(n_samples * 0.10) - 
           int(n_samples * 0.35) - int(n_samples * 0.35)
    }
    
    all_patients = []
    all_labels = []
    
    for esi_level, count in esi_distribution.items():
        for _ in range(count):
            patient = generate_patient(esi_level)
            all_patients.append(patient)
            all_labels.append(esi_level)
    
    # Shuffle
    combined = list(zip(all_patients, all_labels))
    random.shuffle(combined)
    all_patients, all_labels = zip(*combined)
    
    df = pd.DataFrame(all_patients)
    y = np.array(all_labels)
    
    print(f"Generated {len(df)} synthetic patients")
    print(f"Class distribution:")
    for i in range(1, 6):
        count = (y == i).sum()
        print(f"  ESI {i}: {count} ({count/len(y)*100:.1f}%)")
    
    return df, y


def generate_patient(esi_level: int) -> Dict:
    """Generate a single patient with characteristics appropriate for ESI level."""
    
    # Base demographics
    age = generate_age(esi_level)
    gender = np.random.choice(['M', 'F'])
    
    # Vitals based on ESI level
    vitals = generate_vitals(esi_level)
    
    # Symptoms based on ESI level
    symptoms = generate_symptoms(esi_level)
    
    # Temporal features
    duration = generate_duration(esi_level)
    
    patient = {
        'age': age,
        'gender': 1 if gender == 'M' else 0,
        **vitals,
        **symptoms,
        'symptom_duration_hours': duration
    }
    
    return patient


def generate_age(esi_level: int) -> int:
    """Generate age appropriate for ESI level."""
    if esi_level == 1:
        # Emergencies more common in elderly
        return int(np.clip(np.random.normal(65, 15), 18, 95))
    elif esi_level == 2:
        return int(np.clip(np.random.normal(55, 18), 18, 90))
    elif esi_level == 3:
        return int(np.clip(np.random.normal(45, 20), 18, 85))
    elif esi_level == 4:
        return int(np.clip(np.random.normal(35, 15), 18, 75))
    else:  # ESI 5
        return int(np.clip(np.random.normal(30, 12), 18, 70))


def generate_vitals(esi_level: int) -> Dict:
    """Generate vital signs appropriate for ESI level."""
    
    if esi_level == 1:
        # Critical vitals
        return {
            'heart_rate': int(np.clip(np.random.choice([
                np.random.normal(130, 20),  # Tachycardia
                np.random.normal(35, 5)     # Bradycardia
            ]), 20, 200)),
            'bp_systolic': int(np.clip(np.random.choice([
                np.random.normal(80, 10),   # Hypotension
                np.random.normal(200, 15)   # Hypertensive crisis
            ]), 50, 250)),
            'bp_diastolic': int(np.clip(np.random.normal(90, 20), 40, 140)),
            'spo2': int(np.clip(np.random.normal(85, 5), 60, 92)),
            'temperature': float(np.clip(np.random.choice([
                np.random.normal(39.5, 0.5),  # High fever
                np.random.normal(35, 0.5)     # Hypothermia
            ]), 32, 42)),
            'respiratory_rate': int(np.clip(np.random.normal(32, 6), 6, 50))
        }
    
    elif esi_level == 2:
        # Concerning vitals
        return {
            'heart_rate': int(np.clip(np.random.normal(110, 15), 50, 160)),
            'bp_systolic': int(np.clip(np.random.choice([
                np.random.normal(95, 8),
                np.random.normal(170, 12)
            ]), 70, 200)),
            'bp_diastolic': int(np.clip(np.random.normal(95, 15), 50, 120)),
            'spo2': int(np.clip(np.random.normal(92, 3), 88, 96)),
            'temperature': float(np.clip(np.random.normal(38.5, 0.6), 36, 40)),
            'respiratory_rate': int(np.clip(np.random.normal(24, 4), 14, 35))
        }
    
    elif esi_level == 3:
        # Mildly abnormal vitals
        return {
            'heart_rate': int(np.clip(np.random.normal(95, 15), 55, 130)),
            'bp_systolic': int(np.clip(np.random.normal(135, 15), 100, 165)),
            'bp_diastolic': int(np.clip(np.random.normal(85, 10), 60, 100)),
            'spo2': int(np.clip(np.random.normal(96, 2), 93, 99)),
            'temperature': float(np.clip(np.random.normal(37.5, 0.5), 36.5, 38.5)),
            'respiratory_rate': int(np.clip(np.random.normal(18, 3), 12, 24))
        }
    
    elif esi_level == 4:
        # Near-normal vitals
        return {
            'heart_rate': int(np.clip(np.random.normal(80, 12), 55, 110)),
            'bp_systolic': int(np.clip(np.random.normal(125, 12), 100, 145)),
            'bp_diastolic': int(np.clip(np.random.normal(80, 8), 60, 95)),
            'spo2': int(np.clip(np.random.normal(98, 1), 95, 100)),
            'temperature': float(np.clip(np.random.normal(37.1, 0.3), 36.5, 37.8)),
            'respiratory_rate': int(np.clip(np.random.normal(16, 2), 12, 20))
        }
    
    else:  # ESI 5
        # Normal vitals
        return {
            'heart_rate': int(np.clip(np.random.normal(75, 10), 55, 100)),
            'bp_systolic': int(np.clip(np.random.normal(118, 10), 100, 135)),
            'bp_diastolic': int(np.clip(np.random.normal(75, 8), 60, 90)),
            'spo2': int(np.clip(np.random.normal(99, 0.5), 97, 100)),
            'temperature': float(np.clip(np.random.normal(36.8, 0.3), 36.2, 37.5)),
            'respiratory_rate': int(np.clip(np.random.normal(14, 2), 12, 18))
        }


def generate_symptoms(esi_level: int) -> Dict:
    """Generate symptoms appropriate for ESI level."""
    
    # All possible symptoms (binary)
    all_symptoms = [
        'chest_pain', 'arm_pain_left', 'jaw_pain', 'dyspnea', 'shortness_of_breath',
        'facial_droop', 'arm_weakness', 'speech_difficulty', 'abdominal_pain',
        'rigid_abdomen', 'altered_mental_status', 'confusion', 'fever', 'nausea',
        'vomiting', 'dizziness', 'syncope', 'headache', 'seizure',
        'uncontrolled_bleeding', 'severe_pain', 'cough', 'fatigue'
    ]
    
    symptoms = {s: 0 for s in all_symptoms}
    
    if esi_level == 1:
        # Critical symptoms - cardiac, stroke, or respiratory
        scenario = np.random.choice(['cardiac', 'stroke', 'respiratory', 'shock', 'sepsis'],
                                    p=[0.25, 0.20, 0.20, 0.20, 0.15])
        
        if scenario == 'cardiac':
            symptoms['chest_pain'] = 1
            symptoms['arm_pain_left'] = np.random.choice([0, 1], p=[0.3, 0.7])
            symptoms['jaw_pain'] = np.random.choice([0, 1], p=[0.6, 0.4])
            symptoms['dyspnea'] = np.random.choice([0, 1], p=[0.4, 0.6])
            symptoms['severe_pain'] = 1
        
        elif scenario == 'stroke':
            symptoms['facial_droop'] = np.random.choice([0, 1], p=[0.2, 0.8])
            symptoms['arm_weakness'] = np.random.choice([0, 1], p=[0.2, 0.8])
            symptoms['speech_difficulty'] = np.random.choice([0, 1], p=[0.3, 0.7])
            symptoms['altered_mental_status'] = np.random.choice([0, 1], p=[0.4, 0.6])
        
        elif scenario == 'respiratory':
            symptoms['dyspnea'] = 1
            symptoms['shortness_of_breath'] = 1
            symptoms['altered_mental_status'] = np.random.choice([0, 1], p=[0.5, 0.5])
        
        elif scenario == 'shock':
            symptoms['altered_mental_status'] = 1
            symptoms['dizziness'] = 1
            symptoms['confusion'] = 1
            symptoms['uncontrolled_bleeding'] = np.random.choice([0, 1], p=[0.6, 0.4])
        
        else:  # sepsis
            symptoms['fever'] = 1
            symptoms['altered_mental_status'] = 1
            symptoms['confusion'] = 1
    
    elif esi_level == 2:
        # Serious conditions — clear high-acuity symptoms
        scenario = np.random.choice(
            ['chest_pain_acs', 'abdominal_severe', 'neuro_acute', 'infection_severe', 'trauma'],
            p=[0.25, 0.20, 0.20, 0.20, 0.15]
        )
        
        if scenario == 'chest_pain_acs':
            symptoms['chest_pain'] = 1
            symptoms['dyspnea'] = np.random.choice([0, 1], p=[0.4, 0.6])
            symptoms['severe_pain'] = np.random.choice([0, 1], p=[0.3, 0.7])
        
        elif scenario == 'abdominal_severe':
            symptoms['abdominal_pain'] = 1
            symptoms['severe_pain'] = 1
            symptoms['nausea'] = np.random.choice([0, 1], p=[0.2, 0.8])
            symptoms['vomiting'] = np.random.choice([0, 1], p=[0.3, 0.7])
            symptoms['rigid_abdomen'] = np.random.choice([0, 1], p=[0.5, 0.5])
        
        elif scenario == 'neuro_acute':
            symptoms['headache'] = 1
            symptoms['dizziness'] = np.random.choice([0, 1], p=[0.3, 0.7])
            symptoms['syncope'] = np.random.choice([0, 1], p=[0.5, 0.5])
            symptoms['vomiting'] = np.random.choice([0, 1], p=[0.5, 0.5])
        
        elif scenario == 'infection_severe':
            symptoms['fever'] = 1
            symptoms['confusion'] = np.random.choice([0, 1], p=[0.4, 0.6])
            symptoms['altered_mental_status'] = np.random.choice([0, 1], p=[0.5, 0.5])
        
        else:  # trauma
            symptoms['uncontrolled_bleeding'] = 1
            symptoms['severe_pain'] = 1
    
    elif esi_level == 3:
        # Urgent — moderate severity, multiple resource needs
        scenario = np.random.choice(
            ['abdominal_mod', 'pain_multi', 'respiratory_mild', 'gi_illness', 'back_pain'],
            p=[0.25, 0.20, 0.20, 0.20, 0.15]
        )
        
        if scenario == 'abdominal_mod':
            symptoms['abdominal_pain'] = 1
            symptoms['nausea'] = np.random.choice([0, 1], p=[0.3, 0.7])
            symptoms['vomiting'] = np.random.choice([0, 1], p=[0.4, 0.6])
            symptoms['fever'] = np.random.choice([0, 1], p=[0.6, 0.4])
        
        elif scenario == 'pain_multi':
            symptoms['severe_pain'] = 1
            symptoms['headache'] = np.random.choice([0, 1], p=[0.5, 0.5])
            symptoms['nausea'] = np.random.choice([0, 1], p=[0.5, 0.5])
        
        elif scenario == 'respiratory_mild':
            symptoms['shortness_of_breath'] = 1
            symptoms['cough'] = np.random.choice([0, 1], p=[0.4, 0.6])
            symptoms['fever'] = np.random.choice([0, 1], p=[0.4, 0.6])
        
        elif scenario == 'gi_illness':
            symptoms['nausea'] = 1
            symptoms['vomiting'] = 1
            symptoms['abdominal_pain'] = np.random.choice([0, 1], p=[0.4, 0.6])
            symptoms['fever'] = np.random.choice([0, 1], p=[0.6, 0.4])
        
        else:  # back_pain
            symptoms['severe_pain'] = 1
            symptoms['vomiting'] = np.random.choice([0, 1], p=[0.7, 0.3])
    
    elif esi_level == 4:
        # Less urgent — single minor complaint, near-normal vitals
        scenario = np.random.choice(
            ['mild_headache', 'minor_gi', 'uri', 'minor_injury', 'back_ache'],
            p=[0.20, 0.20, 0.25, 0.20, 0.15]
        )
        
        if scenario == 'mild_headache':
            symptoms['headache'] = 1
            # Occasionally mild nausea with headache
            symptoms['nausea'] = np.random.choice([0, 1], p=[0.7, 0.3])
        
        elif scenario == 'minor_gi':
            symptoms['nausea'] = 1
            symptoms['vomiting'] = np.random.choice([0, 1], p=[0.6, 0.4])
        
        elif scenario == 'uri':  # Upper Respiratory Infection
            symptoms['cough'] = 1
            symptoms['fever'] = np.random.choice([0, 1], p=[0.5, 0.5])
            symptoms['fatigue'] = np.random.choice([0, 1], p=[0.5, 0.5])
        
        elif scenario == 'minor_injury':
            symptoms['severe_pain'] = np.random.choice([0, 1], p=[0.7, 0.3])  # Mild pain
        
        else:  # back_ache
            symptoms['fatigue'] = np.random.choice([0, 1], p=[0.6, 0.4])
    
    else:  # ESI 5
        # Non-urgent — minor or no symptoms, stable vitals
        scenario = np.random.choice(
            ['no_symptoms', 'mild_cough', 'mild_fatigue', 'minor_headache'],
            p=[0.50, 0.20, 0.15, 0.15]
        )
        if scenario == 'mild_cough':
            symptoms['cough'] = 1
        elif scenario == 'mild_fatigue':
            symptoms['fatigue'] = 1
        elif scenario == 'minor_headache':
            symptoms['headache'] = 1
        # 50% of ESI 5 have zero symptoms
    
    return symptoms


def generate_duration(esi_level: int) -> float:
    """Generate symptom duration in hours appropriate for ESI level."""
    
    if esi_level == 1:
        # Acute onset
        return float(np.clip(np.random.exponential(1), 0.1, 6))
    elif esi_level == 2:
        return float(np.clip(np.random.exponential(3), 0.5, 24))
    elif esi_level == 3:
        return float(np.clip(np.random.exponential(12), 2, 72))
    elif esi_level == 4:
        return float(np.clip(np.random.exponential(24), 6, 168))
    else:  # ESI 5
        return float(np.clip(np.random.exponential(48), 24, 336))


def save_synthetic_data(n_samples: int = 10000, output_path: str = 'data/synthetic_training_data.csv'):
    """Generate and save synthetic dataset."""
    import os
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    
    X, y = generate_synthetic_dataset(n_samples)
    
    # Add target column
    X['esi_level'] = y
    
    X.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    
    return X


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic ESI training data')
    parser.add_argument('--samples', type=int, default=10000, help='Number of samples')
    parser.add_argument('--output', type=str, default='data/synthetic_training_data.csv', 
                        help='Output path')
    
    args = parser.parse_args()
    
    save_synthetic_data(args.samples, args.output)
