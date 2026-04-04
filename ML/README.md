# RiskScope AI - Machine Learning Module

## [AIM] Overview

This module contains the core ML pipeline for Emergency Severity Index (ESI) prediction. The system uses a multi-layer safety architecture to ensure accurate and safe patient triage.

## [FOLDER] Structure

```
ML/
├── __init__.py              # Module exports
├── config.py                # Configuration & hyperparameters
├── esi_predictor.py         # Core ML model (LightGBM + RF ensemble)
├── safety_engine.py         # Emergency rules & safety layers
├── feature_engineering.py   # 73+ feature extraction pipeline
├── ood_detector.py          # Out-of-distribution detection
├── evaluate.py              # Evaluation metrics & analysis
├── train.py                 # Training script
├── generate_synthetic_data.py # Synthetic data generator
├── test_quick.py            # Quick test suite
└── requirements.txt         # Dependencies
```

## [LAUNCH] Quick Start

### 1. Install Dependencies

```bash
cd ML
pip install -r requirements.txt
```

### 2. Run Quick Tests

```bash
python test_quick.py
```

### 3. Train Model (Synthetic Data)

```bash
python train.py --synthetic
```

### 4. Train Model (Real Data)

```bash
python train.py --data path/to/training_data.csv
```

## [BUILD] Architecture

### 3-Layer Safety System

```
Patient Data
     │
     ▼
┌─────────────────────────────────┐
│  Layer 1: Emergency Rules       │  ← Hard-coded, CANNOT fail
│  (18 ACLS/PALS validated rules) │  
└─────────────────────────────────┘
     │ (if not triggered)
     ▼
┌─────────────────────────────────┐
│  Layer 2: OOD Detection         │  ← Flag unusual cases
│  (Escalate if uncertain)        │
└─────────────────────────────────┘
     │ (if in-distribution)
     ▼
┌─────────────────────────────────┐
│  Layer 3: ML Prediction         │  ← LightGBM + RF Ensemble
│  (60% LGB + 40% RF)             │
└─────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────┐
│  Confidence Calibration         │  ← Escalate if confidence < 60%
└─────────────────────────────────┘
     │
     ▼
   ESI Level 1-5 + Explanation
```

### Model Details

| Component | Technology | Purpose |
|-----------|------------|---------|
| Primary Model | LightGBM | Fast gradient boosting (60% weight) |
| Secondary Model | RandomForest | Interpretable ensemble (40% weight) |
| Class Balance | SMOTE | Handle rare ESI 1-2 cases |
| Explainability | SHAP | "Why this ESI level?" |
| Preprocessing | StandardScaler | Normalize features |

## [CHART] Features (73+)

### Demographics (5)
- Age, age groups (pediatric, adult, elderly), gender

### Vitals (15)
- Raw: HR, BP_sys, BP_dia, SpO2, Temp, RR
- Normalized versions
- Derived: MAP, Pulse Pressure, Shock Index

### Symptoms (21+)
- Binary flags for all symptoms
- Total symptom count

### Clinical Interactions (8)
- Cardiac patterns (chest + arm pain)
- Stroke FAST criteria
- Sepsis indicators
- Acute abdomen signs

### Risk Scores (6)
- SIRS score
- qSOFA score
- NEWS score
- Positive flags for each

### Derangement Indicators (9)
- Critical hypoxia, hypotension, hypertension
- Critical bradycardia, tachycardia
- Critical fever, tachypnea
- Shock pattern
- Total derangement count

### Temporal (4)
- Symptom duration
- Is acute, rapid onset, chronic

## [AIM] Target Metrics

| Metric | Target | Importance |
|--------|--------|------------|
| Cohen's Kappa | ≥ 0.75 | Excellent agreement with nurses |
| ESI 1-2 Sensitivity | ≥ 95% | Must catch emergencies |
| ESI 1-2 Specificity | ≥ 75% | Avoid over-triage |
| Critical Errors | < 0.5% | Level 1 -> Level 4+ mistakes |
| Response Time | < 500ms | Real-time use |

## [TOOL] Usage

### Training

```python
from ML import ESITriagePredictor
from ML.generate_synthetic_data import generate_synthetic_dataset

# Generate or load data
X, y = generate_synthetic_dataset(n_samples=10000)

# Train model
model = ESITriagePredictor(use_smote=True)
model.fit(X, y, validate=True, verbose=True)

# Save model
model.save('models/esi_model.pkl')
```

### Inference

```python
from ML import ESITriagePredictor, ClinicalSafetyEngine, FeatureEngineer

# Load model
model = ESITriagePredictor.load('models/esi_model.pkl')
fe = FeatureEngineer()
safety_engine = ClinicalSafetyEngine(ml_model=model, feature_engineer=fe)

# Triage patient
patient = {
    'age': 58,
    'heart_rate': 105,
    'bp_systolic': 160,
    'spo2': 94,
    'chest_pain': True,
    'arm_pain_left': True,
}

result = safety_engine.triage(patient)
print(f"ESI Level: {result['esi_level']}")
print(f"Confidence: {result['confidence']}")
print(f"Method: {result['method']}")
```

### Explanation

```python
# Get SHAP explanation
explanation = model.explain_prediction(features, top_n=5)
for exp in explanation:
    print(f"{exp['impact']} {exp['feature']}: {exp['value']}")
```

## [ALERT] Emergency Rules

The system includes 18+ hard-coded emergency rules based on ACLS/PALS protocols:

| Rule | Trigger | ESI |
|------|---------|-----|
| Suspected MI | chest_pain + arm_pain_left | 1 |
| Respiratory Failure | SpO2 < 90% | 1 |
| Stroke FAST | facial_droop + arm_weakness | 1 |
| Shock | SBP < 90 + HR > 100 | 1 |
| Sepsis | fever + AMS + tachycardia | 1 |
| ... | ... | ... |

These rules **cannot be overridden** by ML predictions.

## [UP] Evaluation

```python
from ML import ESIEvaluator, run_evaluation

# Quick evaluation
metrics = run_evaluation(model, X_test, y_test, output_dir='results/')

# Detailed evaluation
evaluator = ESIEvaluator()
metrics = evaluator.evaluate(y_true, y_pred, verbose=True)
```

##  Safety Considerations

1. **Rule-based safety layer runs FIRST** - Cannot be bypassed
2. **OOD detection escalates unusual cases** - Fail-safe
3. **Low confidence -> automatic escalation** - Err on caution
4. **Critical error tracking** - Monitor under-triage
5. **Human oversight required** - Decision support only

## [NOTE] License

Internal use for RiskScope AI hackathon project.
