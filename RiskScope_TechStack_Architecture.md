# Complete Tech Stack Architecture Flowchart
## RiskScope AI - ML/Data Science Focused

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE TECH STACK ARCHITECTURE                          │
│                   RiskScope AI - ML & Data Science Focus                     │
└─────────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
                         🖥️ APPLICATION LAYER (Brief)
═══════════════════════════════════════════════════════════════════════════════

    ┌─────────────────┐                         ┌─────────────────┐
    │    FRONTEND     │  ←───── HTTPS ─────→    │    BACKEND      │
    │                 │                         │                 │
    │  React + Vite   │                         │  Flask + Python │
    │  Vanilla CSS    │                         │  REST API       │
    │  Recharts       │                         │  Gunicorn       │
    └────────┬────────┘                         └────────┬────────┘
             │                                           │
             └───────────────────┬───────────────────────┘
                                 │
                                 ▼
═══════════════════════════════════════════════════════════════════════════════
                    🧠 MACHINE LEARNING PIPELINE (Core Focus)
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│                         DATA INGESTION & GENERATION                          │
│                                                                              │
│      ┌─────────────────────────────────────────────────────────────┐        │
│      │                        SYNTHEA                               │        │
│      │              (Synthetic Patient Data Generator)              │        │
│      │                                                              │        │
│      │   • Java-based realistic patient simulation                  │        │
│      │   • 50,000 synthetic patients generated                      │        │
│      │   • Demographics, symptoms, vitals, conditions               │        │
│      │   • ICD-10 codes mapped to ESI levels                        │        │
│      │                                                              │        │
│      │   Future: MIMIC-IV (448,972 real ED visits)                  │        │
│      └─────────────────────────────┬───────────────────────────────┘        │
│                                    │                                         │
│                                    ▼                                         │
│      ┌─────────────────────────────────────────────────────────────┐        │
│      │                    DATA PROCESSING                           │        │
│      │                                                              │        │
│      │   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │        │
│      │   │   PANDAS    │   │   NUMPY     │   │  CLEANING   │       │        │
│      │   │             │   │             │   │             │       │        │
│      │   │ DataFrames  │   │ Numerical   │   │ Missing     │       │        │
│      │   │ CSV loading │   │ operations  │   │ values      │       │        │
│      │   │ Filtering   │   │ Arrays      │   │ Outliers    │       │        │
│      │   └─────────────┘   └─────────────┘   └─────────────┘       │        │
│      │                                                              │        │
│      └─────────────────────────────┬───────────────────────────────┘        │
│                                    │                                         │
│                                    ▼                                         │
└─────────────────────────────────────────────────────────────────────────────┘



┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│                         MODEL TRAINING                                       │
│                                                                              │
│      ┌─────────────────────────────────────────────────────────────┐        │
│      │                  ENSEMBLE ARCHITECTURE                       │        │
│      │                                                              │        │
│      │         ┌─────────────────────────────────────────┐         │        │
│      │         │            TRAINING DATA                 │         │        │
│      │         │         (50,000 patients)                │         │        │
│      │         └───────────────────┬─────────────────────┘         │        │
│      │                             │                                │        │
│      │              ┌──────────────┴──────────────┐                │        │
│      │              ▼                              ▼                │        │
│      │   ┌─────────────────────┐     ┌─────────────────────┐       │        │
│      │   │     LIGHTGBM        │     │   RANDOM FOREST     │       │        │
│      │   │                     │     │                     │       │        │
│      │   │  • Gradient Boosting│     │  • 100 trees        │       │        │
│      │   │  • max_depth: 6     │     │  • max_depth: 8     │       │        │
│      │   │  • num_leaves: 31   │     │  • class_weight:    │       │        │
│      │   │  • learning_rate:   │     │    balanced         │       │        │
│      │   │    0.05             │     │                     │       │        │
│      │   │  • Fast training    │     │  • Interpretable    │       │        │
│      │   │  • High accuracy    │     │  • Robust           │       │        │
│      │   │                     │     │                     │       │        │
│      │   │  Weight: 60%        │     │  Weight: 40%        │       │        │
│      │   └──────────┬──────────┘     └──────────┬──────────┘       │        │
│      │              │                            │                  │        │
│      │              └──────────────┬─────────────┘                  │        │
│      │                             ▼                                │        │
│      │         ┌─────────────────────────────────────────┐         │        │
│      │         │         WEIGHTED ENSEMBLE               │         │        │
│      │         │                                         │         │        │
│      │         │   final_pred = 0.6×LGB + 0.4×RF         │         │        │
│      │         │                                         │         │        │
│      │         │   Output: ESI Level 1-5 + Confidence    │         │        │
│      │         └─────────────────────────────────────────┘         │        │
│      │                                                              │        │
│      └─────────────────────────────┬───────────────────────────────┘        │
│                                    │                                         │
│                                    ▼                                         │
│      ┌─────────────────────────────────────────────────────────────┐        │
│      │                  CLASS IMBALANCE HANDLING                    │        │
│      │                                                              │        │
│      │   Problem: ESI 1-2 (rare) vs ESI 4-5 (common)               │        │
│      │                                                              │        │
│      │   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │        │
│      │   │   SMOTE     │   │  CLASS      │   │  SAMPLE     │       │        │
│      │   │             │   │  WEIGHTS    │   │  WEIGHTS    │       │        │
│      │   │ Synthetic   │   │ Balanced    │   │ Inverse     │       │        │
│      │   │ oversampling│   │ weighting   │   │ frequency   │       │        │
│      │   └─────────────┘   └─────────────┘   └─────────────┘       │        │
│      │                                                              │        │
│      │   Library: imbalanced-learn (imblearn)                       │        │
│      │                                                              │        │
│      └─────────────────────────────────────────────────────────────┘        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│                         MODEL EVALUATION                                     │
│                                                                              │
│      ┌─────────────────────────────────────────────────────────────┐        │
│      │                  VALIDATION STRATEGY                         │        │
│      │                                                              │        │
│      │   ┌─────────────────────────────────────────────────────┐   │        │
│      │   │  5-FOLD CROSS-VALIDATION (Patient-Level Splits)     │   │        │
│      │   │                                                      │   │        │
│      │   │  Fold 1: Train on 80% → Test on 20%                 │   │        │
│      │   │  Fold 2: Train on 80% → Test on 20%                 │   │        │
│      │   │  ...                                                 │   │        │
│      │   │  Average across folds                                │   │        │
│      │   └─────────────────────────────────────────────────────┘   │        │
│      │                                                              │        │
│      └─────────────────────────────┬───────────────────────────────┘        │
│                                    │                                         │
│                                    ▼                                         │
│      ┌─────────────────────────────────────────────────────────────┐        │
│      │                  EVALUATION METRICS                          │        │
│      │                                                              │        │
│      │   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │        │
│      │   │ COHEN'S     │   │ SENSITIVITY │   │ CONFUSION   │       │        │
│      │   │ KAPPA       │   │ (ESI 1-2)   │   │ MATRIX      │       │        │
│      │   │             │   │             │   │             │       │        │
│      │   │ >0.75       │   │ >95%        │   │ Per-class   │       │        │
│      │   │ (Target)    │   │ (Critical)  │   │ accuracy    │       │        │
│      │   │             │   │             │   │             │       │        │
│      │   │ Agreement   │   │ Catch all   │   │ Detailed    │       │        │
│      │   │ with nurses │   │ emergencies │   │ breakdown   │       │        │
│      │   └─────────────┘   └─────────────┘   └─────────────┘       │        │
│      │                                                              │        │
│      │   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │        │
│      │   │ SPECIFICITY │   │ F1 SCORE    │   │ AUC-ROC     │       │        │
│      │   │             │   │             │   │             │       │        │
│      │   │ >75%        │   │ Per-class   │   │ Multi-class │       │        │
│      │   │ (Avoid      │   │ weighted    │   │ one-vs-rest │       │        │
│      │   │ over-triage)│   │ average     │   │             │       │        │
│      │   └─────────────┘   └─────────────┘   └─────────────┘       │        │
│      │                                                              │        │
│      │   Library: scikit-learn (sklearn.metrics)                    │        │
│      │                                                              │        │
│      └─────────────────────────────────────────────────────────────┘        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│                         EXPLAINABILITY (SHAP)                                │
│                                                                              │
│      ┌─────────────────────────────────────────────────────────────┐        │
│      │                  SHAP (SHapley Additive exPlanations)        │        │
│      │                                                              │        │
│      │   ┌─────────────────────────────────────────────────────┐   │        │
│      │   │                  TreeExplainer                       │   │        │
│      │   │                                                      │   │        │
│      │   │   For each prediction:                               │   │        │
│      │   │   • Calculate feature contributions                  │   │        │
│      │   │   • Show "Why ESI Level X?"                         │   │        │
│      │   │   • Positive = increases urgency                     │   │        │
│      │   │   • Negative = decreases urgency                     │   │        │
│      │   └─────────────────────────────────────────────────────┘   │        │
│      │                                                              │        │
│      │   Example Output:                                            │        │
│      │   ┌─────────────────────────────────────────────────────┐   │        │
│      │   │  "Why ESI Level 2?"                                  │   │        │
│      │   │                                                      │   │        │
│      │   │  ████████████████   Chest pain:        +35%          │   │        │
│      │   │  ██████████████     Age > 50:          +28%          │   │        │
│      │   │  ██████████         Tachycardia:       +18%          │   │        │
│      │   │  ████               Arm radiation:     +12%          │   │        │
│      │   │  ▓▓▓▓▓▓▓▓           Normal SpO2:       -15%          │   │        │
│      │   └─────────────────────────────────────────────────────┘   │        │
│      │                                                              │        │
│      │   Fallback: Feature Importance (if SHAP too slow)           │        │
│      │                                                              │        │
│      └─────────────────────────────────────────────────────────────┘        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│                         SAFETY LAYER (Rule-Based)                            │
│                                                                              │
│      ┌─────────────────────────────────────────────────────────────┐        │
│      │          EMERGENCY RULE ENGINE (Runs BEFORE ML)              │        │
│      │                                                              │        │
│      │   Hard-coded clinical rules (ACLS/ESI protocol):            │        │
│      │                                                              │        │
│      │   IF chest_pain AND arm_radiation    → ESI 1 (MI)           │        │
│      │   IF SpO2 < 90%                      → ESI 1 (Resp Failure) │        │
│      │   IF facial_droop AND arm_weakness   → ESI 1 (Stroke)       │        │
│      │   IF BP_systolic < 90 AND HR > 100   → ESI 1 (Shock)        │        │
│      │                                                              │        │
│      │   These rules CANNOT be overridden by ML                     │        │
│      │   Deterministic, auditable, fail-safe                        │        │
│      │                                                              │        │
│      └─────────────────────────────────────────────────────────────┘        │
│                                                                              │
│      ┌─────────────────────────────────────────────────────────────┐        │
│      │          CONFIDENCE CALIBRATION (Post-ML)                    │        │
│      │                                                              │        │
│      │   IF ml_confidence < 60%:                                    │        │
│      │       escalate_one_level()  # ESI 3 → ESI 2                 │        │
│      │       flag_for_review = True                                │        │
│      │                                                              │        │
│      │   Principle: When uncertain, be MORE cautious               │        │
│      │                                                              │        │
│      └─────────────────────────────────────────────────────────────┘        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
                         🚀 DEPLOYMENT & SERVING
═══════════════════════════════════════════════════════════════════════════════

    ┌─────────────────┐           ┌─────────────────┐
    │  MODEL SAVING   │    ──→    │  MODEL LOADING  │
    │                 │           │                 │
    │  joblib.dump()  │           │  joblib.load()  │
    │  model.pkl      │           │  Flask API      │
    │  scaler.pkl     │           │  /api/predict   │
    └─────────────────┘           └─────────────────┘
             │                             │
             │    ┌─────────────────┐      │
             └──→ │    GITHUB       │ ←────┘
                  │  Version Control│
                  └────────┬────────┘
                           │
              ┌────────────┴────────────┐
              ▼                          ▼
    ┌─────────────────┐       ┌─────────────────┐
    │    VERCEL       │       │    RAILWAY      │
    │   (Frontend)    │       │   (Backend+ML)  │
    │                 │       │                 │
    │  React App      │  ←→   │  Flask + Model  │
    └─────────────────┘       └─────────────────┘


═══════════════════════════════════════════════════════════════════════════════
                         📋 ML/DS TECH STACK SUMMARY
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  CATEGORY           TECHNOLOGY              PURPOSE                          │
│  ────────           ──────────              ───────                          │
│                                                                              │
│  DATA GENERATION    Synthea                 50K synthetic patients           │
│                     (Future: MIMIC-IV)      (448K real ED visits)            │
│                                                                              │
│  DATA PROCESSING    Pandas                  DataFrames, CSV handling         │
│                     NumPy                   Numerical operations             │
│                                                                              │
│  PREPROCESSING      scikit-learn            StandardScaler, LabelEncoder     │
│                                             Train/Test split                 │
│                                                                              │
│  MODELING           LightGBM                Fast gradient boosting           │
│                     RandomForest            Interpretable ensemble           │
│                                                                              │
│  IMBALANCE          imbalanced-learn        SMOTE oversampling               │
│                                                                              │
│  EXPLAINABILITY     SHAP                    Feature attribution              │
│                                             "Why this prediction?"           │
│                                                                              │
│  EVALUATION         scikit-learn            Kappa, F1, Confusion Matrix      │
│                                                                              │
│  SERVING            joblib                  Model serialization              │
│                     Flask                   REST API                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                           ML PIPELINE FLOW                                   │
│                                                                              │
│  Synthea → Pandas → Feature Eng → Preprocessing → LightGBM+RF → SHAP → API  │
│                                                                              │
│  Inference Time: < 100ms per patient                                         │
│  Total Triage: < 30 seconds (including user input)                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```
