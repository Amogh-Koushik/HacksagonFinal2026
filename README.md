<p align="center">
  <h1 align="center">🏥 RiskScope AI</h1>
  <p align="center">
    <strong>AI-Powered Emergency Department Triage System</strong>
  </p>
  <p align="center">
    An explainable, safety-first machine learning system that assists emergency department triage by predicting ESI (Emergency Severity Index) levels from patient vitals and symptoms — with built-in clinical safety rules, out-of-distribution detection, and SHAP-based explainability.
  </p>
  <p align="center">
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-architecture">Architecture</a> •
    <a href="#-ml-model">ML Model</a> •
    <a href="#-backend">Backend</a> •
    <a href="#-frontend">Frontend</a> •
    <a href="#-database">Database</a> •
    <a href="#-api-reference">API Reference</a>
  </p>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Tech Stack](#-tech-stack)
- [Architecture](#-architecture)
- [ML Model — How It Works](#-ml-model--how-it-works)
- [Backend — How It Works](#-backend--how-it-works)
- [Frontend — How It Works](#-frontend--how-it-works)
- [Database — How It Works](#-database--how-it-works)
- [Installation & Local Setup](#-installation--local-setup)
- [API Reference](#-api-reference)
- [Project Structure](#-project-structure)
- [Safety System](#-safety-system)
- [Demo Script](#-demo-script)

---

## 🔍 Overview

**RiskScope AI** is an end-to-end AI triage system designed for emergency departments. It classifies patients into **5 ESI levels** (1 = Resuscitation, 5 = Non-Urgent) using a combination of:

1. **Machine Learning** — A weighted ensemble of LightGBM, XGBoost, and Random Forest trained on clinical data
2. **Clinical Safety Rules** — 20+ hardcoded medical protocols (ACLS, FAST stroke, sepsis bundles) that override ML when life-threatening patterns are detected
3. **Confidence Calibration** — Auto-escalation when the model is uncertain
4. **Explainability** — SHAP-based feature importance explanations for every prediction

> **Core Design Principle:** RiskScope doesn't replace doctors — it makes sure no critical patient falls through the cracks. The system will **never** classify a life-threatening patient as non-urgent.

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **ML Training** | Python, LightGBM, XGBoost, scikit-learn, SMOTE | Ensemble model training with class balancing |
| **ML Safety** | Custom Python (rule engine) | 20+ clinical emergency rules |
| **Explainability** | SHAP | Human-readable prediction explanations |
| **OOD Detection** | Isolation Forest (scikit-learn) | Detects out-of-distribution patients |
| **Backend API** | FastAPI, Uvicorn, Pydantic | REST API serving predictions |
| **Database** | Supabase (PostgreSQL) | Patient record storage |
| **Frontend** | React 19, Vite, Tailwind CSS 4 | Clinical dashboard UI |
| **Deployment** | Railway (backend) | Cloud hosting |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React + Vite)                  │
│   LoginPage → DashboardOverview → AddPatient → PatientQueue     │
│                          │                                      │
│              POST /api/v1/predict  &  GET /api/v1/patients      │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP (JSON)
┌──────────────────────────▼──────────────────────────────────────┐
│                     BACKEND (FastAPI)                            │
│                                                                  │
│  ┌─── Layer 1: Safety Service ───────────────────────────────┐  │
│  │  20+ Emergency Rules (MI, Stroke, Sepsis, Shock, etc.)    │  │
│  │  If triggered → ESI 1 or 2 immediately, skip ML           │  │
│  └───────────────────────────────┬───────────────────────────┘  │
│                                  │ (only if no rule fires)       │
│  ┌─── Layer 2: ML Model Service ─┴──────────────────────────┐  │
│  │  PreprocessingService → feature engineering                │  │
│  │  ModelService → KNN prediction from ml.pkl                 │  │
│  │  Returns ESI level + confidence score                      │  │
│  └───────────────────────────────┬───────────────────────────┘  │
│                                  │                               │
│  ┌─── Layer 3: Confidence Calibration ──────────────────────┐  │
│  │  If confidence < 60% → escalate ESI by 1 level            │  │
│  │  (e.g., ESI 3 → ESI 2 for safety)                         │  │
│  └───────────────────────────────┬───────────────────────────┘  │
│                                  │                               │
│  ┌─── Storage Service ──────────┴───────────────────────────┐  │
│  │  Supabase (production) or In-Memory (local dev)           │  │
│  │  Stores patient record + ESI + confidence + method        │  │
│  └───────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

**Request Flow:**
1. Frontend sends patient vitals + symptoms via `POST /api/v1/predict`
2. **Layer 1 (Safety Rules):** 20+ clinical rules run first. If any rule matches (e.g., chest pain + arm radiation = suspected MI), the system immediately returns ESI 1 with the specific ACLS protocol — the ML model is bypassed entirely
3. **Layer 2 (ML Prediction):** If no emergency rule fires, the `PreprocessingService` engineers features (shock index, SIRS score, qSOFA score, etc.), and the `ModelService` runs a KNN prediction from the serialized model
4. **Layer 3 (Confidence Calibration):** If the ML model's confidence is below 60%, the ESI level is escalated by 1 (towards more urgent) as a safety measure
5. The patient record is stored in Supabase (or in-memory fallback) and returned to the frontend

---

## 🧠 ML Model — How It Works

### Data Pipeline

```
Synthea FHIR Data → synthea_to_training.py → Raw CSV
                                                 │
Raw CSV → generate_synthetic_data.py / generate_realistic_data.py → Training CSV
                                                 │
Training CSV → Feature Engineering → Train/Test Split → Model Training
```

**Data Sources:**
- **Synthea:** Synthetic FHIR patient records converted to tabular format via `synthea_to_training.py`
- **Synthetic Generator:** `generate_synthetic_data.py` creates clinically realistic patient distributions with proper vital sign correlations per ESI level

### Feature Engineering

The model uses **26+ engineered features** derived from raw patient data:

| Category | Features | Description |
|----------|----------|-------------|
| **Demographics** | `age`, `gender` | Age in years, binary gender encoding |
| **Vital Signs** | `heart_rate`, `bp_systolic`, `bp_diastolic`, `spo2`, `temperature`, `respiratory_rate` | Raw clinical measurements |
| **Symptom Flags** | `chest_pain`, `arm_pain_left`, `jaw_pain`, `dyspnea`, `facial_droop`, `arm_weakness`, `speech_difficulty`, `seizure`, `uncontrolled_bleeding`, ... | Binary (0/1) symptom indicators |
| **Composite Scores** | `sirs_score`, `qsofa_score`, `shock_index` | Calculated from vitals — SIRS criteria, quick SOFA, HR/SBP ratio |
| **Derived Flags** | `critical_spo2`, `tachycardia`, `hypotension`, `high_fever`, `tachypnea` | Auto-computed from vital sign thresholds |
| **Categoricals** | `complaint_encoded`, `age_group` | Chief complaint mapped to 14 categories, age bucketed into 4 groups |

### Model Architecture

```
                    ┌──────────────────────┐
Training Data ──────┤   5-Fold Stratified   │
  (with SMOTE)      │   Cross-Validation    │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
     ┌────────────────┐ ┌───────────┐ ┌──────────────────┐
     │   LightGBM     │ │  XGBoost  │ │  Random Forest   │
     │  (50% weight)  │ │(30% wt.)  │ │  (20% weight)    │
     │  500 trees     │ │ 300 trees │ │  200 trees       │
     │  depth=8       │ │ depth=8   │ │  depth=12        │
     │  lr=0.03       │ │ lr=0.05   │ │  balanced class  │
     └───────┬────────┘ └─────┬─────┘ └────────┬─────────┘
              │               │                │
              ▼               ▼                ▼
     ┌────────────────────────────────────────────────────┐
     │       Weighted Soft-Vote Ensemble                   │
     │  P(class) = 0.50·P_lgb + 0.30·P_xgb + 0.20·P_rf   │
     │  Final ESI = argmax(P(class))                       │
     └─────────────────────┬──────────────────────────────┘
                           │
                           ▼
               ESI Level (1-5) + Confidence Score
```

**Key Design Decisions:**
- **SMOTE (Synthetic Minority Oversampling):** ESI 1 patients are rare (~5% of ED visits). SMOTE generates synthetic minority samples so the model doesn't ignore critical patients
- **Class-Balanced Weights:** Both LightGBM and RF use `class_weight='balanced'` to penalize misclassification of rare ESI levels
- **Three Diverse Models:** LightGBM (gradient boosting), XGBoost (extreme gradient boosting), and Random Forest (bagging) — different learning paradigms minimize correlated errors
- **Soft Voting:** Probability averaging rather than hard voting gives better calibrated confidence scores

### Safety Layers (ML Module)

The ML module includes its own safety architecture in `safety_engine.py`:

1. **EmergencyRuleEngine:** Hardcoded clinical protocols that override ML (MI, stroke FAST+, sepsis, etc.)
2. **OutOfDistributionDetector (`ood_detector.py`):** Isolation Forest trained on the training distribution. Flags patients that look unlike anything the model has seen
3. **ConfidenceCalibrator (`confidence.py`):** If prediction confidence is below threshold, auto-escalates ESI level
4. **ClinicalSafetyEngine:** Orchestrates all three layers into a single `predict()` call
5. **ExplainabilityEngine (`explain.py`):** SHAP TreeExplainer generates feature importance values, converted to human-readable clinical insights

### Training the Model

```bash
cd ML

# Option 1: Train with synthetic data
python train.py --synthetic --samples 50000

# Option 2: Train with your own CSV
python train.py --data data/training.csv

# Option 3: Test inference only (requires trained model)
python train.py --test-only
```

**Output:** Trained models are saved to `ML/models/`:
- `esi_ensemble_model.pkl` — The ensemble predictor (~20 MB)
- `ood_detector.pkl` — The OOD detector (~18 MB)
- `training_report.txt` — Metrics summary

### Exporting for Backend

After training the ML model, export a lightweight portable model for the backend:

```bash
cd ML
python export_backend_model.py
```

This creates `backend/artifacts/ml.pkl` — a compact KNN-based model with reference data that works reliably across Python versions (avoids cloudpickle segfaults).

---

## ⚡ Backend — How It Works

The backend is a **FastAPI** application that serves the ML model as a REST API.

### Directory Structure

```
backend/
├── app/
│   ├── main.py                          # FastAPI app, CORS, router mounting
│   ├── core/
│   │   └── config.py                    # Pydantic settings (env vars, paths)
│   ├── api/v1/endpoints/
│   │   ├── predict.py                   # POST /predict — main triage endpoint
│   │   ├── patients.py                  # GET/PATCH/DELETE patient records
│   │   └── health.py                    # GET /health — system status
│   ├── models/
│   │   └── schemas.py                   # Pydantic request/response models
│   └── services/
│       ├── preprocessing_service.py     # Feature engineering for API input
│       ├── model_service.py             # Model loading & prediction
│       ├── safety_service.py            # 20+ emergency clinical rules
│       ├── storage_service.py           # Supabase / in-memory storage
│       └── runtime.py                   # Singleton service instances
├── artifacts/
│   ├── ml.pkl                           # Portable ML model
│   └── feature_columns.json            # Expected feature column order
├── requirements.txt
└── .gitignore
```

### Key Services

#### `PreprocessingService`
Transforms raw patient input into the feature vector the model expects:
- Encodes gender (M=1, F=0)
- Maps chief complaint text to 14 categories
- Computes clinical scores: **SIRS score**, **qSOFA score**, **Shock Index** (HR/SBP)
- Derives binary flags: `tachycardia`, `hypotension`, `critical_spo2`, `high_fever`, `tachypnea`
- Buckles age into groups (pediatric, adult, middle-age, elderly)

#### `ModelService`
Loads `artifacts/ml.pkl` at startup and provides `predict(features) → (esi_level, confidence)`:
- Primary: Pure-Python weighted KNN using reference data embedded in the model
- Fallback: Native model `.predict()` / `.predict_proba()` 
- The KNN approach avoids cloudpickle segfaults on Python 3.13

#### `SafetyService`
The backend's own safety layer with **20+ clinical rules** organized by severity:

**ESI 1 Rules (18 rules):**
| Rule | Trigger | Protocol |
|------|---------|----------|
| Suspected MI | chest_pain + arm/jaw radiation | ACLS - Acute Coronary Syndrome |
| Cardiac Arrest | HR < 30 or unresponsive | ACLS - Cardiac Arrest |
| Respiratory Failure | SpO2 < 90% | Respiratory Distress Protocol |
| Stroke FAST+ | 2+ of: facial droop, arm weakness, speech difficulty | Stroke Alert - FAST Protocol |
| Shock | SBP < 90 + HR > 100 | Shock Protocol |
| Sepsis | Temp > 38.3 + HR > 90 + AMS or SBP < 100 | Sepsis Bundle (Hour-1) |
| Hemorrhage | Uncontrolled bleeding | Massive Transfusion Protocol |
| Anaphylaxis | Allergic reaction + dyspnea or SBP < 90 | Anaphylaxis Protocol |
| *...and 10 more* | | |

**ESI 2 Rules (3 rules):** Stable chest pain, high fever (≥39.5°C), moderate hypoxia (SpO2 90-94%)

#### `StorageService`
Dual-mode patient storage:
- **Supabase mode:** If `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` env vars are set, stores to PostgreSQL via Supabase client
- **In-memory mode:** Falls back to an in-memory list if Supabase is unavailable
- Features automatic schema fallback — if a Supabase column doesn't exist, it retries the insert without that column

### Configuration

All settings are managed via environment variables (loaded from `.env`):

```env
# App
APP_NAME=RiskScope Backend
APP_ENV=development
APP_PORT=8000

# CORS
CORS_ALLOWED_ORIGINS=http://localhost:5173,https://your-frontend.vercel.app

# Model
MODEL_PATH=artifacts/ml.pkl
FEATURE_COLUMNS_PATH=artifacts/feature_columns.json
MODEL_VERSION=v1.0.0

# Database (optional — falls back to in-memory)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
SUPABASE_PATIENTS_TABLE=patients

# Swagger docs
ENABLE_SWAGGER=true
```

---

## 🎨 Frontend — How It Works

The frontend is a **React 19** single-page application built with **Vite** and styled with **Tailwind CSS 4**.

### Directory Structure

```
frontend/
├── src/
│   ├── App.jsx                     # Routes and auth guard
│   ├── main.jsx                    # Entry point
│   ├── index.css                   # Global styles
│   ├── pages/
│   │   ├── LoginPage.jsx           # Authentication screen
│   │   ├── DashboardOverview.jsx   # Summary dashboard with stats
│   │   ├── AddPatient.jsx          # Patient intake form
│   │   ├── PatientQueue.jsx        # Live priority queue
│   │   └── PatientDetailsPage.jsx  # Individual patient view
│   ├── components/
│   │   ├── Layout.jsx              # App shell with sidebar
│   │   ├── Sidebar.jsx             # Navigation sidebar
│   │   ├── PatientDetailsPanel.jsx # Detailed patient info panel
│   │   ├── MedicalInputField.jsx   # Styled input for vitals
│   │   ├── BooleanCheckboxField.jsx# Symptom toggle checkboxes
│   │   └── LoadingSkeleton.jsx     # Loading state placeholder
│   ├── contexts/
│   │   ├── AuthContext.jsx         # Login state management
│   │   ├── PatientContext.jsx      # Patient data & API calls
│   │   └── ToastContext.jsx        # Notification system
│   └── utils/
│       └── esiCalculator.js        # Client-side ESI estimation
├── package.json
├── vite.config.js
└── index.html
```

### Pages

| Page | Route | Description |
|------|-------|-------------|
| **LoginPage** | `/login` | Doctor authentication (session-based) |
| **DashboardOverview** | `/` | Summary cards — total patients, ESI distribution, recent triages |
| **AddPatient** | `/add-patient` | Full patient intake form: demographics, vitals, 20+ symptom checkboxes. Submits to `POST /api/v1/predict` |
| **PatientQueue** | `/queue` | Live priority queue sorted by ESI level then arrival time. Color-coded ESI badges. Status management (waiting → treating → discharged) |
| **PatientDetailsPage** | `/patient/:id` | Detailed view: vitals, symptoms, ESI result, method (ML vs rule-based), protocol, clinical action, explainability data |

### How Frontend Connects to Backend

The `PatientContext` manages all API communication:

```javascript
// PatientContext.jsx — key functions
const API_BASE = "https://your-railway-url.up.railway.app/api/v1"  // or localhost:8000

// Submit new patient for triage
addPatient(patientData)     → POST /api/v1/predict    → TriageResponse

// Fetch all patients in queue
fetchPatients()             → GET  /api/v1/patients   → PatientOut[]

// Update patient status
updatePatientStatus(id, s)  → PATCH /api/v1/patients/:id → PatientOut

// Delete patient
deletePatient(id)           → DELETE /api/v1/patients/:id → boolean
```

The frontend connects to the backend via the **API URL configured in** `PatientContext.jsx`. In development, this points to `http://localhost:8000`. In production, it points to the Railway deployment URL.

---

## 🗄 Database — How It Works

### Supabase (PostgreSQL)

RiskScope uses **Supabase** as its production database — a managed PostgreSQL instance with a REST API.

#### `patients` Table Schema

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Auto-generated primary key |
| `created_at` | Timestamp | When the patient was triaged |
| `patient_id` | Text | External patient identifier |
| `name` | Text | Patient name (optional) |
| `age` | Integer | Patient age |
| `gender` | Text | "M" or "F" |
| `heart_rate` | Integer | Heart rate (bpm) |
| `bp_systolic` | Integer | Systolic blood pressure (mmHg) |
| `bp_diastolic` | Integer | Diastolic blood pressure (mmHg) |
| `spo2` | Float | Oxygen saturation (%) |
| `temperature` | Float | Body temperature (°C) |
| `respiratory_rate` | Integer | Breaths per minute |
| `esi_level` | Integer | Predicted ESI (1-5) |
| `confidence` | Float | Model confidence (0-1) |
| `status` | Text | "waiting", "treating", or "discharged" |
| `model_version` | Text | Version of the model used |
| `method` | Text | "ML_PREDICTION" or "RULE_BASED" |
| `protocol` | Text | Clinical protocol (if rule-based) |
| `action` | Text | Recommended clinical action |
| `rule_triggered` | Text | Name of safety rule (if any) |
| `escalated` | Boolean | Whether confidence calibration escalated |
| `recommendation` | Text | Human-readable recommendation |
| `model_input_json` | JSONB | The feature vector used for prediction |
| `vitals_json` | JSONB | Raw vital signs |
| `symptoms_json` | JSONB | Raw symptom data |

#### In-Memory Fallback

If Supabase credentials are not configured, the `StorageService` automatically falls back to an **in-memory list**. This means:
- ✅ The backend works fully without any database
- ✅ Perfect for local development and demos
- ⚠️ Data is lost when the server restarts

#### Setting Up Supabase

1. Create a project at [supabase.com](https://supabase.com)
2. Go to **SQL Editor** and create the `patients` table (the `StorageService` auto-handles missing columns via retry logic)
3. Copy your **Project URL** and **Service Role Key**
4. Add to `backend/.env`:
   ```env
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
   ```

---

## 🚀 Installation & Local Setup

### Prerequisites

- **Python 3.11+** (recommended: 3.12)
- **Node.js 18+** (recommended: 20 LTS)
- **Git**

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-org/riskscope-ai.git
cd riskscope-ai
```

### Step 2: Set Up the ML Module

```bash
# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install ML dependencies
pip install -r ML/requirements.txt
```

#### Train the Model (if no pre-trained model exists)

```bash
cd ML

# Generate synthetic data and train
python train.py --synthetic --samples 50000

# Export portable model for backend
python export_backend_model.py

cd ..
```

This creates:
- `ML/models/esi_ensemble_model.pkl` — Full ensemble model
- `ML/models/ood_detector.pkl` — OOD detector
- `backend/artifacts/ml.pkl` — Portable backend model

### Step 3: Set Up the Backend

```bash
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env  # or create manually
```

Edit `backend/.env`:

```env
APP_NAME=RiskScope Backend
APP_ENV=development
CORS_ALLOWED_ORIGINS=http://localhost:5173
ENABLE_SWAGGER=true

# Optional — leave empty for in-memory storage
SUPABASE_URL=
SUPABASE_SERVICE_ROLE_KEY=
```

#### Start the Backend Server

```bash
# From the backend/ directory
uvicorn app.main:app --reload --port 8000
```

The API will be available at:
- **Root:** http://localhost:8000
- **Swagger Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/api/v1/health

### Step 4: Set Up the Frontend

```bash
cd frontend

# Install Node.js dependencies
npm install

# Start the dev server
npm run dev
```

The frontend will be available at **http://localhost:5173**

> **Note:** Make sure the backend is running on port 8000 before using the frontend. The frontend's `PatientContext.jsx` is configured to hit `http://localhost:8000/api/v1` by default.

### Step 5: Verify Everything Works

1. Open http://localhost:5173 in your browser
2. Log in on the Login page
3. Navigate to "Add Patient"
4. Enter sample vitals (e.g., Age: 58, HR: 105, BP: 160/95, SpO2: 94, Chest Pain: ✓)
5. Submit — you should see an ESI level result with safety details
6. Check "Patient Queue" — the patient should appear sorted by priority

#### Quick API Test (optional)

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "test-001",
    "age": 58,
    "gender": "M",
    "heart_rate": 105,
    "bp_systolic": 160,
    "bp_diastolic": 95,
    "spo2": 94,
    "temperature": 37.2,
    "resp_rate": 22,
    "chest_pain": 1,
    "arm_pain_left": 1
  }'
```

Expected: ESI Level 1, method "RULE_BASED", rule "suspected_mi", protocol "ACLS - Acute Coronary Syndrome"

---

## 📡 API Reference

### `POST /api/v1/predict`

Triage a patient. Runs safety rules → ML prediction → confidence calibration.

**Request Body:**
```json
{
  "patient_id": "P-001",
  "name": "John Doe",
  "age": 58,
  "gender": "M",
  "heart_rate": 105,
  "bp_systolic": 160,
  "bp_diastolic": 95,
  "spo2": 94,
  "temperature": 37.2,
  "resp_rate": 22,
  "complaint": "chest pain",
  "chest_pain": 1,
  "arm_pain_left": 1,
  "dyspnea": 1
}
```

**Response:**
```json
{
  "esi_level": 1,
  "confidence": 1.0,
  "model_version": "v1.0.0",
  "prediction_id": "uuid",
  "created_at": "2026-04-04T18:00:00Z",
  "method": "RULE_BASED",
  "protocol": "ACLS - Acute Coronary Syndrome",
  "action": "12-lead EKG STAT, Aspirin 325mg, IV access, Troponin, Cardiology consult",
  "rule_triggered": "suspected_mi",
  "escalated": false,
  "recommendation": "IMMEDIATE RESUSCITATION — Life-threatening condition",
  "explanation": [{"feature": "suspected_mi", "impact": "+", "value": "CRITICAL"}],
  "patient": { ... },
  "model_input_csv": "age,gender_encoded,...",
  "model_input_row": {"age": 58, "gender_encoded": 1, ...}
}
```

### `POST /api/v1/predict/prepare-csv`

Dry-run: see the feature vector without making a prediction.

### `GET /api/v1/patients`

List all patients, sorted by ESI level (most urgent first).

### `PATCH /api/v1/patients/{id}`

Update patient status: `{"status": "treating"}` or `{"status": "discharged"}`.

### `DELETE /api/v1/patients/{id}`

Remove a patient from the queue.

### `GET /api/v1/health`

System health check — returns model status, storage mode, app version.

---

## 📂 Project Structure

```
riskscope-ai/
│
├── ML/                              # Machine Learning Module
│   ├── config.py                    # Hyperparameters, feature definitions
│   ├── train.py                     # Training pipeline entry point
│   ├── esi_predictor.py             # Ensemble model class
│   ├── feature_engineering.py       # Feature computation logic
│   ├── preprocess.py                # Data preprocessing
│   ├── evaluate.py                  # Model evaluation metrics
│   ├── safety_engine.py             # Clinical safety engine (ML-side)
│   ├── ood_detector.py              # Out-of-distribution detector
│   ├── confidence.py                # Confidence calibration
│   ├── explain.py                   # SHAP explainability engine
│   ├── export_backend_model.py      # Export portable model for backend
│   ├── generate_synthetic_data.py   # Synthetic patient data generator
│   ├── generate_realistic_data.py   # Realistic distribution generator
│   ├── synthea_to_training.py       # FHIR → CSV converter
│   ├── test_model.py                # Model unit tests
│   ├── test_safety_engine.py        # Safety engine tests
│   ├── test_explainability.py       # Explainability tests
│   ├── test_quick.py                # Quick smoke tests
│   └── requirements.txt             # Python ML dependencies
│
├── backend/                         # FastAPI Backend
│   ├── app/
│   │   ├── main.py                  # App entry, CORS, routers
│   │   ├── core/config.py           # Settings from env vars
│   │   ├── api/v1/endpoints/
│   │   │   ├── predict.py           # Triage endpoint
│   │   │   ├── patients.py          # CRUD patient endpoints
│   │   │   └── health.py            # Health check
│   │   ├── models/schemas.py        # Pydantic models
│   │   └── services/
│   │       ├── preprocessing_service.py
│   │       ├── model_service.py
│   │       ├── safety_service.py
│   │       ├── storage_service.py
│   │       └── runtime.py
│   ├── artifacts/                   # Model files (gitignored)
│   ├── requirements.txt
│   └── test_integration.py          # Backend integration tests
│
├── frontend/                        # React Frontend
│   ├── src/
│   │   ├── App.jsx
│   │   ├── pages/                   # 5 page components
│   │   ├── components/              # 6 reusable components
│   │   ├── contexts/                # Auth, Patient, Toast contexts
│   │   └── utils/esiCalculator.js   # Client-side ESI logic
│   ├── package.json
│   └── vite.config.js
│
├── demo_script.py                   # Interactive terminal demo
├── .gitignore
└── README.md                        # This file
```

---

## 🛡 Safety System

RiskScope's safety architecture operates at **three levels**, ensuring that no critical patient is ever classified as non-urgent:

### Level 1: Emergency Rule Engine (Pre-ML)

20+ hardcoded clinical rules based on ACLS/PALS/ESI protocols. These run **before** the ML model and override it completely if a life-threatening pattern is detected.

**Examples:**
- **Suspected MI:** `chest_pain + (arm_pain_left or jaw_pain)` → ESI 1
- **Stroke FAST+:** 2+ of `facial_droop`, `arm_weakness`, `speech_difficulty` → ESI 1
- **Respiratory Failure:** `SpO2 < 90%` → ESI 1
- **Shock:** `SBP < 90 + HR > 100` → ESI 1
- **Sepsis:** `Temp > 38.3 + HR > 90 + (AMS or SBP < 100)` → ESI 1

### Level 2: Out-of-Distribution Detection

An Isolation Forest model identifies patients whose feature patterns differ significantly from the training data. These are flagged for manual physician review.

### Level 3: Confidence Calibration

If the ML model's maximum class probability is below **60%**, the patient is automatically escalated one ESI level toward more urgent (e.g., ESI 3 → ESI 2). This ensures uncertainty always errs on the side of caution.

---

## 🎬 Demo Script

An interactive terminal demo is included for presentations:

```bash
python demo_script.py
```

Features:
- ASCII art banner
- Animated loading bars for each system component
- Color-coded patient vital displays with abnormality indicators
- ESI classification with styled terminal boxes
- Safety engine analysis per patient
- SHAP-style explainability output
- Session summary with architecture overview

> **Tip:** Use **Windows Terminal** (not CMD) for proper ANSI color support.

---

## Live working site 
- link = https://hacksagonfinal202601.vercel.app/
- Credentials : ID = nurse, Password = 123

## 📜 License

MIT License — See [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>Built for IIITM Hacksagon 2026</strong><br>
  <em>"RiskScope doesn't replace doctors — it makes sure no critical patient falls through the cracks."</em>
</p>
