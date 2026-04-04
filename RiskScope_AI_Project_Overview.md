# RiskScope AI - Project Explanation

## What is RiskScope AI?

**RiskScope AI** is an AI-assisted emergency triage system that stratifies patients by urgency level in under 30 seconds, following the globally-recognized **Emergency Severity Index (ESI)** protocol.

> **Key Insight**: We don't diagnose diseases. We prioritize patients so doctors see the sickest first.

---

## The Problem We're Solving

### India's Emergency Department Crisis

| Statistic | Impact |
|-----------|--------|
| **156 million** annual ED visits | System overwhelmed |
| **60%** are non-urgent | Delays critical care |
| **10 minutes** average triage time | Nurse shortage (1:300 ratio) |
| **23%** of critical cases miss golden hour | Preventable deaths |
| **₹8,000 crore** annual loss | Preventable complications |

### Why Existing Solutions Fail

| Solution | Problem |
|----------|---------|
| Traditional ESI | Requires trained nurses (unavailable in rural India) |
| Symptom checkers (WebMD, etc.) | Give diagnoses, not prioritization |
| Hospital queues | First-come-first-served ignores urgency |

---

## Our Solution

### Core Concept

```
Patient enters symptoms -> RiskScope assigns ESI Level 1-5 -> Doctor sees highest priority first
```

### ESI Levels Explained

| Level | Name | Examples | Action |
|-------|------|----------|--------|
| **1** | Immediate | Heart attack, stroke, shock | Resuscitate immediately |
| **2** | Emergent | Chest pain, high fever + confusion | See within 10 minutes |
| **3** | Urgent | Abdominal pain, fractures | See within 30 minutes |
| **4** | Less Urgent | Sprains, minor cuts | See within 60 minutes |
| **5** | Non-Urgent | Cold, sore throat | Can wait hours |

### What Makes Us Different

| Approach | RiskScope AI |
|----------|--------------|
| **Not a diagnosis** | Risk stratification only |
| **Safety-first** | Rules catch emergencies before ML |
| **Explainable** | Shows why each level was assigned |
| **Protocol-aligned** | Follows ACLS/ESI guidelines |

---

## Technical Architecture

### 4-Layer Safety System

```
┌─────────────────────────────────────────────────────────┐
│                    PATIENT INPUT                         │
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│  LAYER 1: EMERGENCY RULES (Hardcoded, Never Fails)      │
│  • Chest pain + arm radiation -> ESI 1                   │
│  • SpO2 < 90% -> ESI 1                                   │
│  • Stroke symptoms (FAST) -> ESI 1                       │
│  [If triggered -> Return ESI 1 immediately]              │
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│  LAYER 2: ML PREDICTION (LightGBM)                      │
│  • Trained on 50K synthetic patients                    │
│  • Predicts ESI 1-5 with probability                    │
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│  LAYER 3: CONFIDENCE CHECK                              │
│  • If confidence < 60% -> Escalate one level             │
│  • Never under-triage when uncertain                    │
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│  LAYER 4: EXPLANATION                                   │
│  • SHAP values show top contributing factors            │
│  • Human-readable "why this level?"                     │
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    FINAL OUTPUT                          │
│  {esi_level, confidence, recommendation, explanation}    │
└─────────────────────────────────────────────────────────┘
```

### Why This Architecture?

| Design Choice | Reason |
|---------------|--------|
| Rules BEFORE ML | Never miss a true emergency due to model error |
| Confidence escalation | Fail safe, not fail dangerous |
| SHAP explanations | Doctors can verify AI reasoning |
| LightGBM over deep learning | Faster, more interpretable, works with less data |

---

## Data Strategy

### For Hackathon Demo: Synthea

| Aspect | Details |
|--------|---------|
| **What** | Synthetic patient generator (open source) |
| **Size** | 50,000 patients with ED visits |
| **Why** | Immediate access, no credentialing |
| **Limitation** | Synthetic, not real clinical data |

### For Production: MIMIC-IV

| Aspect | Details |
|--------|---------|
| **What** | 448,972 real ED visits from Beth Israel Hospital |
| **Credibility** | Used in 300+ research papers, published in Nature |
| **Access** | Free via PhysioNet (requires 2-3 day approval) |

### Why We're Honest About Data

> "This demo uses synthetic data. We acknowledge this limitation. Real clinical validation with MIMIC-IV is our immediate next step."

**Judges respect honesty over pretense.**

---

## Machine Learning Approach

### Model Choice: LightGBM + Random Forest Ensemble

| Why LightGBM | Why NOT Deep Learning |
|--------------|---------------------|
| Trains in minutes | Needs 100K+ samples |
| Inherently explainable | Black box |
| Works with tabular data | Overkill for structured inputs |
| Low latency (<50ms) | Slower inference |

### Features Used

| Category | Features |
|----------|----------|
| **Demographics** | Age, gender |
| **Vitals** | HR, BP, SpO2, temp, respiratory rate |
| **Chief Complaint** | Encoded category (chest pain, cough, etc.) |
| **Symptom Flags** | Binary (chest pain, arm pain, dyspnea, etc.) |
| **Risk Scores** | SIRS criteria, qSOFA (calculated) |

### Validation Metrics

| Metric | Target | Why |
|--------|--------|-----|
| **Cohen's Kappa** | >0.65 | Agreement with "gold standard" ESI labels |
| **ESI 1 Sensitivity** | >95% | Must catch almost all true emergencies |
| **Response Time** | <500ms | Fast enough for real-time use |

---

## Competitive Advantages

### 1. Safety-First Architecture
Most AI health tools: ML -> Output
RiskScope: **Rules -> ML -> Confidence Check -> Output**

### 2. Explainability
Every prediction includes "Why this level?" with top factors.

### 3. Protocol Alignment
Not inventing new triage - implementing ESI with AI assistance.

### 4. Honest About Limitations
We know what we can't do:
- We can't diagnose
- We can't replace doctors
- We need human oversight

---

## Hackathon Demo Flow

```
1. Patient opens website
2. Enters: Age, gender, chief complaint, vitals, symptoms
3. Clicks "Triage Patient"
4. System returns:
   - ESI Level (1-5) with color coding
   - Confidence score
   - Key factors (why this level)
   - Protocol suggestion (for Level 1-2)
5. Doctor dashboard shows queue sorted by urgency
```

---

## 36-Hour Feasibility Assessment

### What's Achievable [OK]

| Component | Hours | Feasibility |
|-----------|-------|-------------|
| Project setup | 3 | [OK] Easy |
| Data generation (Synthea) | 5 | [OK] Straightforward |
| Model training (LightGBM) | 3 | [OK] Fast |
| Safety layer (rules) | 4 | [OK] Deterministic |
| Backend API (Flask) | 4 | [OK] Well-understood |
| Frontend (React) | 10 | [OK] With simplifications |
| Deployment | 2 | [OK] Vercel + Railway |
| Demo prep | 4 | [OK] Critical |

### What's Cut [X]

| Feature | Time Saved | Why Cut |
|---------|------------|---------|
| Mobile responsive | 2 hrs | Demo is on laptop |
| Real-time polling | 3 hrs | Manual refresh works |
| SHAP charts | 2 hrs | Simple list is enough |
| Animations | 1.5 hrs | Zero judge impact |

### Realistic Probability

| Outcome | Probability |
|---------|-------------|
| Working demo | **85%** |
| Top 5 finish | **75%** |
| Top 3 finish | **45%** |
| 1st place | **15%** |

---

## Future Roadmap (Post-Hackathon)

| Phase | Timeline | Goal |
|-------|----------|------|
| **Phase 1** | Month 1-2 | MIMIC-IV integration, real data validation |
| **Phase 2** | Month 3-4 | Shadow deployment at partner hospital |
| **Phase 3** | Month 5-8 | Prospective validation study |
| **Phase 4** | Year 2 | Regulatory clearance (Class IIa medical device) |

---

## Key Differentiators for Judges

1. **We understand the difference between research and demo** - Synthea for hackathon, MIMIC-IV for production

2. **We have a safety architecture, not just accuracy** - Rules catch emergencies before ML

3. **We're honest about limitations** - Not claiming to replace doctors

4. **We follow established protocol** - ESI is the gold standard, not our invention

5. **We have a regulatory path** - Know about FDA 510(k), India MDR 2017

---

## One-Line Pitch

> "RiskScope AI performs ESI-protocol triage in 30 seconds, helping overwhelmed Indian ERs see the sickest patients first."
