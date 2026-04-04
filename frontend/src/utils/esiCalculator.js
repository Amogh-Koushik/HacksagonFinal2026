export function calculateESI(patient) {
  // Support both new format (heart_rate, bp_systolic, etc.) and legacy (heartRate, bloodPressure, etc.)
  const heartRate = patient.heart_rate ?? patient.heartRate
  const spo2 = patient.spo2 ?? patient.oxygenLevel
  const painScale = patient.painScale
  const symptoms = (patient.symptoms || '').toLowerCase()

  // Blood pressure: support new separate fields or legacy combined string
  let systolic = patient.bp_systolic
  let diastolic = patient.bp_diastolic
  if (systolic == null && patient.bloodPressure) {
    const parts = patient.bloodPressure.split('/').map(Number)
    systolic = parts[0] || 120
    diastolic = parts[1] || 80
  }

  // New clinical scores
  const sirsScore = patient.sirs_score
  const qsofaScore = patient.qsofa_score
  const shockIndex = patient.shock_index
  const criticalSpo2 = patient.critical_spo2
  const tachycardia = patient.tachycardia
  const hypotension = patient.hypotension
  const highFever = patient.high_fever
  const tachypnea = patient.tachypnea
  const temperature = patient.temperature
  const respRate = patient.resp_rate

  let score = 0

  // === Symptom-based scoring (legacy support) ===
  if (symptoms) {
    const criticalSymptoms = ['unresponsive', 'cardiac arrest', 'not breathing', 'collapsed', 'unconscious', 'seizure']
    const emergentSymptoms = ['chest pain', 'shortness of breath', 'difficulty breathing', 'stroke', 'severe bleeding', 'head trauma']
    const urgentSymptoms = ['abdominal pain', 'fracture', 'laceration', 'vomiting blood', 'high fever']

    if (criticalSymptoms.some(s => symptoms.includes(s))) score += 40
    else if (emergentSymptoms.some(s => symptoms.includes(s))) score += 25
    else if (urgentSymptoms.some(s => symptoms.includes(s))) score += 15
  }

  // === Heart rate scoring ===
  if (heartRate != null) {
    if (heartRate < 40 || heartRate > 160) score += 35
    else if (heartRate < 50 || heartRate > 140) score += 25
    else if (heartRate < 60 || heartRate > 120) score += 15
    else if (heartRate > 100) score += 8
  }

  // === SpO2 scoring ===
  if (spo2 != null) {
    if (spo2 < 85) score += 40
    else if (spo2 < 90) score += 30
    else if (spo2 < 93) score += 20
    else if (spo2 < 95) score += 10
  }

  // === Blood pressure scoring ===
  if (systolic != null) {
    if (systolic > 200 || systolic < 80) score += 30
    else if (systolic > 170 || systolic < 90) score += 20
    else if (systolic > 150) score += 10
  }
  if (diastolic != null) {
    if (diastolic > 120 || diastolic < 50) score += 15
    else if (diastolic > 100) score += 8
  }

  // === Pain scale (legacy) ===
  if (painScale != null) {
    if (painScale >= 9) score += 20
    else if (painScale >= 7) score += 15
    else if (painScale >= 5) score += 10
    else if (painScale >= 3) score += 5
  }

  // === New clinical score modifiers ===
  if (sirsScore != null) {
    if (sirsScore >= 3) score += 20
    else if (sirsScore >= 2) score += 10
  }

  if (qsofaScore != null) {
    if (qsofaScore >= 2) score += 25
    else if (qsofaScore >= 1) score += 10
  }

  if (shockIndex != null) {
    if (shockIndex > 1.2) score += 20
    else if (shockIndex > 1.0) score += 10
    else if (shockIndex > 0.9) score += 5
  }

  // === Temperature scoring ===
  if (temperature != null) {
    if (temperature >= 40 || temperature <= 34) score += 20
    else if (temperature >= 39 || temperature <= 35) score += 10
    else if (temperature >= 38.5) score += 5
  }

  // === Respiratory rate scoring ===
  if (respRate != null) {
    if (respRate > 30 || respRate < 8) score += 20
    else if (respRate > 24 || respRate < 10) score += 10
    else if (respRate > 20) score += 5
  }

  // === Clinical flag bonuses ===
  if (criticalSpo2 === 1) score += 15
  if (tachycardia === 1) score += 5
  if (hypotension === 1) score += 10
  if (highFever === 1) score += 5
  if (tachypnea === 1) score += 5

  // Final ESI level
  if (score >= 80) return 1
  if (score >= 55) return 2
  if (score >= 35) return 3
  if (score >= 20) return 4
  return 5
}

export const ESI_LABELS = {
  1: { label: 'Resuscitation', description: 'Immediate life-saving intervention', color: 'critical' },
  2: { label: 'Emergent', description: 'High risk, time-sensitive', color: 'warning-high' },
  3: { label: 'Urgent', description: 'Requires attention soon', color: 'warning' },
  4: { label: 'Less Urgent', description: 'Can wait with monitoring', color: 'info' },
  5: { label: 'Non-Urgent', description: 'Minor condition', color: 'safe' },
}

export const ESI_COLORS = {
  1: { bg: 'bg-critical-600', text: 'text-white', light: 'bg-critical-50', border: 'border-critical-200', textColor: 'text-critical-700' },
  2: { bg: 'bg-orange-500', text: 'text-white', light: 'bg-orange-50', border: 'border-orange-200', textColor: 'text-orange-700' },
  3: { bg: 'bg-warning-500', text: 'text-white', light: 'bg-warning-50', border: 'border-warning-200', textColor: 'text-warning-700' },
  4: { bg: 'bg-medical-500', text: 'text-white', light: 'bg-medical-50', border: 'border-medical-200', textColor: 'text-medical-700' },
  5: { bg: 'bg-clinical-500', text: 'text-white', light: 'bg-clinical-50', border: 'border-clinical-200', textColor: 'text-clinical-700' },
}
