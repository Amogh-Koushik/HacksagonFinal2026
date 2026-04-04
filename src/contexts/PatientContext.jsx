import { createContext, useContext, useState, useCallback, useMemo, useEffect } from 'react'
import { calculateESI } from '../utils/esiCalculator'

const PatientContext = createContext(null)
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1'

const DEMO_PATIENTS = [
  {
    id: '1',
    patient_id: 'PT-2024-0072',
    name: 'Emily Watson',
    age: 72,
    gender_encoded: 0,
    heart_rate: 38,
    bp_systolic: 85,
    bp_diastolic: 55,
    spo2: 84,
    temperature: 36.2,
    resp_rate: 28,
    complaint_encoded: 12,
    sirs_score: 3,
    qsofa_score: 3,
    shock_index: 0.45,
    age_group: 4,
    critical_spo2: 1,
    tachycardia: 0,
    hypotension: 1,
    high_fever: 0,
    tachypnea: 1,
    // Legacy fields for backward compat
    symptoms: 'Unresponsive, low pulse, shallow breathing',
    heartRate: 38,
    bloodPressure: '85/55',
    oxygenLevel: 84,
    painScale: 10,
    notes: 'Found collapsed at home',
    addedAt: new Date(Date.now() - 5 * 60000).toISOString(),
    status: 'waiting',
  },
  {
    id: '2',
    patient_id: 'PT-2024-0067',
    name: 'James Carter',
    age: 67,
    gender_encoded: 1,
    heart_rate: 145,
    bp_systolic: 180,
    bp_diastolic: 110,
    spo2: 88,
    temperature: 37.1,
    resp_rate: 26,
    complaint_encoded: 5,
    sirs_score: 2,
    qsofa_score: 2,
    shock_index: 0.81,
    age_group: 4,
    critical_spo2: 1,
    tachycardia: 1,
    hypotension: 0,
    high_fever: 0,
    tachypnea: 1,
    symptoms: 'Chest pain, shortness of breath, dizziness',
    heartRate: 145,
    bloodPressure: '180/110',
    oxygenLevel: 88,
    painScale: 9,
    notes: 'History of cardiac issues',
    addedAt: new Date(Date.now() - 15 * 60000).toISOString(),
    status: 'waiting',
  },
  {
    id: '3',
    patient_id: 'PT-2024-0052',
    name: 'Robert Kim',
    age: 52,
    gender_encoded: 1,
    heart_rate: 125,
    bp_systolic: 150,
    bp_diastolic: 95,
    spo2: 91,
    temperature: 37.4,
    resp_rate: 24,
    complaint_encoded: 8,
    sirs_score: 2,
    qsofa_score: 1,
    shock_index: 0.83,
    age_group: 3,
    critical_spo2: 0,
    tachycardia: 1,
    hypotension: 0,
    high_fever: 0,
    tachypnea: 1,
    symptoms: 'Difficulty breathing, wheezing, chest tightness',
    heartRate: 125,
    bloodPressure: '150/95',
    oxygenLevel: 91,
    painScale: 7,
    notes: 'Known asthma patient',
    addedAt: new Date(Date.now() - 30 * 60000).toISOString(),
    status: 'waiting',
  },
  {
    id: '4',
    patient_id: 'PT-2024-0034',
    name: 'Maria Santos',
    age: 34,
    gender_encoded: 0,
    heart_rate: 110,
    bp_systolic: 130,
    bp_diastolic: 85,
    spo2: 96,
    temperature: 37.8,
    resp_rate: 20,
    complaint_encoded: 15,
    sirs_score: 1,
    qsofa_score: 0,
    shock_index: 0.85,
    age_group: 2,
    critical_spo2: 0,
    tachycardia: 1,
    hypotension: 0,
    high_fever: 0,
    tachypnea: 0,
    symptoms: 'Severe abdominal pain, nausea, vomiting',
    heartRate: 110,
    bloodPressure: '130/85',
    oxygenLevel: 96,
    painScale: 8,
    notes: 'Started 3 hours ago',
    addedAt: new Date(Date.now() - 45 * 60000).toISOString(),
    status: 'waiting',
  },
  {
    id: '5',
    patient_id: 'PT-2024-0045',
    name: 'David Chen',
    age: 45,
    gender_encoded: 1,
    heart_rate: 82,
    bp_systolic: 125,
    bp_diastolic: 80,
    spo2: 98,
    temperature: 36.8,
    resp_rate: 16,
    complaint_encoded: 22,
    sirs_score: 0,
    qsofa_score: 0,
    shock_index: 0.66,
    age_group: 2,
    critical_spo2: 0,
    tachycardia: 0,
    hypotension: 0,
    high_fever: 0,
    tachypnea: 0,
    symptoms: 'Migraine, sensitivity to light',
    heartRate: 82,
    bloodPressure: '125/80',
    oxygenLevel: 98,
    painScale: 6,
    notes: 'Recurring migraine',
    addedAt: new Date(Date.now() - 90 * 60000).toISOString(),
    status: 'waiting',
  },
  {
    id: '6',
    patient_id: 'PT-2024-0028',
    name: 'Sarah Johnson',
    age: 28,
    gender_encoded: 0,
    heart_rate: 78,
    bp_systolic: 120,
    bp_diastolic: 75,
    spo2: 99,
    temperature: 36.6,
    resp_rate: 14,
    complaint_encoded: 30,
    sirs_score: 0,
    qsofa_score: 0,
    shock_index: 0.65,
    age_group: 1,
    critical_spo2: 0,
    tachycardia: 0,
    hypotension: 0,
    high_fever: 0,
    tachypnea: 0,
    symptoms: 'Ankle sprain, swelling',
    heartRate: 78,
    bloodPressure: '120/75',
    oxygenLevel: 99,
    painScale: 4,
    notes: 'Sports injury',
    addedAt: new Date(Date.now() - 60 * 60000).toISOString(),
    status: 'waiting',
  },
]

export function PatientProvider({ children }) {
  const [patients, setPatients] = useState([])

  const normalizePatient = useCallback((patient) => {
    const normalized = {
      ...patient,
      id: String(patient.id),
      addedAt: patient.addedAt || patient.added_at || patient.created_at || new Date().toISOString(),
      status: patient.status || 'waiting',
      esi: patient.esi ?? patient.esi_level,
    }

    if (normalized.heartRate == null && normalized.heart_rate != null) {
      normalized.heartRate = normalized.heart_rate
    }
    if (normalized.oxygenLevel == null && normalized.spo2 != null) {
      normalized.oxygenLevel = normalized.spo2
    }
    if (normalized.bloodPressure == null) {
      const s = normalized.bp_systolic
      const d = normalized.bp_diastolic
      if (s != null || d != null) {
        normalized.bloodPressure = `${s ?? '—'}/${d ?? '—'}`
      }
    }

    return normalized
  }, [])

  const fetchPatients = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/patients`)
      if (!response.ok) throw new Error('Failed to fetch patients')
      const data = await response.json()
      setPatients(data.map(normalizePatient))
    } catch {
      setPatients(DEMO_PATIENTS.map(p => ({ ...p, esi: calculateESI(p) })))
    }
  }, [normalizePatient])

  useEffect(() => {
    fetchPatients()
  }, [fetchPatients])

  const addPatient = useCallback(async (patientData) => {
    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patientData),
      })

      if (!response.ok) throw new Error('Failed to add patient')

      const data = await response.json()
      const patient = normalizePatient(data.patient)
      setPatients(prev => [...prev, patient])
      return patient
    } catch {
      const fallbackPatient = {
        ...patientData,
        id: crypto.randomUUID(),
        addedAt: new Date().toISOString(),
        status: 'waiting',
        esi: calculateESI(patientData),
      }
      setPatients(prev => [...prev, fallbackPatient])
      return fallbackPatient
    }
  }, [normalizePatient])

  const removePatient = useCallback(async (id) => {
    try {
      const response = await fetch(`${API_BASE_URL}/patients/${id}`, { method: 'DELETE' })
      if (!response.ok) throw new Error('Failed to delete patient')
    } catch {
      // Keep UI responsive even if backend call fails.
    }
    setPatients(prev => prev.filter(p => p.id !== id))
  }, [])

  const updateStatus = useCallback(async (id, status) => {
    if (status === 'discharged') {
      try {
        const response = await fetch(`${API_BASE_URL}/patients/${id}`, { method: 'DELETE' })
        if (!response.ok) throw new Error('Failed to discharge and remove patient')
      } catch {
        // Keep UI responsive even if backend call fails.
      }
      setPatients(prev => prev.filter(p => p.id !== id))
      return
    }

    try {
      const response = await fetch(`${API_BASE_URL}/patients/${id}/status`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status }),
      })
      if (!response.ok) throw new Error('Failed to update status')
      const updated = await response.json()
      const normalized = normalizePatient(updated)
      setPatients(prev => prev.map(p => p.id === id ? normalized : p))
      return
    } catch {
      setPatients(prev => prev.map(p => p.id === id ? { ...p, status } : p))
    }
  }, [normalizePatient])

  const sortedPatients = useMemo(() => {
    return [...patients].sort((a, b) => {
      if (a.esi !== b.esi) return a.esi - b.esi
      return new Date(a.addedAt) - new Date(b.addedAt)
    })
  }, [patients])

  const stats = useMemo(() => {
    const waiting = patients.filter(p => p.status === 'waiting')
    const critical = waiting.filter(p => p.esi <= 2)
    const avgRisk = waiting.length > 0
      ? (waiting.reduce((sum, p) => sum + p.esi, 0) / waiting.length).toFixed(1)
      : 0
    return {
      totalWaiting: waiting.length,
      criticalCount: critical.length,
      avgRiskLevel: avgRisk,
      totalPatients: patients.length,
    }
  }, [patients])

  return (
    <PatientContext.Provider value={{ patients: sortedPatients, stats, addPatient, removePatient, updateStatus }}>
      {children}
    </PatientContext.Provider>
  )
}

export const usePatients = () => {
  const ctx = useContext(PatientContext)
  if (!ctx) throw new Error('usePatients must be used within PatientProvider')
  return ctx
}
