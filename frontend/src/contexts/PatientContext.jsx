import { createContext, useContext, useState, useCallback, useMemo, useEffect } from 'react'

const PatientContext = createContext(null)
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1'

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
      setPatients([])
    }
  }, [normalizePatient])

  useEffect(() => {
    fetchPatients()
  }, [fetchPatients])

  const addPatient = useCallback(async (patientData) => {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patientData),
    })

    if (!response.ok) {
      let detail = 'Failed to add patient'
      try {
        const payload = await response.json()
        if (payload?.detail) detail = String(payload.detail)
      } catch {
        // Ignore parse errors and keep default message.
      }
      throw new Error(detail)
    }

    const data = await response.json()
    const patient = normalizePatient(data.patient)
    setPatients(prev => [...prev, patient])
    return patient
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
