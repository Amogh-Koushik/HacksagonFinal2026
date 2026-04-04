import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { usePatients } from '../contexts/PatientContext'
import { useToast } from '../contexts/ToastContext'
import { ESI_LABELS } from '../utils/esiCalculator'
import {
  UserPlus, Heart, Activity, Gauge, Check, AlertCircle,
  Wind, Thermometer, Hash, User, Zap, Brain, ShieldAlert, FileText
} from 'lucide-react'
import MedicalInputField from '../components/MedicalInputField'
import BooleanCheckboxField from '../components/BooleanCheckboxField'

const stagger = { hidden: { opacity: 0 }, show: { opacity: 1, transition: { staggerChildren: 0.05 } } }
const fadeUp = { hidden: { opacity: 0, y: 12 }, show: { opacity: 1, y: 0, transition: { duration: 0.3 } } }

const GENDER_OPTIONS = [
  { value: '1', label: 'Male' },
  { value: '0', label: 'Female' },
]

const AGE_GROUP_OPTIONS = [
  { value: '1', label: 'Group 1 - Child (0-17)' },
  { value: '2', label: 'Group 2 - Adult (18-44)' },
  { value: '3', label: 'Group 3 - Middle Age (45-64)' },
  { value: '4', label: 'Group 4 - Senior (65+)' },
]

const INITIAL_FORM = {
  patient_id: '',
  age: '',
  gender_encoded: '',
  heart_rate: '',
  bp_systolic: '',
  bp_diastolic: '',
  spo2: '',
  temperature: '',
  resp_rate: '',
  complaint: '',
  sirs_score: '',
  qsofa_score: '',
  shock_index: '',
  age_group: '',
  critical_spo2: 0,
  tachycardia: 0,
  hypotension: 0,
  high_fever: 0,
  tachypnea: 0,
  notes: '',
}

export default function AddPatient() {
  const { addPatient } = usePatients()
  const { showToast } = useToast()
  const idRef = useRef(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [lastAdded, setLastAdded] = useState(null)

  const [form, setForm] = useState({ ...INITIAL_FORM })

  useEffect(() => { idRef.current?.focus() }, [])

  const updateField = (field) => (val) => {
    setForm(prev => ({ ...prev, [field]: val }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!form.patient_id.trim()) return
    setIsSubmitting(true)
    await new Promise(r => setTimeout(r, 500))

    // Build patient object - convert numeric fields
    const numericFields = [
      'age', 'gender_encoded', 'heart_rate', 'bp_systolic', 'bp_diastolic',
      'spo2', 'temperature', 'resp_rate', 'sirs_score',
      'qsofa_score', 'shock_index', 'age_group'
    ]

    const patientData = { patient_id: form.patient_id.trim() }
    
    // Handle numeric fields - convert empty to null
    numericFields.forEach(f => {
      if (form[f] === null || form[f] === undefined || form[f] === '') {
        patientData[f] = null
      } else {
        patientData[f] = Number(form[f])
      }
    })

    // Handle boolean fields - already 0 or 1
    patientData.critical_spo2 = Number(form.critical_spo2)
    patientData.tachycardia = Number(form.tachycardia)
    patientData.hypotension = Number(form.hypotension)
    patientData.high_fever = Number(form.high_fever)
    patientData.tachypnea = Number(form.tachypnea)

    // Handle notes
    patientData.complaint = form.complaint.trim() || null
    patientData.notes = form.notes.trim() || null

    const patient = await addPatient(patientData)
    setLastAdded(patient)
    showToast(
      `Patient ${form.patient_id} added — ESI ${patient.esi} (${ESI_LABELS[patient.esi].label})`,
      patient.esi <= 2 ? 'critical' : 'success'
    )
    setForm({ ...INITIAL_FORM })
    setIsSubmitting(false)
    idRef.current?.focus()
  }

  const inputCls = "w-full px-4 py-3 rounded-xl border border-slate-200 bg-slate-50/50 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-medical-500/20 focus:border-medical-400 transition-all"

  return (
    <motion.div variants={stagger} initial="hidden" animate="show">
      <motion.div variants={fadeUp} className="mb-8">
        <h1 className="text-2xl font-bold text-slate-800 tracking-tight">Add Patient</h1>
        <p className="text-slate-500 text-sm mt-1">Enter patient clinical parameters for AI-assisted triage assessment</p>
      </motion.div>

      <AnimatePresence>
        {lastAdded && (
          <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}
            className={`mb-6 flex items-center gap-3 p-4 rounded-xl border ${
              lastAdded.esi <= 2 ? 'bg-critical-50 border-critical-200 text-critical-800' : 'bg-clinical-50 border-clinical-200 text-clinical-800'
            }`}>
            {lastAdded.esi <= 2 ? <AlertCircle className="w-5 h-5 shrink-0" /> : <Check className="w-5 h-5 shrink-0" />}
            <div>
              <p className="text-sm font-medium">Patient {lastAdded.patient_id} — ESI {lastAdded.esi} ({ESI_LABELS[lastAdded.esi].label})</p>
              {lastAdded.esi <= 2 && <p className="text-xs mt-0.5 opacity-80">⚠ Critical — requires immediate attention</p>}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <motion.div variants={fadeUp}>
        <form onSubmit={handleSubmit} className="bg-white rounded-2xl border border-slate-200/60 shadow-sm overflow-hidden">
          <div className="px-8 py-5 border-b border-slate-100 flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-medical-50 border border-medical-100 flex items-center justify-center">
              <UserPlus className="w-5 h-5 text-medical-600" />
            </div>
            <div>
              <h2 className="text-[15px] font-semibold text-slate-800">Patient Intake Form</h2>
              <p className="text-xs text-slate-500">Complete clinical parameters for ESI assessment</p>
            </div>
          </div>

          <div className="p-8 space-y-8">
            {/* Patient ID */}
            <div>
              <h3 className="text-sm font-semibold text-slate-700 mb-4 flex items-center gap-2">
                <Hash className="w-4 h-4 text-slate-400" /> Patient Identification
              </h3>
              <div className="max-w-md">
                <label className="block text-sm font-medium text-slate-600 mb-1.5 flex items-center gap-1.5">
                  <Hash className="w-3.5 h-3.5 text-medical-400" /> Patient ID <span className="text-critical-400">*</span>
                </label>
                <input
                  ref={idRef}
                  type="text"
                  value={form.patient_id}
                  onChange={(e) => setForm(prev => ({ ...prev, patient_id: e.target.value }))}
                  placeholder="Enter patient ID"
                  required
                  className="w-full px-4 py-3 rounded-xl border border-slate-200 bg-slate-50/50 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-medical-500/20 focus:border-medical-400 transition-all"
                />
              </div>
            </div>

            {/* Demographics */}
            <div>
              <h3 className="text-sm font-semibold text-slate-700 mb-4 flex items-center gap-2">
                <User className="w-4 h-4 text-slate-400" /> Demographics
              </h3>
              <div className="rounded-2xl border border-slate-100 bg-slate-50/30 p-5">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                  <MedicalInputField
                    label="Age"
                    value={form.age}
                    onChange={updateField('age')}
                    type="number"
                    placeholder="Years"
                    min="0"
                    max="150"
                    icon={User}
                    iconColor="text-medical-400"
                  />

                  <div>
                    <label className="block text-sm font-medium text-slate-600 mb-1.5 flex items-center gap-1.5">
                      <User className="w-3.5 h-3.5 text-medical-400" /> Gender
                    </label>
                    <select
                      value={form.gender_encoded}
                      onChange={(e) => updateField('gender_encoded')(e.target.value)}
                      className={`${inputCls} appearance-none cursor-pointer`}
                    >
                      <option value="">Select gender</option>
                      {GENDER_OPTIONS.map((opt) => (
                        <option key={opt.value} value={opt.value}>{opt.label}</option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-slate-600 mb-1.5 flex items-center gap-1.5">
                      <User className="w-3.5 h-3.5 text-medical-400" /> Age Group
                    </label>
                    <select
                      value={form.age_group}
                      onChange={(e) => updateField('age_group')(e.target.value)}
                      className={`${inputCls} appearance-none cursor-pointer`}
                    >
                      <option value="">Select age group</option>
                      {AGE_GROUP_OPTIONS.map((opt) => (
                        <option key={opt.value} value={opt.value}>{opt.label}</option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>
            </div>

            {/* Vital Signs */}
            <div>
              <h3 className="text-sm font-semibold text-slate-700 mb-4 flex items-center gap-2">
                <Activity className="w-4 h-4 text-slate-400" /> Vital Signs
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
                <MedicalInputField
                  label="Heart Rate (bpm)"
                  value={form.heart_rate}
                  onChange={updateField('heart_rate')}
                  type="number"
                  placeholder="60–100"
                  min="20"
                  max="300"
                  icon={Heart}
                  iconColor="text-critical-400"
                />
                <MedicalInputField
                  label="BP Systolic (mmHg)"
                  value={form.bp_systolic}
                  onChange={updateField('bp_systolic')}
                  type="number"
                  placeholder="120"
                  min="40"
                  max="300"
                  icon={Gauge}
                  iconColor="text-medical-400"
                />
                <MedicalInputField
                  label="BP Diastolic (mmHg)"
                  value={form.bp_diastolic}
                  onChange={updateField('bp_diastolic')}
                  type="number"
                  placeholder="80"
                  min="20"
                  max="200"
                  icon={Gauge}
                  iconColor="text-medical-400"
                />
                <MedicalInputField
                  label="SpO₂ (%)"
                  value={form.spo2}
                  onChange={updateField('spo2')}
                  type="number"
                  placeholder="95–100"
                  min="50"
                  max="100"
                  icon={Wind}
                  iconColor="text-clinical-500"
                />
                <MedicalInputField
                  label="Temperature (°C)"
                  value={form.temperature}
                  onChange={updateField('temperature')}
                  type="number"
                  placeholder="36.5–37.5"
                  min="30"
                  max="45"
                  step="0.1"
                  icon={Thermometer}
                  iconColor="text-warning-500"
                />
                <MedicalInputField
                  label="Resp Rate (/min)"
                  value={form.resp_rate}
                  onChange={updateField('resp_rate')}
                  type="number"
                  placeholder="12–20"
                  min="0"
                  max="80"
                  icon={Activity}
                  iconColor="text-medical-400"
                />
              </div>
            </div>

            {/* Clinical Assessment */}
            <div>
              <h3 className="text-sm font-semibold text-slate-700 mb-4 flex items-center gap-2">
                <Brain className="w-4 h-4 text-slate-400" /> Clinical Assessment
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-5">
                <MedicalInputField
                  label="SIRS Score"
                  value={form.sirs_score}
                  onChange={updateField('sirs_score')}
                  type="number"
                  placeholder="0–4"
                  min="0"
                  max="4"
                  icon={Zap}
                  iconColor="text-warning-500"
                />
                <MedicalInputField
                  label="qSOFA Score"
                  value={form.qsofa_score}
                  onChange={updateField('qsofa_score')}
                  type="number"
                  placeholder="0–3"
                  min="0"
                  max="3"
                  icon={Zap}
                  iconColor="text-critical-400"
                />
                <MedicalInputField
                  label="Shock Index"
                  value={form.shock_index}
                  onChange={updateField('shock_index')}
                  type="number"
                  placeholder="0.5–1.0"
                  min="0"
                  max="5"
                  step="0.01"
                  icon={ShieldAlert}
                  iconColor="text-critical-400"
                />
              </div>
            </div>

            {/* Clinical Flags */}
            <div>
              <h3 className="text-sm font-semibold text-slate-700 mb-4 flex items-center gap-2">
                <ShieldAlert className="w-4 h-4 text-slate-400" /> Clinical Flags
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
                <BooleanCheckboxField
                  label="Critical SpO₂"
                  value={form.critical_spo2}
                  onChange={updateField('critical_spo2')}
                  icon={Wind}
                  iconColor="text-critical-400"
                />
                <BooleanCheckboxField
                  label="Tachycardia"
                  value={form.tachycardia}
                  onChange={updateField('tachycardia')}
                  icon={Heart}
                  iconColor="text-critical-400"
                />
                <BooleanCheckboxField
                  label="Hypotension"
                  value={form.hypotension}
                  onChange={updateField('hypotension')}
                  icon={Gauge}
                  iconColor="text-warning-500"
                />
                <BooleanCheckboxField
                  label="High Fever"
                  value={form.high_fever}
                  onChange={updateField('high_fever')}
                  icon={Thermometer}
                  iconColor="text-warning-500"
                />
                <BooleanCheckboxField
                  label="Tachypnea"
                  value={form.tachypnea}
                  onChange={updateField('tachypnea')}
                  icon={Activity}
                  iconColor="text-warning-500"
                />
              </div>
            </div>

            {/* Complaint */}
            <div>
              <h3 className="text-sm font-semibold text-slate-700 mb-4 flex items-center gap-2">
                <AlertCircle className="w-4 h-4 text-slate-400" /> Complaint
              </h3>
              <input
                type="text"
                value={form.complaint}
                onChange={(e) => updateField('complaint')(e.target.value)}
                placeholder="Patient complaint in plain text"
                className={inputCls}
              />
            </div>

            {/* Notes */}
            <div>
              <h3 className="text-sm font-semibold text-slate-700 mb-4 flex items-center gap-2">
                <FileText className="w-4 h-4 text-slate-400" /> Notes
              </h3>
              <textarea
                value={form.notes}
                onChange={(e) => setForm(prev => ({ ...prev, notes: e.target.value }))}
                placeholder="Additional notes about the patient..."
                rows="3"
                className="w-full px-4 py-3 rounded-xl border border-slate-200 bg-slate-50/50 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-medical-500/20 focus:border-medical-400 transition-all resize-none"
              />
            </div>
          </div>

          {/* Submit */}
          <div className="px-8 py-5 border-t border-slate-100 bg-slate-50/50 flex justify-end">
            <button type="submit" disabled={isSubmitting}
              className="px-8 py-3 rounded-xl bg-medical-600 hover:bg-medical-700 active:bg-medical-800 text-white font-medium text-sm transition-all disabled:opacity-60 disabled:cursor-not-allowed flex items-center gap-2 cursor-pointer">
              {isSubmitting ? (
                <>
                  <motion.div animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                    className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full" />
                  Processing...
                </>
              ) : (
                <><UserPlus className="w-4 h-4" /> Add to Queue</>
              )}
            </button>
          </div>
        </form>
      </motion.div>
    </motion.div>
  )
}
