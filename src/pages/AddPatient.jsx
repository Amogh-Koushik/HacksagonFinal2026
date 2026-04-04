import { useState, useRef, useEffect } from 'react'
import { motion as Motion, AnimatePresence } from 'framer-motion'
import { usePatients } from '../contexts/PatientContext'
import { useToast } from '../contexts/ToastContext'
import { ESI_LABELS } from '../utils/esiCalculator'
import {
  UserPlus, Activity, Gauge, Check, AlertCircle,
  Wind, Thermometer, Hash, User, Heart, Brain, Clock
} from 'lucide-react'
import MedicalInputField from '../components/MedicalInputField'
import BooleanCheckboxField from '../components/BooleanCheckboxField'

const stagger = { hidden: { opacity: 0 }, show: { opacity: 1, transition: { staggerChildren: 0.05 } } }
const fadeUp = { hidden: { opacity: 0, y: 12 }, show: { opacity: 1, y: 0, transition: { duration: 0.3 } } }

const GENDER_OPTIONS = [
  { value: 'M', label: 'Male' },
  { value: 'F', label: 'Female' },
]

const VITAL_FIELDS = [
  { key: 'heart_rate', label: 'heart_rate (BPM)' },
  { key: 'bp_systolic', label: 'bp_systolic (mmHg)' },
  { key: 'bp_diastolic', label: 'bp_diastolic (mmHg)' },
  { key: 'spo2', label: 'spo2 (Oxygen saturation %)' },
  { key: 'temperature', label: 'temperature (Celsius)' },
  { key: 'respiratory_rate', label: 'respiratory_rate (Breaths per min)' },
]

const SYMPTOM_FIELDS = [
  { key: 'chest_pain', label: 'Chest Pain' },
  { key: 'arm_pain_left', label: 'Arm Pain (Left)' },
  { key: 'jaw_pain', label: 'Jaw Pain' },
  { key: 'dyspnea', label: 'Dyspnea (Difficulty Breathing)' },
  { key: 'shortness_of_breath', label: 'Shortness of Breath' },
  { key: 'facial_droop', label: 'Facial Droop' },
  { key: 'arm_weakness', label: 'Arm Weakness' },
  { key: 'speech_difficulty', label: 'Speech Difficulty' },
  { key: 'abdominal_pain', label: 'Abdominal Pain' },
  { key: 'rigid_abdomen', label: 'Rigid Abdomen' },
  { key: 'altered_mental_status', label: 'Altered Mental Status' },
  { key: 'confusion', label: 'Confusion' },
  { key: 'fever', label: 'Fever' },
  { key: 'nausea', label: 'Nausea' },
  { key: 'vomiting', label: 'Vomiting' },
  { key: 'dizziness', label: 'Dizziness' },
  { key: 'syncope', label: 'Syncope (Fainting)' },
  { key: 'headache', label: 'Headache' },
  { key: 'seizure', label: 'Seizure' },
  { key: 'uncontrolled_bleeding', label: 'Uncontrolled Bleeding' },
  { key: 'severe_pain', label: 'Severe Pain' },
]

const INITIAL_FORM = {
  patient_id: '',
  age: '',
  gender: '',
  heart_rate: '',
  bp_systolic: '',
  bp_diastolic: '',
  spo2: '',
  temperature: '',
  respiratory_rate: '',
  symptom_duration_hours: '',
  notes: '',
  ...Object.fromEntries(SYMPTOM_FIELDS.map(({ key }) => [key, 0])),
}

function genderToEncoded(gender) {
  if (gender === 'M') return 1
  if (gender === 'F') return 0
  return null
}

function buildSymptomText(form) {
  const selected = SYMPTOM_FIELDS.filter(({ key }) => Number(form[key]) === 1).map(({ label }) => label)
  return selected.join(', ')
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
    try {
      await new Promise(r => setTimeout(r, 300))

      const patientData = {
        patient_id: form.patient_id.trim(),
        age: form.age === '' ? null : Number(form.age),
        gender: form.gender || null,
        gender_encoded: genderToEncoded(form.gender),
        heart_rate: form.heart_rate === '' ? null : Number(form.heart_rate),
        bp_systolic: form.bp_systolic === '' ? null : Number(form.bp_systolic),
        bp_diastolic: form.bp_diastolic === '' ? null : Number(form.bp_diastolic),
        spo2: form.spo2 === '' ? null : Number(form.spo2),
        temperature: form.temperature === '' ? null : Number(form.temperature),
        respiratory_rate: form.respiratory_rate === '' ? null : Number(form.respiratory_rate),
        resp_rate: form.respiratory_rate === '' ? null : Number(form.respiratory_rate),
        symptom_duration_hours: form.symptom_duration_hours === '' ? null : Number(form.symptom_duration_hours),
        notes: form.notes.trim() || null,
      }

      SYMPTOM_FIELDS.forEach(({ key }) => {
        patientData[key] = Number(form[key])
      })

      const symptomText = buildSymptomText(form)
      patientData.symptoms = symptomText || null
      patientData.complaint = symptomText || null

      const patient = await addPatient(patientData)
      setLastAdded(patient)
      showToast(
        `Patient ${form.patient_id} added — ESI ${patient.esi} (${ESI_LABELS[patient.esi].label})`,
        patient.esi <= 2 ? 'critical' : 'success'
      )

      setForm({ ...INITIAL_FORM })
      idRef.current?.focus()
    } catch (error) {
      showToast(error?.message || 'Unable to get ESI from ML model', 'critical')
    } finally {
      setIsSubmitting(false)
    }
  }

  const inputCls = 'w-full px-4 py-3 rounded-xl border border-slate-200 bg-slate-50/50 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-medical-500/20 focus:border-medical-400 transition-all'

  return (
    <Motion.div variants={stagger} initial="hidden" animate="show">
      <Motion.div variants={fadeUp} className="mb-8">
        <h1 className="text-2xl font-bold text-slate-800 tracking-tight">Add Patient</h1>
        <p className="text-slate-500 text-sm mt-1">Enter demographics, vitals, symptoms, and duration for AI-assisted triage</p>
      </Motion.div>

      <AnimatePresence>
        {lastAdded && (
          <Motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}
            className={`mb-6 flex items-center gap-3 p-4 rounded-xl border ${
              lastAdded.esi <= 2 ? 'bg-critical-50 border-critical-200 text-critical-800' : 'bg-clinical-50 border-clinical-200 text-clinical-800'
            }`}>
            {lastAdded.esi <= 2 ? <AlertCircle className="w-5 h-5 shrink-0" /> : <Check className="w-5 h-5 shrink-0" />}
            <div>
              <p className="text-sm font-medium">Patient {lastAdded.patient_id} — ESI {lastAdded.esi} ({ESI_LABELS[lastAdded.esi].label})</p>
              {lastAdded.esi <= 2 && <p className="text-xs mt-0.5 opacity-80">Critical case requires immediate attention</p>}
            </div>
          </Motion.div>
        )}
      </AnimatePresence>

      <Motion.div variants={fadeUp}>
        <form onSubmit={handleSubmit} className="bg-white rounded-2xl border border-slate-200/60 shadow-sm overflow-hidden">
          <div className="px-8 py-5 border-b border-slate-100 flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-medical-50 border border-medical-100 flex items-center justify-center">
              <UserPlus className="w-5 h-5 text-medical-600" />
            </div>
            <div>
              <h2 className="text-[15px] font-semibold text-slate-800">Patient Intake Form</h2>
              <p className="text-xs text-slate-500">36 structured inputs for triage prediction</p>
            </div>
          </div>

          <div className="p-8 space-y-8">
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
                  className={inputCls}
                />
              </div>
            </div>

            <div>
              <h3 className="text-sm font-semibold text-slate-700 mb-4 flex items-center gap-2">
                <User className="w-4 h-4 text-slate-400" /> Demographics
              </h3>
              <div className="rounded-2xl border border-slate-100 bg-slate-50/30 p-5">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                  <MedicalInputField
                    label="Age"
                    value={form.age}
                    onChange={updateField('age')}
                    type="number"
                    placeholder="Years"
                    min="0"
                    max="120"
                    icon={User}
                    iconColor="text-medical-400"
                    required
                  />

                  <div>
                    <label className="block text-sm font-medium text-slate-600 mb-1.5 flex items-center gap-1.5">
                      <User className="w-3.5 h-3.5 text-medical-400" /> Gender <span className="text-critical-400">*</span>
                    </label>
                    <select
                      value={form.gender}
                      onChange={(e) => updateField('gender')(e.target.value)}
                      required
                      className={`${inputCls} appearance-none cursor-pointer`}
                    >
                      <option value="">Select gender</option>
                      {GENDER_OPTIONS.map((opt) => (
                        <option key={opt.value} value={opt.value}>{opt.label}</option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>
            </div>

            <div>
              <h3 className="text-sm font-semibold text-slate-700 mb-4 flex items-center gap-2">
                <Activity className="w-4 h-4 text-slate-400" /> Vital Signs
              </h3>
              <div className="mb-4 rounded-xl border border-warning-200 bg-warning-50 px-4 py-3 text-xs text-warning-800">
                Important: Please fill all required vital inputs exactly as listed.
                <div className="mt-1 text-[11px] text-warning-700">
                  {VITAL_FIELDS.map(v => v.label).join(' | ')}
                </div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
                <MedicalInputField
                  label="heart_rate (BPM)"
                  value={form.heart_rate}
                  onChange={updateField('heart_rate')}
                  type="number"
                  placeholder="60-100"
                  min="20"
                  max="250"
                  icon={Heart}
                  iconColor="text-critical-400"
                  required
                />
                <MedicalInputField
                  label="bp_systolic (mmHg)"
                  value={form.bp_systolic}
                  onChange={updateField('bp_systolic')}
                  type="number"
                  placeholder="120"
                  min="50"
                  max="260"
                  icon={Gauge}
                  iconColor="text-medical-400"
                  required
                />
                <MedicalInputField
                  label="bp_diastolic (mmHg)"
                  value={form.bp_diastolic}
                  onChange={updateField('bp_diastolic')}
                  type="number"
                  placeholder="80"
                  min="30"
                  max="180"
                  icon={Gauge}
                  iconColor="text-medical-400"
                  required
                />
                <MedicalInputField
                  label="spo2 (Oxygen saturation %)"
                  value={form.spo2}
                  onChange={updateField('spo2')}
                  type="number"
                  placeholder="95-100"
                  min="50"
                  max="100"
                  icon={Wind}
                  iconColor="text-clinical-500"
                  required
                />
                <MedicalInputField
                  label="temperature (Celsius)"
                  value={form.temperature}
                  onChange={updateField('temperature')}
                  type="number"
                  placeholder="36.5-37.5"
                  min="30"
                  max="45"
                  step="0.1"
                  icon={Thermometer}
                  iconColor="text-warning-500"
                  required
                />
                <MedicalInputField
                  label="respiratory_rate (Breaths per min)"
                  value={form.respiratory_rate}
                  onChange={updateField('respiratory_rate')}
                  type="number"
                  placeholder="12-20"
                  min="0"
                  max="80"
                  icon={Activity}
                  iconColor="text-medical-400"
                  required
                />
              </div>
            </div>

            <div>
              <h3 className="text-sm font-semibold text-slate-700 mb-4 flex items-center gap-2">
                <Brain className="w-4 h-4 text-slate-400" /> Clinical Symptoms (Yes / No)
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {SYMPTOM_FIELDS.map(({ key, label }) => (
                  <BooleanCheckboxField
                    key={key}
                    label={label}
                    value={form[key]}
                    onChange={updateField(key)}
                  />
                ))}
              </div>
            </div>

            <div>
              <h3 className="text-sm font-semibold text-slate-700 mb-4 flex items-center gap-2">
                <Clock className="w-4 h-4 text-slate-400" /> Duration
              </h3>
              <div className="max-w-md">
                <MedicalInputField
                  label="symptom_duration_hours"
                  value={form.symptom_duration_hours}
                  onChange={updateField('symptom_duration_hours')}
                  type="number"
                  placeholder="How many hours ago symptoms started"
                  min="0"
                  max="720"
                  step="0.1"
                  icon={Clock}
                  iconColor="text-medical-400"
                />
              </div>
            </div>

            <div>
              <h3 className="text-sm font-semibold text-slate-700 mb-4 flex items-center gap-2">
                <AlertCircle className="w-4 h-4 text-slate-400" /> Notes (Optional)
              </h3>
              <textarea
                value={form.notes}
                onChange={(e) => setForm(prev => ({ ...prev, notes: e.target.value }))}
                placeholder="Additional notes about this patient"
                rows="3"
                className="w-full px-4 py-3 rounded-xl border border-slate-200 bg-slate-50/50 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-medical-500/20 focus:border-medical-400 transition-all resize-none"
              />
            </div>
          </div>

          <div className="px-8 py-5 border-t border-slate-100 bg-slate-50/50 flex justify-end">
            <button type="submit" disabled={isSubmitting}
              className="px-8 py-3 rounded-xl bg-medical-600 hover:bg-medical-700 active:bg-medical-800 text-white font-medium text-sm transition-all disabled:opacity-60 disabled:cursor-not-allowed flex items-center gap-2 cursor-pointer">
              {isSubmitting ? (
                <>
                  <Motion.div animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                    className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full" />
                  Processing...
                </>
              ) : (
                <><UserPlus className="w-4 h-4" /> Add to Queue</>
              )}
            </button>
          </div>
        </form>
      </Motion.div>
    </Motion.div>
  )
}
