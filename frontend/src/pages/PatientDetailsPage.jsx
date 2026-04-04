import { motion as Motion } from 'framer-motion'
import { useNavigate, useParams } from 'react-router-dom'
import { usePatients } from '../contexts/PatientContext'
import { ESI_LABELS, ESI_COLORS } from '../utils/esiCalculator'
import {
  ArrowLeft,
  AlertTriangle,
  Heart,
  Wind,
  Gauge,
  Thermometer,
  Activity,
  Shield,
  User,
  Hash,
  FileText,
  Brain,
  Sparkles,
} from 'lucide-react'

const fadeUp = { hidden: { opacity: 0, y: 12 }, show: { opacity: 1, y: 0, transition: { duration: 0.3 } } }
const stagger = { hidden: { opacity: 0 }, show: { opacity: 1, transition: { staggerChildren: 0.06 } } }

const SYMPTOM_FLAGS = [
  ['Chest Pain', 'chest_pain'],
  ['Arm Pain (Left)', 'arm_pain_left'],
  ['Jaw Pain', 'jaw_pain'],
  ['Dyspnea', 'dyspnea'],
  ['Shortness of Breath', 'shortness_of_breath'],
  ['Facial Droop', 'facial_droop'],
  ['Arm Weakness', 'arm_weakness'],
  ['Speech Difficulty', 'speech_difficulty'],
  ['Abdominal Pain', 'abdominal_pain'],
  ['Rigid Abdomen', 'rigid_abdomen'],
  ['Altered Mental Status', 'altered_mental_status'],
  ['Confusion', 'confusion'],
  ['Fever', 'fever'],
  ['Nausea', 'nausea'],
  ['Vomiting', 'vomiting'],
  ['Dizziness', 'dizziness'],
  ['Syncope', 'syncope'],
  ['Headache', 'headache'],
  ['Seizure', 'seizure'],
  ['Uncontrolled Bleeding', 'uncontrolled_bleeding'],
  ['Severe Pain', 'severe_pain'],
]

function formatValue(val, suffix = '') {
  if (val === null || val === undefined || val === '') return 'N/A'
  return `${val}${suffix}`
}

function getRiskStatus(esi) {
  if (esi <= 2) return { label: 'Critical', className: 'bg-critical-100 text-critical-700 border-critical-200' }
  if (esi <= 3) return { label: 'Moderate', className: 'bg-warning-100 text-warning-700 border-warning-200' }
  return { label: 'Stable', className: 'bg-clinical-100 text-clinical-700 border-clinical-200' }
}

function ParamRow({ label, value, icon: Icon, iconColor = 'text-slate-400' }) {
  const isNA = value === null || value === undefined || value === '' || value === 'N/A'
  return (
    <div className="flex items-center justify-between gap-4 py-2.5 border-b border-slate-100/90 last:border-b-0">
      <div className="flex items-center gap-2 text-sm text-slate-600 min-w-0">
        {Icon && <Icon className={`w-3.5 h-3.5 ${iconColor}`} />}
        <span className="truncate">{label}</span>
      </div>
      <span className={`text-sm font-semibold text-right ${isNA ? 'text-slate-300 italic' : 'text-slate-900'}`}>
        {isNA ? 'N/A' : value}
      </span>
    </div>
  )
}

function FlagBadge({ label, value }) {
  if (value === null || value === undefined || value === '') {
    return (
      <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-center">
        <p className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">{label}</p>
        <p className="text-xs font-semibold text-slate-300 mt-0.5 italic">N/A</p>
      </div>
    )
  }

  const isPositive = Number(value) === 1 || value === true || value === '1'
  return (
    <div
      className={`rounded-xl border px-3 py-2.5 text-center transition-colors ${
        isPositive ? 'border-critical-200 bg-critical-50 shadow-sm shadow-critical-100/50' : 'border-slate-200 bg-white'
      }`}
    >
      <p className={`text-[10px] font-semibold uppercase tracking-wide ${isPositive ? 'text-critical-500' : 'text-slate-500'}`}>
        {label}
      </p>
      <p className={`text-xs font-bold mt-1 ${isPositive ? 'text-critical-700' : 'text-slate-700'}`}>
        {isPositive ? 'YES' : 'NO'}
      </p>
    </div>
  )
}

function SectionCard({ title, icon: Icon, children }) {
  return (
    <div className="bg-white rounded-2xl border border-slate-200/90 p-5 md:p-6 shadow-[0_8px_30px_-20px_rgba(15,23,42,0.35)]">
      <h3 className="text-xs md:text-sm font-bold text-slate-500 uppercase tracking-[0.12em] mb-3 md:mb-4 flex items-center gap-2">
        {Icon && <Icon className="w-4 h-4 text-slate-400" />}
        {title}
      </h3>
      {children}
    </div>
  )
}

export default function PatientDetailsPage() {
  const navigate = useNavigate()
  const { id } = useParams()
  const { patients } = usePatients()

  const patient = patients.find((p) => p.id === id)

  if (!patient) {
    return (
      <Motion.div variants={fadeUp} initial="hidden" animate="show" className="p-8 text-center">
        <h1 className="text-xl font-bold text-slate-800">Patient Not Found</h1>
        <p className="text-slate-500 mt-2">The patient you're looking for doesn't exist.</p>
        <button
          onClick={() => navigate('/queue')}
          className="mt-4 px-4 py-2 rounded-lg bg-medical-600 text-white font-medium hover:bg-medical-700 transition-colors"
        >
          Back to Queue
        </button>
      </Motion.div>
    )
  }

  const risk = getRiskStatus(patient.esi)
  const colors = ESI_COLORS[patient.esi]

  return (
    <Motion.div variants={stagger} initial="hidden" animate="show" className="min-h-screen bg-[radial-gradient(circle_at_15%_15%,rgba(14,165,233,0.10),transparent_40%),radial-gradient(circle_at_85%_20%,rgba(34,197,94,0.10),transparent_38%),linear-gradient(to_bottom,#f8fafc,#eef2f7)]">
      <div className="bg-white/95 backdrop-blur border-b border-slate-200 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 md:px-6 py-4 flex items-center gap-3">
          <button
            onClick={() => navigate('/queue')}
            className="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-slate-100 transition-colors text-slate-600 font-medium"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Queue
          </button>
          <div className="min-w-0">
            <h1 className="text-lg font-bold text-slate-800 truncate">{patient.patient_id || `Patient ${patient.id}`}</h1>
            <p className="text-xs text-slate-500">Full Clinical Details</p>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 md:px-6 py-6 md:py-8 space-y-6">
        <Motion.div variants={fadeUp} className="relative overflow-hidden bg-white rounded-2xl border border-slate-200 shadow-[0_10px_35px_-20px_rgba(15,23,42,0.45)] p-5 md:p-6">
          <div className="absolute -top-12 -right-12 w-44 h-44 rounded-full bg-medical-200/20 blur-2xl" />
          <div className="absolute -bottom-16 -left-10 w-40 h-40 rounded-full bg-clinical-200/20 blur-2xl" />
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div className="flex items-center gap-4">
              <div className={`w-16 h-16 rounded-2xl ${colors.bg} ${colors.text} flex items-center justify-center text-2xl font-black shadow-lg`}>
                {patient.esi}
              </div>
              <div>
                <p className="text-xl md:text-2xl font-bold text-slate-800 tracking-tight">{ESI_LABELS[patient.esi].label}</p>
                <p className="text-sm text-slate-500 mt-0.5">{ESI_LABELS[patient.esi].description}</p>
              </div>
            </div>

            <div className="relative flex flex-wrap items-center gap-2 z-10">
              <span className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold border ${risk.className}`}>
                {patient.esi <= 2 && <AlertTriangle className="w-3 h-3" />}
                {risk.label}
              </span>
              <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold border border-slate-200 text-slate-700 bg-white shadow-sm">
                <Sparkles className="w-3.5 h-3.5" /> ESI: {patient.esi}
              </span>
              <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold border border-slate-200 text-slate-700 bg-white shadow-sm">
                Confidence: {formatValue(patient.confidence)}
              </span>
              <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold border border-slate-200 text-slate-700 bg-white shadow-sm">
                Version: {formatValue(patient.model_version)}
              </span>
            </div>
          </div>
        </Motion.div>

        <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
          <Motion.div variants={fadeUp} className="xl:col-span-4 space-y-6">
            <SectionCard title="Identification" icon={Hash}>
              <ParamRow label="Patient ID" value={patient.patient_id || patient.id} icon={Hash} iconColor="text-medical-400" />
            </SectionCard>

            <SectionCard title="Demographics" icon={User}>
              <ParamRow label="Age" value={formatValue(patient.age, ' yrs')} icon={User} iconColor="text-medical-400" />
              <ParamRow label="Gender" value={formatValue(patient.gender)} icon={User} iconColor="text-medical-400" />
              <ParamRow label="Gender (Encoded)" value={formatValue(patient.gender_encoded)} icon={User} iconColor="text-medical-400" />
              <ParamRow label="Age Group" value={formatValue(patient.age_group)} />
            </SectionCard>

            <SectionCard title="Model Output" icon={Brain}>
              <ParamRow label="ESI Level" value={formatValue(patient.esi)} />
              <ParamRow label="Confidence" value={formatValue(patient.confidence)} />
              <ParamRow label="Model Version" value={formatValue(patient.model_version)} />
              <ParamRow label="Status" value={formatValue(patient.status)} />
            </SectionCard>
          </Motion.div>

          <Motion.div variants={fadeUp} className="xl:col-span-4 space-y-6">
            <SectionCard title="Vital Signs" icon={Activity}>
              <ParamRow label="Heart Rate" value={formatValue(patient.heart_rate, ' bpm')} icon={Heart} iconColor="text-critical-400" />
              <ParamRow label="BP Systolic" value={formatValue(patient.bp_systolic, ' mmHg')} icon={Gauge} iconColor="text-medical-400" />
              <ParamRow label="BP Diastolic" value={formatValue(patient.bp_diastolic, ' mmHg')} icon={Gauge} iconColor="text-medical-400" />
              <ParamRow label="SpO2" value={formatValue(patient.spo2, '%')} icon={Wind} iconColor="text-clinical-500" />
              <ParamRow label="Temperature" value={formatValue(patient.temperature, ' °C')} icon={Thermometer} iconColor="text-warning-500" />
              <ParamRow
                label="Respiratory Rate"
                value={formatValue(patient.respiratory_rate ?? patient.resp_rate, ' /min')}
                icon={Activity}
                iconColor="text-medical-400"
              />
            </SectionCard>

            <SectionCard title="Clinical Scores" icon={Shield}>
              <ParamRow label="Complaint" value={formatValue(patient.complaint)} />
              <ParamRow label="Complaint Encoded" value={formatValue(patient.complaint_encoded)} />
              <ParamRow label="SIRS Score" value={formatValue(patient.sirs_score)} />
              <ParamRow label="qSOFA Score" value={formatValue(patient.qsofa_score)} />
              <ParamRow label="Shock Index" value={formatValue(patient.shock_index)} />
              <ParamRow label="Symptom Duration" value={formatValue(patient.symptom_duration_hours, ' hrs')} />
            </SectionCard>
          </Motion.div>

          <Motion.div variants={fadeUp} className="xl:col-span-4 space-y-6">
            <SectionCard title="Clinical Flags" icon={AlertTriangle}>
              <div className="grid grid-cols-2 gap-2 mb-3">
                <FlagBadge label="Critical SpO2" value={patient.critical_spo2} />
                <FlagBadge label="Tachycardia" value={patient.tachycardia} />
                <FlagBadge label="Hypotension" value={patient.hypotension} />
                <FlagBadge label="High Fever" value={patient.high_fever} />
                <FlagBadge label="Tachypnea" value={patient.tachypnea} />
              </div>
              <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-[0.12em] mb-2">Symptom Flags</h4>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {SYMPTOM_FLAGS.map(([label, key]) => (
                  <FlagBadge key={key} label={label} value={patient[key]} />
                ))}
              </div>
            </SectionCard>

            {patient.notes && (
              <SectionCard title="Notes" icon={FileText}>
                <p className="text-sm text-slate-700 leading-relaxed whitespace-pre-wrap">{patient.notes}</p>
              </SectionCard>
            )}
          </Motion.div>
        </div>
      </div>
    </Motion.div>
  )
}
