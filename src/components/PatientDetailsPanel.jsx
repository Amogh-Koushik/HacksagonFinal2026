import { motion as Motion, AnimatePresence } from 'framer-motion'
import { X, AlertTriangle, Heart, Wind, Gauge, Thermometer, Activity, Shield, User, Hash } from 'lucide-react'
import { ESI_LABELS, ESI_COLORS } from '../utils/esiCalculator'

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
  if (esi <= 2) return { label: 'CRITICAL', className: 'bg-critical-100 text-critical-700 border-critical-200' }
  if (esi <= 3) return { label: 'MODERATE', className: 'bg-warning-100 text-warning-700 border-warning-200' }
  return { label: 'STABLE', className: 'bg-clinical-100 text-clinical-700 border-clinical-200' }
}

function ParamRow({ label, value, icon: Icon, iconColor = 'text-slate-400' }) {
  const isNA = value === null || value === undefined || value === '' || value === 'N/A'
  return (
    <div className="flex items-center justify-between py-2.5 border-b border-slate-100 last:border-b-0">
      <div className="flex items-center gap-2 text-sm text-slate-500">
        {Icon && <Icon className={`w-3.5 h-3.5 ${iconColor}`} />}
        {label}
      </div>
      <span className={`text-sm font-semibold ${isNA ? 'text-slate-300 italic' : 'text-slate-800'}`}>
        {isNA ? 'N/A' : value}
      </span>
    </div>
  )
}

function ESIVisualIndicator({ esi }) {
  if (esi <= 2) {
    return (
      <Motion.div
        className="relative rounded-2xl overflow-hidden"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        {/* Red pulse background */}
        <Motion.div
          className="absolute inset-0 bg-critical-500/10 rounded-2xl"
          animate={{ opacity: [0.3, 0.8, 0.3] }}
          transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
        />
        <div className="relative px-6 py-5 flex items-center gap-4">
          <Motion.div
            className="w-14 h-14 rounded-xl bg-critical-600 flex items-center justify-center shadow-lg shadow-critical-500/30"
            animate={{ scale: [1, 1.08, 1] }}
            transition={{ duration: 1.2, repeat: Infinity, ease: 'easeInOut' }}
          >
            <span className="text-white text-xl font-black">{esi}</span>
          </Motion.div>
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <h3 className="text-lg font-bold text-critical-800">{ESI_LABELS[esi].label}</h3>
              <Motion.div
                animate={{ opacity: [1, 0.3, 1] }}
                transition={{ duration: 0.8, repeat: Infinity }}
              >
                <AlertTriangle className="w-5 h-5 text-critical-500" />
              </Motion.div>
            </div>
            <p className="text-sm text-critical-600 mt-0.5">{ESI_LABELS[esi].description}</p>
          </div>
        </div>
      </Motion.div>
    )
  }

  if (esi === 5) {
    return (
      <Motion.div
        className="relative rounded-2xl overflow-hidden"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        {/* Gentle green breathing glow */}
        <Motion.div
          className="absolute inset-0 rounded-2xl"
          style={{
            background: 'radial-gradient(ellipse at center, rgba(23, 180, 99, 0.1), transparent 70%)',
          }}
          animate={{ opacity: [0.4, 1, 0.4] }}
          transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
        />
        <div className="relative px-6 py-5 flex items-center gap-4">
          <Motion.div
            className="w-14 h-14 rounded-xl bg-clinical-500 flex items-center justify-center shadow-lg shadow-clinical-400/20"
            animate={{ scale: [1, 1.03, 1] }}
            transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
          >
            <span className="text-white text-xl font-black">{esi}</span>
          </Motion.div>
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <h3 className="text-lg font-bold text-clinical-700">{ESI_LABELS[esi].label}</h3>
              <Motion.div
                animate={{ opacity: [0.5, 1, 0.5] }}
                transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
              >
                <Shield className="w-4 h-4 text-clinical-400" />
              </Motion.div>
            </div>
            <p className="text-sm text-clinical-600 mt-0.5">{ESI_LABELS[esi].description}</p>
          </div>
        </div>
      </Motion.div>
    )
  }

  // ESI 3–4: Neutral clinical
  const colors = ESI_COLORS[esi]
  return (
    <div className="px-6 py-5 flex items-center gap-4">
      <div className={`w-14 h-14 rounded-xl ${colors.bg} flex items-center justify-center`}>
        <span className={`text-xl font-black ${colors.text}`}>{esi}</span>
      </div>
      <div className="flex-1">
        <h3 className="text-lg font-bold text-slate-800">{ESI_LABELS[esi].label}</h3>
        <p className="text-sm text-slate-500 mt-0.5">{ESI_LABELS[esi].description}</p>
      </div>
    </div>
  )
}

export default function PatientDetailsPanel({ patient, onClose }) {
  if (!patient) return null

  const risk = getRiskStatus(patient.esi)

  return (
    <AnimatePresence>
      {patient && (
        <>
          {/* Backdrop */}
          <Motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 bg-black/20 backdrop-blur-sm z-40"
            onClick={onClose}
          />

          {/* Panel */}
          <Motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', stiffness: 300, damping: 30 }}
            className="fixed top-0 right-0 h-full w-full max-w-lg bg-white shadow-2xl z-50 flex flex-col border-l border-slate-200"
          >
            {/* Header */}
            <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between shrink-0">
              <div>
                <h2 className="text-[15px] font-semibold text-slate-800">Patient Details</h2>
                <p className="text-xs text-slate-400 mt-0.5">Clinical parameters & triage result</p>
              </div>
              <button
                onClick={onClose}
                className="w-8 h-8 rounded-lg hover:bg-slate-100 flex items-center justify-center transition-colors cursor-pointer"
              >
                <X className="w-4 h-4 text-slate-400" />
              </button>
            </div>

            {/* Body */}
            <div className="flex-1 overflow-y-auto">
              {/* ESI Visual */}
              <div className={`border-b border-slate-100 ${
                patient.esi <= 2 ? 'bg-critical-50/30' : patient.esi === 5 ? 'bg-clinical-50/30' : 'bg-slate-50/50'
              }`}>
                <ESIVisualIndicator esi={patient.esi} />
              </div>

              {/* Risk Status Badge */}
              <div className="px-6 py-4 border-b border-slate-100 flex items-center gap-3">
                <span className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold border ${risk.className}`}>
                  {patient.esi <= 2 && <AlertTriangle className="w-3 h-3" />}
                  {risk.label}
                </span>
                <span className="text-xs text-slate-400">
                  Predicted ESI Level: <strong className="text-slate-600">{patient.esi}</strong>
                </span>
              </div>

              {/* Patient ID */}
              <div className="px-6 py-4 border-b border-slate-100">
                <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-3">Identification</h3>
                <ParamRow label="Patient ID" value={patient.patient_id || patient.id} icon={Hash} iconColor="text-medical-400" />
              </div>

              {/* Demographics */}
              <div className="px-6 py-4 border-b border-slate-100">
                <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-3">Demographics</h3>
                <ParamRow label="Age" value={formatValue(patient.age, ' yrs')} icon={User} iconColor="text-medical-400" />
                <ParamRow label="Gender" value={formatValue(patient.gender)} icon={User} iconColor="text-medical-400" />
                <ParamRow label="Gender (Encoded)" value={formatValue(patient.gender_encoded)} icon={User} iconColor="text-medical-400" />
                <ParamRow label="Age Group" value={formatValue(patient.age_group)} />
              </div>

              {/* Vital Signs */}
              <div className="px-6 py-4 border-b border-slate-100">
                <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-3">Vital Signs</h3>
                <ParamRow label="Heart Rate" value={formatValue(patient.heart_rate, ' bpm')} icon={Heart} iconColor="text-critical-400" />
                <ParamRow label="BP Systolic" value={formatValue(patient.bp_systolic, ' mmHg')} icon={Gauge} iconColor="text-medical-400" />
                <ParamRow label="BP Diastolic" value={formatValue(patient.bp_diastolic, ' mmHg')} icon={Gauge} iconColor="text-medical-400" />
                <ParamRow label="SpO₂" value={formatValue(patient.spo2, '%')} icon={Wind} iconColor="text-clinical-500" />
                <ParamRow label="Temperature" value={formatValue(patient.temperature, '°C')} icon={Thermometer} iconColor="text-warning-500" />
                <ParamRow label="Respiratory Rate" value={formatValue(patient.respiratory_rate ?? patient.resp_rate, ' /min')} icon={Activity} iconColor="text-medical-400" />
              </div>

              {/* Clinical Scores */}
              <div className="px-6 py-4 border-b border-slate-100">
                <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-3">Clinical Scores</h3>
                <ParamRow label="Complaint" value={formatValue(patient.complaint)} />
                <ParamRow label="Complaint Encoded" value={formatValue(patient.complaint_encoded)} />
                <ParamRow label="SIRS Score" value={formatValue(patient.sirs_score)} />
                <ParamRow label="qSOFA Score" value={formatValue(patient.qsofa_score)} />
                <ParamRow label="Shock Index" value={formatValue(patient.shock_index)} />
                <ParamRow label="Symptom Duration" value={formatValue(patient.symptom_duration_hours, ' hrs')} />
              </div>

              <div className="px-6 py-4 border-b border-slate-100">
                <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-3">Model Output</h3>
                <ParamRow label="ESI Level" value={formatValue(patient.esi)} />
                <ParamRow label="Confidence" value={formatValue(patient.confidence)} />
                <ParamRow label="Model Version" value={formatValue(patient.model_version)} />
                <ParamRow label="Status" value={formatValue(patient.status)} />
              </div>

              {/* Clinical Flags */}
              <div className="px-6 py-4">
                <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-3">Clinical Flags</h3>
                <div className="grid grid-cols-2 gap-2 mb-3">
                  <FlagBadge label="Critical SpO₂" value={patient.critical_spo2} />
                  <FlagBadge label="Tachycardia" value={patient.tachycardia} />
                  <FlagBadge label="Hypotension" value={patient.hypotension} />
                  <FlagBadge label="High Fever" value={patient.high_fever} />
                  <FlagBadge label="Tachypnea" value={patient.tachypnea} />
                </div>
                <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-2">Symptom Flags</h4>
                <div className="grid grid-cols-2 gap-2">
                  {SYMPTOM_FLAGS.map(([label, key]) => (
                    <FlagBadge key={key} label={label} value={patient[key]} />
                  ))}
                </div>
              </div>
            </div>
          </Motion.div>
        </>
      )}
    </AnimatePresence>
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
    <div className={`rounded-lg border px-3 py-2 text-center ${
      isPositive ? 'border-critical-200 bg-critical-50' : 'border-clinical-200 bg-clinical-50'
    }`}>
      <p className={`text-[10px] font-semibold uppercase tracking-wider ${
        isPositive ? 'text-critical-500' : 'text-clinical-500'
      }`}>{label}</p>
      <p className={`text-xs font-bold mt-0.5 ${
        isPositive ? 'text-critical-700' : 'text-clinical-700'
      }`}>{isPositive ? 'YES' : 'NO'}</p>
    </div>
  )
}
