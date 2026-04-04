import { motion } from 'framer-motion'
import { useNavigate, useParams } from 'react-router-dom'
import { usePatients } from '../contexts/PatientContext'
import { ESI_LABELS, ESI_COLORS } from '../utils/esiCalculator'
import { ArrowLeft, AlertTriangle, Heart, Wind, Gauge, Thermometer, Activity, Shield, User, Hash, FileText } from 'lucide-react'

const fadeUp = { hidden: { opacity: 0, y: 12 }, show: { opacity: 1, y: 0, transition: { duration: 0.3 } } }
const stagger = { hidden: { opacity: 0 }, show: { opacity: 1, transition: { staggerChildren: 0.05 } } }

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

function ESIVisualIndicator({ esi }) {
  if (esi <= 2) {
    return (
      <motion.div
        className="relative rounded-2xl overflow-hidden"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        {/* Red pulse background */}
        <motion.div
          className="absolute inset-0 bg-critical-500/10 rounded-2xl"
          animate={{ opacity: [0.3, 0.8, 0.3] }}
          transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
        />
        <div className="relative px-6 py-5 flex items-center gap-4">
          <motion.div
            className="w-14 h-14 rounded-xl bg-critical-600 flex items-center justify-center shadow-lg shadow-critical-500/30"
            animate={{ scale: [1, 1.08, 1] }}
            transition={{ duration: 1.2, repeat: Infinity, ease: 'easeInOut' }}
          >
            <span className="text-white text-xl font-black">{esi}</span>
          </motion.div>
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <h3 className="text-lg font-bold text-critical-800">{ESI_LABELS[esi].label}</h3>
              <motion.div
                animate={{ opacity: [1, 0.3, 1] }}
                transition={{ duration: 0.8, repeat: Infinity }}
              >
                <AlertTriangle className="w-5 h-5 text-critical-500" />
              </motion.div>
            </div>
            <p className="text-sm text-critical-600 mt-0.5">{ESI_LABELS[esi].description}</p>
          </div>
        </div>
      </motion.div>
    )
  }

  if (esi === 5) {
    return (
      <motion.div
        className="relative rounded-2xl overflow-hidden"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        {/* Gentle green breathing glow */}
        <motion.div
          className="absolute inset-0 rounded-2xl"
          style={{
            background: 'radial-gradient(ellipse at center, rgba(23, 180, 99, 0.1), transparent 70%)',
          }}
          animate={{ opacity: [0.4, 1, 0.4] }}
          transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
        />
        <div className="relative px-6 py-5 flex items-center gap-4">
          <motion.div
            className="w-14 h-14 rounded-xl bg-clinical-500 flex items-center justify-center shadow-lg shadow-clinical-400/20"
            animate={{ scale: [1, 1.03, 1] }}
            transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
          >
            <span className="text-white text-xl font-black">{esi}</span>
          </motion.div>
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <h3 className="text-lg font-bold text-clinical-700">{ESI_LABELS[esi].label}</h3>
              <motion.div
                animate={{ opacity: [0.5, 1, 0.5] }}
                transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
              >
                <Shield className="w-4 h-4 text-clinical-400" />
              </motion.div>
            </div>
            <p className="text-sm text-clinical-600 mt-0.5">{ESI_LABELS[esi].description}</p>
          </div>
        </div>
      </motion.div>
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

export default function PatientDetailsPage() {
  const navigate = useNavigate()
  const { id } = useParams()
  const { patients } = usePatients()

  const patient = patients.find(p => p.id === id)

  if (!patient) {
    return (
      <motion.div variants={fadeUp} initial="hidden" animate="show" className="p-8 text-center">
        <h1 className="text-xl font-bold text-slate-800">Patient Not Found</h1>
        <p className="text-slate-500 mt-2">The patient you're looking for doesn't exist.</p>
        <button
          onClick={() => navigate('/queue')}
          className="mt-4 px-4 py-2 rounded-lg bg-medical-600 text-white font-medium hover:bg-medical-700 transition-colors"
        >
          Back to Queue
        </button>
      </motion.div>
    )
  }

  const risk = getRiskStatus(patient.esi)
  const isCritical = patient.esi <= 2

  return (
    <motion.div variants={stagger} initial="hidden" animate="show" className="min-h-screen bg-slate-50/50">
      {/* Header with back button */}
      <div className="bg-white border-b border-slate-200 sticky top-0 z-40">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center gap-4">
          <button
            onClick={() => navigate('/queue')}
            className="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-slate-100 transition-colors text-slate-600 font-medium"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Queue
          </button>
          <div>
            <h1 className="text-lg font-bold text-slate-800">{patient.patient_id || `Patient ${patient.id}`}</h1>
            <p className="text-xs text-slate-500">Full Clinical Details</p>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="max-w-6xl mx-auto px-6 py-8 space-y-6">
        {/* ESI Section */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className={`rounded-2xl border overflow-hidden ${
            isCritical ? 'border-critical-100 bg-critical-50/30' : 'border-slate-200 bg-white'
          }`}
        >
          <ESIVisualIndicator esi={patient.esi} />
        </motion.div>

        {/* Risk Status */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
          className="flex items-center gap-4"
        >
          <span className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold border ${risk.className}`}>
            {patient.esi <= 2 && <AlertTriangle className="w-3 h-3" />}
            {risk.label}
          </span>
          <span className="text-sm text-slate-600">
            Predicted ESI Level: <strong>{patient.esi}</strong>
          </span>
        </motion.div>

        {/* Two column layout for details */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left column */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="space-y-6"
          >
            {/* Patient ID */}
            <div className="bg-white rounded-2xl border border-slate-200 p-6">
              <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-4 flex items-center gap-2">
                <Hash className="w-4 h-4" /> Identification
              </h3>
              <ParamRow label="Patient ID" value={patient.patient_id || patient.id} icon={Hash} iconColor="text-medical-400" />
            </div>

            {/* Demographics */}
            <div className="bg-white rounded-2xl border border-slate-200 p-6">
              <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-4 flex items-center gap-2">
                <User className="w-4 h-4" /> Demographics
              </h3>
              <div className="space-y-2.5">
                <ParamRow label="Age" value={formatValue(patient.age, ' yrs')} icon={User} iconColor="text-medical-400" />
                <ParamRow label="Gender (Encoded)" value={formatValue(patient.gender_encoded)} icon={User} iconColor="text-medical-400" />
                <ParamRow label="Age Group" value={formatValue(patient.age_group)} />
              </div>
            </div>

            {/* Vital Signs */}
            <div className="bg-white rounded-2xl border border-slate-200 p-6">
              <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-4 flex items-center gap-2">
                <Activity className="w-4 h-4" /> Vital Signs
              </h3>
              <div className="space-y-2.5">
                <ParamRow label="Heart Rate" value={formatValue(patient.heart_rate, ' bpm')} icon={Heart} iconColor="text-critical-400" />
                <ParamRow label="BP Systolic" value={formatValue(patient.bp_systolic, ' mmHg')} icon={Gauge} iconColor="text-medical-400" />
                <ParamRow label="BP Diastolic" value={formatValue(patient.bp_diastolic, ' mmHg')} icon={Gauge} iconColor="text-medical-400" />
                <ParamRow label="SpO₂" value={formatValue(patient.spo2, '%')} icon={Wind} iconColor="text-clinical-500" />
                <ParamRow label="Temperature" value={formatValue(patient.temperature, '°C')} icon={Thermometer} iconColor="text-warning-500" />
                <ParamRow label="Resp Rate" value={formatValue(patient.resp_rate, ' /min')} icon={Activity} iconColor="text-medical-400" />
              </div>
            </div>
          </motion.div>

          {/* Right column */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.25 }}
            className="space-y-6"
          >
            {/* Clinical Scores */}
            <div className="bg-white rounded-2xl border border-slate-200 p-6">
              <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-4">Clinical Scores</h3>
              <div className="space-y-2.5">
                <ParamRow label="Complaint" value={formatValue(patient.complaint || patient.complaint_encoded)} />
                <ParamRow label="SIRS Score" value={formatValue(patient.sirs_score)} />
                <ParamRow label="qSOFA Score" value={formatValue(patient.qsofa_score)} />
                <ParamRow label="Shock Index" value={formatValue(patient.shock_index)} />
              </div>
            </div>

            {/* Clinical Flags */}
            <div className="bg-white rounded-2xl border border-slate-200 p-6">
              <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-4">Clinical Flags</h3>
              <div className="grid grid-cols-2 gap-2">
                <FlagBadge label="Critical SpO₂" value={patient.critical_spo2} />
                <FlagBadge label="Tachycardia" value={patient.tachycardia} />
                <FlagBadge label="Hypotension" value={patient.hypotension} />
                <FlagBadge label="High Fever" value={patient.high_fever} />
                <FlagBadge label="Tachypnea" value={patient.tachypnea} />
              </div>
            </div>

            {/* Notes */}
            {patient.notes && (
              <div className="bg-white rounded-2xl border border-slate-200 p-6">
                <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-4 flex items-center gap-2">
                  <FileText className="w-4 h-4" /> Notes
                </h3>
                <p className="text-sm text-slate-700 leading-relaxed">{patient.notes}</p>
              </div>
            )}
          </motion.div>
        </div>
      </div>
    </motion.div>
  )
}
