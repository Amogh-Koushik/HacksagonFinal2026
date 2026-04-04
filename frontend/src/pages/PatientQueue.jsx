import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { usePatients } from '../contexts/PatientContext'
import { useToast } from '../contexts/ToastContext'
import { ESI_LABELS, ESI_COLORS } from '../utils/esiCalculator'
import { TableSkeleton } from '../components/LoadingSkeleton'
import {
  Clock, Heart, Wind, Gauge, AlertTriangle,
  CheckCircle2, XCircle, Eye, ChevronDown, ChevronUp,
  ClipboardList, Search, Filter, FileText
} from 'lucide-react'

const fadeUp = { hidden: { opacity: 0, y: 12 }, show: { opacity: 1, y: 0, transition: { duration: 0.3 } } }
const stagger = { hidden: { opacity: 0 }, show: { opacity: 1, transition: { staggerChildren: 0.05 } } }

function ESIBadge({ level }) {
  const colors = ESI_COLORS[level]
  return (
    <span className={`inline-flex items-center justify-center w-8 h-8 rounded-lg text-xs font-bold ${colors.bg} ${colors.text}`}>
      {level}
    </span>
  )
}

function StatusBadge({ status }) {
  const styles = {
    waiting: 'bg-warning-50 text-warning-700 border-warning-200',
    treating: 'bg-medical-50 text-medical-700 border-medical-200',
    discharged: 'bg-clinical-50 text-clinical-700 border-clinical-200',
  }
  const labels = { waiting: 'Waiting', treating: 'In Treatment', discharged: 'Discharged' }
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-semibold border ${styles[status]}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${
        status === 'waiting' ? 'bg-warning-500' : status === 'treating' ? 'bg-medical-500 animate-pulse' : 'bg-clinical-500'
      }`} />
      {labels[status]}
    </span>
  )
}

function formatVital(val, suffix = '') {
  if (val === null || val === undefined || val === '') return '—'
  return `${val}${suffix}`
}

function PatientRow({ patient, onStatusChange, onRemove, onSelect }) {
  const [expanded, setExpanded] = useState(false)
  const now = new Date()
  const waitMin = Math.round((now - new Date(patient.addedAt)) / 60000)
  const isCritical = patient.esi <= 2

  const handleRowClick = () => {
    onSelect(patient)
  }

  const handleExpandClick = (e) => {
    e.stopPropagation()
    setExpanded(!expanded)
  }

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, x: -20 }}
      transition={{ duration: 0.25 }}
      className={`border-b last:border-b-0 transition-colors ${
        isCritical ? 'bg-critical-50/40 border-critical-100' : 'border-slate-100 hover:bg-slate-50/50'
      }`}
    >
      {/* Main Row */}
      <div className="px-6 py-4 flex items-center gap-4 cursor-pointer" onClick={handleRowClick}>
        <ESIBadge level={patient.esi} />

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <p className={`text-sm font-semibold ${isCritical ? 'text-critical-900' : 'text-slate-800'}`}>
              {patient.patient_id || patient.name || `Patient ${patient.id}`}
            </p>
            {isCritical && (
              <span className="flex items-center gap-1 text-[10px] font-bold text-critical-600 bg-critical-100 px-1.5 py-0.5 rounded-md uppercase">
                <AlertTriangle className="w-3 h-3" /> Critical
              </span>
            )}
          </div>
          <p className="text-xs text-slate-500 mt-0.5 truncate">
            {patient.notes ? `📝 ${patient.notes}` : (patient.symptoms || `HR: ${formatVital(patient.heart_rate)} • SpO₂: ${formatVital(patient.spo2, '%')} • BP: ${formatVital(patient.bp_systolic)}/${formatVital(patient.bp_diastolic)}`)}
          </p>
        </div>

        <div className="hidden md:flex items-center gap-6 text-xs text-slate-500 shrink-0">
          <span className="flex items-center gap-1">
            <Heart className="w-3 h-3 text-critical-400" />
            {formatVital(patient.heart_rate || patient.heartRate)}
          </span>
          <span className="flex items-center gap-1">
            <Wind className="w-3 h-3 text-clinical-500" />
            {formatVital(patient.spo2 || patient.oxygenLevel, '%')}
          </span>
          <span className="flex items-center gap-1">
            <Gauge className="w-3 h-3 text-medical-400" />
            {patient.bloodPressure || `${formatVital(patient.bp_systolic)}/${formatVital(patient.bp_diastolic)}`}
          </span>
        </div>

        <div className="flex items-center gap-1.5 text-xs text-slate-400 shrink-0 w-16 justify-end">
          <Clock className="w-3 h-3" />{waitMin}m
        </div>

        <StatusBadge status={patient.status} />

        <div className="w-5 text-slate-400" onClick={handleExpandClick}>
          {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </div>
      </div>

      {/* Expanded Detail */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className={`px-6 pb-5 pt-1 ${isCritical ? 'bg-critical-50/20' : 'bg-slate-50/30'}`}>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                <DetailItem label="ESI Level" value={`${patient.esi} — ${ESI_LABELS[patient.esi].label}`} />
                <DetailItem label="Age" value={patient.age != null ? `${patient.age} years` : 'N/A'} />
                <DetailItem label="SIRS Score" value={patient.sirs_score != null ? patient.sirs_score : (patient.painScale != null ? `${patient.painScale}/10` : 'N/A')} />
                <DetailItem label="Wait Time" value={`${waitMin} minutes`} />
              </div>

              <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-4">
                <VitalCard icon={Heart} label="Heart Rate" value={`${formatVital(patient.heart_rate || patient.heartRate)} bpm`}
                  status={(() => {
                    const hr = patient.heart_rate || patient.heartRate
                    if (hr == null) return 'normal'
                    return hr > 120 || hr < 50 ? 'danger' : hr > 100 ? 'warning' : 'normal'
                  })()} />
                <VitalCard icon={Wind} label="SpO₂" value={`${formatVital(patient.spo2 || patient.oxygenLevel)}%`}
                  status={(() => {
                    const sp = patient.spo2 || patient.oxygenLevel
                    if (sp == null) return 'normal'
                    return sp < 90 ? 'danger' : sp < 95 ? 'warning' : 'normal'
                  })()} />
                <VitalCard icon={Gauge} label="Blood Pressure"
                  value={patient.bloodPressure || `${formatVital(patient.bp_systolic)}/${formatVital(patient.bp_diastolic)}`}
                  status={(() => {
                    const s = patient.bp_systolic || (patient.bloodPressure ? Number(patient.bloodPressure.split('/')[0]) : null)
                    if (s == null) return 'normal'
                    return s > 160 || s < 90 ? 'danger' : s > 140 ? 'warning' : 'normal'
                  })()} />
              </div>

              {patient.notes && (
                <div className="bg-white rounded-lg border border-slate-200 p-3 mb-4">
                  <p className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-1">Notes</p>
                  <p className="text-sm text-slate-700">{patient.notes}</p>
                </div>
              )}

              <div className="flex items-center gap-2 pt-2">
                <button onClick={() => onSelect(patient)}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-600 text-white text-xs font-medium hover:bg-slate-700 transition-colors cursor-pointer">
                  <Eye className="w-3 h-3" /> View Full Details
                </button>
                {patient.status === 'waiting' && (
                  <button onClick={() => onStatusChange(patient.id, 'treating')}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-medical-600 text-white text-xs font-medium hover:bg-medical-700 transition-colors cursor-pointer">
                    <Eye className="w-3 h-3" /> Begin Treatment
                  </button>
                )}
                {patient.status === 'treating' && (
                  <button onClick={() => onStatusChange(patient.id, 'discharged')}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-clinical-600 text-white text-xs font-medium hover:bg-clinical-700 transition-colors cursor-pointer">
                    <CheckCircle2 className="w-3 h-3" /> Discharge
                  </button>
                )}
                <button onClick={() => onRemove(patient.id)}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-200 text-slate-500 text-xs font-medium hover:bg-critical-50 hover:text-critical-600 hover:border-critical-200 transition-colors cursor-pointer">
                  <XCircle className="w-3 h-3" /> Remove
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

function DetailItem({ label, value }) {
  return (
    <div>
      <p className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">{label}</p>
      <p className="text-sm font-medium text-slate-700 mt-0.5">{value}</p>
    </div>
  )
}

function VitalCard({ icon: Icon, label, value, status }) {
  const cls = {
    normal: 'bg-clinical-50 border-clinical-200 text-clinical-700',
    warning: 'bg-warning-50 border-warning-200 text-warning-700',
    danger: 'bg-critical-50 border-critical-200 text-critical-700',
  }
  return (
    <div className={`rounded-lg border p-3 flex items-center gap-3 ${cls[status]}`}>
      <Icon className="w-4 h-4 shrink-0" />
      <div>
        <p className="text-[10px] font-semibold uppercase tracking-wider opacity-70">{label}</p>
        <p className="text-sm font-bold">{value}</p>
      </div>
    </div>
  )
}

export default function PatientQueue() {
  const navigate = useNavigate()
  const { patients, updateStatus, removePatient } = usePatients()
  const { showToast } = useToast()
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')
  const [filterESI, setFilterESI] = useState(0)

  useEffect(() => { const t = setTimeout(() => setLoading(false), 500); return () => clearTimeout(t) }, [])

  const handleStatusChange = (id, status) => {
    updateStatus(id, status)
    const p = patients.find(p => p.id === id)
    const displayName = p?.patient_id || p?.name || 'Patient'
    showToast(
      `${displayName} → ${status === 'treating' ? 'In Treatment' : 'Discharged and removed'}`,
      'info'
    )
  }

  const handleRemove = (id) => {
    const p = patients.find(p => p.id === id)
    const displayName = p?.patient_id || p?.name || 'Patient'
    removePatient(id)
    showToast(`${displayName} removed from queue`, 'warning')
  }

  const handleSelectPatient = (patient) => {
    navigate(`/patient/${patient.id}`)
  }

  const filtered = patients.filter(p => {
    const searchTerm = search.toLowerCase()
    const matchesSearch = !search || 
      (p.name && p.name.toLowerCase().includes(searchTerm)) ||
      (p.patient_id && p.patient_id.toLowerCase().includes(searchTerm)) ||
      (p.symptoms && p.symptoms.toLowerCase().includes(searchTerm))
    if (!matchesSearch) return false
    if (filterESI > 0 && p.esi !== filterESI) return false
    return true
  })

  if (loading) return <div><div className="mb-8"><h1 className="text-2xl font-bold text-slate-800">Patient Queue</h1></div><TableSkeleton rows={6} /></div>

  return (
    <motion.div variants={stagger} initial="hidden" animate="show">
      <motion.div variants={fadeUp} className="mb-6 flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-800 tracking-tight">Patient Queue</h1>
          <p className="text-slate-500 text-sm mt-1">Sorted by ESI priority — most critical first</p>
        </div>
        <div className="flex items-center gap-3">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
            <input type="text" value={search} onChange={e => setSearch(e.target.value)}
              placeholder="Search patients..."
              className="pl-9 pr-4 py-2 rounded-xl border border-slate-200 bg-white text-sm text-slate-700 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-medical-500/20 focus:border-medical-400 transition-all w-52" />
          </div>
          <div className="relative">
            <Filter className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-400" />
            <select value={filterESI} onChange={e => setFilterESI(Number(e.target.value))}
              className="pl-8 pr-8 py-2 rounded-xl border border-slate-200 bg-white text-sm text-slate-700 focus:outline-none focus:ring-2 focus:ring-medical-500/20 focus:border-medical-400 transition-all appearance-none cursor-pointer">
              <option value={0}>All ESI</option>
              {[1,2,3,4,5].map(l => <option key={l} value={l}>ESI {l} — {ESI_LABELS[l].label}</option>)}
            </select>
          </div>
        </div>
      </motion.div>

      {/* Queue Summary Bar */}
      <motion.div variants={fadeUp} className="mb-4 flex items-center gap-3 flex-wrap">
        {[1,2,3,4,5].map(level => {
          const count = patients.filter(p => p.esi === level).length
          if (count === 0) return null
          const colors = ESI_COLORS[level]
          return (
            <button key={level} onClick={() => setFilterESI(filterESI === level ? 0 : level)}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs font-semibold transition-all cursor-pointer ${
                filterESI === level ? `${colors.light} ${colors.border} ${colors.textColor}` : 'bg-white border-slate-200 text-slate-500 hover:bg-slate-50'
              }`}>
              <span className={`w-2 h-2 rounded-full ${colors.bg}`} />
              ESI {level}: {count}
            </button>
          )
        })}
        {filterESI > 0 && (
          <button onClick={() => setFilterESI(0)} className="text-xs text-medical-600 hover:underline ml-1 cursor-pointer">Clear filter</button>
        )}
      </motion.div>

      {/* Queue Table */}
      <motion.div variants={fadeUp} className="bg-white rounded-2xl border border-slate-200/60 shadow-sm overflow-hidden">
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <ClipboardList className="w-5 h-5 text-medical-500" />
            <h2 className="text-[15px] font-semibold text-slate-800">Active Queue</h2>
            <span className="text-xs text-slate-400 font-medium">{filtered.length} patient{filtered.length !== 1 ? 's' : ''}</span>
          </div>
          <div className="flex items-center gap-2 text-[10px] font-semibold text-slate-400 uppercase tracking-widest">
            <span className="w-2 h-2 rounded-full bg-clinical-400 animate-pulse" />
            Live
          </div>
        </div>

        {/* Column Headers */}
        <div className="px-6 py-2.5 border-b border-slate-100 bg-slate-50/50 flex items-center gap-4 text-[10px] font-semibold text-slate-400 uppercase tracking-widest">
          <span className="w-8">ESI</span>
          <span className="flex-1">Patient</span>
          <span className="hidden md:block w-40">Vitals</span>
          <span className="w-16 text-right">Wait</span>
          <span className="w-28">Status</span>
          <span className="w-5" />
        </div>

        {/* Rows */}
        {filtered.length === 0 ? (
          <div className="px-6 py-16 text-center">
            <p className="text-slate-400 text-sm">No patients match your criteria</p>
          </div>
        ) : (
          <AnimatePresence>
            {filtered.map(patient => (
              <PatientRow
                key={patient.id}
                patient={patient}
                onStatusChange={handleStatusChange}
                onRemove={handleRemove}
                onSelect={handleSelectPatient}
              />
            ))}
          </AnimatePresence>
        )}
      </motion.div>
    </motion.div>
  )
}
