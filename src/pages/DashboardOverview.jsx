import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { usePatients } from '../contexts/PatientContext'
import { ESI_LABELS } from '../utils/esiCalculator'
import { PageSkeleton } from '../components/LoadingSkeleton'
import {
  Users, AlertTriangle, TrendingUp, HeartPulse,
  Stethoscope, ArrowRight, Clock
} from 'lucide-react'

const stagger = { hidden: { opacity: 0 }, show: { opacity: 1, transition: { staggerChildren: 0.07 } } }
const fadeUp = { hidden: { opacity: 0, y: 14 }, show: { opacity: 1, y: 0, transition: { duration: 0.35 } } }

function StatCard({ icon: Icon, label, value, sub, color }) {
  const cls = {
    blue: 'bg-medical-50 text-medical-600 border-medical-100',
    red: 'bg-critical-50 text-critical-600 border-critical-100',
    green: 'bg-clinical-50 text-clinical-600 border-clinical-100',
    amber: 'bg-warning-50 text-warning-600 border-warning-100',
  }
  return (
    <motion.div variants={fadeUp} className="bg-white rounded-2xl border border-slate-200/60 p-6 shadow-sm hover:shadow-md transition-shadow duration-300">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-[13px] text-slate-500 font-medium">{label}</p>
          <p className="text-3xl font-bold text-slate-800 mt-1.5 tracking-tight">{value}</p>
          {sub && <p className="text-[11px] text-slate-400 mt-1">{sub}</p>}
        </div>
        <div className={`w-11 h-11 rounded-xl border flex items-center justify-center ${cls[color]}`}>
          <Icon className="w-5 h-5" />
        </div>
      </div>
    </motion.div>
  )
}

export default function DashboardOverview() {
  const { stats, patients } = usePatients()
  const navigate = useNavigate()
  const [loading, setLoading] = useState(true)

  useEffect(() => { const t = setTimeout(() => setLoading(false), 600); return () => clearTimeout(t) }, [])

  if (loading) return <PageSkeleton />

  const criticalPatients = patients.filter(p => p.esi <= 2)
  const now = new Date()

  return (
    <motion.div variants={stagger} initial="hidden" animate="show">
      <motion.div variants={fadeUp} className="mb-8">
        <h1 className="text-2xl font-bold text-slate-800 tracking-tight">Dashboard Overview</h1>
        <p className="text-slate-500 text-sm mt-1">AI-assisted emergency triage monitoring</p>
      </motion.div>

      {/* Info Banner */}
      <motion.div variants={fadeUp} className="mb-6 bg-medical-50/60 border border-medical-100 rounded-2xl p-5 flex items-start gap-4">
        <div className="w-10 h-10 rounded-xl bg-medical-100 flex items-center justify-center shrink-0">
          <Stethoscope className="w-5 h-5 text-medical-600" />
        </div>
        <div>
          <h3 className="text-sm font-semibold text-medical-800">AI Triage Assistant</h3>
          <p className="text-[13px] text-medical-700/70 mt-0.5 leading-relaxed">
            This assistant helps nurses and hospital staff triage patients by predicting
            Emergency Severity Index (ESI) risk levels. Patients are automatically prioritized
            so critically ill patients are treated first.
          </p>
        </div>
      </motion.div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4 mb-8">
        <StatCard icon={Users} label="Patients Waiting" value={stats.totalWaiting} sub="In queue" color="blue" />
        <StatCard icon={AlertTriangle} label="Critical Patients" value={stats.criticalCount} sub="ESI 1–2 priority" color="red" />
        <StatCard icon={TrendingUp} label="Avg. Risk Level" value={`ESI ${stats.avgRiskLevel}`} sub="Lower = more critical" color="amber" />
        <StatCard icon={HeartPulse} label="Queue Status"
          value={stats.criticalCount > 0 ? 'Alert' : 'Stable'}
          sub={stats.criticalCount > 0 ? 'Critical patients present' : 'No critical cases'}
          color={stats.criticalCount > 0 ? 'red' : 'green'} />
      </div>

      {/* Critical Alert */}
      {criticalPatients.length > 0 && (
        <motion.div variants={fadeUp} className="mb-8">
          <div className="bg-critical-50 border border-critical-200 rounded-2xl overflow-hidden">
            <div className="px-6 py-4 border-b border-critical-100 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-2 h-2 rounded-full bg-critical-500 animate-pulse-critical" />
                <h3 className="text-sm font-semibold text-critical-800">Critical — Immediate Attention Required</h3>
              </div>
              <button onClick={() => navigate('/queue')}
                className="text-xs font-medium text-critical-600 hover:text-critical-700 flex items-center gap-1 transition-colors cursor-pointer">
                View Queue <ArrowRight className="w-3 h-3" />
              </button>
            </div>
            <div className="divide-y divide-critical-100">
              {criticalPatients.map(p => {
                const waitMin = Math.round((now - new Date(p.addedAt)) / 60000)
                return (
                  <div key={p.id} className="px-6 py-3.5 flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <span className={`inline-flex items-center justify-center w-8 h-8 rounded-lg text-xs font-bold text-white ${p.esi === 1 ? 'bg-critical-600' : 'bg-orange-500'}`}>
                        {p.esi}
                      </span>
                      <div>
                        <p className="text-sm font-medium text-critical-900">{p.patient_id || p.name || `Patient ${p.id}`}</p>
                        <p className="text-xs text-critical-600">{ESI_LABELS[p.esi].label}{p.symptoms ? ` — ${p.symptoms.split(',')[0]}` : ''}</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-1.5 text-xs text-critical-500">
                      <Clock className="w-3 h-3" />{waitMin}m
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </motion.div>
      )}

      {/* ESI Reference */}
      <motion.div variants={fadeUp}>
        <h3 className="text-sm font-semibold text-slate-700 mb-3">ESI Reference Guide</h3>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          {[1,2,3,4,5].map(level => {
            const info = ESI_LABELS[level]
            const count = patients.filter(p => p.esi === level).length
            const bg = { 1:'bg-critical-50 border-critical-200', 2:'bg-orange-50 border-orange-200', 3:'bg-warning-50 border-warning-200', 4:'bg-medical-50 border-medical-200', 5:'bg-clinical-50 border-clinical-200' }
            const tx = { 1:'text-critical-700', 2:'text-orange-700', 3:'text-warning-700', 4:'text-medical-700', 5:'text-clinical-700' }
            return (
              <div key={level} className={`${bg[level]} border rounded-xl p-4`}>
                <div className="flex items-center justify-between mb-1.5">
                  <span className={`text-lg font-bold ${tx[level]}`}>ESI {level}</span>
                  <span className={`text-sm font-bold ${tx[level]}`}>{count}</span>
                </div>
                <p className={`text-xs font-semibold ${tx[level]}`}>{info.label}</p>
                <p className={`text-[11px] ${tx[level]} opacity-60 mt-0.5`}>{info.description}</p>
              </div>
            )
          })}
        </div>
      </motion.div>
    </motion.div>
  )
}
