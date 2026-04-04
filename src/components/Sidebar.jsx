import { NavLink, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { useAuth } from '../contexts/AuthContext'
import { usePatients } from '../contexts/PatientContext'
import { LayoutDashboard, UserPlus, ClipboardList, LogOut, Activity, Shield } from 'lucide-react'

const navItems = [
  { path: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { path: '/add-patient', icon: UserPlus, label: 'Add Patient' },
  { path: '/queue', icon: ClipboardList, label: 'Patient Queue' },
]

export default function Sidebar() {
  const { logout, user } = useAuth()
  const { stats } = usePatients()
  const navigate = useNavigate()

  const handleLogout = () => {
    logout()
    navigate('/login', { replace: true })
  }

  return (
    <aside className="w-[264px] bg-slate-900 flex flex-col shrink-0 border-r border-slate-800">
      {/* Header */}
      <div className="p-6 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-medical-600 flex items-center justify-center">
            <Shield className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-white font-semibold text-sm tracking-tight">RiskScope AI</h1>
            <p className="text-slate-500 text-xs">Triage Assistant</p>
          </div>
        </div>
      </div>

      {/* System Status */}
      <div className="px-6 py-3.5 border-b border-slate-800 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-clinical-400 animate-pulse" />
          <span className="text-xs text-slate-400 font-medium">System Online</span>
        </div>
        {stats.criticalCount > 0 && (
          <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-critical-600/20 text-critical-400 text-xs font-semibold">
            <span className="w-1.5 h-1.5 rounded-full bg-critical-400 animate-pulse-critical" />
            {stats.criticalCount} Critical
          </span>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4 space-y-1">
        <p className="px-4 mb-2 text-[10px] font-semibold text-slate-600 uppercase tracking-widest">Navigation</p>
        {navItems.map(item => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.path === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-4 py-2.5 rounded-xl text-[13px] font-medium transition-all duration-200 ${
                isActive
                  ? 'bg-medical-600/15 text-medical-300'
                  : 'text-slate-400 hover:bg-slate-800/70 hover:text-slate-200'
              }`
            }
          >
            {({ isActive }) => (
              <>
                <item.icon className={`w-[18px] h-[18px] ${isActive ? 'text-medical-400' : ''}`} />
                <span>{item.label}</span>
                {item.path === '/queue' && stats.totalWaiting > 0 && (
                  <span className={`ml-auto text-[11px] font-bold px-2 py-0.5 rounded-full ${
                    isActive ? 'bg-medical-500/20 text-medical-300' : 'bg-slate-800 text-slate-400'
                  }`}>
                    {stats.totalWaiting}
                  </span>
                )}
                {isActive && (
                  <motion.div
                    layoutId="sidebar-active"
                    className="ml-auto w-1.5 h-1.5 rounded-full bg-medical-400"
                    transition={{ type: 'spring', stiffness: 350, damping: 30 }}
                    style={item.path === '/queue' ? { display: 'none' } : {}}
                  />
                )}
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* User & Logout */}
      <div className="p-4 border-t border-slate-800">
        <div className="px-4 py-2 mb-2">
          <p className="text-[10px] text-slate-600 uppercase tracking-wider font-semibold">Signed in as</p>
          <p className="text-sm text-slate-300 font-medium mt-0.5">{user?.name || 'Staff'}</p>
          <p className="text-[11px] text-slate-500 capitalize">{user?.role || 'Nurse'}</p>
        </div>
        <button
          onClick={handleLogout}
          className="w-full flex items-center gap-3 px-4 py-2.5 rounded-xl text-[13px] font-medium text-slate-500 hover:bg-critical-500/10 hover:text-critical-400 transition-all duration-200"
        >
          <LogOut className="w-[18px] h-[18px]" />
          Sign Out
        </button>
      </div>
    </aside>
  )
}
