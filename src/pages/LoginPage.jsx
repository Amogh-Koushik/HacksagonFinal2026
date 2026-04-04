import { useState, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { useAuth } from '../contexts/AuthContext'
import { Shield, Lock, User, AlertCircle } from 'lucide-react'

export default function LoginPage() {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const { login, isAuthenticated } = useAuth()
  const navigate = useNavigate()
  const usernameRef = useRef(null)

  useEffect(() => {
    if (isAuthenticated) navigate('/', { replace: true })
  }, [isAuthenticated, navigate])

  useEffect(() => { usernameRef.current?.focus() }, [])

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setIsLoading(true)
    await new Promise(r => setTimeout(r, 800))
    const result = login(username, password)
    if (result.success) {
      navigate('/', { replace: true })
    } else {
      setError('Invalid credentials. Access denied.')
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50/30 to-slate-100 flex items-center justify-center p-4">
      <div className="absolute inset-0 opacity-[0.02]" style={{
        backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23000' fill-opacity='1'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`
      }} />

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-[420px] relative"
      >
        <div className="bg-slate-800 rounded-t-2xl px-6 py-3 flex items-center gap-3">
          <div className="w-2 h-2 rounded-full bg-clinical-400 animate-pulse" />
          <span className="text-[11px] text-slate-400 font-medium tracking-wider uppercase">
            RiskScope AI — Hospital Triage System
          </span>
        </div>

        <div className="bg-white rounded-b-2xl shadow-xl shadow-slate-200/50 border border-slate-200/60 border-t-0 p-8">
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-medical-50 border border-medical-100 mb-4">
              <Shield className="w-8 h-8 text-medical-600" />
            </div>
            <h1 className="text-xl font-semibold text-slate-800">Staff Authentication</h1>
            <p className="text-sm text-slate-500 mt-1">Authorized personnel only</p>
          </div>

          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-6 flex items-center gap-3 px-4 py-3 rounded-xl bg-critical-50 border border-critical-200 text-critical-700 text-sm"
            >
              <AlertCircle className="w-4 h-4 shrink-0" />
              {error}
            </motion.div>
          )}

          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1.5">Username</label>
              <div className="relative">
                <User className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
                <input
                  ref={usernameRef}
                  type="text"
                  value={username}
                  onChange={e => setUsername(e.target.value)}
                  placeholder="Enter username"
                  className="w-full pl-11 pr-4 py-3 rounded-xl border border-slate-200 bg-slate-50/50 text-slate-800 placeholder-slate-400 text-sm focus:outline-none focus:ring-2 focus:ring-medical-500/20 focus:border-medical-400 transition-all"
                  required
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1.5">Password</label>
              <div className="relative">
                <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
                <input
                  type="password"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  placeholder="Enter password"
                  className="w-full pl-11 pr-4 py-3 rounded-xl border border-slate-200 bg-slate-50/50 text-slate-800 placeholder-slate-400 text-sm focus:outline-none focus:ring-2 focus:ring-medical-500/20 focus:border-medical-400 transition-all"
                  required
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="w-full py-3 rounded-xl bg-medical-600 hover:bg-medical-700 active:bg-medical-800 text-white font-medium text-sm transition-all disabled:opacity-60 disabled:cursor-not-allowed flex items-center justify-center gap-2 cursor-pointer"
            >
              {isLoading ? (
                <>
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                    className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full"
                  />
                  Authenticating...
                </>
              ) : 'Sign In'}
            </button>
          </form>

          <div className="mt-6 pt-5 border-t border-slate-100">
            <p className="text-[11px] text-slate-400 text-center leading-relaxed">
              This system is restricted to authorized hospital staff.<br />
              Unauthorized access is prohibited and monitored.
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  )
}
