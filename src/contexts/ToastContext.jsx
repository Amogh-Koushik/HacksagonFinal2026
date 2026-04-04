import { createContext, useContext, useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { CheckCircle2, AlertTriangle, X, Info } from 'lucide-react'

const ToastContext = createContext(null)

let toastId = 0

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([])

  const showToast = useCallback((message, type = 'success') => {
    const id = ++toastId
    setToasts(prev => [...prev, { id, message, type }])
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id))
    }, 4000)
  }, [])

  const removeToast = useCallback((id) => {
    setToasts(prev => prev.filter(t => t.id !== id))
  }, [])

  return (
    <ToastContext.Provider value={{ showToast }}>
      {children}
      <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-3 max-w-sm">
        <AnimatePresence>
          {toasts.map(toast => (
            <motion.div
              key={toast.id}
              initial={{ opacity: 0, y: 20, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 10, scale: 0.95 }}
              transition={{ duration: 0.25 }}
              className={`flex items-start gap-3 px-4 py-3.5 rounded-xl shadow-lg border backdrop-blur-sm ${
                toast.type === 'success' ? 'bg-clinical-50/95 border-clinical-200 text-clinical-800' :
                toast.type === 'critical' ? 'bg-critical-50/95 border-critical-200 text-critical-800' :
                toast.type === 'warning' ? 'bg-warning-50/95 border-warning-200 text-warning-800' :
                'bg-medical-50/95 border-medical-200 text-medical-800'
              }`}
            >
              {toast.type === 'success' && <CheckCircle2 className="w-5 h-5 text-clinical-500 shrink-0 mt-0.5" />}
              {toast.type === 'critical' && <AlertTriangle className="w-5 h-5 text-critical-500 shrink-0 mt-0.5" />}
              {toast.type === 'warning' && <AlertTriangle className="w-5 h-5 text-warning-500 shrink-0 mt-0.5" />}
              {toast.type === 'info' && <Info className="w-5 h-5 text-medical-500 shrink-0 mt-0.5" />}
              <p className="text-sm font-medium flex-1">{toast.message}</p>
              <button onClick={() => removeToast(toast.id)} className="shrink-0 mt-0.5 opacity-50 hover:opacity-100 transition-opacity">
                <X className="w-4 h-4" />
              </button>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </ToastContext.Provider>
  )
}

export const useToast = () => {
  const ctx = useContext(ToastContext)
  if (!ctx) throw new Error('useToast must be used within ToastProvider')
  return ctx
}
