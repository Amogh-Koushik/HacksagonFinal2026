import { useState } from 'react'
import { Outlet } from 'react-router-dom'
import { Menu, Shield } from 'lucide-react'
import Sidebar from './Sidebar'

export default function Layout() {
  const [mobileNavOpen, setMobileNavOpen] = useState(false)

  return (
    <div className="flex min-h-screen bg-slate-50/80">
      <Sidebar mobileOpen={mobileNavOpen} setMobileOpen={setMobileNavOpen} />
      <main className="flex-1 overflow-auto">
        <div className="lg:hidden sticky top-0 z-30 bg-white/95 backdrop-blur border-b border-slate-200 px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-medical-600 flex items-center justify-center">
              <Shield className="w-4 h-4 text-white" />
            </div>
            <div>
              <p className="text-sm font-semibold text-slate-800 leading-tight">RiskScope AI</p>
              <p className="text-[11px] text-slate-500 leading-tight">Triage Assistant</p>
            </div>
          </div>
          <button
            type="button"
            onClick={() => setMobileNavOpen(true)}
            className="w-9 h-9 rounded-lg border border-slate-200 bg-white text-slate-600 hover:bg-slate-50 flex items-center justify-center"
            aria-label="Open navigation menu"
          >
            <Menu className="w-4.5 h-4.5" />
          </button>
        </div>

        <div className="max-w-[1400px] mx-auto p-4 sm:p-6 lg:p-8">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
