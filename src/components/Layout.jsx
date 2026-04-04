import { Outlet } from 'react-router-dom'
import Sidebar from './Sidebar'

export default function Layout() {
  return (
    <div className="flex h-screen bg-slate-50/80">
      <Sidebar />
      <main className="flex-1 overflow-auto">
        <div className="max-w-[1400px] mx-auto p-8">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
