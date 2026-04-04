import { Routes, Route, Navigate } from 'react-router-dom'
import { useAuth } from './contexts/AuthContext'
import LoginPage from './pages/LoginPage'
import Layout from './components/Layout'
import DashboardOverview from './pages/DashboardOverview'
import AddPatient from './pages/AddPatient'
import PatientQueue from './pages/PatientQueue'
import PatientDetailsPage from './pages/PatientDetailsPage'

function ProtectedRoute({ children }) {
  const { isAuthenticated } = useAuth()
  if (!isAuthenticated) return <Navigate to="/login" replace />
  return children
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/" element={<ProtectedRoute><Layout /></ProtectedRoute>}>
        <Route index element={<DashboardOverview />} />
        <Route path="add-patient" element={<AddPatient />} />
        <Route path="queue" element={<PatientQueue />} />
      </Route>
      <Route path="/patient/:id" element={<ProtectedRoute><PatientDetailsPage /></ProtectedRoute>} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
