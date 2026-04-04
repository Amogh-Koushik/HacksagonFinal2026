import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { AuthProvider } from './contexts/AuthContext'
import { PatientProvider } from './contexts/PatientContext'
import { ToastProvider } from './contexts/ToastContext'
import App from './App'
import './index.css'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
      <AuthProvider>
        <PatientProvider>
          <ToastProvider>
            <App />
          </ToastProvider>
        </PatientProvider>
      </AuthProvider>
    </BrowserRouter>
  </StrictMode>
)
