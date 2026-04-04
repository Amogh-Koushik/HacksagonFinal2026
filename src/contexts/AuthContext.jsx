import { createContext, useContext, useState, useCallback } from 'react'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser] = useState(() => {
    const stored = localStorage.getItem('hospital_user')
    return stored ? JSON.parse(stored) : null
  })

  const login = useCallback((username, password) => {
    if (username === 'nurse' && password === '123') {
      const userData = { username, role: 'nurse', name: 'Nurse Station' }
      setUser(userData)
      localStorage.setItem('hospital_user', JSON.stringify(userData))
      return { success: true }
    }
    return { success: false, error: 'Invalid credentials' }
  }, [])

  const logout = useCallback(() => {
    setUser(null)
    localStorage.removeItem('hospital_user')
  }, [])

  return (
    <AuthContext.Provider value={{ user, isAuthenticated: !!user, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}
