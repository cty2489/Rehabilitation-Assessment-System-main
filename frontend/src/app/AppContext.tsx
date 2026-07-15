import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  ReactNode,
} from 'react'
import { fetchAuthSession, loginUser, logoutUser } from '../api'
import { Route } from '../types'

// --------------------------------------------------------------------------- //
// Auth. The backend stores a short-lived signed session in an HttpOnly cookie. //
// --------------------------------------------------------------------------- //
interface AuthValue {
  user: string | null
  ready: boolean
  login: (username: string, password: string) => Promise<void>
  logout: () => void
}

const AuthContext = createContext<AuthValue | null>(null)

// --------------------------------------------------------------------------- //
// Routing (no react-router; a small route enum drives the AppShell).          //
// `selectedPatientId` lets pages deep-link into a patient detail view.        //
// --------------------------------------------------------------------------- //
interface RouteValue {
  route: Route
  selectedPatientId: number | null
  navigate: (route: Route, patientId?: number | null) => void
}

const RouteContext = createContext<RouteValue | null>(null)

export function AppProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<string | null>(null)
  const [authReady, setAuthReady] = useState(false)
  const [route, setRoute] = useState<Route>('dashboard')
  const [selectedPatientId, setSelectedPatientId] = useState<number | null>(null)

  useEffect(() => {
    let active = true
    void fetchAuthSession()
      .then((session) => {
        if (!active) return
        setUser(session.user)
      })
      .catch(() => {
        if (!active) return
        setUser(null)
      })
      .finally(() => {
        if (active) setAuthReady(true)
      })
    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    const clearExpiredSession = () => {
      setUser(null)
      setAuthReady(true)
    }
    window.addEventListener('rehab:unauthorized', clearExpiredSession)
    return () => window.removeEventListener('rehab:unauthorized', clearExpiredSession)
  }, [])

  const login = useCallback(async (username: string, password: string) => {
    const resp = await loginUser(username.trim(), password)
    const name = resp.user || username.trim() || '医生'
    localStorage.removeItem('rehab_auth_token')
    setUser(name)
    setAuthReady(true)
    setRoute('dashboard')
  }, [])

  const logout = useCallback(() => {
    void logoutUser()
    setUser(null)
  }, [])

  const navigate = useCallback((next: Route, patientId: number | null = null) => {
    setRoute(next)
    setSelectedPatientId(patientId)
  }, [])

  const authValue = useMemo<AuthValue>(
    () => ({ user, ready: authReady, login, logout }),
    [user, authReady, login, logout],
  )
  const routeValue = useMemo<RouteValue>(
    () => ({ route, selectedPatientId, navigate }),
    [route, selectedPatientId, navigate],
  )

  return (
    <AuthContext.Provider value={authValue}>
      <RouteContext.Provider value={routeValue}>{children}</RouteContext.Provider>
    </AuthContext.Provider>
  )
}

export function useAuth(): AuthValue {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AppProvider')
  return ctx
}

export function useRoute(): RouteValue {
  const ctx = useContext(RouteContext)
  if (!ctx) throw new Error('useRoute must be used within AppProvider')
  return ctx
}
