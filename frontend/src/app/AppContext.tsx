import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  ReactNode,
} from 'react'
import { clearAuthToken, getAuthToken, loginUser, setAuthToken } from '../api'
import { Route } from '../types'

const USER_KEY = 'rehab_user'

// --------------------------------------------------------------------------- //
// Auth. The backend issues a demo-scoped bearer token after password login.    //
// --------------------------------------------------------------------------- //
interface AuthValue {
  user: string | null
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
  const [user, setUser] = useState<string | null>(
    () => (getAuthToken() ? localStorage.getItem(USER_KEY) : null),
  )
  const [route, setRoute] = useState<Route>('dashboard')
  const [selectedPatientId, setSelectedPatientId] = useState<number | null>(null)

  const login = useCallback(async (username: string, password: string) => {
    const resp = await loginUser(username.trim(), password)
    const name = resp.user || username.trim() || '医生'
    setAuthToken(resp.access_token)
    localStorage.setItem(USER_KEY, name)
    setUser(name)
    setRoute('dashboard')
  }, [])

  const logout = useCallback(() => {
    clearAuthToken()
    localStorage.removeItem(USER_KEY)
    setUser(null)
  }, [])

  const navigate = useCallback((next: Route, patientId: number | null = null) => {
    setRoute(next)
    setSelectedPatientId(patientId)
  }, [])

  const authValue = useMemo<AuthValue>(() => ({ user, login, logout }), [user, login, logout])
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
