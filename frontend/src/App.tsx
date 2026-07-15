import AppShell from './components/AppShell'
import LoginPage from './pages/LoginPage'
import { useAuth } from './app/AppContext'

export default function App() {
  const { user, ready } = useAuth()
  if (!ready) return null
  return user ? <AppShell /> : <LoginPage />
}
