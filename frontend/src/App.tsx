import AppShell from './components/AppShell'
import LoginPage from './pages/LoginPage'
import { useAuth } from './app/AppContext'

export default function App() {
  const { user } = useAuth()
  return user ? <AppShell /> : <LoginPage />
}
