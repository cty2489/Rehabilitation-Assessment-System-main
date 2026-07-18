import { useState } from 'react'
import { Activity, LockKeyhole, LogIn, ShieldCheck, UserRound } from 'lucide-react'
import { useAuth } from '../app/AppContext'

export default function LoginPage() {
  const { login } = useAuth()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      await login(username, password)
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err)
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="login-page">
      <form className="login-card" onSubmit={submit}>
        <div className="login-brand">
          <div className="brand-logo" aria-hidden="true">
            <Activity />
          </div>
          <div>
            <h1>智能康复评估平台</h1>
            <div className="subtitle">EEG · EMG · IMU 多模态康复评估</div>
          </div>
        </div>

        <div className="field">
          <label>用户名</label>
          <div className="login-input-wrap">
            <UserRound aria-hidden="true" />
            <input
              type="text"
              value={username}
              placeholder="请输入用户名"
              onChange={(e) => setUsername(e.target.value)}
              autoComplete="username"
              autoFocus
            />
          </div>
        </div>
        <div className="field">
          <label>密码</label>
          <div className="login-input-wrap">
            <LockKeyhole aria-hidden="true" />
            <input
              type="password"
              value={password}
              placeholder="请输入密码"
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
            />
          </div>
        </div>

        {error && <div className="error-banner" role="alert">{error}</div>}

        <button className="button login-button" type="submit" disabled={loading}>
          <LogIn aria-hidden="true" />
          {loading ? '登录中...' : '登录'}
        </button>
        <p className="login-hint"><ShieldCheck aria-hidden="true" />受控医疗业务系统</p>
      </form>
    </div>
  )
}
