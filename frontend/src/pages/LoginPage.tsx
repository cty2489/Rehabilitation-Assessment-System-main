import { useState } from 'react'
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
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M2 12h3l2 -6 3 12 3 -8 2 4h7" />
            </svg>
          </div>
          <div>
            <h1>智能康复评估平台</h1>
            <div className="subtitle">EEG · EMG · IMU 多模态融合　/　CMK-AGN × Yi-1.5-6B</div>
          </div>
        </div>

        <div className="field">
          <label>用户名</label>
          <input
            type="text"
            value={username}
            placeholder="请输入用户名"
            onChange={(e) => setUsername(e.target.value)}
            autoFocus
          />
        </div>
        <div className="field">
          <label>密码</label>
          <input
            type="password"
            value={password}
            placeholder="请输入密码"
            onChange={(e) => setPassword(e.target.value)}
          />
        </div>

        {error && <div className="error-banner">{error}</div>}

        <button className="button login-button" type="submit" disabled={loading}>
          {loading ? '登录中...' : '登录'}
        </button>
        <p className="login-hint">请输入演示账号密码</p>
      </form>
    </div>
  )
}
