import { useState } from 'react'
import { useAuth } from '../app/AppContext'

// Frontend-only demo login: any credentials enter the system. The username is
// shown in the top bar and persisted in localStorage.
export default function LoginPage() {
  const { login } = useAuth()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')

  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    login(username || '医生')
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

        <button className="button login-button" type="submit">
          登录
        </button>
        <p className="login-hint">演示环境：输入任意账号即可登录</p>
      </form>
    </div>
  )
}
