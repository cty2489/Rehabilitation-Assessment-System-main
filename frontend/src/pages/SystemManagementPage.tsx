import { useEffect, useState } from 'react'
import { fetchHealth } from '../api'
import { useAuth } from '../app/AppContext'
import { HealthStatus } from '../types'

export default function SystemManagementPage() {
  const { user, logout } = useAuth()
  const [health, setHealth] = useState<HealthStatus | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch((e) => setError(String(e.message || e)))
  }, [])

  return (
    <div>
      <div className="page-head">
        <div>
          <h1 className="page-title">系统管理</h1>
          <p className="page-sub">账户、模型状态与系统信息</p>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div className="grid-2-cards">
        <div className="card">
          <h2>当前账户<span className="h2-suffix">Account</span></h2>
          <div className="info-grid">
            <div className="info-item">
              <span className="info-label">用户名</span>
              <span className="info-value">{user}</span>
            </div>
            <div className="info-item">
              <span className="info-label">角色</span>
              <span className="info-value">临床医生（演示）</span>
            </div>
          </div>
          <div className="actions">
            <button className="button secondary" onClick={logout}>
              退出登录
            </button>
          </div>
        </div>

        <div className="card">
          <h2>系统状态<span className="h2-suffix">Status</span></h2>
          <div className="info-grid">
            <div className="info-item">
              <span className="info-label">平台版本</span>
              <span className="info-value">v1.1.5 · Clinical OS</span>
            </div>
            <div className="info-item">
              <span className="info-label">后端状态</span>
              <span className="info-value">
                {health ? (
                  <span className="badge badge-ok">{health.status}</span>
                ) : '—'}
              </span>
            </div>
            <div className="info-item">
              <span className="info-label">已加载模型</span>
              <span className="info-value">
                {health ? health.models_loaded.join('、') || '无' : '—'}
              </span>
            </div>
            <div className="info-item">
              <span className="info-label">报告模型</span>
              <span className="info-value">{health?.report_model || '—'}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
