import { FormEvent, useCallback, useEffect, useState } from 'react'
import { Ban, Copy, KeyRound, LogOut, PauseCircle, PlayCircle, RefreshCw, X } from 'lucide-react'
import {
  createDeviceCredential,
  fetchDeviceCredentials,
  fetchHealth,
  revokeDeviceCredential,
  rotateDeviceCredential,
  updateDeviceCredential,
} from '../api'
import { useAuth } from '../app/AppContext'
import {
  DeviceCredentialRecord,
  DeviceCredentialSecret,
  DeviceCredentialStatus,
  HealthStatus,
} from '../types'

function statusMeta(status: DeviceCredentialStatus): { label: string; className: string } {
  if (status === 'active') return { label: '启用', className: 'badge-ok' }
  if (status === 'disabled') return { label: '停用', className: 'badge-warn' }
  return { label: '已撤销', className: 'badge-neutral' }
}

function timeText(value: string | null): string {
  return value || '—'
}

export default function SystemManagementPage() {
  const { user, logout } = useAuth()
  const [health, setHealth] = useState<HealthStatus | null>(null)
  const [credentials, setCredentials] = useState<DeviceCredentialRecord[]>([])
  const [deviceId, setDeviceId] = useState('')
  const [label, setLabel] = useState('')
  const [secret, setSecret] = useState<DeviceCredentialSecret | null>(null)
  const [copied, setCopied] = useState(false)
  const [loading, setLoading] = useState(true)
  const [creating, setCreating] = useState(false)
  const [busyId, setBusyId] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)

  const loadCredentials = useCallback(async () => {
    const payload = await fetchDeviceCredentials()
    setCredentials(payload.items)
  }, [])

  useEffect(() => {
    Promise.all([fetchHealth(), loadCredentials()])
      .then(([nextHealth]) => setHealth(nextHealth))
      .catch((e) => setError(String(e.message || e)))
      .finally(() => setLoading(false))
  }, [loadCredentials])

  const createCredential = async (event: FormEvent) => {
    event.preventDefault()
    if (!deviceId.trim() || !label.trim()) return
    setCreating(true)
    setError(null)
    setSecret(null)
    try {
      const created = await createDeviceCredential(deviceId.trim(), label.trim())
      setSecret(created)
      setDeviceId('')
      setLabel('')
      await loadCredentials()
    } catch (e) {
      setError(String((e as Error).message || e))
    } finally {
      setCreating(false)
    }
  }

  const setStatus = async (record: DeviceCredentialRecord, status: 'active' | 'disabled') => {
    setBusyId(record.id)
    setError(null)
    try {
      await updateDeviceCredential(record.id, { status })
      await loadCredentials()
    } catch (e) {
      setError(String((e as Error).message || e))
    } finally {
      setBusyId(null)
    }
  }

  const rotate = async (record: DeviceCredentialRecord) => {
    if (!window.confirm(`重新生成 ${record.device_id} 的设备码？旧码将立即失效。`)) return
    setBusyId(record.id)
    setError(null)
    setSecret(null)
    try {
      const rotated = await rotateDeviceCredential(record.id)
      setSecret(rotated)
      await loadCredentials()
    } catch (e) {
      setError(String((e as Error).message || e))
    } finally {
      setBusyId(null)
    }
  }

  const revoke = async (record: DeviceCredentialRecord) => {
    if (!window.confirm(`撤销 ${record.device_id} 的设备码？该设备将立即无法访问云端。`)) return
    setBusyId(record.id)
    setError(null)
    try {
      await revokeDeviceCredential(record.id)
      if (secret?.credential.id === record.id) setSecret(null)
      await loadCredentials()
    } catch (e) {
      setError(String((e as Error).message || e))
    } finally {
      setBusyId(null)
    }
  }

  const copySecret = async () => {
    if (!secret) return
    try {
      await navigator.clipboard.writeText(secret.token)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1600)
    } catch {
      setError('浏览器未允许写入剪贴板')
    }
  }

  return (
    <div>
      <div className="page-head">
        <div>
          <h1 className="page-title">系统管理</h1>
          <p className="page-sub">账户、设备凭证与系统状态</p>
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
              <span className="info-value">系统管理员</span>
            </div>
          </div>
          <div className="actions">
            <button className="button secondary" onClick={logout}>
              <LogOut aria-hidden="true" />
              退出登录
            </button>
          </div>
        </div>

        <div className="card">
          <h2>系统状态<span className="h2-suffix">Status</span></h2>
          <div className="info-grid">
            <div className="info-item">
              <span className="info-label">平台版本</span>
              <span className="info-value">
                {health?.app_version || '—'}
                {health?.build_commit && health.build_commit !== 'unknown' ? ` · ${health.build_commit.slice(0, 8)}` : ''}
              </span>
            </div>
            <div className="info-item">
              <span className="info-label">后端状态</span>
              <span className="info-value">
                {health ? <span className="badge badge-ok">{health.status}</span> : '—'}
              </span>
            </div>
            <div className="info-item">
              <span className="info-label">已加载模型</span>
              <span className="info-value">{health ? health.models_loaded.join('、') || '无' : '—'}</span>
            </div>
            <div className="info-item">
              <span className="info-label">报告模型</span>
              <span className="info-value">{health?.report_model || '—'}</span>
            </div>
          </div>
        </div>
      </div>

      <section className="card settings-wide-card credential-manager">
        <div className="credential-head">
          <div>
            <h2>设备凭证<span className="h2-suffix">Device Access</span></h2>
            <p className="credential-summary">
              {loading ? '正在读取…' : `共 ${credentials.length} 个凭证，${credentials.filter((item) => item.status === 'active').length} 个启用`}
            </p>
          </div>
        </div>

        <form className="credential-create" onSubmit={createCredential}>
          <label className="field">
            <span>设备 ID</span>
            <input
              value={deviceId}
              onChange={(event) => setDeviceId(event.target.value)}
              placeholder="device_004"
              pattern="[A-Za-z0-9._-]+"
              maxLength={128}
              required
            />
          </label>
          <label className="field">
            <span>设备名称</span>
            <input
              value={label}
              onChange={(event) => setLabel(event.target.value)}
              placeholder="训练设备 004"
              maxLength={128}
              required
            />
          </label>
          <button className="button credential-create-button" disabled={creating}>
            <KeyRound aria-hidden="true" />
            {creating ? '生成中…' : '生成设备码'}
          </button>
        </form>

        {secret && (
          <div className="credential-secret" role="status">
            <div className="credential-secret-main">
              <span className="credential-secret-label">{secret.credential.device_id} 新设备码（仅显示一次）</span>
              <code>{secret.token}</code>
            </div>
            <div className="credential-secret-actions">
              <button className="button secondary tiny" onClick={copySecret}>
                <Copy aria-hidden="true" />
                {copied ? '已复制' : '复制'}
              </button>
              <button className="button secondary tiny" onClick={() => setSecret(null)}>
                <X aria-hidden="true" />
                关闭
              </button>
            </div>
          </div>
        )}

        <div className="credential-table-wrap">
          <table className="data-table credential-table">
            <thead>
              <tr>
                <th>设备</th>
                <th>凭证</th>
                <th>状态</th>
                <th>最近使用</th>
                <th>任务</th>
                <th>创建时间</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {!loading && credentials.length === 0 && (
                <tr><td colSpan={7} className="credential-empty">暂无设备凭证</td></tr>
              )}
              {credentials.map((record) => {
                const status = statusMeta(record.status)
                const busy = busyId === record.id
                return (
                  <tr key={record.id}>
                    <td>
                      <strong>{record.label || record.device_id}</strong>
                      <span className="credential-device-id">{record.device_id}</span>
                    </td>
                    <td>
                      <code className="credential-hint">{record.token_hint}</code>
                      {record.access_scope === 'shared' && <span className="badge badge-warn">共享权限</span>}
                    </td>
                    <td><span className={`badge ${status.className}`}>{status.label}</span></td>
                    <td>{timeText(record.last_used_at)}</td>
                    <td>{record.job_count ?? 0}</td>
                    <td>{timeText(record.created_at)}</td>
                    <td>
                      <div className="credential-actions">
                        {record.status === 'active' && (
                          <button className="button secondary tiny" disabled={busy} onClick={() => setStatus(record, 'disabled')}>
                            <PauseCircle aria-hidden="true" />
                            停用
                          </button>
                        )}
                        {record.status === 'disabled' && (
                          <button className="button secondary tiny" disabled={busy} onClick={() => setStatus(record, 'active')}>
                            <PlayCircle aria-hidden="true" />
                            启用
                          </button>
                        )}
                        <button className="button secondary tiny" disabled={busy} onClick={() => rotate(record)}>
                          <RefreshCw aria-hidden="true" />
                          {record.status === 'revoked' ? '重新生成' : '轮换'}
                        </button>
                        {record.status !== 'revoked' && (
                          <button className="button danger tiny" disabled={busy} onClick={() => revoke(record)}>
                            <Ban aria-hidden="true" />
                            撤销
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )
}
