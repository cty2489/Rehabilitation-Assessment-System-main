import { useEffect, useState } from 'react'
import { fetchHealth, fetchLlmSettings, updateLlmModelSettings, updateLlmSettings } from '../api'
import { useAuth } from '../app/AppContext'
import { HealthStatus, LlmModelOption, LlmSettings } from '../types'

const providerLabel: Record<string, string> = {
  remote: '远程服务',
  local: '本地权重',
  deepseek: 'API 服务',
}

function modelStatus(model: LlmModelOption): { label: string; className: string } {
  if (model.is_active) return { label: '当前使用', className: 'badge-ok' }
  if (model.available) return { label: '可选择', className: 'badge-neutral' }
  return { label: '未就绪', className: 'badge-warn' }
}

function modelLocation(model: LlmModelOption): string {
  if (model.provider === 'remote') return model.remote_url || '—'
  if (model.provider === 'deepseek') return model.model_id || '—'
  return model.weight_path || model.model_id || '—'
}

export default function SystemManagementPage() {
  const { user, logout } = useAuth()
  const [health, setHealth] = useState<HealthStatus | null>(null)
  const [llmSettings, setLlmSettings] = useState<LlmSettings | null>(null)
  const [selectedModelId, setSelectedModelId] = useState('')
  const [modelLocations, setModelLocations] = useState<Record<string, string>>({})
  const [saving, setSaving] = useState(false)
  const [savingModelId, setSavingModelId] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  function syncModelLocations(nextSettings: LlmSettings) {
    const next: Record<string, string> = {}
    nextSettings.models.forEach((model) => {
      next[model.id] = model.provider === 'remote'
        ? (model.remote_url || '')
        : (model.weight_path || '')
    })
    setModelLocations(next)
  }

  useEffect(() => {
    Promise.all([fetchHealth(), fetchLlmSettings()])
      .then(([nextHealth, nextSettings]) => {
        setHealth(nextHealth)
        setLlmSettings(nextSettings)
        setSelectedModelId(nextSettings.active_model_id)
        syncModelLocations(nextSettings)
      })
      .catch((e) => setError(String(e.message || e)))
  }, [])

  const activeModel = llmSettings?.active_model || null
  const selectedModel = llmSettings?.models.find((model) => model.id === selectedModelId) || activeModel
  const dirty = Boolean(llmSettings && selectedModelId && selectedModelId !== llmSettings.active_model_id)

  async function saveModelSelection() {
    if (!selectedModelId) return
    setSaving(true)
    setError(null)
    setMessage(null)
    try {
      const next = await updateLlmSettings(selectedModelId)
      setLlmSettings(next)
      setSelectedModelId(next.active_model_id)
      syncModelLocations(next)
      setMessage('大模型设置已保存，下一次生成报告将使用该模型。')
    } catch (e) {
      setError(String((e as Error).message || e))
    } finally {
      setSaving(false)
    }
  }

  async function saveModelLocation(model: LlmModelOption) {
    const value = (modelLocations[model.id] || '').trim()
    setSavingModelId(model.id)
    setError(null)
    setMessage(null)
    try {
      const payload = model.provider === 'remote'
        ? { remote_url: value }
        : { weight_path: value }
      const next = await updateLlmModelSettings(model.id, payload)
      setLlmSettings(next)
      setSelectedModelId(next.active_model_id)
      syncModelLocations(next)
      setMessage(`${model.name} 的模型配置已保存。`)
    } catch (e) {
      setError(String((e as Error).message || e))
    } finally {
      setSavingModelId(null)
    }
  }

  return (
    <div>
      <div className="page-head">
        <div>
          <h1 className="page-title">系统管理</h1>
          <p className="page-sub">账户、模型状态与系统信息</p>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}
      {message && <div className="success-banner">{message}</div>}

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
              <span className="info-value">v1.0 · Clinical OS</span>
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
              <span className="info-value">{health?.report_model || activeModel?.name || '—'}</span>
            </div>
          </div>
        </div>
      </div>

      <div className="card settings-wide-card">
        <h2>大模型设置<span className="h2-suffix">LLM</span></h2>

        <div className="llm-settings-head">
          <div className="field llm-select-field">
            <label>报告生成模型</label>
            <select
              value={selectedModelId}
              onChange={(e) => {
                setSelectedModelId(e.target.value)
                setMessage(null)
              }}
              disabled={!llmSettings || saving}
            >
              {(llmSettings?.models || []).map((model) => (
                <option
                  key={model.id}
                  value={model.id}
                  disabled={!model.available && !model.is_active}
                >
                  {model.origin ? `${model.origin} · ` : ''}{model.name}
                </option>
              ))}
            </select>
          </div>

          <div className="llm-active-summary">
            <span className="info-label">当前配置</span>
            <span className="llm-active-name">{activeModel?.name || '—'}</span>
            <span className="model-muted">
              {activeModel ? `${providerLabel[activeModel.provider] || activeModel.provider} · ${activeModel.vendor || '—'}` : '—'}
            </span>
          </div>

          <button
            className="button"
            onClick={saveModelSelection}
            disabled={!dirty || saving}
          >
            {saving ? '保存中…' : '保存设置'}
          </button>
        </div>

        {selectedModel && (
          <div className="llm-selected-detail">
            <div className="info-item">
              <span className="info-label">待使用模型</span>
              <span className="info-value">{selectedModel.name}</span>
            </div>
            <div className="info-item">
              <span className="info-label">调用方式</span>
              <span className="info-value">{providerLabel[selectedModel.provider] || selectedModel.provider}</span>
            </div>
            <div className="info-item">
              <span className="info-label">模型位置</span>
              <span className="info-value model-path">{modelLocation(selectedModel)}</span>
            </div>
            <div className="info-item">
              <span className="info-label">就绪状态</span>
              <span className="info-value">
                <span className={`badge ${modelStatus(selectedModel).className}`}>
                  {modelStatus(selectedModel).label}
                </span>
              </span>
            </div>
          </div>
        )}

        <div className="model-table-wrap">
          <table className="data-table compact model-table">
            <thead>
              <tr>
                <th>模型</th>
                <th>来源</th>
                <th>方式</th>
                <th>位置</th>
                <th>状态</th>
              </tr>
            </thead>
            <tbody>
              {(llmSettings?.models || []).map((model) => {
                const status = modelStatus(model)
                return (
                  <tr key={model.id} className={model.is_active ? 'active-row' : undefined}>
                    <td>
                      <strong>{model.name}</strong>
                      <span className="model-id">{model.id}</span>
                    </td>
                    <td>{model.origin || '—'}</td>
                    <td>{providerLabel[model.provider] || model.provider}</td>
                    <td className="model-path">
                      <div className="model-path-edit">
                        <input
                          className="model-path-input"
                          value={modelLocations[model.id] ?? modelLocation(model)}
                          onChange={(e) => {
                            setModelLocations((prev) => ({
                              ...prev,
                              [model.id]: e.target.value,
                            }))
                            setMessage(null)
                          }}
                          disabled={savingModelId === model.id}
                          aria-label={`${model.name} 模型位置`}
                        />
                        <button
                          className="button secondary tiny"
                          onClick={() => saveModelLocation(model)}
                          disabled={savingModelId === model.id}
                        >
                          {savingModelId === model.id ? '保存中…' : '保存'}
                        </button>
                      </div>
                      {model.provider === 'local' && (
                        <span className="model-muted">
                          {model.weight_exists ? ' · 权重已找到' : ' · 权重待放置'}
                        </span>
                      )}
                    </td>
                    <td><span className={`badge ${status.className}`}>{status.label}</span></td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
