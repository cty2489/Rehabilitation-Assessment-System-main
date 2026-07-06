import { useEffect, useState } from 'react'
import { fetchHealth, fetchLlmSettings, updateLlmSettings } from '../api'
import { HealthStatus, LlmModelOption, LlmSettings } from '../types'

const providerLabel: Record<string, string> = {
  remote: '远程服务',
  local: '本地模型',
  deepseek: 'API 服务',
}

function modelStatus(model: LlmModelOption): { label: string; className: string } {
  if (model.is_active) return { label: '当前使用', className: 'badge-ok' }
  if (model.available) return { label: '可切换', className: 'badge-neutral' }
  if (model.status === 'candidate' || model.report_ready === false) {
    return { label: '候选待验证', className: 'badge-warn' }
  }
  return { label: '未就绪', className: 'badge-warn' }
}

function modelMeta(model: LlmModelOption): string {
  return [
    model.origin,
    model.vendor,
    providerLabel[model.provider] || model.provider,
  ].filter(Boolean).join(' · ')
}

export default function ModelSettingsPage() {
  const [health, setHealth] = useState<HealthStatus | null>(null)
  const [settings, setSettings] = useState<LlmSettings | null>(null)
  const [selectedModelId, setSelectedModelId] = useState('')
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const load = () => {
    setError(null)
    Promise.all([fetchHealth(), fetchLlmSettings()])
      .then(([nextHealth, nextSettings]) => {
        setHealth(nextHealth)
        setSettings(nextSettings)
        setSelectedModelId(nextSettings.active_model_id)
      })
      .catch((e) => setError(String(e.message || e)))
  }

  useEffect(() => {
    load()
  }, [])

  const activeModel = settings?.active_model || null
  const selectedModel = settings?.models.find((model) => model.id === selectedModelId) || activeModel
  const dirty = Boolean(settings && selectedModelId && selectedModelId !== settings.active_model_id)

  async function saveModelSelection() {
    if (!selectedModelId) return
    setSaving(true)
    setError(null)
    setMessage(null)
    try {
      const next = await updateLlmSettings(selectedModelId)
      setSettings(next)
      setSelectedModelId(next.active_model_id)
      setHealth(await fetchHealth())
      setMessage('模型设置已保存，下一次生成报告将使用该模型。')
    } catch (e) {
      setError(String((e as Error).message || e))
    } finally {
      setSaving(false)
    }
  }

  return (
    <div>
      <div className="page-head">
        <div>
          <h1 className="page-title">模型设置</h1>
          <p className="page-sub">选择报告生成使用的大模型</p>
        </div>
        <button className="button secondary" onClick={load} disabled={saving}>
          刷新状态
        </button>
      </div>

      {error && <div className="error-banner">{error}</div>}
      {message && <div className="success-banner">{message}</div>}

      <div className="model-settings-layout">
        <div className="card model-current-card">
          <h2>当前模型<span className="h2-suffix">Active</span></h2>
          <div className="model-current-name">{activeModel?.name || health?.report_model || '—'}</div>
          <div className="model-current-meta">
            {activeModel ? modelMeta(activeModel) : health?.report_provider || '—'}
          </div>
          <div className="model-current-status">
            <span className="badge badge-ok">{health?.status || '—'}</span>
            <span className="badge badge-neutral">{health?.report_model || activeModel?.id || '—'}</span>
          </div>
        </div>

        <div className="card model-switch-card">
          <h2>切换模型<span className="h2-suffix">Switch</span></h2>
          <div className="field">
            <label>报告生成模型</label>
            <select
              value={selectedModelId}
              onChange={(e) => {
                setSelectedModelId(e.target.value)
                setMessage(null)
              }}
              disabled={!settings || saving}
            >
              {(settings?.models || []).map((model) => (
                <option
                  key={model.id}
                  value={model.id}
                  disabled={!model.available && !model.is_active}
                >
                  {model.name}
                </option>
              ))}
            </select>
          </div>
          {selectedModel && (
            <div className="model-switch-summary">
              <span className={`badge ${modelStatus(selectedModel).className}`}>
                {modelStatus(selectedModel).label}
              </span>
              <span>{modelMeta(selectedModel) || selectedModel.id}</span>
            </div>
          )}
          <div className="actions">
            <button className="button" onClick={saveModelSelection} disabled={!dirty || saving}>
              {saving ? '保存中…' : '保存设置'}
            </button>
          </div>
        </div>
      </div>

      <div className="card settings-wide-card">
        <h2>候选模型<span className="h2-suffix">Candidates</span></h2>
        <div className="model-candidate-grid">
          {(settings?.models || []).map((model) => {
            const status = modelStatus(model)
            return (
              <div key={model.id} className={`model-candidate ${model.is_active ? 'active' : ''}`}>
                <div className="model-candidate-head">
                  <strong>{model.name}</strong>
                  <span className={`badge ${status.className}`}>{status.label}</span>
                </div>
                <div className="model-current-meta">{modelMeta(model)}</div>
                <p>{model.description || '—'}</p>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
