import { useState } from 'react'
import { downloadAssessmentExport, type AssessmentExportKind } from '../api'
import { AssessmentRecord } from '../types'
import { fmtDateTime } from '../util'
import MarkdownReport from './MarkdownReport'

const HAND_TONE_DESC: Record<string, string> = {
  '0': '未见肌张力增高',
  '1': '轻度增高',
  '1+': '轻中度增高',
  '2': '中度增高',
  '3': '重度增高',
  '4': '强直状态',
}

const BRUNNSTROM_DESC: Record<number, string> = {
  1: '弛缓期，无主动运动',
  2: '联合反应出现',
  3: '可引出共同运动',
  4: '部分分离运动',
  5: '分离运动明显',
  6: '接近正常',
}

interface BiomarkerCoverage {
  available: number
  total: number
  missing: string[]
}

const shortHash = (value?: string | null) =>
  value ? `${value.slice(0, 12)}...${value.slice(-6)}` : '—'

function asStringList(value: unknown): string[] {
  if (Array.isArray(value)) return value.map((item) => String(item))
  if (typeof value === 'string' && value.trim()) return [value]
  return []
}

function num(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function biomarkerCoverage(value: unknown): BiomarkerCoverage | null {
  if (!value || typeof value !== 'object') return null
  const obj = value as { coverage?: unknown; groups?: unknown; flat?: unknown }
  const coverage = obj.coverage
  if (coverage && typeof coverage === 'object') {
    const cov = coverage as { available?: unknown; total?: unknown; missing_keys?: unknown }
    const available = num(cov.available)
    const total = num(cov.total)
    if (available != null && total != null) {
      return {
        available,
        total,
        missing: Array.isArray(cov.missing_keys) ? cov.missing_keys.map(String) : [],
      }
    }
  }

  if (Array.isArray(obj.groups)) {
    const total = obj.groups.reduce((sum, group) => {
      const markers = (group as { markers?: unknown }).markers
      return sum + (Array.isArray(markers) ? markers.length : 0)
    }, 0)
    return total > 0 ? { available: total, total, missing: [] } : null
  }

  if (obj.flat && typeof obj.flat === 'object') {
    const total = Object.keys(obj.flat).length
    return total > 0 ? { available: total, total, missing: [] } : null
  }

  return null
}

function reportBadge(record: AssessmentRecord) {
  if (record.report_status === 'failed') {
    return <span className="badge badge-warn">报告未生成</span>
  }
  if (record.report_status === 'manual') {
    return <span className="badge badge-neutral">手工记录</span>
  }
  return <span className="badge badge-ok">报告已生成</span>
}

function Meta({ label, value, title }: { label: string; value: React.ReactNode; title?: string }) {
  return (
    <div className="record-meta-item" title={title}>
      <span className="record-meta-label">{label}</span>
      <span className="record-meta-value">{value || '—'}</span>
    </div>
  )
}

// Renders provenance, core motor indicators, biomarker coverage, and the report for one
// persisted assessment record in a patient history timeline.
export default function RecordDetail({ record }: { record: AssessmentRecord }) {
  const [downloadError, setDownloadError] = useState<string | null>(null)
  const [downloading, setDownloading] = useState<AssessmentExportKind | null>(null)
  const coverage = biomarkerCoverage(record.biomarkers)
  const warnings = asStringList(record.parse_warnings)
  const source = record.institution || record.source || '—'
  const model = [record.llm_provider, record.llm_model].filter(Boolean).join(' / ') || '—'
  const trials = record.trials || []
  const biomarkerItems = record.biomarker_items || []

  const download = async (kind: AssessmentExportKind) => {
    setDownloadError(null)
    setDownloading(kind)
    try {
      await downloadAssessmentExport(record.id, kind)
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err)
      setDownloadError(msg)
    } finally {
      setDownloading(null)
    }
  }

  return (
    <div className="record-detail">
      <div className="record-detail-toolbar">
        <div className="record-status-line">
          {reportBadge(record)}
          {coverage ? (
            <span className="badge badge-ok">
              Biomarker {coverage.available}/{coverage.total}
            </span>
          ) : (
            <span className="badge badge-neutral">Biomarker 未记录</span>
          )}
          {record.n_trials != null && <span className="badge badge-neutral">{record.n_trials} trials</span>}
        </div>
        <div className="record-export-actions" aria-label="导出评估结果">
          <button className="button secondary" onClick={() => download('json')} disabled={!!downloading}>
            {downloading === 'json' ? '生成中...' : 'JSON'}
          </button>
          <button className="button secondary" onClick={() => download('pdf')} disabled={!!downloading}>
            {downloading === 'pdf' ? '生成中...' : 'PDF'}
          </button>
          <button className="button secondary" onClick={() => download('zip')} disabled={!!downloading}>
            {downloading === 'zip' ? '生成中...' : 'ZIP'}
          </button>
        </div>
      </div>

      {downloadError && <div className="error-banner">导出失败：{downloadError}</div>}

      <div className="record-meta-grid">
        <Meta label="数据来源" value={source} />
        <Meta label="记录生成时间" value={fmtDateTime(record.created_at)} />
        <Meta label="数据采集时间" value={record.assessment_time ? fmtDateTime(record.assessment_time) : '—'} />
        <Meta label="Session" value={record.session_id || '—'} />
        <Meta label="Assessment ID" value={record.assessment_id || '—'} />
        <Meta label="数据包" value={record.package_name || '—'} />
        <Meta label="SHA-256" value={shortHash(record.package_hash)} title={record.package_hash || undefined} />
        <Meta label="LLM" value={model} />
        <Meta label="DL Checkpoints" value={record.model_version || '—'} />
      </div>

      {warnings.length > 0 && (
        <div className="record-warning-strip">
          <strong>解析警告：</strong>
          {warnings.join('；')}
        </div>
      )}

      {coverage && coverage.missing.length > 0 && (
        <div className="record-warning-strip">
          <strong>缺失 biomarker：</strong>
          {coverage.missing.join('、')}
        </div>
      )}

      {trials.length > 0 && (
        <details className="record-subsection" open>
          <summary>运动 / Trial 明细（{trials.length} 条）</summary>
          <div className="mini-table-wrap">
            <table className="mini-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>动作</th>
                  <th>类型</th>
                  <th>EEG</th>
                  <th>EMG/IMU</th>
                  <th>状态</th>
                </tr>
              </thead>
              <tbody>
                {trials.map((trial, index) => (
                  <tr key={trial.id || index}>
                    <td>{trial.trial_index ?? index + 1}</td>
                    <td>{trial.action_name || '—'}</td>
                    <td>{trial.assessment_type || '—'}</td>
                    <td title={trial.eeg_file || undefined}>{trial.eeg_name || trial.eeg_file || '—'}</td>
                    <td title={trial.emg_file || undefined}>{trial.emg_name || trial.emg_file || '—'}</td>
                    <td>{trial.status || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}

      {biomarkerItems.length > 0 && (
        <details className="record-subsection">
          <summary>生物标志物明细（{biomarkerItems.length} 项）</summary>
          <div className="mini-table-wrap">
            <table className="mini-table">
              <thead>
                <tr>
                  <th>分组</th>
                  <th>指标</th>
                  <th>值</th>
                  <th>单位</th>
                  <th>有效试次</th>
                </tr>
              </thead>
              <tbody>
                {biomarkerItems.map((marker) => (
                  <tr key={marker.id}>
                    <td>{marker.group_label || marker.group_key || '—'}</td>
                    <td>
                      {marker.marker_name || marker.marker_key}
                      {!marker.available && <span className="mini-muted">（未提取）</span>}
                    </td>
                    <td>{marker.value_text || '—'}</td>
                    <td>{marker.unit || '—'}</td>
                    <td>{marker.n_valid ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}

      <div className="results-grid">
        <div className="result-card">
          <div className="label">FMA-UE 手部分数</div>
          <div className="value">
            {Math.round(record.fma_ue)}
            <span className="unit">/ 20 分</span>
          </div>
          <div className="progress-bar">
            <div style={{ width: `${(record.fma_ue / 20) * 100}%` }} />
          </div>
        </div>
        <div className="result-card">
          <div className="label">手部肌张力 · Hand MAS</div>
          <div className="value">
            {record.hand_tone}
            <span className="unit">级</span>
          </div>
          <div className="meta">{HAND_TONE_DESC[record.hand_tone] || '—'}</div>
        </div>
        <div className="result-card">
          <div className="label">手功能 · Brunnstrom 分期</div>
          <div className="value">
            Brunnstrom {record.hand_function}
            <span className="unit">期</span>
          </div>
          <div className="meta">{BRUNNSTROM_DESC[record.hand_function] || '—'}</div>
        </div>
      </div>

      <h4 className="record-report-title">中文康复建议</h4>
      {record.report ? (
        <MarkdownReport text={record.report} />
      ) : (
        <div className="error-banner">AI 康复报告未生成（评估指标已保留）。</div>
      )}
    </div>
  )
}
