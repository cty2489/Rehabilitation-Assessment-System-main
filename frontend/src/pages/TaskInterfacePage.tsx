import { useEffect, useState } from 'react'
import PatientForm from '../components/PatientForm'
import ProgressSteps from '../components/ProgressSteps'
import ResultsPanel from '../components/ResultsPanel'
import ReportDisplay from '../components/ReportDisplay'
import { useRoute } from '../app/AppContext'
import {
  EvalPackageParse,
  Institution,
  type AssessmentExportKind,
  deleteMysqlAssessment,
  downloadAssessmentExport,
  enrollPatient,
  fetchMysqlAssessment,
  fetchMysqlAssessments,
  fetchOnlineStatus,
  parseEvalPackage,
  submitOffline,
} from '../api'
import {
  EnrollmentRequest,
  MysqlAssessmentDetail,
  MysqlAssessmentItem,
  ParalysisSide,
  PatientInfo,
  Sex,
} from '../types'
import { useAssessmentStream } from '../hooks/useAssessmentStream'
import { fmtDateTime } from '../util'

type Mode = 'offline' | 'online'

const INITIAL_PATIENT: PatientInfo = {
  patient_id: '',
  name: '',
  sex: '男',
  age: '',
  diagnosis: '',
  disease_days: '',
  paralysis_side: '左',
}

const shortHash = (value?: string | null) =>
  value ? `${value.slice(0, 12)}...${value.slice(-6)}` : '-'

function toStringList(value: unknown): string[] {
  if (Array.isArray(value)) return value.map((item) => String(item))
  if (typeof value === 'string' && value.trim()) return [value]
  return []
}

function countBiomarkers(value: unknown): number | null {
  if (!value || typeof value !== 'object') return null
  const groups = (value as { groups?: unknown }).groups
  if (Array.isArray(groups)) {
    return groups.reduce((sum, group) => {
      const markers = (group as { markers?: unknown }).markers
      return sum + (Array.isArray(markers) ? markers.length : 0)
    }, 0)
  }
  const flat = (value as { flat?: unknown }).flat
  if (flat && typeof flat === 'object') return Object.keys(flat).length
  return null
}

function formatJSON(value: unknown): string {
  if (value == null) return ''
  if (typeof value === 'string') return value
  return JSON.stringify(value, null, 2)
}

function prefillToPatient(p: EvalPackageParse['patient_prefill']): PatientInfo {
  return {
    patient_id: p.patient_id || '',
    name: p.name || '',
    sex: (p.sex === '女' ? '女' : '男') as Sex,
    age: typeof p.age === 'number' ? p.age : '',
    diagnosis: p.diagnosis || '',
    disease_days: typeof p.disease_days === 'number' ? p.disease_days : '',
    paralysis_side: (p.paralysis_side === '右' ? '右' : '左') as ParalysisSide,
  }
}

export default function TaskInterfacePage() {
  const { navigate } = useRoute()
  const [mode, setMode] = useState<Mode>('offline')

  // Offline-mode local state
  const [institution, setInstitution] = useState<Institution>('hospital')
  const [zipFile, setZipFile] = useState<File | null>(null)
  const [parsing, setParsing] = useState(false)
  const [parsed, setParsed] = useState<EvalPackageParse | null>(null)
  const [patient, setPatient] = useState<PatientInfo>(INITIAL_PATIENT)
  const [submitting, setSubmitting] = useState(false)
  const [localError, setLocalError] = useState<string | null>(null)

  const stream = useAssessmentStream()
  // Bumped after an enrollment or a completed device assessment to refresh the
  // MySQL records list.
  const [recordsReload, setRecordsReload] = useState(0)

  const error = localError || stream.error

  // A finished device assessment wrote a new MySQL record — refresh the list.
  useEffect(() => {
    if (stream.phase === 'done') setRecordsReload((n) => n + 1)
  }, [stream.phase])

  const onPickZip = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] || null
    setZipFile(f)
    setParsed(null)
    setLocalError(null)
  }

  const handleParse = async () => {
    if (!zipFile) {
      setLocalError('请先选择 zip 数据包')
      return
    }
    setLocalError(null)
    setParsing(true)
    try {
      const res = await parseEvalPackage(institution, zipFile)
      setParsed(res)
      setPatient(prefillToPatient(res.patient_prefill))
    } catch (err) {
      setLocalError(`解析失败：${err instanceof Error ? err.message : String(err)}`)
    } finally {
      setParsing(false)
    }
  }

  const handleStart = async () => {
    setLocalError(null)
    if (!zipFile || !parsed) {
      setLocalError('请先解析数据包')
      return
    }
    if (!patient.patient_id.trim() || !patient.name.trim()) {
      setLocalError('请填写患者编号与姓名')
      return
    }
    if (!patient.diagnosis) {
      setLocalError('请选择诊断类型')
      return
    }

    const form = new FormData()
    form.append('institution', institution)
    form.append('package', zipFile)
    form.append('patient_id', patient.patient_id)
    form.append('name', patient.name)
    form.append('sex', patient.sex)
    if (patient.age !== '') form.append('age', String(patient.age))
    form.append('diagnosis', patient.diagnosis)
    if (patient.disease_days !== '') form.append('disease_days', String(patient.disease_days))
    form.append('paralysis_side', patient.paralysis_side)

    setSubmitting(true)
    try {
      const data = await submitOffline(form)
      stream.start(data.session_id)
    } catch (err) {
      setLocalError(`提交失败：${err instanceof Error ? err.message : String(err)}`)
    } finally {
      setSubmitting(false)
    }
  }

  const handleRestart = () => {
    stream.reset()
    setLocalError(null)
  }

  const exportName = (ext: string) =>
    `对接接口报告_${patient.patient_id || 'patient'}_${stream.sessionId?.slice(0, 6) ?? ''}.${ext}`

  const handleExportMarkdown = () => {
    const body = stream.reportText.trim() ? stream.reportText : '# （AI 康复报告未生成）\n'
    const blob = new Blob([body], { type: 'text/markdown;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = exportName('md')
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const handleExportWord = () => {
    if (!stream.sessionId) return
    const a = document.createElement('a')
    a.href = `/api/assess/${stream.sessionId}/report.docx`
    a.download = exportName('docx')
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  const processing = stream.phase !== 'idle'

  return (
    <div>
      <div className="page-head">
        <div>
          <h1 className="page-title">任务一与任务三对接接口页面</h1>
          <p className="page-sub">数据采集（任务一）→ 评估系统（任务三）　/　在线 · 离线双模式</p>
        </div>
        {processing && (
          <button className="button secondary" onClick={handleRestart}>
            ← 返回
          </button>
        )}
      </div>

      {/* Mode tabs */}
      <div className="card" style={{ display: 'flex', gap: 12 }}>
        <button
          className={`button ${mode === 'offline' ? '' : 'secondary'}`}
          onClick={() => setMode('offline')}
          disabled={processing}
        >
          离线模式（上传数据包）
        </button>
        <button
          className={`button ${mode === 'online' ? '' : 'secondary'}`}
          onClick={() => setMode('online')}
          disabled={processing}
        >
          在线模式（设备端实时采集）
        </button>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {mode === 'online' && <OnlinePanel />}

      {mode === 'offline' && (
        <>
          {!processing && (
            <>
              <div className="card">
                <h2>
                  数据包导入
                  <span className="h2-suffix">Offline · Package</span>
                </h2>
                <div className="grid-2">
                  <div className="field">
                    <label>数据来源机构</label>
                    <select
                      value={institution}
                      onChange={(e) => {
                        setInstitution(e.target.value as Institution)
                        setParsed(null)
                      }}
                    >
                      <option value="hospital">医院端（Delsys 56列 / 32导联 BDF）</option>
                      <option value="device">设备端（穿戴设备 8通道 / 8导联 BDF）</option>
                    </select>
                  </div>
                  <div className="field">
                    <label>评估数据包（.zip）</label>
                    <input type="file" accept=".zip" onChange={onPickZip} />
                  </div>
                </div>
                <div className="actions">
                  <button className="button secondary" onClick={handleParse} disabled={parsing || !zipFile}>
                    {parsing ? '解析中…' : '解析数据包'}
                  </button>
                </div>

                {parsed && (
                  <div className="report-display" style={{ marginTop: 12 }}>
                    <p>
                      机构：<strong>{parsed.institution === 'hospital' ? '医院端' : '设备端'}</strong>
                      　·　纳入 active 主动评估试次：<strong>{parsed.n_trials}</strong> 个
                    </p>
                    <p style={{ color: '#6b7280', fontSize: 13 }}>
                      Package SHA-256: {shortHash(parsed.package_hash)}
                    </p>
                    {parsed.enrolled && (
                      <p style={{ color: '#047857' }}>
                        ✓ 该患者已入组，已按 MySQL 档案回填基本信息。
                      </p>
                    )}
                    {parsed.warnings.length > 0 && (
                      <ul>
                        {parsed.warnings.map((w, i) => (
                          <li key={i} style={{ color: '#b45309' }}>⚠ {w}</li>
                        ))}
                      </ul>
                    )}
                    {parsed.n_trials === 0 && (
                      <p style={{ color: '#b91c1c' }}>没有可用试次，无法运行分析（设备端样例多为占位空文件）。</p>
                    )}
                  </div>
                )}
              </div>

              {parsed && parsed.n_trials > 0 && (
                <>
                  <PatientForm value={patient} onChange={setPatient} />
                  <div className="actions">
                    <button className="button" onClick={handleStart} disabled={submitting}>
                      {submitting ? '提交中…' : '开始分析'}
                    </button>
                  </div>
                </>
              )}

              <EnrollmentPanel onEnrolled={() => setRecordsReload((n) => n + 1)} />
              <MysqlRecordsPanel reload={recordsReload} />
            </>
          )}

          {processing && (
            <>
              <ProgressSteps steps={stream.steps} />
              {stream.coverage && (
                <div className="card">
                  <h2>
                    生物标志物提取覆盖率
                    <span className="h2-suffix">Biomarker · Coverage</span>
                  </h2>
                  <p>
                    本次成功提取 <strong>{stream.coverage.available}</strong> / {stream.coverage.total} 项生物标志物。
                    {stream.coverage.missing_keys.length > 0 && (
                      <>
                        {' '}缺失 {stream.coverage.missing_keys.length} 项（多因该采集格式不支持或数据不足，已自动标注「数据不足」，不影响报告生成）。
                      </>
                    )}
                  </p>
                  {stream.coverage.missing_keys.length > 0 && (
                    <p style={{ color: '#6b7280', fontSize: 13 }}>
                      缺失项：{stream.coverage.missing_keys.join('、')}
                    </p>
                  )}
                </div>
              )}
              <ResultsPanel results={stream.results} />
              <ReportDisplay
                text={stream.reportText}
                streaming={stream.reportStreaming}
                onExportMarkdown={handleExportMarkdown}
                onExportWord={handleExportWord}
                onRestart={handleRestart}
                onViewPatient={() => navigate('patients', null)}
                done={stream.phase === 'done'}
              />
            </>
          )}
        </>
      )}
    </div>
  )
}

// 患者入组（写入 MySQL 基本信息 + 可选第一次评估记录） --------------------- //
// 表单内部用字符串承载数值输入（允许真正为空），提交时再转成 EnrollmentRequest。
interface EnrollForm {
  patient_id: string
  name: string
  sex: Sex
  age: string
  diagnosis: string
  paralysis_side: ParalysisSide
  disease_days: string
  fma_ue: string
  bi: string
  hand_tone: string
  hand_function: string
}

const INITIAL_ENROLL: EnrollForm = {
  patient_id: '',
  name: '',
  sex: '男',
  age: '',
  diagnosis: '',
  paralysis_side: '左',
  disease_days: '',
  fma_ue: '',
  bi: '',
  hand_tone: '',
  hand_function: '',
}

function EnrollmentPanel({ onEnrolled }: { onEnrolled: () => void }) {
  const [open, setOpen] = useState(false)
  const [form, setForm] = useState<EnrollForm>(INITIAL_ENROLL)
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)

  const set = (k: keyof EnrollForm, v: string) => setForm((f) => ({ ...f, [k]: v }))

  const num = (v: string): number | null => {
    if (v === '') return null
    const n = Number(v)
    return Number.isFinite(n) ? n : null
  }

  const submit = async () => {
    setErr(null)
    setMsg(null)
    if (!form.patient_id.trim() || !form.name.trim()) {
      setErr('请填写患者编号与姓名')
      return
    }
    const payload: EnrollmentRequest = {
      patient_id: form.patient_id.trim(),
      name: form.name.trim(),
      sex: form.sex,
      age: num(form.age),
      diagnosis: form.diagnosis || null,
      paralysis_side: form.paralysis_side || null,
      disease_days: num(form.disease_days),
      fma_ue: num(form.fma_ue),
      bi: num(form.bi),
      hand_tone: form.hand_tone || null,
      hand_function: num(form.hand_function),
    }
    setBusy(true)
    try {
      await enrollPatient(payload)
      setMsg(`已入组：${payload.patient_id}`)
      setForm(INITIAL_ENROLL)
      onEnrolled()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="card">
      <h2>
        患者入组（MySQL）
        <span className="h2-suffix">Enrollment · Hospital</span>
        <button
          className="button secondary"
          style={{ marginLeft: 12, padding: '2px 10px' }}
          onClick={() => setOpen((o) => !o)}
        >
          {open ? '收起' : '展开'}
        </button>
      </h2>
      <p style={{ color: '#6b7280', fontSize: 13 }}>
        医院完成诊断后录入患者基本信息（最小集）与第一次评估记录（手工分数）。之后设备端上传评估时按 patient_id 自动关联。
      </p>
      {open && (
        <>
          <div className="grid-2">
            <div className="field">
              <label>患者编号 *</label>
              <input value={form.patient_id} onChange={(e) => set('patient_id', e.target.value)} />
            </div>
            <div className="field">
              <label>姓名 *</label>
              <input value={form.name} onChange={(e) => set('name', e.target.value)} />
            </div>
            <div className="field">
              <label>性别</label>
              <select value={form.sex} onChange={(e) => set('sex', e.target.value as Sex)}>
                <option value="男">男</option>
                <option value="女">女</option>
              </select>
            </div>
            <div className="field">
              <label>年龄</label>
              <input type="number" value={form.age ?? ''} onChange={(e) => set('age', e.target.value)} />
            </div>
            <div className="field">
              <label>诊断</label>
              <input value={form.diagnosis ?? ''} onChange={(e) => set('diagnosis', e.target.value)} />
            </div>
            <div className="field">
              <label>偏瘫侧</label>
              <select
                value={form.paralysis_side ?? '左'}
                onChange={(e) => set('paralysis_side', e.target.value as ParalysisSide)}
              >
                <option value="左">左</option>
                <option value="右">右</option>
              </select>
            </div>
            <div className="field">
              <label>病程（天）</label>
              <input
                type="number"
                value={form.disease_days ?? ''}
                onChange={(e) => set('disease_days', e.target.value)}
              />
            </div>
          </div>
          <h3 style={{ marginTop: 16 }}>第一次评估记录（医院给出，可留空）</h3>
          <div className="grid-2">
            <div className="field">
              <label>FMA-UE (0–20)</label>
              <input type="number" value={form.fma_ue ?? ''} onChange={(e) => set('fma_ue', e.target.value)} />
            </div>
            <div className="field">
              <label>Barthel 指数 (0–100)</label>
              <input type="number" value={form.bi ?? ''} onChange={(e) => set('bi', e.target.value)} />
            </div>
            <div className="field">
              <label>手部肌张力</label>
              <input
                value={form.hand_tone ?? ''}
                placeholder='0 / 1 / 1+ / 2 / 3 / 4'
                onChange={(e) => set('hand_tone', e.target.value)}
              />
            </div>
            <div className="field">
              <label>Brunnstrom 分期 (1–6)</label>
              <input
                type="number"
                value={form.hand_function ?? ''}
                onChange={(e) => set('hand_function', e.target.value)}
              />
            </div>
          </div>
          <div className="actions">
            <button className="button" onClick={submit} disabled={busy}>
              {busy ? '提交中…' : '提交入组'}
            </button>
          </div>
          {msg && <p style={{ color: '#047857' }}>✓ {msg}</p>}
          {err && <div className="error-banner">{err}</div>}
        </>
      )}
    </div>
  )
}

// 设备评估记录列表（MySQL） ------------------------------------------------ //
function MysqlRecordsPanel({ reload }: { reload: number }) {
  const [items, setItems] = useState<MysqlAssessmentItem[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(false)
  const [detailLoading, setDetailLoading] = useState(false)
  const [selected, setSelected] = useState<MysqlAssessmentDetail | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [downloading, setDownloading] = useState<AssessmentExportKind | null>(null)

  const refresh = async () => {
    setLoading(true)
    setErr(null)
    try {
      const data = await fetchMysqlAssessments(100, 0)
      setItems(data.items)
      setTotal(data.total)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    refresh()
  }, [reload])

  const onOpen = async (id: number) => {
    if (selected?.id === id) {
      setSelected(null)
      return
    }
    setDetailLoading(true)
    setErr(null)
    try {
      setSelected(await fetchMysqlAssessment(id))
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setDetailLoading(false)
    }
  }

  const onDelete = async (id: number) => {
    if (!window.confirm(`确认删除记录 #${id}？`)) return
    try {
      await deleteMysqlAssessment(id)
      if (selected?.id === id) setSelected(null)
      refresh()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
  }

  const onDownload = async (id: number, kind: AssessmentExportKind) => {
    setDownloading(kind)
    setErr(null)
    try {
      await downloadAssessmentExport(id, kind)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setDownloading(null)
    }
  }

  const selectedWarnings = toStringList(selected?.parse_warnings)
  const markerCount = countBiomarkers(selected?.biomarkers)

  return (
    <div className="card">
      <h2>
        设备评估记录（MySQL）
        <span className="h2-suffix">Device · Records</span>
        <button
          className="button secondary"
          style={{ marginLeft: 12, padding: '2px 10px' }}
          onClick={refresh}
          disabled={loading}
        >
          {loading ? '刷新中…' : '刷新'}
        </button>
      </h2>
      {err && <div className="error-banner">{err}</div>}
      <p style={{ color: '#6b7280', fontSize: 13 }}>共 {total} 条记录（测试期可逐条删除）。</p>
      {items.length === 0 ? (
        <p style={{ color: '#6b7280' }}>暂无记录。</p>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table className="data-table" style={{ width: '100%' }}>
            <thead>
              <tr>
                <th>时间</th>
                <th>患者</th>
                <th>来源</th>
                <th>Trials</th>
                <th>FMA-UE</th>
                <th>BI</th>
                <th>肌张力</th>
                <th>Brunnstrom</th>
                <th>Model</th>
                <th>数据包</th>
                <th>报告</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {items.map((r) => (
                <tr key={r.id}>
                  <td>{fmtDateTime(r.created_at)}</td>
                  <td>
                    {r.name || '—'}
                    <br />
                    <span style={{ color: '#6b7280', fontSize: 12 }}>{r.patient_id}</span>
                  </td>
                  <td>
                    {r.institution || r.source}
                    <br />
                    <span style={{ color: '#6b7280', fontSize: 12 }}>{r.source}</span>
                  </td>
                  <td>{r.n_trials ?? '-'}</td>
                  <td>{r.fma_ue}</td>
                  <td>{r.bi}</td>
                  <td>{r.hand_tone}</td>
                  <td>{r.hand_function}</td>
                  <td style={{ fontSize: 12, color: '#6b7280' }}>{r.llm_provider || '-'} / {r.llm_model || '-'}</td>
                  <td style={{ fontSize: 12, color: '#6b7280' }}>
                    {r.package_name || '—'}
                    <br />
                    {shortHash(r.package_hash)}
                  </td>
                  <td>{r.report_status}</td>
                  <td style={{ whiteSpace: 'nowrap' }}>
                    <button
                      className="button secondary"
                      style={{ padding: '2px 10px', marginRight: 6 }}
                      onClick={() => onOpen(r.id)}
                      disabled={detailLoading}
                    >
                      {selected?.id === r.id ? 'Hide' : 'View'}
                    </button>
                    <button
                      className="button secondary"
                      style={{ padding: '2px 10px' }}
                      onClick={() => onDelete(r.id)}
                    >
                      删除
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {detailLoading && <p style={{ color: '#6b7280', fontSize: 13 }}>Loading detail...</p>}
      {selected && (
        <div className="report-display" style={{ marginTop: 12 }}>
          <div className="record-detail-toolbar">
            <h3 style={{ marginTop: 0 }}>MySQL structured detail #{selected.id}</h3>
            <div className="record-export-actions" aria-label="导出评估结果">
              <button className="button secondary" onClick={() => onDownload(selected.id, 'json')} disabled={!!downloading}>
                {downloading === 'json' ? '生成中...' : 'JSON'}
              </button>
              <button className="button secondary" onClick={() => onDownload(selected.id, 'pdf')} disabled={!!downloading}>
                {downloading === 'pdf' ? '生成中...' : 'PDF'}
              </button>
              <button className="button secondary" onClick={() => onDownload(selected.id, 'zip')} disabled={!!downloading}>
                {downloading === 'zip' ? '生成中...' : 'ZIP'}
              </button>
            </div>
          </div>
          <div className="grid-2">
            <p>
              Patient: <strong>{selected.name || selected.patient_id}</strong>
              <br />
              <span style={{ color: '#6b7280', fontSize: 12 }}>
                {selected.patient_id} / {selected.sex || '-'} / {selected.age ?? '-'}
              </span>
            </p>
            <p>
              Diagnosis: {selected.diagnosis || '-'}
              <br />
              <span style={{ color: '#6b7280', fontSize: 12 }}>
                Side: {selected.paralysis_side || '-'} / Days: {selected.disease_days ?? '-'}
              </span>
            </p>
            <p>
              Source: {selected.institution || selected.source} / {selected.n_trials ?? '-'} trials
              <br />
              <span style={{ color: '#6b7280', fontSize: 12 }}>
                session={selected.session_id || '-'} / assessment={selected.assessment_id || '-'}
              </span>
            </p>
            <p>
              LLM: {selected.llm_provider || '-'} / {selected.llm_model || '-'}
              <br />
              <span style={{ color: '#6b7280', fontSize: 12 }}>{selected.model_version || '-'}</span>
            </p>
          </div>
          <p style={{ color: '#6b7280', fontSize: 13 }}>
            package={selected.package_name || '-'} / sha256={shortHash(selected.package_hash)} /
            biomarkers={markerCount ?? '-'}
          </p>
          {selectedWarnings.length > 0 && (
            <ul>
              {selectedWarnings.map((warning, index) => (
                <li key={index} style={{ color: '#b45309' }}>{warning}</li>
              ))}
            </ul>
          )}
          {selected.prediction_json != null && (
            <details open>
              <summary>Prediction JSON</summary>
              <pre style={{ whiteSpace: 'pre-wrap', maxHeight: 220, overflow: 'auto', fontSize: 12 }}>
                {formatJSON(selected.prediction_json)}
              </pre>
            </details>
          )}
          {selected.biomarkers != null && (
            <details>
              <summary>Biomarker JSON</summary>
              <pre style={{ whiteSpace: 'pre-wrap', maxHeight: 260, overflow: 'auto', fontSize: 12 }}>
                {formatJSON(selected.biomarkers)}
              </pre>
            </details>
          )}
          {selected.report && (
            <details>
              <summary>AI report</summary>
              <pre style={{ whiteSpace: 'pre-wrap', maxHeight: 360, overflow: 'auto', fontSize: 13 }}>
                {selected.report}
              </pre>
            </details>
          )}
        </div>
      )}
    </div>
  )
}

function OnlinePanel() {
  const [status, setStatus] = useState<{ status: string; device_url: string; message: string } | null>(null)
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  const refresh = async () => {
    setLoading(true)
    setErr(null)
    try {
      setStatus(await fetchOnlineStatus())
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    refresh()
  }, [])

  return (
    <div className="card">
      <h2>
        在线模式 · 设备端实时采集
        <span className="h2-suffix">Online · Device Stream</span>
      </h2>
      <div className="error-banner" style={{ background: '#fffbeb', color: '#92400e', borderColor: '#fcd34d' }}>
        设备端实时采集接口尚未对接，当前为占位界面。请使用「离线模式」上传数据包进行分析。
      </div>
      <div className="field" style={{ marginTop: 12 }}>
        <label>设备数据流地址（DEVICE_STREAM_URL）</label>
        <input type="text" value={status?.device_url || ''} placeholder="（未配置）" readOnly />
      </div>
      <div className="actions">
        <button className="button" disabled title="设备端接口待对接">
          连接设备（待对接）
        </button>
        <button className="button secondary" onClick={refresh} disabled={loading}>
          {loading ? '查询中…' : '刷新状态'}
        </button>
      </div>
      {err && <div className="error-banner">{err}</div>}
      {status && <p style={{ color: '#6b7280', fontSize: 13 }}>状态：{status.status}　·　{status.message}</p>}
    </div>
  )
}
