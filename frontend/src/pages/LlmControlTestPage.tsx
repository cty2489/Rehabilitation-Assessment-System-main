import { useEffect, useState } from 'react'
import { CloudUpload, FlaskConical, Play } from 'lucide-react'
import PatientForm from '../components/PatientForm'
import MetricScoreCards from '../components/MetricScoreCards'
import ReportDisplay from '../components/ReportDisplay'
import { useAssessmentStream } from '../hooks/useAssessmentStream'
import {
  EvalPackageParse,
  Institution,
  PackageUploadProgress,
  createLlmControlTest,
  fetchLlmSettings,
  parseEvalPackage,
} from '../api'
import { LlmSettings, PatientInfo } from '../types'

type TruthInputs = {
  FMA_UE: string
  hand_tone: '' | '0' | '1' | '1+' | '2' | '3' | '4'
  hand_function: string
}

const INITIAL_PATIENT: PatientInfo = {
  patient_id: 'LLMTEST-001',
  name: '对照测试病例',
  sex: '男',
  age: '',
  diagnosis: '',
  disease_days: '',
  paralysis_side: '',
}

const INITIAL_TRUTH: TruthInputs = {
  FMA_UE: '',
  hand_tone: '',
  hand_function: '',
}

const shortHash = (value?: string | null) =>
  value ? `${value.slice(0, 12)}…${value.slice(-8)}` : '—'

const formatFileSize = (bytes: number) => {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function prefillToPatient(prefill: EvalPackageParse['patient_prefill']): PatientInfo {
  return {
    patient_id: prefill.patient_id || '',
    name: prefill.name || '',
    sex: prefill.sex === '女' ? '女' : '男',
    age: prefill.age ?? '',
    diagnosis: prefill.diagnosis || '',
    disease_days: prefill.disease_days ?? '',
    paralysis_side: prefill.paralysis_side === '右'
      ? '右'
      : prefill.paralysis_side === '左'
        ? '左'
        : '',
  }
}

export default function LlmControlTestPage() {
  const [settings, setSettings] = useState<LlmSettings | null>(null)
  const [patient, setPatient] = useState<PatientInfo>(INITIAL_PATIENT)
  const [truth, setTruth] = useState<TruthInputs>(INITIAL_TRUTH)
  const [institution, setInstitution] = useState<Institution>('hospital')
  const [zipFile, setZipFile] = useState<File | null>(null)
  const [parsed, setParsed] = useState<EvalPackageParse | null>(null)
  const [parsing, setParsing] = useState(false)
  const [uploadProgress, setUploadProgress] = useState<PackageUploadProgress | null>(null)
  const [localError, setLocalError] = useState<string | null>(null)
  const [fingerprint, setFingerprint] = useState<string | null>(null)
  const [submittedModel, setSubmittedModel] = useState<string | null>(null)
  const stream = useAssessmentStream()

  useEffect(() => {
    fetchLlmSettings()
      .then(setSettings)
      .catch((error) => setLocalError(String(error.message || error)))
  }, [])

  function pickZip(event: React.ChangeEvent<HTMLInputElement>) {
    setZipFile(event.target.files?.[0] || null)
    setParsed(null)
    setUploadProgress(null)
    setLocalError(null)
  }

  async function parseZip() {
    if (!zipFile) {
      setLocalError('请先选择医院端评估数据包（.zip）。')
      return
    }
    setLocalError(null)
    setParsing(true)
    setUploadProgress({ phase: 'uploading', loadedBytes: 0, totalBytes: zipFile.size, percent: 0 })
    try {
      const result = await parseEvalPackage(institution, zipFile, setUploadProgress)
      setParsed(result)
      setPatient(prefillToPatient(result.patient_prefill))
    } catch (error) {
      setParsed(null)
      setLocalError(`数据包解析失败：${String((error as Error).message || error)}`)
    } finally {
      setParsing(false)
      setUploadProgress(null)
    }
  }

  async function submit() {
    setLocalError(null)
    if (!patient.patient_id.trim() || !patient.name.trim() || !patient.diagnosis) {
      setLocalError('请完整填写患者编号、姓名和诊断。')
      return
    }
    if (!patient.paralysis_side) {
      setLocalError('请选择偏瘫侧。')
      return
    }
    if (!parsed) {
      setLocalError('请先上传并解析医院端评估数据包。')
      return
    }
    if (truth.FMA_UE === '' || truth.hand_tone === '' || truth.hand_function === '') {
      setLocalError('请填写 FMA 手部分数、Hand MAS 和 Brunnstrom 分期三个真值。')
      return
    }

    const fma = Number(truth.FMA_UE)
    const stage = Number(truth.hand_function)
    if (!Number.isFinite(fma) || fma < 0 || fma > 20) {
      setLocalError('FMA 手部分数必须在 0–20 之间。')
      return
    }
    if (!Number.isInteger(stage) || stage < 1 || stage > 6) {
      setLocalError('Brunnstrom 分期必须为 1–6 的整数。')
      return
    }

    const form = new FormData()
    form.append('patient_id', patient.patient_id)
    form.append('name', patient.name)
    form.append('sex', patient.sex)
    form.append('diagnosis', patient.diagnosis)
    form.append('paralysis_side', patient.paralysis_side)
    if (patient.age !== '') form.append('age', String(patient.age))
    if (patient.disease_days !== '') form.append('disease_days', String(patient.disease_days))
    form.append('FMA_UE', String(fma))
    form.append('hand_tone', truth.hand_tone)
    form.append('hand_function', String(stage))
    form.append('institution', institution)
    form.append('upload_id', parsed.upload_id)

    try {
      const session = await createLlmControlTest(form)
      setFingerprint(session.input_fingerprint)
      setSubmittedModel(session.report_model_id)
      stream.start(session.session_id)
    } catch (error) {
      setLocalError(String((error as Error).message || error))
    }
  }

  function exportMarkdown() {
    const blob = new Blob([stream.reportText], { type: 'text/markdown;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `LLM对照测试_${fingerprint?.slice(0, 12) || 'report'}.md`
    document.body.appendChild(anchor)
    anchor.click()
    document.body.removeChild(anchor)
    URL.revokeObjectURL(url)
  }

  function exportWord() {
    if (!stream.sessionId) return
    const anchor = document.createElement('a')
    anchor.href = `/api/assess/${stream.sessionId}/report.docx`
    anchor.download = `LLM对照测试_${fingerprint?.slice(0, 12) || 'report'}.docx`
    document.body.appendChild(anchor)
    anchor.click()
    document.body.removeChild(anchor)
  }

  const disabled = stream.phase === 'processing' || parsing
  const activeModel = settings?.active_model?.name || settings?.active_model_id || '读取中…'

  return (
    <div>
      <div className="page-head">
        <div>
          <h1 className="page-title">LLM 对照测试</h1>
          <p className="page-sub">信号计算生物标志物 · 手动替换三个 DL 评分 · 复用同一 Planner/RAG/报告链路</p>
        </div>
      </div>

      <div className="llm-control-notice">
        <FlaskConical aria-hidden="true" />
        <div>
          <strong>仅用于控制变量测试</strong>
          <span>26 项生物标志物仍由上传的原始信号计算；仅 FMA、Hand MAS、Brunnstrom 使用人工真值。本入口不写入患者档案或正式评估记录。</span>
        </div>
      </div>

      {(localError || stream.error) && (
        <div className="error-banner">{localError || stream.error}</div>
      )}

      <div className="card llm-control-model-card">
        <h2>本次报告模型<span className="h2-suffix">Controlled · LLM</span></h2>
        <div className="llm-control-model-row">
          <strong>{submittedModel || activeModel}</strong>
          <span>模型在提交时锁定；如需切换，请先到“模型设置”修改后再开始新测试。</span>
        </div>
      </div>

      <div className="card">
        <h2>评估数据包<span className="h2-suffix">Hospital · ZIP</span></h2>
        <p className="muted">与医院端评估使用同一种ZIP：系统按manifest.json自动匹配EEG和EMG/IMU，不再逐个选择文件。</p>
        <div className="grid-2">
          <div className="field">
            <label>数据来源机构</label>
            <select
              value={institution}
              disabled={disabled}
              onChange={(event) => {
                setInstitution(event.target.value as Institution)
                setParsed(null)
              }}
            >
              <option value="hospital">医院端（Delsys 56列 / 32导联 BDF）</option>
              <option value="device">设备端（穿戴设备 8通道 / 8导联 BDF）</option>
            </select>
          </div>
          <div className="field">
            <label>评估数据包（.zip）</label>
            <input type="file" accept=".zip" onChange={pickZip} disabled={disabled} />
          </div>
        </div>
        <div className="actions">
          <button className="button secondary" onClick={parseZip} disabled={disabled || !zipFile}>
            <CloudUpload aria-hidden="true" />
            {parsing ? '上传并校验中…' : '解析数据包'}
          </button>
        </div>
        {parsing && uploadProgress && (
          <div className="package-upload-progress" role="status" aria-live="polite">
            <progress max={100} value={uploadProgress.percent} />
            <span>
              {uploadProgress.phase === 'uploading'
                ? `${formatFileSize(uploadProgress.loadedBytes)} / ${formatFileSize(uploadProgress.totalBytes)}`
                : '上传完成，正在校验数据包'}
            </span>
          </div>
        )}
        {parsed && (
          <div className="report-display" style={{ marginTop: 12 }}>
            <p>已识别可用试次：<strong>{parsed.n_trials}</strong> 个</p>
            <p className="muted">Package SHA-256: {shortHash(parsed.package_hash)}</p>
            {parsed.warnings.length > 0 && <p className="muted">提示：{parsed.warnings.join('；')}</p>}
          </div>
        )}
      </div>

      <PatientForm value={patient} onChange={setPatient} disabled={disabled} />

      <div className="card">
        <h2>人工评分真值<span className="h2-suffix">Manual · Ground Truth</span></h2>
        <p className="muted llm-control-help">只替代三个 DL 临床评分；生物标志物不会在此处手动填写。</p>
        <div className="llm-control-score-grid">
          <div className="field">
            <label>FMA-UE 手部分数（0–20）</label>
            <input type="number" min={0} max={20} step="any" value={truth.FMA_UE} disabled={disabled} onChange={(event) => setTruth({ ...truth, FMA_UE: event.target.value })} />
          </div>
          <div className="field">
            <label>手部肌张力 Hand MAS</label>
            <select value={truth.hand_tone} disabled={disabled} onChange={(event) => setTruth({ ...truth, hand_tone: event.target.value as TruthInputs['hand_tone'] })}>
              <option value="">请选择</option>
              {['0', '1', '1+', '2', '3', '4'].map((value) => <option key={value} value={value}>{value} 级</option>)}
            </select>
          </div>
          <div className="field">
            <label>Brunnstrom 手功能分期</label>
            <select value={truth.hand_function} disabled={disabled} onChange={(event) => setTruth({ ...truth, hand_function: event.target.value })}>
              <option value="">请选择</option>
              {[1, 2, 3, 4, 5, 6].map((value) => <option key={value} value={value}>{value} 期</option>)}
            </select>
          </div>
        </div>
      </div>

      <div className="actions">
        <button className="button" onClick={submit} disabled={disabled || !parsed}>
          <Play aria-hidden="true" />
          {disabled ? 'LLM 生成中…' : '计算生物标志物并生成测试报告'}
        </button>
      </div>

      {stream.phase !== 'idle' && (
        <>
          <div className="card llm-control-run-card">
            <h2>运行状态<span className="h2-suffix">Signal Biomarkers · No DL · No DB Write</span></h2>
            <div className="llm-control-run-meta">
              <span className={`badge ${stream.phase === 'done' ? 'badge-ok' : 'badge-neutral'}`}>{stream.phase === 'done' ? '已完成' : 'Planner / RAG / LLM 运行中'}</span>
              <span>生物标志物：{stream.coverage?.available ?? '计算中…'}/26</span>
              <span>输入指纹：{fingerprint?.slice(0, 16) || '—'}</span>
            </div>
          </div>
          <div className="card">
            <h2>人工真值（非模型预测）<span className="h2-suffix">Verified · Input</span></h2>
            <MetricScoreCards fmaUe={truth.FMA_UE === '' ? undefined : Number(truth.FMA_UE)} handTone={truth.hand_tone || undefined} handFunction={truth.hand_function === '' ? undefined : Number(truth.hand_function)} />
          </div>
          <ReportDisplay text={stream.reportText} streaming={stream.reportStreaming} onExportMarkdown={exportMarkdown} onExportWord={exportWord} onRestart={stream.reset} done={stream.phase === 'done'} />
        </>
      )}
    </div>
  )
}
