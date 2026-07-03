import { useCallback, useEffect, useRef, useState } from 'react'
import PatientForm from '../components/PatientForm'
import FileUpload from '../components/FileUpload'
import ProgressSteps from '../components/ProgressSteps'
import ResultsPanel from '../components/ResultsPanel'
import ReportDisplay from '../components/ReportDisplay'
import { useRoute } from '../app/AppContext'
import { authHeaders } from '../api'
import {
  PatientInfo,
  PredictionEntry,
  SSEEvent,
  StepKey,
  StepState,
  TaskKey,
} from '../types'

type Phase = 'input' | 'processing' | 'done'

// No prefilled values — age/disease_days start empty; diagnosis starts unselected.
const INITIAL_PATIENT: PatientInfo = {
  patient_id: '',
  name: '',
  sex: '男',
  age: '',
  diagnosis: '',
  disease_days: '',
  paralysis_side: '左',
}

const STEP_DEFS: { key: StepKey; label: string }[] = [
  { key: 'parse', label: '文件解析与校验' },
  { key: 'preprocess', label: '信号预处理' },
  { key: 'alignment', label: '多模态时序对齐' },
  { key: 'feature_extract', label: '多尺度特征提取' },
  { key: 'graph_fusion', label: '跨模态图注意力融合' },
  { key: 'inference', label: '模型推理' },
  { key: 'report', label: 'AI 报告生成' },
]

function freshSteps(): StepState[] {
  return STEP_DEFS.map((s) => ({ ...s, status: 'pending', details: [] }))
}

export default function AssessmentPage() {
  const { navigate } = useRoute()
  const [phase, setPhase] = useState<Phase>('input')
  const [patient, setPatient] = useState<PatientInfo>(INITIAL_PATIENT)
  const [eegFiles, setEegFiles] = useState<File[]>([])
  const [emgFiles, setEmgFiles] = useState<File[]>([])
  const [steps, setSteps] = useState<StepState[]>(freshSteps)
  const [results, setResults] = useState<Partial<Record<TaskKey, PredictionEntry>>>({})
  const [reportText, setReportText] = useState('')
  const [reportStreaming, setReportStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [savedPatientId, setSavedPatientId] = useState<string | null>(null)
  const esRef = useRef<EventSource | null>(null)

  useEffect(() => {
    return () => {
      esRef.current?.close()
    }
  }, [])

  const updateStep = useCallback(
    (key: StepKey, mutator: (s: StepState) => StepState) => {
      setSteps((prev) => prev.map((s) => (s.key === key ? mutator(s) : s)))
    },
    [],
  )

  const handleEvent = useCallback(
    (event: SSEEvent) => {
      switch (event.type) {
        case 'step_start':
          updateStep(event.step, (s) => ({ ...s, status: 'running', label: event.label || s.label }))
          if (event.step === 'report') {
            setReportStreaming(true)
          }
          break
        case 'step_detail':
          updateStep(event.step, (s) => ({ ...s, details: [...s.details, event.detail] }))
          break
        case 'step_done':
          updateStep(event.step, (s) => ({ ...s, status: 'done' }))
          if (event.step === 'report') {
            setReportStreaming(false)
          }
          break
        case 'prediction':
          setResults((prev) => ({
            ...prev,
            [event.task]: {
              task: event.task,
              label: event.label,
              value: event.value,
              range: event.range,
            },
          }))
          break
        case 'report_chunk':
          setReportText((prev) => prev + event.chunk)
          break
        case 'done':
          setPhase('done')
          setReportStreaming(false)
          esRef.current?.close()
          break
        case 'error':
          // A report-generation error must NOT block completion: predictions are
          // already shown and the record is still saved server-side. Surface the
          // message but keep the done-phase buttons available.
          setError(event.message)
          break
      }
    },
    [updateStep],
  )

  const handleSubmit = async () => {
    setError(null)
    if (!patient.patient_id.trim() || !patient.name.trim()) {
      setError('请填写患者编号与姓名')
      return
    }
    if (!patient.diagnosis) {
      setError('请选择诊断类型')
      return
    }
    if (eegFiles.length === 0 || emgFiles.length === 0) {
      setError('请上传至少一组 EEG / EMG 文件')
      return
    }
    if (eegFiles.length !== emgFiles.length) {
      setError(`EEG (${eegFiles.length}) 与 EMG (${emgFiles.length}) 文件数量必须一致`)
      return
    }

    const form = new FormData()
    form.append('patient_id', patient.patient_id)
    form.append('name', patient.name)
    form.append('sex', patient.sex)
    // age / disease_days are optional — only send when provided.
    if (patient.age !== '') form.append('age', String(patient.age))
    form.append('diagnosis', patient.diagnosis)
    if (patient.disease_days !== '') form.append('disease_days', String(patient.disease_days))
    form.append('paralysis_side', patient.paralysis_side)
    eegFiles.forEach((f) => form.append('eeg_files', f))
    emgFiles.forEach((f) => form.append('emg_files', f))

    setSteps(freshSteps())
    setResults({})
    setReportText('')
    setReportStreaming(false)
    setSavedPatientId(patient.patient_id)
    setPhase('processing')

    try {
      const res = await fetch('/api/assess', {
        method: 'POST',
        headers: authHeaders(),
        body: form,
      })
      if (!res.ok) {
        const detail = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(detail.detail || `HTTP ${res.status}`)
      }
      const data: { session_id: string } = await res.json()
      setSessionId(data.session_id)

      const es = new EventSource(`/api/assess/${data.session_id}/stream`)
      esRef.current = es
      es.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data) as SSEEvent
          handleEvent(msg)
        } catch (parseErr) {
          console.error('SSE parse error', parseErr, e.data)
        }
      }
      es.onerror = () => {
        // EventSource auto-reconnects; leave UI as-is during processing.
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err)
      setError(`提交失败：${msg}`)
      setPhase('input')
    }
  }

  // Used both by the in-processing 返回 button and the post-done 重新评估 button.
  const handleRestart = () => {
    esRef.current?.close()
    esRef.current = null
    setPhase('input')
    setSteps(freshSteps())
    setResults({})
    setReportText('')
    setReportStreaming(false)
    setError(null)
    setSessionId(null)
  }

  // The report text is now the full multi-section Markdown report (it already
  // contains patient info + all sections), so we export it verbatim.
  const exportName = (ext: string) =>
    `康复评估报告_${patient.patient_id || 'patient'}_${sessionId?.slice(0, 6) ?? ''}.${ext}`

  const handleExportMarkdown = () => {
    const body = reportText.trim() ? reportText : '# （AI 康复报告未生成）\n'
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

  // Word export: the backend renders the same Markdown into a .docx.
  const handleExportWord = () => {
    if (!sessionId) return
    const a = document.createElement('a')
    a.href = `/api/assess/${sessionId}/report.docx`
    a.download = exportName('docx')
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  return (
    <div>
      <div className="page-head">
        <div>
          <h1 className="page-title">康复评估</h1>
          <p className="page-sub">EEG · EMG · IMU 多模态融合　/　CMK-AGN × Yi-1.5-6B</p>
        </div>
        {phase === 'processing' && (
          <button className="button secondary" onClick={handleRestart}>
            ← 返回
          </button>
        )}
      </div>

      {error && <div className="error-banner">{error}</div>}

      {phase === 'input' && (
        <>
          <PatientForm value={patient} onChange={setPatient} />
          <FileUpload
            eegFiles={eegFiles}
            emgFiles={emgFiles}
            onChange={(eeg, emg) => {
              setEegFiles(eeg)
              setEmgFiles(emg)
            }}
          />
          <div className="actions">
            <button className="button" onClick={handleSubmit}>
              开始评估
            </button>
          </div>
        </>
      )}

      {phase !== 'input' && (
        <>
          <ProgressSteps steps={steps} />
          <ResultsPanel results={results} />
          <ReportDisplay
            text={reportText}
            streaming={reportStreaming}
            onExportMarkdown={handleExportMarkdown}
            onExportWord={handleExportWord}
            onRestart={handleRestart}
            onViewPatient={
              savedPatientId
                ? () => navigate('patients', null)
                : undefined
            }
            done={phase === 'done'}
          />
        </>
      )}
    </div>
  )
}
