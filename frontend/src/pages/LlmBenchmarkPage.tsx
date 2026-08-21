import { useEffect, useState } from 'react'
import { ClipboardCheck, Upload } from 'lucide-react'
import PatientForm from '../components/PatientForm'
import { useRoute } from '../app/AppContext'
import {
  EvalPackageParse,
  PackageUploadProgress,
  fetchBenchmarkPresets,
  parseEvalPackage,
  prepareLlmBenchmarkBatch,
  prepareLlmBenchmarkSingle,
} from '../api'
import { BenchmarkBatchPrepareResponse, BenchmarkPresetsResponse, PatientInfo } from '../types'

const INITIAL_PATIENT: PatientInfo = {
  patient_id: 'CASE001',
  name: '评测病例',
  sex: '男',
  age: '',
  diagnosis: '',
  disease_days: '',
  paralysis_side: '左',
}

function prefillToPatient(prefill: EvalPackageParse['patient_prefill']): PatientInfo {
  return {
    patient_id: prefill.patient_id || '',
    name: prefill.name || '',
    sex: prefill.sex === '女' ? '女' : '男',
    age: prefill.age ?? '',
    diagnosis: prefill.diagnosis || '',
    disease_days: prefill.disease_days ?? '',
    paralysis_side: prefill.paralysis_side === '右' ? '右' : '左',
  }
}

export default function LlmBenchmarkPage() {
  const { navigate } = useRoute()
  const [presets, setPresets] = useState<BenchmarkPresetsResponse | null>(null)
  const [preset, setPreset] = useState('benchmark_stage1')
  const [institution, setInstitution] = useState<'hospital' | 'device'>('hospital')
  const [patient, setPatient] = useState<PatientInfo>(INITIAL_PATIENT)
  const [scores, setScores] = useState({ fma_wrist: '', fma_hand: '', hand_mas: '', brunnstrom_hand: '' })
  const [singleZipFile, setSingleZipFile] = useState<File | null>(null)
  const [singleParsed, setSingleParsed] = useState<EvalPackageParse | null>(null)
  const [singleParsing, setSingleParsing] = useState(false)
  const [singleUploadProgress, setSingleUploadProgress] = useState<PackageUploadProgress | null>(null)
  const [batchFile, setBatchFile] = useState<File | null>(null)
  const [batchResult, setBatchResult] = useState<BenchmarkBatchPrepareResponse | null>(null)
  const [singleResult, setSingleResult] = useState<Record<string, unknown> | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    void fetchBenchmarkPresets().then(setPresets).catch((err) => setError(String(err.message || err)))
  }, [])

  async function prepareBatch() {
    if (!batchFile) {
      setError('请先选择包含患者资料和clinical_scores.xlsx的ZIP。')
      return
    }
    setBusy(true)
    setError(null)
    setBatchResult(null)
    try {
      setBatchResult(await prepareLlmBenchmarkBatch(institution, batchFile, preset))
    } catch (err) {
      setError(String((err as Error).message || err))
    } finally {
      setBusy(false)
    }
  }

  async function parseSingleZip() {
    if (!singleZipFile) {
      setError('请先选择单病例评估数据包（.zip）。')
      return
    }
    setError(null)
    setSingleParsing(true)
    setSingleUploadProgress({
      phase: 'uploading',
      loadedBytes: 0,
      totalBytes: singleZipFile.size,
      percent: 0,
    })
    try {
      const result = await parseEvalPackage(institution, singleZipFile, setSingleUploadProgress)
      setSingleParsed(result)
      setPatient(prefillToPatient(result.patient_prefill))
    } catch (err) {
      setSingleParsed(null)
      setError(`数据包解析失败：${String((err as Error).message || err)}`)
    } finally {
      setSingleParsing(false)
      setSingleUploadProgress(null)
    }
  }

  async function prepareSingle() {
    if (!patient.patient_id || !patient.diagnosis || !singleParsed) {
      setError('请解析单病例ZIP，并完整填写患者信息。')
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
    form.append('fma_wrist', scores.fma_wrist)
    form.append('fma_hand', scores.fma_hand)
    form.append('hand_mas', scores.hand_mas)
    form.append('brunnstrom_hand', scores.brunnstrom_hand)
    form.append('institution', institution)
    form.append('preset', preset)
    form.append('upload_id', singleParsed.upload_id)

    setBusy(true)
    setError(null)
    setSingleResult(null)
    try {
      setSingleResult(await prepareLlmBenchmarkSingle(form) as unknown as Record<string, unknown>)
    } catch (err) {
      setError(String((err as Error).message || err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div>
      <div className="page-head">
        <div>
          <h1 className="page-title">LLM 评测准备</h1>
          <p className="page-sub">固定医生真值、biomarker和实验开关；当前只准备输入，不启动真实模型。</p>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div className="card">
        <h2><ClipboardCheck aria-hidden="true" /> 实验Preset</h2>
        <div className="grid-2">
          <div className="field">
            <label>实验配置</label>
            <select value={preset} onChange={(event) => setPreset(event.target.value)} disabled={busy}>
              {Object.keys(presets?.presets || {
                benchmark_stage1: true,
                benchmark_stage2_baseline: true,
                benchmark_stage2_rag: true,
                benchmark_stage2_kg: true,
                benchmark_stage2_rag_kg: true,
              }).map((name) => <option key={name} value={name}>{name}</option>)}
            </select>
          </div>
          <div className="field">
            <label>资料包机构格式</label>
            <select
              value={institution}
              onChange={(event) => {
                setInstitution(event.target.value as 'hospital' | 'device')
                setSingleParsed(null)
              }}
              disabled={busy || singleParsing}
            >
              <option value="hospital">hospital</option>
              <option value="device">device</option>
            </select>
          </div>
        </div>
        <p className="muted">Stage 1默认：manual_clinical_scores、RAG OFF、KG OFF。后续只改变模型或指定消融Preset。</p>
      </div>

      <div className="card">
        <h2><Upload aria-hidden="true" /> 批量ZIP入口</h2>
        <p className="muted">ZIP中放置多个患者目录（每个目录含manifest.json和原始信号）以及一个clinical_scores.xlsx。</p>
        <div className="actions">
          <input type="file" accept=".zip" disabled={busy} onChange={(event) => setBatchFile(event.target.files?.[0] || null)} />
          <button className="button" onClick={prepareBatch} disabled={busy || !batchFile}>校验并准备批量输入</button>
        </div>
        {batchResult && (
          <pre className="code-block">{JSON.stringify(batchResult, null, 2)}</pre>
        )}
      </div>

      <div className="card">
        <h2>单病例Benchmark入口</h2>
        <p className="muted">上传与医院端评估相同的单个ZIP，系统按manifest自动匹配信号；临床评分仍由医生填写。</p>
        <div className="grid-2">
          <div className="field">
            <label>单病例评估数据包（.zip）</label>
            <input
              type="file"
              accept=".zip"
              disabled={busy || singleParsing}
              onChange={(event) => {
                setSingleZipFile(event.target.files?.[0] || null)
                setSingleParsed(null)
                setSingleUploadProgress(null)
                setError(null)
              }}
            />
          </div>
          <div className="actions">
            <button className="button secondary" onClick={parseSingleZip} disabled={busy || singleParsing || !singleZipFile}>
              {singleParsing ? '上传并校验中…' : '解析数据包'}
            </button>
          </div>
        </div>
        {singleParsing && singleUploadProgress && (
          <div className="package-upload-progress" role="status" aria-live="polite">
            <progress max={100} value={singleUploadProgress.percent} />
            <span>{singleUploadProgress.phase === 'uploading' ? `上传中 ${singleUploadProgress.percent}%` : '上传完成，正在校验数据包'}</span>
          </div>
        )}
        {singleParsed && (
          <p className="muted">已识别可用试次：<strong>{singleParsed.n_trials}</strong> 个</p>
        )}
        <PatientForm value={patient} onChange={setPatient} disabled={busy || singleParsing} />
        <div className="grid-2">
          <div className="field"><label>FMA腕（fma_wrist，非负数）</label><input type="number" min={0} value={scores.fma_wrist} onChange={(event) => setScores({ ...scores, fma_wrist: event.target.value })} /></div>
          <div className="field"><label>FMA手（fma_hand，0–20）</label><input type="number" min={0} max={20} value={scores.fma_hand} onChange={(event) => setScores({ ...scores, fma_hand: event.target.value })} /></div>
          <div className="field"><label>手部MAS（hand_mas）</label><select value={scores.hand_mas} onChange={(event) => setScores({ ...scores, hand_mas: event.target.value })}><option value="">请选择</option>{['0', '1', '1+', '2', '3', '4'].map((value) => <option key={value} value={value}>{value}级</option>)}</select></div>
          <div className="field"><label>Brunnstrom手功能（brunnstrom_hand）</label><select value={scores.brunnstrom_hand} onChange={(event) => setScores({ ...scores, brunnstrom_hand: event.target.value })}><option value="">请选择</option>{[1, 2, 3, 4, 5, 6].map((value) => <option key={value} value={value}>{value}期</option>)}</select></div>
        </div>
        <div className="actions">
          <button className="button" onClick={prepareSingle} disabled={busy || singleParsing || !singleParsed}>计算biomarker并准备Benchmark输入</button>
          <button className="button secondary" onClick={() => navigate('llm-control-test')} disabled={busy}>返回单病例调试模式</button>
        </div>
        {singleResult && <pre className="code-block">{JSON.stringify(singleResult, null, 2)}</pre>}
      </div>
    </div>
  )
}
