import { useEffect, useRef, useState } from 'react'
import { CheckCircle2, Download, FileArchive, FileJson, FileText, LoaderCircle, RefreshCw, ShieldCheck, Trash2, UploadCloud, XCircle } from 'lucide-react'
import {
  downloadStrategyReport,
  fetchStrategyReportCapabilities,
  fetchStrategyReportJobs,
  generateStrategyReport,
  type PackageUploadProgress,
  type StrategyReportCapabilities,
  type StrategyReportDownloadKind,
  type StrategyReportJob,
} from '../api'

type RowStatus = 'queued' | 'uploading' | 'generating' | 'success' | 'review' | 'error'

interface UploadRow {
  id: string
  file?: File
  fileName: string
  fileSize: number
  status: RowStatus
  progress: number
  result?: StrategyReportJob
  error?: string
  downloading?: StrategyReportDownloadKind
}

const formatBytes = (bytes: number) => {
  if (!bytes) return ''
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`
}

const rowId = (file: File) => `${file.name}-${file.size}-${file.lastModified}-${crypto.randomUUID()}`

const statusForJob = (job: StrategyReportJob): RowStatus => {
  if (job.processing_status === 'QUEUED' || job.processing_status === 'RUNNING') return 'generating'
  if (job.processing_status === 'FAILED') return 'error'
  return job.formal_report_created ? 'success' : 'review'
}

export default function StrategyReportPage() {
  const [capabilities, setCapabilities] = useState<StrategyReportCapabilities | null>(null)
  const [capabilityError, setCapabilityError] = useState('')
  const [historyError, setHistoryError] = useState('')
  const [rows, setRows] = useState<UploadRow[]>([])
  const [running, setRunning] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const hiddenJobsRef = useRef(new Set<string>())

  useEffect(() => {
    fetchStrategyReportCapabilities()
      .then(setCapabilities)
      .catch((error: unknown) => setCapabilityError(error instanceof Error ? error.message : '接口状态读取失败'))
  }, [])

  const mergeRemoteJobs = (jobs: StrategyReportJob[]) => {
    setRows((current) => {
      const next = [...current]
      for (const job of jobs) {
        if (hiddenJobsRef.current.has(job.job_id)) continue
        let index = next.findIndex((row) => row.id === job.job_id || row.result?.job_id === job.job_id)
        if (index < 0) {
          index = next.findIndex((row) => (
            !row.result
            && row.fileName === job.source_upload_name
            && row.fileSize === job.source_upload_size
            && (row.status === 'uploading' || row.status === 'generating' || row.status === 'error')
          ))
        }
        const existing = index >= 0 ? next[index] : undefined
        const patch: UploadRow = {
          ...(existing || {
            id: job.job_id,
            fileName: job.source_upload_name || '历史报告任务',
            fileSize: job.source_upload_size || 0,
            status: 'generating' as const,
            progress: 0,
          }),
          id: job.job_id,
          fileName: job.source_upload_name || existing?.fileName || '历史报告任务',
          fileSize: job.source_upload_size || existing?.fileSize || 0,
          status: statusForJob(job),
          progress: job.progress_percent,
          result: job,
          error: job.processing_status === 'FAILED' ? (job.error_message || '报告生成失败，请查看人工复核信息') : undefined,
        }
        if (index >= 0) next[index] = patch
        else next.push(patch)
      }
      return next
    })
  }

  const refreshJobs = async (showBusy = false) => {
    if (showBusy) setRefreshing(true)
    try {
      const response = await fetchStrategyReportJobs(50)
      mergeRemoteJobs(response.jobs)
      setHistoryError('')
    } catch (error) {
      setHistoryError(error instanceof Error ? error.message : '任务状态读取失败')
    } finally {
      if (showBusy) setRefreshing(false)
    }
  }

  useEffect(() => {
    let active = true
    const poll = async () => {
      if (!active) return
      try {
        const response = await fetchStrategyReportJobs(50)
        if (!active) return
        mergeRemoteJobs(response.jobs)
        setHistoryError('')
      } catch (error) {
        if (active) setHistoryError(error instanceof Error ? error.message : '任务状态读取失败')
      }
    }
    void poll()
    const timer = window.setInterval(() => void poll(), 3000)
    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [])

  const addFiles = (files: FileList | null) => {
    if (!files) return
    const accepted = Array.from(files).filter((file) => file.name.toLowerCase().endsWith('.zip'))
    setRows((current) => [
      ...current,
      ...accepted.map((file) => ({
        id: rowId(file),
        file,
        fileName: file.name,
        fileSize: file.size,
        status: 'queued' as const,
        progress: 0,
      })),
    ])
    if (inputRef.current) inputRef.current.value = ''
  }

  const patchRow = (id: string, patch: Partial<UploadRow>) => {
    setRows((current) => current.map((row) => row.id === id ? { ...row, ...patch } : row))
  }

  const run = async () => {
    const pending = rows.filter((row) => row.file && (row.status === 'queued' || row.status === 'error'))
    if (!pending.length || running) return
    setRunning(true)
    for (const row of pending) {
      if (!row.file) continue
      patchRow(row.id, { status: 'uploading', progress: 0, error: undefined, result: undefined })
      try {
        const result = await generateStrategyReport(row.file, (progress: PackageUploadProgress) => {
          patchRow(row.id, {
            status: progress.phase === 'uploading' ? 'uploading' : 'generating',
            progress: progress.phase === 'uploading' ? progress.percent : 5,
          })
        })
        patchRow(row.id, {
          id: result.job_id,
          status: statusForJob(result),
          progress: result.progress_percent,
          result,
        })
      } catch (error) {
        patchRow(row.id, {
          status: 'error',
          progress: 0,
          error: error instanceof Error ? error.message : '报告任务提交失败',
        })
        void refreshJobs()
      }
    }
    setRunning(false)
  }

  const download = async (row: UploadRow, kind: StrategyReportDownloadKind) => {
    if (!row.result) return
    patchRow(row.id, { downloading: kind })
    try {
      await downloadStrategyReport(row.result, kind)
    } catch (error) {
      patchRow(row.id, { error: error instanceof Error ? error.message : '下载失败' })
    } finally {
      patchRow(row.id, { downloading: undefined })
    }
  }

  const removeRow = (row: UploadRow) => {
    if (row.result?.job_id) hiddenJobsRef.current.add(row.result.job_id)
    setRows((current) => current.filter((item) => item.id !== row.id))
  }

  const clearRows = () => {
    rows.forEach((row) => {
      if (row.result?.job_id) hiddenJobsRef.current.add(row.result.job_id)
    })
    setRows([])
  }

  const pendingCount = rows.filter((row) => row.file && (row.status === 'queued' || row.status === 'error')).length
  const successCount = rows.filter((row) => row.status === 'success').length
  const activeCount = rows.filter((row) => row.status === 'uploading' || row.status === 'generating').length

   return (
    <div className="strategy-report-page">
      <div className="page-head">
        <div>
          <h1 className="page-title">康复训练策略报告生成</h1>
          <p className="page-sub">上传完成后由服务器后台生成；关闭或刷新页面不会中断任务，重新进入后可自动恢复结果。</p>
        </div>
        <div className="strategy-template-badge"><ShieldCheck />审定模板 v11</div>
      </div>

      <section className="strategy-boundary" aria-label="生成规则">
        <ShieldCheck aria-hidden="true" />
        <div>
          <strong>真实解析、后台生成、通过发布门后才可下载</strong>
          <p>
            系统会安全解析本次 ZIP 的 manifest 与 EEG/EMG/IMU 原始数据，实际运行 biomarker、评分、QualityGate、planner_rag、Validator 和 publication gate。
            上传被服务器接收后会立即生成任务编号，网页每3秒查询进度；不需要保持同一条长连接。
          </p>
        </div>
      </section>

      <section className="card strategy-upload-card">
        <h2>选择病人数据包 <span className="h2-suffix">支持多选 ZIP，系统逐份提交后台任务</span></h2>
        <label className="strategy-drop-zone">
          <UploadCloud aria-hidden="true" />
          <span><strong>点击选择一个或多个 ZIP 文件</strong><small>上传成功后可以离开页面，报告会继续生成。</small></span>
          <input ref={inputRef} type="file" accept=".zip,application/zip" multiple onChange={(event) => addFiles(event.target.files)} disabled={running} />
        </label>
        <div className="strategy-toolbar">
          <span>
            {capabilityError
              ? `接口状态：${capabilityError}`
              : capabilities
                ? `异步真实流水线 · 模板 ${capabilities.template_version} · 支持刷新恢复 · 输出 PDF/JSON/Markdown/ZIP`
                : '正在读取生成接口状态…'}
          </span>
          <div>
            <button className="button secondary" type="button" disabled={refreshing} onClick={() => void refreshJobs(true)}>
              <RefreshCw className={refreshing ? 'spin' : ''} />刷新任务
            </button>
            <button className="button secondary" type="button" disabled={running || !rows.length} onClick={clearRows}>清空列表</button>
            <button className="button primary" type="button" disabled={running || !pendingCount || !!capabilityError} onClick={run}>
              {running ? <><LoaderCircle className="spin" />正在上传</> : <>开始生成{pendingCount ? `（${pendingCount}份）` : ''}</>}
            </button>
          </div>
        </div>
      </section>

      <section className="card strategy-results-card">
        <h2>生成清单 <span className="h2-suffix">已完成 {successCount}，处理中 {activeCount}</span></h2>
        {historyError && <p className="strategy-error">任务状态暂时无法刷新：{historyError}</p>}
        {!rows.length && <div className="strategy-empty"><FileArchive />尚未选择数据包，也没有可恢复的近期任务</div>}
        <div className="strategy-file-list">
          {rows.map((row) => (
            <article className={`strategy-file-row status-${row.status}`} key={row.id}>
              <div className="strategy-file-status" aria-hidden="true">
                {row.status === 'success' ? <CheckCircle2 /> : row.status === 'review' || row.status === 'error' ? <XCircle /> : row.status === 'queued' ? <FileArchive /> : <LoaderCircle className="spin" />}
              </div>
              <div className="strategy-file-main">
                <div className="strategy-file-title"><strong>{row.fileName}</strong><span>{formatBytes(row.fileSize)}</span></div>
                {(row.status === 'uploading' || row.status === 'generating') && (
                  <div className="strategy-progress">
                    <span style={{ width: `${row.progress}%` }} />
                    <small>{row.status === 'uploading' ? `正在上传 ${row.progress}%` : row.result?.current_stage || '任务已提交，等待服务器处理'}</small>
                  </div>
                )}
                {row.status === 'queued' && <p className="muted">等待上传</p>}
                {row.status === 'error' && <p className="strategy-error">{row.error}</p>}
                {row.status === 'review' && row.result && (
                  <p className="strategy-error">
                    MANUAL_REVIEW：已保存内部结果，但发布门禁止生成或下载患者正式报告。
                    {row.result.publication_warnings?.length ? ` ${row.result.publication_warnings.join('；')}` : ''}
                  </p>
                )}
                {row.status === 'success' && row.result && (
                  <div className="strategy-success-meta">
                    <span>{row.result.report_number}</span>
                    <span>{row.result.publication_status}</span>
                    <span>{row.result.clinical_score_mode === 'doctor_clinical_score' ? '医生临床评分' : 'DL评分'}</span>
                    <span>{row.result.strategy_count}条策略</span>
                    {row.result.biomarker_coverage && <span>{row.result.biomarker_coverage.available}/{row.result.biomarker_coverage.total}项指标</span>}
                  </div>
                )}
              </div>
              <div className="strategy-file-actions">
                {row.status === 'success' && row.result?.formal_report_created ? (
                  <>
                    <button className="button primary" onClick={() => download(row, 'pdf')} disabled={!!row.downloading}><FileText />PDF</button>
                    <button className="button secondary" onClick={() => download(row, 'json')} disabled={!!row.downloading}><FileJson />JSON</button>
                    <button className="button secondary" onClick={() => download(row, 'zip')} disabled={!!row.downloading}><Download />全部</button>
                  </>
                ) : (
                  <button className="icon-btn" title="从当前列表移除" disabled={row.status === 'uploading'} onClick={() => removeRow(row)}><Trash2 /></button>
                )}
              </div>
            </article>
          ))}
        </div>
      </section>
    </div>
  )
}
