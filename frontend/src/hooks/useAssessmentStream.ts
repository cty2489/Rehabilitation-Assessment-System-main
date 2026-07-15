import { useCallback, useEffect, useRef, useState } from 'react'
import { cancelAssessment } from '../api'
import {
  BiomarkerCoverage,
  PredictionEntry,
  SSEEvent,
  StepKey,
  StepState,
  TaskKey,
} from '../types'

export type StreamPhase = 'idle' | 'processing' | 'done'

const STEP_DEFS: { key: StepKey; label: string }[] = [
  { key: 'parse', label: '文件解析与校验' },
  { key: 'preprocess', label: '信号预处理' },
  { key: 'alignment', label: '多模态时序对齐' },
  { key: 'feature_extract', label: '多尺度特征提取' },
  { key: 'graph_fusion', label: '跨模态图注意力融合' },
  { key: 'inference', label: '模型推理' },
  { key: 'report', label: 'AI 报告生成' },
]

export function freshSteps(): StepState[] {
  return STEP_DEFS.map((s) => ({ ...s, status: 'pending', details: [] }))
}

/**
 * Shared consumer for the `/api/assess/{id}/stream` SSE flow. Owns the steps /
 * predictions / report-text / coverage state and the EventSource lifecycle, so
 * both the 康复评估 page and the 任务一与任务三对接 page drive the exact same
 * backend pipeline without duplicating the event handling.
 */
export function useAssessmentStream() {
  const [phase, setPhase] = useState<StreamPhase>('idle')
  const [steps, setSteps] = useState<StepState[]>(freshSteps)
  const [results, setResults] = useState<Partial<Record<TaskKey, PredictionEntry>>>({})
  const [reportText, setReportText] = useState('')
  const [reportStreaming, setReportStreaming] = useState(false)
  const [queueAhead, setQueueAhead] = useState(0)
  const [coverage, setCoverage] = useState<BiomarkerCoverage | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const esRef = useRef<EventSource | null>(null)

  useEffect(() => () => esRef.current?.close(), [])

  const updateStep = useCallback((key: StepKey, mutator: (s: StepState) => StepState) => {
    setSteps((prev) => prev.map((s) => (s.key === key ? mutator(s) : s)))
  }, [])

  const handleEvent = useCallback(
    (event: SSEEvent) => {
      switch (event.type) {
        case 'step_start':
          updateStep(event.step, (s) => ({ ...s, status: 'running', label: event.label || s.label }))
          setQueueAhead(0)
          if (event.step === 'report') {
            setReportStreaming(true)
          }
          break
        case 'assessment_queued':
        case 'report_queued':
          setQueueAhead(event.ahead)
          break
        case 'step_detail':
          updateStep(event.step, (s) => ({ ...s, details: [...s.details, event.detail] }))
          break
        case 'step_done':
          updateStep(event.step, (s) => ({ ...s, status: 'done' }))
          if (event.step === 'report') setReportStreaming(false)
          break
        case 'prediction':
          setResults((prev) => ({
            ...prev,
            [event.task]: { task: event.task, label: event.label, value: event.value, range: event.range },
          }))
          break
        case 'biomarker_coverage':
          setCoverage({ available: event.available, total: event.total, missing_keys: event.missing_keys })
          break
        case 'report_chunk':
          setReportText((prev) => prev + event.chunk)
          break
        case 'done':
          setPhase('done')
          setReportStreaming(false)
          esRef.current?.close()
          break
        case 'cancelled':
          setError(event.message || '评估任务已取消')
          setPhase('idle')
          setReportStreaming(false)
          esRef.current?.close()
          break
        case 'error':
          setError(event.message)
          break
      }
    },
    [updateStep],
  )

  /** Reset transient state and begin consuming the stream for a session id. */
  const start = useCallback(
    (newSessionId: string) => {
      setSteps(freshSteps())
      setResults({})
      setReportText('')
      setReportStreaming(false)
      setQueueAhead(0)
      setCoverage(null)
      setError(null)
      setSessionId(newSessionId)
      setPhase('processing')

      const es = new EventSource(`/api/assess/${newSessionId}/stream`)
      esRef.current = es
      es.onmessage = (e) => {
        try {
          handleEvent(JSON.parse(e.data) as SSEEvent)
        } catch (parseErr) {
          console.error('SSE parse error', parseErr, e.data)
        }
      }
      es.onerror = () => {
        // EventSource auto-reconnects; leave UI as-is during processing.
      }
    },
    [handleEvent],
  )

  const reset = useCallback(() => {
    if (phase === 'processing' && sessionId) void cancelAssessment(sessionId)
    esRef.current?.close()
    esRef.current = null
    setPhase('idle')
    setSteps(freshSteps())
    setResults({})
    setReportText('')
    setReportStreaming(false)
    setQueueAhead(0)
    setCoverage(null)
    setError(null)
    setSessionId(null)
  }, [phase, sessionId])

  return {
    phase,
    steps,
    results,
    reportText,
    reportStreaming,
    queueAhead,
    coverage,
    error,
    sessionId,
    setError,
    start,
    reset,
  }
}
