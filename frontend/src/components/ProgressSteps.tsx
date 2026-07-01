import { useEffect, useRef } from 'react'
import { StepState } from '../types'

interface Props {
  steps: StepState[]
}

export default function ProgressSteps({ steps }: Props) {
  return (
    <div className="card">
      <h2>
        处理进度
        <span className="h2-suffix">Pipeline · Realtime</span>
      </h2>
      <div className="progress-steps">
        {steps.map((s, idx) => (
          <StepRow key={s.key} step={s} index={idx} />
        ))}
      </div>
    </div>
  )
}

const STATUS_LABEL: Record<string, string> = {
  pending: 'Pending',
  running: 'Running',
  done: 'Complete',
}

function StepRow({ step, index }: { step: StepState; index: number }) {
  const detailsRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (detailsRef.current) {
      detailsRef.current.scrollTop = detailsRef.current.scrollHeight
    }
  }, [step.details.length])

  return (
    <div className={`step-row ${step.status}`}>
      <span className="step-bullet">
        {step.status === 'pending' ? index + 1 : ''}
      </span>
      <div className="step-content">
        <div className="step-label">
          <span className="step-num">STEP {String(index + 1).padStart(2, '0')}</span>
          <span>{step.label}</span>
          <span className="step-status-chip">{STATUS_LABEL[step.status] || step.status}</span>
        </div>
        {step.details.length > 0 && (
          <div className="step-details" ref={detailsRef}>
            {step.details.map((d, i) => (
              <div key={i}>{d}</div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
