import MetricScoreCards from './MetricScoreCards'
import { PredictionEntry, TaskKey } from '../types'

interface Props {
  results: Partial<Record<TaskKey, PredictionEntry>>
}

export default function ResultsPanel({ results }: Props) {
  const entries: TaskKey[] = ['FMA_UE', 'hand_tone', 'hand_function']
  const visible = entries.filter((k) => results[k] !== undefined)
  if (visible.length === 0) return null

  const fma = results.FMA_UE
  const handTone = results.hand_tone
  const handFunction = results.hand_function
  const fmaValue = fma == null ? undefined : Number(fma.value)
  const handFunctionValue = handFunction == null ? undefined : Number(handFunction.value)

  return (
    <div className="card">
      <h2>
        评估结果
        <span className="h2-suffix">Clinical · Scores</span>
      </h2>
      <MetricScoreCards
        fmaUe={Number.isFinite(fmaValue) ? fmaValue : undefined}
        handTone={handTone == null ? undefined : String(handTone.value)}
        handFunction={Number.isFinite(handFunctionValue) ? handFunctionValue : undefined}
      />
    </div>
  )
}
