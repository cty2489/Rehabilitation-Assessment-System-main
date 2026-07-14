import { PredictionEntry, TaskKey } from '../types'

interface Props {
  results: Partial<Record<TaskKey, PredictionEntry>>
}

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

export default function ResultsPanel({ results }: Props) {
  const entries: TaskKey[] = ['FMA_UE', 'hand_tone', 'hand_function']
  const visible = entries.filter((k) => results[k] !== undefined)
  if (visible.length === 0) return null

  return (
    <div className="card">
      <h2>
        评估结果
        <span className="h2-suffix">Clinical · Scores</span>
      </h2>
      <div className="results-grid">
        {visible.map((k) => (
          <ResultCard key={k} entry={results[k]!} />
        ))}
      </div>
    </div>
  )
}

function ResultCard({ entry }: { entry: PredictionEntry }) {
  if (entry.task === 'FMA_UE') {
    const v = typeof entry.value === 'number' ? entry.value : parseFloat(String(entry.value))
    return (
      <div className="result-card">
        <div className="label">{entry.label}</div>
        <div className="value">
          {v.toFixed(0)}
          <span className="unit">/ 20 分</span>
        </div>
        <div className="progress-bar">
          <div style={{ width: `${(v / 20) * 100}%` }} />
        </div>
      </div>
    )
  }
  if (entry.task === 'hand_tone') {
    const v = String(entry.value)
    return (
      <div className="result-card">
        <div className="label">手部肌张力 · Hand MAS（Modified Ashworth）</div>
        <div className="value">
          {v}<span className="unit">级</span>
        </div>
        <div className="meta">{HAND_TONE_DESC[v] || '—'}</div>
      </div>
    )
  }
  // hand_function
  const v = typeof entry.value === 'number' ? entry.value : parseInt(String(entry.value), 10)
  return (
    <div className="result-card">
      <div className="label">手功能 · Brunnstrom 分期</div>
      <div className="value">
        Brunnstrom {v}<span className="unit">期</span>
      </div>
      <div className="meta">{BRUNNSTROM_DESC[v] || '—'}</div>
    </div>
  )
}
