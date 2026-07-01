import { AssessmentRecord } from '../types'
import MarkdownReport from './MarkdownReport'

const HAND_TONE_DESC: Record<string, string> = {
  '0': '正常张力',
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

// Renders the 4 indicators + Chinese report for one persisted assessment record.
// Reuses the .result-card / .results-grid styling from the assessment page.
export default function RecordDetail({ record }: { record: AssessmentRecord }) {
  return (
    <div className="record-detail">
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
          <div className="label">Barthel 指数</div>
          <div className="value">
            {Math.round(record.bi)}
            <span className="unit">/ 100 分</span>
          </div>
          <div className="progress-bar">
            <div style={{ width: `${record.bi}%` }} />
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
