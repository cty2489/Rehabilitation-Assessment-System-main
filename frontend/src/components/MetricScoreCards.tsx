import type { ReactNode } from 'react'

export const HAND_TONE_DESC: Record<string, string> = {
  '0': '未见肌张力增高',
  '1': '轻度增高',
  '1+': '轻中度增高',
  '2': '中度增高',
  '3': '重度增高',
  '4': '强直状态',
}

export const BRUNNSTROM_DESC: Record<number, string> = {
  1: '弛缓期，无主动运动',
  2: '联合反应出现',
  3: '可引出共同运动',
  4: '部分分离运动',
  5: '分离运动明显',
  6: '接近正常',
}

interface MetricScoreCardsProps {
  fmaUe?: number | null
  handTone?: string | null
  handFunction?: number | null
}

function validNumber(value: number | null | undefined): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function clampFma(value: number): number {
  return Math.min(20, Math.max(0, value))
}

export default function MetricScoreCards({
  fmaUe,
  handTone,
  handFunction,
}: MetricScoreCardsProps) {
  const cards: ReactNode[] = []

  if (validNumber(fmaUe)) {
    const score = Math.round(fmaUe)
    cards.push(
      <div className="result-card" key="fma-ue">
        <div className="label">FMA-UE 手部分数</div>
        <div className="value">
          {score}
          <span className="unit">/ 20 分</span>
        </div>
        <div className="progress-bar">
          <div style={{ width: `${(clampFma(fmaUe) / 20) * 100}%` }} />
        </div>
      </div>,
    )
  }

  if (handTone != null && String(handTone).trim()) {
    const value = String(handTone)
    cards.push(
      <div className="result-card" key="hand-tone">
        <div className="label">手部肌张力 · Hand MAS（Modified Ashworth）</div>
        <div className="value">
          {value}
          <span className="unit">级</span>
        </div>
        <div className="meta">{HAND_TONE_DESC[value] || '—'}</div>
      </div>,
    )
  }

  if (validNumber(handFunction)) {
    cards.push(
      <div className="result-card" key="hand-function">
        <div className="label">手功能 · Brunnstrom 分期</div>
        <div className="value">
          Brunnstrom {handFunction}
          <span className="unit">期</span>
        </div>
        <div className="meta">{BRUNNSTROM_DESC[handFunction] || '—'}</div>
      </div>,
    )
  }

  if (cards.length === 0) return null
  return <div className="results-grid">{cards}</div>
}
