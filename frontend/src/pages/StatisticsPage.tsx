import { useEffect, useState } from 'react'
import { fetchStats } from '../api'
import { StatsSummary } from '../types'
import { BarChart } from './DashboardPage'

const HAND_FN_LABEL: Record<string, string> = {
  '1': 'Brunnstrom 1 期',
  '2': 'Brunnstrom 2 期',
  '3': 'Brunnstrom 3 期',
  '4': 'Brunnstrom 4 期',
  '5': 'Brunnstrom 5 期',
  '6': 'Brunnstrom 6 期',
}

export default function StatisticsPage() {
  const [stats, setStats] = useState<StatsSummary | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchStats().then(setStats).catch((e) => setError(String(e.message || e)))
  }, [])

  const handFn = stats
    ? Object.fromEntries(
        Object.entries(stats.hand_function_distribution).map(([k, v]) => [
          HAND_FN_LABEL[k] || k,
          v,
        ]),
      )
    : {}

  const byDay = stats
    ? Object.fromEntries(stats.assessments_by_day.map((d) => [d.date.slice(5), d.count]))
    : {}

  return (
    <div>
      <div className="page-head">
        <div>
          <h1 className="page-title">统计分析</h1>
          <p className="page-sub">基于全部评估记录的指标分布与趋势</p>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div className="stat-row">
        <div className="stat-card"><div className="stat-value">{stats?.patient_count ?? '—'}</div><div className="stat-label">患者总数</div></div>
        <div className="stat-card"><div className="stat-value">{stats?.assessment_count ?? '—'}</div><div className="stat-label">评估次数</div></div>
        <div className="stat-card"><div className="stat-value">{stats?.avg_fma_ue ?? '—'}</div><div className="stat-label">平均 FMA-UE</div></div>
        <div className="stat-card"><div className="stat-value">{stats?.avg_bi ?? '—'}</div><div className="stat-label">平均 Barthel</div></div>
      </div>

      <div className="grid-2-cards">
        <div className="card">
          <h2>诊断分布<span className="h2-suffix">Diagnosis</span></h2>
          {stats && Object.keys(stats.diagnosis_distribution).length ? (
            <BarChart data={stats.diagnosis_distribution} />
          ) : <p className="muted">暂无数据</p>}
        </div>
        <div className="card">
          <h2>手功能分期分布<span className="h2-suffix">Brunnstrom</span></h2>
          {Object.keys(handFn).length ? <BarChart data={handFn} /> : <p className="muted">暂无数据</p>}
        </div>
      </div>

      <div className="card">
        <h2>评估量趋势（按日）<span className="h2-suffix">Volume</span></h2>
        {Object.keys(byDay).length ? <BarChart data={byDay} /> : <p className="muted">暂无数据</p>}
      </div>
    </div>
  )
}
