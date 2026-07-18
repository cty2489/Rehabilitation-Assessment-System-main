import { useEffect, useState } from 'react'
import {
  ClipboardCheck,
  FileWarning,
  Hand,
  Plus,
  UsersRound,
  type LucideIcon,
} from 'lucide-react'
import { fetchAssessments, fetchStats } from '../api'
import { useAuth, useRoute } from '../app/AppContext'
import { AssessmentOverviewItem, StatsSummary } from '../types'
import { fmtDateTime } from '../util'

export default function DashboardPage() {
  const { user } = useAuth()
  const { navigate } = useRoute()
  const [stats, setStats] = useState<StatsSummary | null>(null)
  const [recent, setRecent] = useState<AssessmentOverviewItem[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchStats().then(setStats).catch((e) => setError(String(e.message || e)))
    fetchAssessments(8, 0).then((r) => setRecent(r.items)).catch(() => {})
  }, [])

  return (
    <div>
      <div className="page-head">
        <div>
          <h1 className="page-title">欢迎，{user}</h1>
          <p className="page-sub">智能康复评估平台 · 总览</p>
        </div>
        <button className="button" onClick={() => navigate('assessment')}>
          <Plus aria-hidden="true" />
          开始新评估
        </button>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div className="stat-row">
        <StatCard label="患者总数" value={stats?.patient_count ?? '—'} icon={UsersRound} />
        <StatCard label="评估总次数" value={stats?.assessment_count ?? '—'} icon={ClipboardCheck} tone="blue" />
        <StatCard label="报告失败数" value={stats?.report_failed_count ?? '—'} icon={FileWarning} tone="warn" />
        <StatCard label="平均 FMA-UE" value={stats?.avg_fma_ue ?? '—'} icon={Hand} tone="green" />
      </div>

      <div className="grid-2-cards">
        <div className="card">
          <h2>诊断分布<span className="h2-suffix">Diagnosis</span></h2>
          {stats && Object.keys(stats.diagnosis_distribution).length > 0 ? (
            <BarChart data={stats.diagnosis_distribution} />
          ) : (
            <p className="muted">暂无数据</p>
          )}
        </div>

        <div className="card">
          <h2>近期评估<span className="h2-suffix">Recent</span></h2>
          {recent.length === 0 ? (
            <p className="muted">暂无评估记录</p>
          ) : (
            <table className="data-table compact">
              <thead>
                <tr><th>时间</th><th>患者</th><th>FMA</th><th>Brunnstrom</th></tr>
              </thead>
              <tbody>
                {recent.map((r) => (
                  <tr
                    key={r.id}
                    className="clickable"
                    onClick={() => navigate('patients', r.patient_db_id)}
                  >
                    <td>{fmtDateTime(r.created_at)}</td>
                    <td>{r.name}</td>
                    <td>{Math.round(r.fma_ue)}</td>
                    <td>{r.hand_function} 期</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  )
}

export function StatCard({
  label,
  value,
  icon: Icon,
  tone,
}: {
  label: string
  value: React.ReactNode
  icon: LucideIcon
  tone?: 'blue' | 'green' | 'warn'
}) {
  return (
    <div className={`stat-card ${tone || ''}`}>
      <div className="stat-card-icon" aria-hidden="true"><Icon /></div>
      <div>
        <div className="stat-value">{value}</div>
        <div className="stat-label">{label}</div>
      </div>
    </div>
  )
}

export function BarChart({ data }: { data: Record<string, number> }) {
  const max = Math.max(1, ...Object.values(data))
  return (
    <div className="bar-chart">
      {Object.entries(data).map(([k, v]) => (
        <div key={k} className="bar-row">
          <span className="bar-label">{k}</span>
          <span className="bar-track">
            <span className="bar-fill" style={{ width: `${(v / max) * 100}%` }} />
          </span>
          <span className="bar-value">{v}</span>
        </div>
      ))}
    </div>
  )
}
