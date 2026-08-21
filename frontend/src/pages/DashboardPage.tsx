import { useEffect, useState } from 'react'
import { ClipboardPlus, FileWarning, FolderOpen, ListChecks, Plus, type LucideIcon } from 'lucide-react'
import { fetchStats } from '../api'
import { useAuth, useRoute } from '../app/AppContext'
import { StatsSummary } from '../types'

export default function DashboardPage() {
  const { user } = useAuth()
  const { navigate } = useRoute()
  const [stats, setStats] = useState<StatsSummary | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchStats().then(setStats).catch((e) => setError(String(e.message || e)))
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

      <div className="dashboard-workspace">
        <div className="card dashboard-work-card">
          <h2>待处理事项<span className="h2-suffix">Work Queue</span></h2>
          <div className="dashboard-queue-item">
            <FileWarning aria-hidden="true" />
            <div>
              <strong>{stats?.report_failed_count ?? '—'} 份报告需要复核</strong>
              <p>进入评估记录查看报告状态和具体原因。</p>
            </div>
            <button className="button secondary" onClick={() => navigate('records')}>打开记录</button>
          </div>
        </div>
        <div className="card dashboard-shortcuts-card">
          <h2>常用入口<span className="h2-suffix">Shortcuts</span></h2>
          <div className="dashboard-shortcuts">
            <button className="dashboard-shortcut" onClick={() => navigate('assessment')}>
              <ClipboardPlus aria-hidden="true" /><span>开始新评估</span>
            </button>
            <button className="dashboard-shortcut" onClick={() => navigate('patients')}>
              <FolderOpen aria-hidden="true" /><span>打开患者档案</span>
            </button>
            <button className="dashboard-shortcut" onClick={() => navigate('stats')}>
              <ListChecks aria-hidden="true" /><span>进入统计分析</span>
            </button>
          </div>
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
