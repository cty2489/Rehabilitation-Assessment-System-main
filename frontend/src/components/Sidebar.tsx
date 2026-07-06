import { useRoute } from '../app/AppContext'
import { Route } from '../types'

const NAV: { route: Route; label: string; icon: string }[] = [
  { route: 'dashboard', label: '仪表盘', icon: '▦' },
  { route: 'patients', label: '患者管理', icon: '👤' },
  { route: 'assessment', label: '康复评估', icon: '✚' },
  { route: 'records', label: '评估记录总览', icon: '🗂' },
  { route: 'stats', label: '统计分析', icon: '📊' },
  { route: 'task-interface', label: '任务一与任务三对接接口页面', icon: '🔗' },
  { route: 'llm-settings', label: '模型设置', icon: '◇' },
  { route: 'system', label: '系统管理', icon: '⚙' },
]

export default function Sidebar() {
  const { route, navigate } = useRoute()

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="brand-logo" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="M2 12h3l2 -6 3 12 3 -8 2 4h7" />
          </svg>
        </div>
        <span>智能康复评估</span>
      </div>
      <nav className="sidebar-nav">
        {NAV.map((item) => (
          <button
            key={item.route}
            className={`sidebar-item ${route === item.route ? 'active' : ''}`}
            onClick={() => navigate(item.route)}
          >
            <span className="sidebar-icon" aria-hidden="true">
              {item.icon}
            </span>
            {item.label}
          </button>
        ))}
      </nav>
    </aside>
  )
}
