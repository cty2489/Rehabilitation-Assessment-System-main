import {
  Activity,
  BrainCircuit,
  Cable,
  ChartNoAxesCombined,
  ClipboardPlus,
  ClipboardCheck,
  Files,
  FileOutput,
  FlaskConical,
  LayoutDashboard,
  LibraryBig,
  Network,
  Settings,
  UsersRound,
  type LucideIcon,
} from 'lucide-react'
import { useRoute } from '../app/AppContext'
import { Route } from '../types'

const NAV: { route: Route; label: string; icon: LucideIcon; group?: 'system' }[] = [
  { route: 'dashboard', label: '仪表盘', icon: LayoutDashboard },
  { route: 'patients', label: '患者管理', icon: UsersRound },
  { route: 'assessment', label: '康复评估', icon: ClipboardPlus },
  { route: 'strategy-reports', label: '训练策略报告', icon: FileOutput },
  { route: 'records', label: '评估记录', icon: Files },
  { route: 'stats', label: '统计分析', icon: ChartNoAxesCombined },
  { route: 'task-interface', label: '设备接口', icon: Cable },
  { route: 'knowledge', label: 'RAG 知识库', icon: LibraryBig },
  { route: 'knowledge-graph', label: '知识结构（管理员）', icon: Network, group: 'system' },
  { route: 'llm-settings', label: '模型设置', icon: BrainCircuit, group: 'system' },
  { route: 'llm-control-test', label: 'LLM 对照测试', icon: FlaskConical, group: 'system' },
  { route: 'llm-benchmark', label: 'LLM 评测准备', icon: ClipboardCheck, group: 'system' },
  { route: 'system', label: '系统管理', icon: Settings },
]

export default function Sidebar() {
  const { route, navigate } = useRoute()

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="brand-logo" aria-hidden="true">
          <Activity />
        </div>
        <span className="sidebar-brand-copy">
          <strong>智能康复评估</strong>
          <small>Clinical OS</small>
        </span>
      </div>
      <nav className="sidebar-nav" aria-label="主导航">
        {NAV.map((item, index) => {
          const Icon = item.icon
          return (
            <div key={item.route} className={item.group && index > 0 ? 'sidebar-system-group' : undefined}>
              {item.group && NAV[index - 1]?.group !== item.group && <span className="sidebar-group-label">平台配置</span>}
              <button
                className={`sidebar-item ${route === item.route ? 'active' : ''}`}
                onClick={() => navigate(item.route)}
                title={item.label}
                aria-current={route === item.route ? 'page' : undefined}
              >
                <span className="sidebar-icon" aria-hidden="true">
                  <Icon />
                </span>
                <span className="sidebar-item-label">{item.label}</span>
              </button>
            </div>
          )
        })}
      </nav>
      <div className="sidebar-foot">多模态康复评估</div>
    </aside>
  )
}
