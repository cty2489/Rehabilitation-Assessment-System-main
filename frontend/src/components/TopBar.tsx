import { useEffect, useRef, useState } from 'react'
import { Activity, BrainCircuit, ChevronDown, LogOut, Settings } from 'lucide-react'
import { useAuth, useRoute } from '../app/AppContext'
import { Route } from '../types'

const ROUTE_LABEL: Record<Route, string> = {
  dashboard: '工作台总览',
  patients: '患者管理',
  assessment: '康复评估',
  records: '评估记录',
  stats: '统计分析',
  'task-interface': '设备接口',
  knowledge: 'RAG 知识库',
  'rag-guidelines': '知识与研究证据',
  'rag-guidelines-test': '知识与研究证据',
  'llm-settings': '模型设置',
  system: '系统管理',
}

export default function TopBar() {
  const { user, logout } = useAuth()
  const { route, navigate } = useRoute()
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement | null>(null)

  // Close the dropdown on outside click.
  useEffect(() => {
    if (!open) return
    const onClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onClick)
    return () => document.removeEventListener('mousedown', onClick)
  }, [open])

  const initial = (user || '医').slice(0, 1)

  return (
    <header className="appbar">
      <div className="appbar-context">
        <Activity aria-hidden="true" />
        <span>{ROUTE_LABEL[route]}</span>
      </div>
      <div className="appbar-spacer" />
      <div className="topbar-user" ref={ref}>
        <button
          className="user-trigger"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          aria-haspopup="menu"
        >
          <span className="user-avatar">{initial}</span>
          <span className="user-name">{user}</span>
          <ChevronDown className="user-caret" aria-hidden="true" />
        </button>
        {open && (
          <div className="user-menu" role="menu">
            <button
              className="user-menu-item"
              role="menuitem"
              onClick={() => {
                setOpen(false)
                navigate('system')
              }}
            >
              <Settings aria-hidden="true" />
              系统管理
            </button>
            <button
              className="user-menu-item"
              role="menuitem"
              onClick={() => {
                setOpen(false)
                navigate('llm-settings')
              }}
            >
              <BrainCircuit aria-hidden="true" />
              模型设置
            </button>
            <div className="user-menu-sep" />
            <button className="user-menu-item danger" role="menuitem" onClick={logout}>
              <LogOut aria-hidden="true" />
              退出登录
            </button>
          </div>
        )}
      </div>
    </header>
  )
}
