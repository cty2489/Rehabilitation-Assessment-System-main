import { useEffect, useRef, useState } from 'react'
import { useAuth, useRoute } from '../app/AppContext'

export default function TopBar() {
  const { user, logout } = useAuth()
  const { navigate } = useRoute()
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
      <div className="appbar-spacer" />
      <div className="topbar-user" ref={ref}>
        <button className="user-trigger" onClick={() => setOpen((v) => !v)}>
          <span className="user-avatar">{initial}</span>
          <span className="user-name">{user}</span>
          <span className="user-caret" aria-hidden="true">▾</span>
        </button>
        {open && (
          <div className="user-menu">
            <button
              className="user-menu-item"
              onClick={() => {
                setOpen(false)
                navigate('system')
              }}
            >
              系统管理
            </button>
            <button
              className="user-menu-item"
              onClick={() => {
                setOpen(false)
                navigate('system')
              }}
            >
              个人设置
            </button>
            <div className="user-menu-sep" />
            <button className="user-menu-item danger" onClick={logout}>
              退出登录
            </button>
          </div>
        )}
      </div>
    </header>
  )
}
