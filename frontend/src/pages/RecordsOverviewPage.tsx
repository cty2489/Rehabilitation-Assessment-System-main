import { useEffect, useState } from 'react'
import { Search } from 'lucide-react'
import { fetchAssessments } from '../api'
import { useRoute } from '../app/AppContext'
import { AssessmentOverviewItem } from '../types'
import { fmtDateTime } from '../util'

export default function RecordsOverviewPage() {
  const { navigate } = useRoute()
  const [items, setItems] = useState<AssessmentOverviewItem[] | null>(null)
  const [total, setTotal] = useState(0)
  const [query, setQuery] = useState('')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchAssessments(200, 0)
      .then((r) => {
        setItems(r.items)
        setTotal(r.total)
      })
      .catch((e) => setError(String(e.message || e)))
  }, [])

  const filtered = (items || []).filter(
    (r) =>
      !query ||
      r.name.includes(query) ||
      r.patient_id.toLowerCase().includes(query.toLowerCase()),
  )

  return (
    <div>
      <div className="page-head">
        <div>
          <h1 className="page-title">评估记录总览</h1>
          <p className="page-sub">跨患者按时间排列的全部评估记录（共 {total} 条）</p>
        </div>
        <label className="search-field">
          <Search aria-hidden="true" />
          <span className="sr-only">搜索评估记录</span>
          <input
            className="search-input"
            placeholder="搜索患者姓名 / 编号"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </label>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div className="card">
        {items === null ? (
          <p className="muted">加载中…</p>
        ) : filtered.length === 0 ? (
          <p className="muted">暂无记录。</p>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>评估时间</th>
                <th>患者编号</th>
                <th>姓名</th>
                <th>FMA 手部</th>
                <th>手部 MAS</th>
                <th>Brunnstrom 手部</th>
                <th>报告</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((r) => (
                <tr
                  key={r.id}
                  className="clickable"
                  onClick={() => navigate('patients', r.patient_db_id)}
                >
                  <td>{fmtDateTime(r.created_at)}</td>
                  <td>{r.patient_id}</td>
                  <td>{r.name}</td>
                  <td>{Math.round(r.fma_ue)}/20 分</td>
                  <td>{r.hand_tone} 级</td>
                  <td>{r.hand_function} 期</td>
                  <td>
                    {r.report_status === 'failed' ? (
                      <span className="badge badge-warn">未生成</span>
                    ) : (
                      <span className="badge badge-ok">已生成</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
