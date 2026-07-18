import { useEffect, useMemo, useState } from 'react'
import { RefreshCw } from 'lucide-react'
import {
  fetchKnowledgeCoverage,
  fetchKnowledgeEntries,
  fetchKnowledgeEntry,
  fetchKnowledgeSources,
  fetchKnowledgeStatus,
} from '../api'
import {
  KnowledgeCoverageResponse,
  KnowledgeEntriesResponse,
  KnowledgeEntryDetail,
  KnowledgeEntrySummary,
  KnowledgeSource,
  KnowledgeSourcesResponse,
  KnowledgeStatusResponse,
} from '../types'

type KnowledgeTab = 'overview' | 'coverage' | 'entries' | 'sources'

const TABS: { id: KnowledgeTab; label: string }[] = [
  { id: 'overview', label: '概览' },
  { id: 'coverage', label: '26项指标知识' },
  { id: 'entries', label: '全部知识' },
  { id: 'sources', label: '参考文献' },
]

const RETRIEVAL_FLOW = [
  {
    title: '评估上下文',
    detail: '患者信息、深度模型结果与可用 biomarker',
  },
  {
    title: '检索问题',
    detail: '按临床量表及 EEG、EMG、IMU 自动构造查询',
  },
  {
    title: '双路检索',
    detail: '总体知识向量召回，26 项指标按 system_key 精确匹配',
  },
  {
    title: '报告引用',
    detail: '证据进入报告生成，正文以【1】【2】关联参考文献',
  },
]

const CATEGORY_ORDER = ['临床量表', 'EMG', 'EEG', 'IMU', '康复建议', '安全边界']

function formatTime(value: string): string {
  if (!value) return '—'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString('zh-CN')
}

function displayKnowledgeTitle(title: string): string {
  return title
    .replace(/（当前[^）]*）/g, '')
    .replace(/（方向未标定）/g, '')
    .replace(/\s{2,}/g, ' ')
    .trim()
}

function ragServiceLabel(status: KnowledgeStatusResponse): string {
  if (status.rag.mode === 'off') return '未启用'
  if (!status.rag.service.reachable) return '未连接'
  return status.rag.service.collection_matches ? '在线' : '索引待同步'
}

function reportAccessLabel(status: KnowledgeStatusResponse): string {
  if (status.rag.mode === 'assist') {
    return status.rag.assist_approved ? '已接入报告' : '待启用'
  }
  if (status.rag.mode === 'shadow') return '仅检索记录'
  return '未接入'
}

function EntryTable({
  items,
  onSelect,
}: {
  items: KnowledgeEntrySummary[]
  onSelect: (knowledgeId: string) => void
}) {
  return (
    <div className="knowledge-table-wrap">
      <table className="knowledge-table">
        <thead>
          <tr>
            <th>知识条目</th>
            <th>知识类型</th>
            <th>关联指标</th>
            <th>参考文献</th>
          </tr>
        </thead>
        <tbody>
          {items.length === 0 && (
            <tr><td colSpan={4} className="knowledge-empty">没有符合条件的知识条目</td></tr>
          )}
          {items.map((entry) => (
            <tr key={entry.knowledge_id} className="knowledge-row">
              <td>
                <button className="knowledge-entry-link" onClick={() => onSelect(entry.knowledge_id)}>
                  <strong>{displayKnowledgeTitle(entry.title)}</strong>
                  <span>{entry.knowledge_id} · v{entry.entry_version}</span>
                </button>
              </td>
              <td>{entry.category}</td>
              <td><code>{entry.system_key}</code></td>
              <td>{entry.source_ids.length} 项</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function EvidenceDrawer({ entry, loading, onClose }: {
  entry: KnowledgeEntryDetail | null
  loading: boolean
  onClose: () => void
}) {
  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', closeOnEscape)
    return () => window.removeEventListener('keydown', closeOnEscape)
  }, [onClose])

  return (
    <div className="knowledge-drawer-backdrop" onMouseDown={(event) => {
      if (event.target === event.currentTarget) onClose()
    }}>
      <aside className="knowledge-drawer" role="dialog" aria-modal="true" aria-label="RAG知识条目详情">
        <header className="knowledge-drawer-head">
          <div>
            <span className="knowledge-drawer-id">{entry?.knowledge_id || '读取中'}</span>
            <h2>{entry ? displayKnowledgeTitle(entry.title) : '正在读取知识条目'}</h2>
          </div>
          <button className="knowledge-close" onClick={onClose} aria-label="关闭" title="关闭">×</button>
        </header>
        {loading && <div className="knowledge-drawer-loading">正在读取知识内容…</div>}
        {entry && (
          <div className="knowledge-drawer-body">
            <div className="knowledge-drawer-meta">
              <span>{entry.category}</span>
              <code>{entry.system_key}</code>
              <span>{entry.source_ids.length} 项来源</span>
              <span>版本 {entry.entry_version}</span>
            </div>

            <section className="knowledge-detail-section knowledge-detail-primary">
              <h3>知识摘要</h3>
              <p>{entry.content || '—'}</p>
            </section>
            {entry.allowed_interpretation && (
              <section className="knowledge-detail-section">
                <h3>报告解读参考</h3>
                <p>{entry.allowed_interpretation}</p>
              </section>
            )}
            {entry.reference_range_policy && (
              <section className="knowledge-detail-section">
                <h3>数值解释说明</h3>
                <p>{entry.reference_range_policy}</p>
              </section>
            )}
            {entry.applicable_population.length > 0 && (
              <section className="knowledge-detail-section">
                <h3>适用范围</h3>
                <p>{entry.applicable_population.join('；')}</p>
              </section>
            )}

            <section className="knowledge-detail-section">
              <h3>参考文献</h3>
              {entry.sources.length === 0 ? (
                <p>暂无结构化来源</p>
              ) : (
                <div className="knowledge-source-list">
                  {entry.sources.map((source) => (
                    <article key={source.source_id} className="knowledge-source-item">
                      <span>{source.source_id}{source.evidence_tier ? ` · 来源等级 ${source.evidence_tier}` : ''}</span>
                      <strong>{source.title}</strong>
                      <p>{source.scope || source.note || '—'}</p>
                      {source.url && (
                        <a href={source.url} target="_blank" rel="noopener noreferrer">查看原始来源</a>
                      )}
                    </article>
                  ))}
                </div>
              )}
            </section>
          </div>
        )}
      </aside>
    </div>
  )
}

export default function KnowledgeGovernancePage() {
  const [tab, setTab] = useState<KnowledgeTab>('overview')
  const [status, setStatus] = useState<KnowledgeStatusResponse | null>(null)
  const [entries, setEntries] = useState<KnowledgeEntriesResponse | null>(null)
  const [coverage, setCoverage] = useState<KnowledgeCoverageResponse | null>(null)
  const [sources, setSources] = useState<KnowledgeSourcesResponse | null>(null)
  const [selectedEntry, setSelectedEntry] = useState<KnowledgeEntryDetail | null>(null)
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [drawerLoading, setDrawerLoading] = useState(false)
  const [query, setQuery] = useState('')
  const [category, setCategory] = useState('')
  const [sourceQuery, setSourceQuery] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  async function load() {
    setLoading(true)
    setError(null)
    try {
      const nextStatus = await fetchKnowledgeStatus()
      setStatus(nextStatus)
      if (!nextStatus.available) {
        setEntries(null)
        setCoverage(null)
        setSources(null)
        return
      }
      const [nextEntries, nextCoverage, nextSources] = await Promise.all([
        fetchKnowledgeEntries(),
        fetchKnowledgeCoverage(),
        fetchKnowledgeSources(),
      ])
      setEntries(nextEntries)
      setCoverage(nextCoverage)
      setSources(nextSources)
    } catch (nextError) {
      setError(String((nextError as Error).message || nextError))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [])

  async function openEntry(knowledgeId: string) {
    setDrawerOpen(true)
    setDrawerLoading(true)
    setSelectedEntry(null)
    try {
      const payload = await fetchKnowledgeEntry(knowledgeId)
      setSelectedEntry(payload.entry)
    } catch (nextError) {
      setDrawerOpen(false)
      setError(String((nextError as Error).message || nextError))
    } finally {
      setDrawerLoading(false)
    }
  }

  const filteredEntries = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase()
    return (entries?.items || []).filter((entry) => {
      if (category && entry.category !== category) return false
      if (!needle) return true
      return [entry.title, entry.knowledge_id, entry.system_key, entry.category]
        .join(' ')
        .toLocaleLowerCase()
        .includes(needle)
    })
  }, [entries, query, category])

  const filteredSources = useMemo(() => {
    const needle = sourceQuery.trim().toLocaleLowerCase()
    if (!needle) return sources?.items || []
    return (sources?.items || []).filter((source) => (
      [source.source_id, source.title, source.source_type, source.scope, source.evidence_tier]
        .join(' ')
        .toLocaleLowerCase()
        .includes(needle)
    ))
  }, [sources, sourceQuery])

  const categoryStats = useMemo(() => {
    const counts = new Map<string, number>()
    for (const entry of entries?.items || []) {
      counts.set(entry.category, (counts.get(entry.category) || 0) + 1)
    }
    return [...counts.entries()]
      .map(([name, count]) => ({ name, count }))
      .sort((left, right) => {
        const leftIndex = CATEGORY_ORDER.indexOf(left.name)
        const rightIndex = CATEGORY_ORDER.indexOf(right.name)
        return (leftIndex < 0 ? 99 : leftIndex) - (rightIndex < 0 ? 99 : rightIndex)
      })
  }, [entries])

  const largestCategory = Math.max(1, ...categoryStats.map((item) => item.count))

  return (
    <div className="knowledge-page">
      <div className="page-head">
        <div>
          <h1 className="page-title">RAG 知识库</h1>
          <p className="page-sub">康复评估知识、26 项指标映射与报告引用来源</p>
        </div>
        <button className="button secondary" onClick={load} disabled={loading} title="刷新RAG知识库">
          <RefreshCw size={15} aria-hidden="true" />
          {loading ? '读取中' : '刷新'}
        </button>
      </div>

      {error && <div className="error-banner">{error}</div>}
      {loading && !status && <div className="knowledge-initial-loading">正在读取 RAG 知识库…</div>}
      {status && !status.available && (
        <div className="error-banner">{status.error || 'RAG 知识库尚未准备完成'}</div>
      )}

      {status?.available && (
        <>
          <section className="knowledge-metrics" aria-label="RAG知识库摘要">
            <div><span>知识条目</span><strong>{status.counts.total_entries}</strong></div>
            <div><span>26项指标知识</span><strong>{status.counts.mapped_biomarkers}/26</strong></div>
            <div><span>参考文献</span><strong>{status.counts.sources}</strong></div>
            <div><span>检索服务</span><strong className="metric-text">{ragServiceLabel(status)}</strong></div>
            <div><span>报告接入</span><strong className="metric-text">{reportAccessLabel(status)}</strong></div>
          </section>

          <div className="knowledge-tabs" role="tablist" aria-label="RAG知识库视图">
            {TABS.map((item) => (
              <button
                key={item.id}
                role="tab"
                aria-selected={tab === item.id}
                className={tab === item.id ? 'active' : ''}
                onClick={() => setTab(item.id)}
              >
                {item.label}
              </button>
            ))}
          </div>

          {tab === 'overview' && (
            <div className="knowledge-panel" role="tabpanel">
              <section className="knowledge-section">
                <div className="knowledge-section-head">
                  <div><h2>报告检索链路</h2><p>当前系统生成康复报告时使用的 RAG 数据路径</p></div>
                  <span className={`knowledge-runtime ${status.rag.service.reachable ? 'online' : 'offline'}`}>
                    RAG · {ragServiceLabel(status)}
                  </span>
                </div>
                <ol className="knowledge-flow">
                  {RETRIEVAL_FLOW.map((step, index) => (
                    <li key={step.title}>
                      <span className="knowledge-flow-index">{String(index + 1).padStart(2, '0')}</span>
                      <div>
                        <strong>{step.title}</strong>
                        <p>{step.detail}</p>
                      </div>
                    </li>
                  ))}
                </ol>
              </section>

              <section className="knowledge-section">
                <div className="knowledge-section-head">
                  <div><h2>知识库内容</h2><p>{status.counts.total_entries} 条结构化知识，关联 {status.counts.sources} 项来源</p></div>
                </div>
                <div className="knowledge-overview-grid">
                  <div className="knowledge-category-list" aria-label="知识类型分布">
                    {categoryStats.map((item) => (
                      <div className="knowledge-category-row" key={item.name}>
                        <div><span>{item.name}</span><strong>{item.count}</strong></div>
                        <span className="knowledge-category-track">
                          <span style={{ width: `${Math.max(8, Math.round(item.count / largestCategory * 100))}%` }} />
                        </span>
                      </div>
                    ))}
                  </div>
                  <dl className="knowledge-rag-meta">
                    <div><dt>知识版本</dt><dd>{status.versions.content_release || '—'}</dd></div>
                    <div><dt>向量集合</dt><dd>{status.versions.index_collection || '—'}</dd></div>
                    <div><dt>索引更新时间</dt><dd>{formatTime(status.versions.index_built_at_utc)}</dd></div>
                    <div><dt>报告引用方式</dt><dd>正文数字引用【n】</dd></div>
                  </dl>
                </div>
              </section>
            </div>
          )}

          {tab === 'coverage' && coverage && (
            <div className="knowledge-panel" role="tabpanel">
              <div className="knowledge-panel-head">
                <div>
                  <h2>26 项 biomarker 知识映射</h2>
                  <p>已建立 {coverage.mapped}/{coverage.expected} 项 system_key 精确映射，报告按指标键取回对应知识</p>
                </div>
              </div>
              <EntryTable items={coverage.items} onSelect={openEntry} />
            </div>
          )}

          {tab === 'entries' && entries && (
            <div className="knowledge-panel" role="tabpanel">
              <div className="knowledge-filterbar">
                <input
                  className="knowledge-search"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="搜索知识名称、编号或关联指标"
                  aria-label="搜索知识条目"
                />
                <select value={category} onChange={(event) => setCategory(event.target.value)} aria-label="按知识类型筛选">
                  <option value="">全部类型</option>
                  {entries.filters.categories.map((item) => <option key={item} value={item}>{item}</option>)}
                </select>
                <span>{filteredEntries.length} 条知识</span>
              </div>
              <EntryTable items={filteredEntries} onSelect={openEntry} />
            </div>
          )}

          {tab === 'sources' && sources && (
            <div className="knowledge-panel" role="tabpanel">
              <div className="knowledge-filterbar">
                <input
                  className="knowledge-search"
                  value={sourceQuery}
                  onChange={(event) => setSourceQuery(event.target.value)}
                  placeholder="搜索文献题名、类型或来源编号"
                  aria-label="搜索参考文献"
                />
                <span>{filteredSources.length} 项来源</span>
              </div>
              <div className="knowledge-table-wrap">
                <table className="knowledge-table knowledge-source-table">
                  <thead><tr><th>参考文献</th><th>类型</th><th>来源等级</th><th>年份</th><th>支持内容</th><th>关联知识</th></tr></thead>
                  <tbody>
                    {filteredSources.map((source: KnowledgeSource) => (
                      <tr key={source.source_id}>
                        <td>
                          <strong>{source.title}</strong>
                          <span className="knowledge-source-id">{source.source_id}</span>
                          {source.url && <a href={source.url} target="_blank" rel="noopener noreferrer">查看原始来源</a>}
                        </td>
                        <td>{source.source_type || '—'}</td>
                        <td><span className="knowledge-tier">{source.evidence_tier || '—'}</span></td>
                        <td>{source.year || '—'}</td>
                        <td>{source.scope || '—'}</td>
                        <td>{source.knowledge_ids.length}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}

      {drawerOpen && (
        <EvidenceDrawer
          entry={selectedEntry}
          loading={drawerLoading}
          onClose={() => setDrawerOpen(false)}
        />
      )}
    </div>
  )
}
