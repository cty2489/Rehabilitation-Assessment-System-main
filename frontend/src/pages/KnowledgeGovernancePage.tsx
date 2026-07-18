import { useEffect, useMemo, useState } from 'react'
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
  { id: 'coverage', label: '26项映射' },
  { id: 'entries', label: '知识条目' },
  { id: 'sources', label: '来源文献' },
]

const STATUS_CLASS: Record<string, string> = {
  blocked_current_implementation: 'knowledge-badge-blocked',
  research_only: 'knowledge-badge-research',
  conditional_after_protocol_fix: 'knowledge-badge-conditional',
  guideline_candidate_pending_expert: 'knowledge-badge-guideline',
}

function statusClass(status: string): string {
  return STATUS_CLASS[status] || 'knowledge-badge-neutral'
}

function shortStatus(status: string): string {
  const labels: Record<string, string> = {
    blocked_current_implementation: '阻断',
    research_only: '仅研究',
    conditional_after_protocol_fix: '条件候选',
    guideline_candidate_pending_expert: '指南候选',
  }
  return labels[status] || status || '未知'
}

function formatTime(value: string): string {
  if (!value) return '—'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString('zh-CN')
}

function KnowledgeStatusBadge({ entry }: { entry: KnowledgeEntrySummary }) {
  return (
    <span className={`knowledge-badge ${statusClass(entry.knowledge_status)}`} title={entry.knowledge_status_label}>
      {shortStatus(entry.knowledge_status)}
    </span>
  )
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
            <th>模态</th>
            <th>系统键</th>
            <th>治理状态</th>
            <th>临床可用</th>
            <th>来源</th>
          </tr>
        </thead>
        <tbody>
          {items.length === 0 && (
            <tr><td colSpan={6} className="knowledge-empty">没有符合条件的知识条目</td></tr>
          )}
          {items.map((entry) => (
            <tr key={entry.knowledge_id} className="knowledge-row">
              <td>
                <button className="knowledge-entry-link" onClick={() => onSelect(entry.knowledge_id)}>
                  <strong>{entry.title}</strong>
                  <span>{entry.knowledge_id} · v{entry.entry_version}</span>
                </button>
              </td>
              <td>{entry.category}</td>
              <td><code>{entry.system_key}</code></td>
              <td><KnowledgeStatusBadge entry={entry} /></td>
              <td>
                <span className={`knowledge-readiness ${entry.clinical_ready ? 'ready' : 'not-ready'}`}>
                  {entry.clinical_ready ? '是' : '否'}
                </span>
              </td>
              <td>{entry.source_ids.length}</td>
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
      <aside className="knowledge-drawer" role="dialog" aria-modal="true" aria-label="知识条目详情">
        <header className="knowledge-drawer-head">
          <div>
            <span className="knowledge-drawer-id">{entry?.knowledge_id || '读取中'}</span>
            <h2>{entry?.title || '正在读取知识条目'}</h2>
          </div>
          <button className="knowledge-close" onClick={onClose} aria-label="关闭" title="关闭">×</button>
        </header>
        {loading && <div className="knowledge-drawer-loading">正在读取证据…</div>}
        {entry && (
          <div className="knowledge-drawer-body">
            <div className="knowledge-drawer-meta">
              <KnowledgeStatusBadge entry={entry} />
              <code>{entry.system_key}</code>
              <span>{entry.category}</span>
              <span>版本 {entry.entry_version}</span>
            </div>

            <section className="knowledge-detail-section">
              <h3>核心结论</h3>
              <p>{entry.content || '—'}</p>
            </section>
            <section className="knowledge-detail-section knowledge-detail-allowed">
              <h3>允许解释</h3>
              <p>{entry.allowed_interpretation || '—'}</p>
            </section>
            <section className="knowledge-detail-section knowledge-detail-prohibited">
              <h3>禁止解释</h3>
              <p>{entry.prohibited_interpretation || '—'}</p>
            </section>
            <section className="knowledge-detail-section">
              <h3>采集与算法要求</h3>
              <p>{entry.acquisition_and_algorithm_requirements || '—'}</p>
            </section>
            <section className="knowledge-detail-section">
              <h3>参考范围政策</h3>
              <p>{entry.reference_range_policy || '—'}</p>
            </section>
            <section className="knowledge-detail-section">
              <h3>实施动作</h3>
              <p>{entry.implementation_action || '—'}</p>
            </section>

            <section className="knowledge-detail-section">
              <h3>证据来源</h3>
              <div className="knowledge-source-list">
                {entry.sources.map((source) => (
                  <article key={source.source_id} className="knowledge-source-item">
                    <span>{source.source_id} · {source.evidence_tier ? `证据等级 ${source.evidence_tier}` : '未分级'}</span>
                    <strong>{source.title}</strong>
                    <p>{source.scope || source.note || '—'}</p>
                    {source.url && (
                      <a href={source.url} target="_blank" rel="noopener noreferrer">查看原始来源</a>
                    )}
                  </article>
                ))}
              </div>
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
  const [entryStatus, setEntryStatus] = useState('')
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
      if (entryStatus && entry.knowledge_status !== entryStatus) return false
      if (!needle) return true
      return [entry.title, entry.knowledge_id, entry.system_key, entry.category]
        .join(' ')
        .toLocaleLowerCase()
        .includes(needle)
    })
  }, [entries, query, category, entryStatus])

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

  const statusCount = (key: string, biomarkersOnly = false) => {
    const item = status?.status_counts.find((candidate) => candidate.status === key)
    return biomarkersOnly ? item?.biomarker_count || 0 : item?.count || 0
  }

  return (
    <div className="knowledge-page">
      <div className="page-head">
        <div>
          <h1 className="page-title">知识与证据治理</h1>
          <p className="page-sub">内部试运行知识、系统指标映射与来源证据</p>
        </div>
        <button className="button secondary" onClick={load} disabled={loading} title="刷新知识状态">
          <span aria-hidden="true">↻</span>{loading ? '读取中' : '刷新'}
        </button>
      </div>

      {error && <div className="error-banner">{error}</div>}
      {loading && !status && <div className="knowledge-initial-loading">正在读取知识发布包…</div>}
      {status?.trial_release.warning && (
        <div className="knowledge-trial-warning">
          <strong>内部试运行</strong>
          <span>{status.trial_release.warning}</span>
        </div>
      )}
      {status && !status.available && (
        <div className="error-banner">{status.error || '知识发布包尚未准备完成'}</div>
      )}

      {status?.available && (
        <>
          <section className="knowledge-metrics" aria-label="知识治理摘要">
            <div><span>指标映射</span><strong>{status.counts.mapped_biomarkers}/26</strong></div>
            <div className="metric-critical"><span>临床可用</span><strong>{status.counts.clinical_ready_biomarkers}/26</strong></div>
            <div className="metric-danger"><span>阻断</span><strong>{statusCount('blocked_current_implementation', true)}</strong></div>
            <div><span>仅研究</span><strong>{statusCount('research_only', true)}</strong></div>
            <div className="metric-warning"><span>条件候选</span><strong>{statusCount('conditional_after_protocol_fix', true)}</strong></div>
            <div><span>专家确认</span><strong>{status.counts.expert_verified_entries}</strong></div>
          </section>

          <div className="knowledge-tabs" role="tablist" aria-label="知识治理视图">
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
                  <div><h2>运行状态</h2><p>应用、内容、索引和模型版本分别记录</p></div>
                  <span className={`knowledge-runtime ${status.rag.service.reachable ? 'online' : 'offline'}`}>
                    RAG {status.rag.mode.toUpperCase()} · {status.rag.service.status}
                  </span>
                </div>
                <dl className="knowledge-version-grid">
                  <div><dt>内容发布</dt><dd>{status.versions.content_release || '—'}</dd></div>
                  <div><dt>索引集合</dt><dd>{status.versions.index_collection || '—'}</dd></div>
                  <div><dt>源文档Schema</dt><dd>{status.versions.source_document || '—'}</dd></div>
                  <div><dt>索引构建时间</dt><dd>{formatTime(status.versions.index_built_at_utc)}</dd></div>
                  <div><dt>应用版本</dt><dd>{status.versions.application || '—'}</dd></div>
                  <div><dt>报告模型</dt><dd>{status.versions.report_model || '—'}</dd></div>
                </dl>
              </section>

              <section className="knowledge-section">
                <div className="knowledge-section-head">
                  <div><h2>治理边界</h2><p>{status.counts.total_entries}条知识，{status.counts.sources}项结构化来源</p></div>
                </div>
                <div className="knowledge-boundaries">
                  <div>
                    <h3>允许用途</h3>
                    <ul>{(status.trial_release.allowed_usage || []).map((item) => <li key={item}>{item}</li>)}</ul>
                  </div>
                  <div className="prohibited">
                    <h3>禁止用途</h3>
                    <ul>{(status.trial_release.prohibited_usage || []).map((item) => <li key={item}>{item}</li>)}</ul>
                  </div>
                </div>
              </section>

              <section className="knowledge-section">
                <div className="knowledge-section-head"><div><h2>条目状态</h2><p>总条目与26项生物标志物分别统计</p></div></div>
                <div className="knowledge-table-wrap">
                  <table className="knowledge-table compact">
                    <thead><tr><th>治理状态</th><th>全部条目</th><th>26项指标</th></tr></thead>
                    <tbody>
                      {status.status_counts.map((item) => (
                        <tr key={item.status}>
                          <td><span className={`knowledge-badge ${statusClass(item.status)}`}>{shortStatus(item.status)}</span></td>
                          <td>{item.count}</td>
                          <td>{item.biomarker_count}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {!status.validation.valid && (
                  <div className="knowledge-validation">
                    {status.validation.issues.map((issue) => <span key={issue}>{issue}</span>)}
                  </div>
                )}
              </section>
            </div>
          )}

          {tab === 'coverage' && coverage && (
            <div className="knowledge-panel" role="tabpanel">
              <div className="knowledge-panel-head">
                <div><h2>系统指标映射</h2><p>已映射 {coverage.mapped}/{coverage.expected}，临床可用 {coverage.clinical_ready}/{coverage.expected}</p></div>
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
                  placeholder="搜索名称、编号或系统键"
                  aria-label="搜索知识条目"
                />
                <select value={category} onChange={(event) => setCategory(event.target.value)} aria-label="按模态筛选">
                  <option value="">全部模态</option>
                  {entries.filters.categories.map((item) => <option key={item} value={item}>{item}</option>)}
                </select>
                <select value={entryStatus} onChange={(event) => setEntryStatus(event.target.value)} aria-label="按治理状态筛选">
                  <option value="">全部状态</option>
                  {entries.filters.statuses.map((item) => <option key={item.status} value={item.status}>{shortStatus(item.status)}</option>)}
                </select>
                <span>{filteredEntries.length} 条</span>
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
                  placeholder="搜索题名、类型、范围或来源编号"
                  aria-label="搜索来源文献"
                />
                <span>{filteredSources.length} 项来源</span>
              </div>
              <div className="knowledge-table-wrap">
                <table className="knowledge-table knowledge-source-table">
                  <thead><tr><th>来源</th><th>类型</th><th>等级</th><th>年份</th><th>适用范围</th><th>关联条目</th></tr></thead>
                  <tbody>
                    {filteredSources.map((source: KnowledgeSource) => (
                      <tr key={source.source_id}>
                        <td>
                          <strong>{source.title}</strong>
                          <span className="knowledge-source-id">{source.source_id}</span>
                          {source.url && <a href={source.url} target="_blank" rel="noopener noreferrer">原始来源</a>}
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
