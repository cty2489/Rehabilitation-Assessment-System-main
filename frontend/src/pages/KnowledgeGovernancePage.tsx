import { useEffect, useMemo, useState } from 'react'
import { RefreshCw } from 'lucide-react'
import {
  fetchKnowledgeCoverage,
  fetchKnowledgeEntries,
  fetchKnowledgeEntry,
  fetchKnowledgeSources,
  fetchKnowledgeStatus,
  fetchGuidelineStatus,
  fetchRagVersions,
  searchGuidelines,
  searchSelectedRag,
} from '../api'
import {
  KnowledgeCoverageResponse,
  KnowledgeEntriesResponse,
  KnowledgeEntryDetail,
  KnowledgeEntrySummary,
  KnowledgeSource,
  KnowledgeSourcesResponse,
  KnowledgeStatusResponse,
  GuidelineTestSearchResponse,
  GuidelineTestStatus,
  RagTestSearchHit,
  RagVersionsResponse,
} from '../types'

type KnowledgeTab = 'overview' | 'search' | 'coverage' | 'entries' | 'sources'

const TABS: { id: KnowledgeTab; label: string }[] = [
  { id: 'overview', label: '概览' },
  { id: 'search', label: '统一检索' },
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
    detail: '按知识角色、证据等级和任务边界进行召回',
  },
  {
    title: '报告引用',
    detail: '证据进入报告生成，正文以【1】【2】关联参考文献',
  },
]

const CATEGORY_ORDER = ['临床量表', 'EMG', 'EEG', 'IMU', '康复建议', '安全边界']

const CANDIDATE_ROLE_LABELS: Record<string, string> = {
  core_direct: '核心临床证据',
  guideline: '指南/共识',
  textbook_reference: '教材参考',
  training: '训练建议',
  core_background: '背景知识',
  method_reference: '方法参考',
  research_reference: '研究参考',
}

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

function UnifiedSearchPanel() {
  const [mode, setMode] = useState<'knowledge' | 'research'>('knowledge')
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(3)
  const [loading, setLoading] = useState(false)
  const [knowledgeHits, setKnowledgeHits] = useState<RagTestSearchHit[]>([])
  const [researchStatus, setResearchStatus] = useState<GuidelineTestStatus | null>(null)
  const [researchResult, setResearchResult] = useState<GuidelineTestSearchResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchGuidelineStatus()
      .then(setResearchStatus)
      .catch((nextError) => setError(String((nextError as Error).message || nextError)))
  }, [])

  async function runSearch() {
    const nextQuery = query.trim()
    if (!nextQuery) return
    setLoading(true)
    setError(null)
    try {
      if (mode === 'knowledge') {
        const payload = await searchSelectedRag(nextQuery, topK)
        setKnowledgeHits(payload.results[0]?.hits || [])
        setResearchResult(null)
      } else {
        const payload = await searchGuidelines(nextQuery, topK)
        setResearchResult(payload)
        setKnowledgeHits([])
      }
    } catch (nextError) {
      setError(String((nextError as Error).message || nextError))
    } finally {
      setLoading(false)
    }
  }

  const researchReady = Boolean(
    researchStatus?.enabled
      && researchStatus.service_reachable
      && researchStatus.allow_demo
      && researchStatus.mode === 'test_only'
      && researchStatus.allowed_rag_mode === 'test_only',
  )
  const canSearch = mode === 'knowledge' || researchReady

  return (
    <div className="knowledge-panel" role="tabpanel">
      <div className="knowledge-panel-head">
        <div>
          <h2>统一检索</h2>
          <p>同一个检索入口，按需要切换全部知识或研究证据；结果仍保留各自的证据边界。</p>
        </div>
      </div>
      <div className="knowledge-tabs knowledge-search-mode" role="tablist" aria-label="检索范围">
        <button className={mode === 'knowledge' ? 'active' : ''} onClick={() => setMode('knowledge')} role="tab" aria-selected={mode === 'knowledge'}>
          全部知识（论文·指南·教材）
        </button>
        <button className={mode === 'research' ? 'active' : ''} onClick={() => setMode('research')} role="tab" aria-selected={mode === 'research'}>
          研究证据
        </button>
      </div>
      {mode === 'research' && (
        <div className="rag-evidence-boundary-banner">
          <span className="rag-evidence-boundary-badge">使用边界</span>
          <span>研究证据仅用于知识与证据展示，不构成临床诊断、患者分期或医疗建议。</span>
        </div>
      )}
      {error && <div className="error-banner">{error}</div>}
      {mode === 'research' && researchStatus && !researchReady && (
        <div className="rag-disabled-banner"><strong>研究证据检索当前不可用</strong></div>
      )}
      <div className="knowledge-filterbar">
        <input
          className="knowledge-search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={(event) => { if (event.key === 'Enter') runSearch() }}
          placeholder={mode === 'research' ? '例如：卒中后上肢康复训练应遵循哪些原则？' : '例如：MAS定义、FMA评定、Brunnstrom分期'}
          aria-label="统一检索词"
          disabled={!canSearch || loading}
        />
        <select value={topK} onChange={(event) => setTopK(Number(event.target.value))} disabled={!canSearch || loading} aria-label="返回结果数">
          {[1, 2, 3, 4, 5].map((value) => <option key={value} value={value}>{value} 条</option>)}
        </select>
        <button className="button secondary" onClick={runSearch} disabled={!canSearch || loading || !query.trim()}>
          {loading ? '检索中…' : '检索'}
        </button>
      </div>
      {mode === 'knowledge' && knowledgeHits.length > 0 && (
        <div className="knowledge-source-list" aria-label="统一知识检索结果">
          {knowledgeHits.map((hit) => (
            <article key={hit.chunk_id} className="knowledge-source-item">
              <span>{hit.uid || hit.knowledge_id} · {(hit.metadata?.source_type as string) || 'source'} · 相似度 {hit.score.toFixed(3)}{hit.page_number ? ` · 第 ${hit.page_number} 页` : ' · 页码不可用'}</span>
              <strong>{hit.title || hit.knowledge_id}</strong>
              <p>{hit.text ? `${hit.text.slice(0, 300)}${hit.text.length > 300 ? '…' : ''}` : '—'}</p>
              {hit.source_page_url && <a href={hit.source_page_url} target="_blank" rel="noopener noreferrer">查看原文（{hit.page_number ? `第 ${hit.page_number} 页` : '页码不可用'}）</a>}
            </article>
          ))}
        </div>
      )}
      {mode === 'research' && researchResult && (
        <div className="knowledge-source-list" aria-label="研究证据检索结果">
          {researchResult.results.length === 0 ? <p className="muted">未找到相关结果，请尝试更换关键词。</p> : researchResult.results.map((hit) => (
            <article key={hit.chunk_id || hit.rank} className="knowledge-source-item">
              <span>【{hit.citation_index}】 · {hit.source_type || 'source'} · {hit.evidence_scope || '研究证据'} · 相似度 {hit.score.toFixed(3)}</span>
              <strong>{hit.title || hit.source_id}</strong>
              <p>{hit.text ? `${hit.text.slice(0, 360)}${hit.text.length > 360 ? '…' : ''}` : '—'}</p>
              <span>{hit.page_locator || '页码不可用'}{hit.source_detail?.evidence_tier ? ` · ${hit.source_detail.evidence_tier}` : ''}</span>
              {hit.source_detail?.source_url && <a href={hit.source_detail.source_url} target="_blank" rel="noopener noreferrer">查看原始来源</a>}
            </article>
          ))}
        </div>
      )}
    </div>
  )
}

export default function KnowledgeGovernancePage({ initialTab = 'overview' }: { initialTab?: KnowledgeTab }) {
  const [tab, setTab] = useState<KnowledgeTab>(initialTab)
  const [status, setStatus] = useState<KnowledgeStatusResponse | null>(null)
  const [ragVersions, setRagVersions] = useState<RagVersionsResponse | null>(null)
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
      const nextRagVersions = await fetchRagVersions()
      setRagVersions(nextRagVersions)
      const nextStatus = await fetchKnowledgeStatus()
      setStatus(nextStatus)
      if (nextRagVersions.release) {
        setEntries(null)
        setCoverage(null)
        setSources(null)
        return
      }
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
    if (ragVersions?.release) {
      return Object.entries(ragVersions.release.role_counts)
        .map(([name, count]) => ({ name: CANDIDATE_ROLE_LABELS[name] || name, count }))
        .sort((left, right) => right.count - left.count)
    }
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
  }, [entries, ragVersions])

  const largestCategory = Math.max(1, ...categoryStats.map((item) => item.count))
  const candidateRelease = ragVersions?.release
  const pageAvailable = Boolean(candidateRelease) || Boolean(status?.available)
  const visibleTabs = candidateRelease ? TABS.slice(0, 2) : TABS

  return (
    <div className="knowledge-page">
      <div className="page-head">
        <div>
          <h1 className="page-title">RAG 知识库</h1>
          <p className="page-sub">论文、指南、教材、页码来源与报告引用</p>
        </div>
        <button className="button secondary" onClick={load} disabled={loading} title="刷新RAG知识库">
          <RefreshCw size={15} aria-hidden="true" />
          {loading ? '读取中' : '刷新'}
        </button>
      </div>

      {error && <div className="error-banner">{error}</div>}
      {loading && !status && <div className="knowledge-initial-loading">正在读取 RAG 知识库…</div>}
      {status && !status.available && !candidateRelease && (
        <div className="error-banner">{status.error || 'RAG 知识库尚未准备完成'}</div>
      )}

      {pageAvailable && (
        <>
          <section className="knowledge-metrics" aria-label="RAG知识库摘要">
            <div><span>{candidateRelease ? '知识切片' : '知识条目'}</span><strong>{candidateRelease ? candidateRelease.chunks.toLocaleString() : status?.counts.total_entries}</strong></div>
            <div><span>{candidateRelease ? 'Embedding' : '26项指标知识'}</span><strong>{candidateRelease ? candidateRelease.embeddings.toLocaleString() : `${status?.counts.mapped_biomarkers || 0}/26`}</strong></div>
            <div><span>{candidateRelease ? '原始来源' : '参考文献'}</span><strong>{candidateRelease ? candidateRelease.sources.toLocaleString() : status?.counts.sources}</strong></div>
            <div><span>检索服务</span><strong className="metric-text">{candidateRelease ? (ragVersions?.health.v1_candidate?.status === 'ok' ? '在线' : '未连接') : (status ? ragServiceLabel(status) : '未连接')}</strong></div>
            <div><span>报告接入</span><strong className="metric-text">{candidateRelease ? '已接入新版' : (status ? reportAccessLabel(status) : '未接入')}</strong></div>
          </section>

          <div className="knowledge-tabs" role="tablist" aria-label="RAG知识库视图">
            {visibleTabs.map((item) => (
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
                  <div><h2>报告检索链路</h2><p>{candidateRelease ? '当前系统生成康复报告时使用完整 v1_candidate' : '当前系统生成康复报告时使用的 RAG 数据路径'}</p></div>
                  <span className={`knowledge-runtime ${candidateRelease ? 'online' : (status?.rag.service.reachable ? 'online' : 'offline')}`}>
                    RAG · {candidateRelease ? '新版在线' : (status ? ragServiceLabel(status) : '未连接')}
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
                  <div><h2>知识库内容</h2><p>按知识角色和证据边界分布</p></div>
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
                    <div><dt>知识版本</dt><dd>{candidateRelease ? 'v1_candidate' : (status?.versions.content_release || '—')}</dd></div>
                    <div><dt>向量集合</dt><dd>{candidateRelease ? candidateRelease.collection : (status?.versions.index_collection || '—')}</dd></div>
                    <div><dt>向量规格</dt><dd>{candidateRelease ? `${candidateRelease.embedding_dimensions}维 · ${candidateRelease.distance}` : formatTime(status?.versions.index_built_at_utc || '')}</dd></div>
                    <div><dt>原始文件</dt><dd>{candidateRelease ? `${candidateRelease.source_files} 个，可按页查看` : '正文数字引用【n】'}</dd></div>
                  </dl>
                </div>
              </section>

            </div>
          )}

          {tab === 'search' && (
            <UnifiedSearchPanel />
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
