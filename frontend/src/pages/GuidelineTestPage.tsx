import { useCallback, useEffect, useState } from 'react'
import { fetchGuidelineStatus, searchGuidelines } from '../api'
import type {
  GuidelineTestHit,
  GuidelineTestSearchResponse,
  GuidelineTestStatus,
} from '../types'

const EXAMPLE_QUESTIONS = [
  '传感器评估中的上肢能力和真实日常表现有什么区别？',
  '系统评价中 IMU 指标与 ICF 上肢临床评估的总体相关性是多少？',
  '论文报告的 Brunnstrom 手臂和手部预测 Spearman 相关性是多少？',
  'b710 的 r=0.85 是否能表述为已经确定的强相关？',
  '卒中后上肢康复训练应遵循哪些原则？',
]

function renderClickableText(text: string): React.ReactNode {
  // Only render safe HTTPS links as clickable
  const urlRegex = /(https:\/\/[^\s<>")\]]+)/g
  const safeUrlRegex = /^https:\/\/[^\s<>")\]]+$/
  const parts = text.split(urlRegex)
  return parts.map((part, i) => {
    if (safeUrlRegex.test(part)) {
      return (
        <a key={i} href={part} target="_blank" rel="noopener noreferrer">
          {part}
        </a>
      )
    }
    return part
  })
}

export default function GuidelineTestPage() {
  const [status, setStatus] = useState<GuidelineTestStatus | null>(null)
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(3)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<GuidelineTestSearchResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchGuidelineStatus()
      .then(setStatus)
      .catch((err: unknown) => {
        if (err && typeof err === 'object' && 'status' in err) {
          const apiErr = err as { status: number; message: string }
          if (apiErr.status === 401) {
            setError('请先登录后再使用知识库检索功能。')
          } else {
            setError(apiErr.message)
          }
        } else {
          setError(err instanceof Error ? err.message : String(err))
        }
      })
  }, [])

  const handleSearch = useCallback(async () => {
    const q = query.trim()
    if (!q) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await searchGuidelines(q, topK)
      setResult(res)
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'status' in err) {
        const apiErr = err as { status: number; message: string }
        if (apiErr.status === 401) {
          setError('请先登录后再使用知识库检索功能。')
        } else {
          setError(apiErr.message)
        }
      } else {
        setError(err instanceof Error ? err.message : String(err))
      }
    } finally {
      setLoading(false)
    }
  }, [query, topK])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !loading) handleSearch()
  }

  const handleExampleClick = (q: string) => {
    setQuery(q)
  }

  const isBlocked = result?.reason_code !== 'in_scope' && result?.reason_code !== undefined
  const hasResults = result && result.results.length > 0
  const isReady = Boolean(
    status?.enabled
      && status.service_reachable
      && status.allow_demo
      && status.mode === 'test_only'
      && status.allowed_rag_mode === 'test_only'
      && status.clinical_ready === false,
  )

  return (
    <div className="rag-guideline-page">
      <div className="page-head">
        <div>
          <h1 className="page-title">康复知识与研究证据检索</h1>
          <p className="page-sub">
            检索知识库中的康复知识、解释边界与 IMU 研究证据
          </p>
        </div>
      </div>

      {/* Evidence boundary banner */}
      <div className="rag-evidence-boundary-banner">
        <span className="rag-evidence-boundary-badge">使用边界</span>
        <span>
          研究证据检索结果仅用于知识与证据展示，不构成临床诊断、患者分期、治疗处方或医疗建议。
        </span>
      </div>

      {/* Status indicator */}
      {status && !status.enabled && (
        <div className="rag-disabled-banner">
          <span className="rag-disabled-icon">!</span>
          <div>
            <strong>知识库检索服务尚未启用</strong>
            <p>请联系管理员启用知识库检索服务后刷新页面。</p>
          </div>
        </div>
      )}

      {status?.error && (
        <div className="error-banner">{status.error}</div>
      )}

      {error && (
        <div className="error-banner">{error}</div>
      )}

      {/* Search form */}
      <div className="rag-search-section">
        <p className="rag-hint">
          可检索康复原则、研究结论、统计限制和适用边界。
        </p>
        <div className="rag-example-row">
          <span className="rag-example-label">示例问题：</span>
          <div className="rag-example-list">
            {EXAMPLE_QUESTIONS.map((q) => (
              <button
                key={q}
                className="rag-example-btn"
                onClick={() => handleExampleClick(q)}
                disabled={!isReady || loading}
              >
                {q}
              </button>
            ))}
          </div>
        </div>
        <div className="rag-search-row">
          <input
            type="text"
            className="rag-search-input"
            placeholder="例如：IMU 与 ICF 上肢评估的相关性证据是什么？"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            maxLength={2000}
            disabled={!isReady || loading}
          />
          <select
            className="rag-topk-select"
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            disabled={!isReady || loading}
          >
            <option value={1}>1 条</option>
            <option value={2}>2 条</option>
            <option value={3}>3 条</option>
            <option value={4}>4 条</option>
            <option value={5}>5 条</option>
          </select>
          <button
            className="button"
            onClick={handleSearch}
            disabled={!isReady || loading || !query.trim()}
          >
            {loading ? '检索中...' : '检索'}
          </button>
        </div>
      </div>

      {/* Loading state */}
      {loading && (
        <div className="rag-loading-card">
          <div className="rag-spinner" />
          <span>正在检索知识库...</span>
        </div>
      )}

      {/* Results */}
      {result && !loading && (
        <div className="rag-results-section">
          <div className="rag-results-header">
            <span>检索到 {result.results.length} 条结果</span>
            <span className="rag-results-time">耗时 {result.elapsed_ms}ms</span>
          </div>

          {/* Blocked / out-of-scope */}
          {isBlocked && (
            <div className="rag-blocked-banner">
              <span className="rag-blocked-icon">i</span>
              <div>
                <strong>
                  {result.blocked_message || '该问题超出当前知识库的使用边界，请改问研究结论、适用范围或证据局限。'}
                </strong>
              </div>
            </div>
          )}

          {/* Empty results */}
          {!isBlocked && result.results.length === 0 && (
            <div className="rag-empty-state">
              <p>未找到相关结果，请尝试更换关键词。</p>
            </div>
          )}

          {/* Result hits - flat layout */}
          {hasResults && (
            <div className="rag-hits-list">
              {result.results.map((hit) => (
                <GuidelineHitItem key={hit.chunk_id || hit.rank} hit={hit} />
              ))}
            </div>
          )}

          {/* Citations reference list */}
          {hasResults && result.citations && result.citations.length > 0 && (
            <div className="rag-citations-section">
              <h3 className="rag-citations-title">参考文献</h3>
              <ol className="rag-citations-list">
                {result.citations.map((cit) => (
                  <li key={cit.index} id={`cit-${cit.index}`}>
                    <span className="rag-cit-source">{cit.source_id}</span>
                    {cit.title && <span> {cit.title}</span>}
                    {cit.year && <span> ({cit.year})</span>}
                    {cit.doi && (
                      <span>
                        {' '}
                        DOI:{' '}
                        {cit.doi.startsWith('https://') ? (
                          <a href={cit.doi} target="_blank" rel="noopener noreferrer">
                            {cit.doi}
                          </a>
                        ) : (
                          cit.doi
                        )}
                      </span>
                    )}
                    {cit.page_locator && <span>, {cit.page_locator}</span>}
                  </li>
                ))}
              </ol>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function GuidelineHitItem({ hit }: { hit: GuidelineTestHit }) {
  const [expanded, setExpanded] = useState(false)
  const scorePercent = Math.round(hit.score * 100)
  const doiHref = hit.doi
    ? (hit.doi.startsWith('https://') ? hit.doi : `https://doi.org/${hit.doi}`)
    : ''
  const citationIndices = hit.citation_indices?.length
    ? hit.citation_indices
    : [hit.citation_index]
  const sourceDetail = hit.source_detail
  const localExcerpt = sourceDetail.local_excerpt || hit.text
  const hasResearchMetadata = Boolean(
    hit.research_only
      || hit.research_type
      || hit.sample_size
      || hit.evidence_scope
      || hit.applicable_scope
      || hit.limitations.length
      || hit.non_clinical_statement,
  )

  return (
    <div className="rag-hit-item">
      <div className="rag-hit-header">
        <span className="rag-hit-citation">
          {citationIndices.map((index) => (
            <a key={index} href={`#cit-${index}`}>【{index}】</a>
          ))}
        </span>
        <span className="rag-hit-title">{hit.title || hit.source_id}</span>
        <span className="rag-hit-score" title={`相关度: ${scorePercent}%`}>
          {scorePercent}%
        </span>
      </div>

      <p className={`rag-hit-text ${expanded ? 'expanded' : ''}`}>
        {renderClickableText(hit.text)}
      </p>

      <div className="rag-source-card">
        <div className="rag-source-card-head">
          <strong>证据来源与访问</strong>
          {sourceDetail.source_type && <span>{sourceDetail.source_type}</span>}
          {sourceDetail.evidence_tier && <span>证据层级 {sourceDetail.evidence_tier}</span>}
        </div>
        <p className="rag-source-access">{sourceDetail.access_status}</p>
        <details className="rag-source-excerpt">
          <summary>查看系统内证据摘录</summary>
          <p>{localExcerpt}</p>
        </details>
        <div className="rag-source-links">
          {sourceDetail.source_url ? (
            <a href={sourceDetail.source_url} target="_blank" rel="noopener noreferrer">
              查看原始来源
            </a>
          ) : (
            <span>原始来源链接尚未归档</span>
          )}
          <span>{sourceDetail.rights_status}</span>
        </div>
      </div>

      {hasResearchMetadata && (
        <div className="rag-research-block">
          <div className="rag-research-labels">
            <span className="rag-research-badge">研究证据</span>
            {hit.research_type && <span className="rag-research-type">{hit.research_type}</span>}
            {hit.source_type && <span className="rag-research-source-type">{hit.source_type}</span>}
          </div>

          <dl className="rag-research-meta">
            {hit.sample_size && (
              <div>
                <dt>样本信息</dt>
                <dd>{hit.sample_size}</dd>
              </div>
            )}
            {hit.evidence_scope && (
              <div>
                <dt>证据范围</dt>
                <dd>{hit.evidence_scope}</dd>
              </div>
            )}
            {hit.applicable_scope && (
              <div>
                <dt>适用范围</dt>
                <dd>{hit.applicable_scope}</dd>
              </div>
            )}
            {hit.doi && (
              <div>
                <dt>DOI</dt>
                <dd>
                  <a href={doiHref} target="_blank" rel="noopener noreferrer">{hit.doi}</a>
                </dd>
              </div>
            )}
            {hit.license && (
              <div>
                <dt>许可</dt>
                <dd>{hit.license}</dd>
              </div>
            )}
          </dl>

          {hit.limitations.length > 0 && (
            <div className="rag-research-limitations">
              <strong>研究局限</strong>
              <ul>
                {hit.limitations.map((limitation) => (
                  <li key={limitation}>{limitation}</li>
                ))}
              </ul>
            </div>
          )}

          {hit.non_clinical_statement && (
            <p className="rag-research-boundary">{hit.non_clinical_statement}</p>
          )}
        </div>
      )}

      {hit.text.length > 200 && (
        <button
          className="rag-expand-btn"
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? '收起' : '展开全文'}
        </button>
      )}
    </div>
  )
}
