import { useId, useMemo, useState } from 'react'
import { BrainCircuit, Database, GitBranch, Network, Search } from 'lucide-react'
import {
  KnowledgeGraphDisplay,
  KnowledgeGraphDisplayNode,
  KnowledgeGraphDisplayPath,
} from '../types'

interface Props {
  graph: KnowledgeGraphDisplay | null
}

type LayerKey = 'source' | 'indicator' | 'functional_finding' | 'clinical_dimension' | 'rag_topic'

interface LayerDefinition {
  key: LayerKey
  label: string
  caption: string
}

interface PositionedNode {
  id: string
  label: string
  detail?: string
  x: number
  y: number
  layer: LayerKey
}

const LAYERS: LayerDefinition[] = [
  { key: 'source', label: '本次数据', caption: '字段、值与状态' },
  { key: 'indicator', label: '图谱指标', caption: '模型预测 / EEG / EMG' },
  { key: 'functional_finding', label: '功能表现', caption: '指标反映的表现' },
  { key: 'clinical_dimension', label: '临床维度', caption: '本次分析方向' },
  { key: 'rag_topic', label: 'RAG 主题', caption: '进入检索与 Planner' },
]

const RELATION_LABELS: Record<string, string> = {
  ASSOCIATED_WITH: '关联',
  MEASURES: '测量/反映',
  DESCRIBES: '描述',
  SUPPORTS: '支持',
  BELONGS_TO: '属于',
  SUGGESTS_TOPIC: '生成主题',
}

const STATE_LABELS: Record<string, string> = {
  observed: '已记录',
  within_reference: '参考范围内',
  above_reference: '高于参考范围',
  below_reference: '低于参考范围',
  direction_only: '方向性观察',
  not_classifiable: '单次不分类',
  missing: '数据缺失',
}

const LAYER_X = [28, 236, 444, 652, 860]
const NODE_WIDTH = 168
const NODE_HEIGHT = 58
const NODE_GAP = 72
const GRAPH_WIDTH = 1058

function numberValue(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0
}

function pathNode(path: KnowledgeGraphDisplayPath, layer: LayerKey): KnowledgeGraphDisplayNode {
  return path[layer]
}

function nodeKey(layer: LayerKey, node: KnowledgeGraphDisplayNode): string {
  return `${layer}:${node.id || node.label}`
}

function modeLabel(graph: KnowledgeGraphDisplay): string {
  if (graph.mode === 'llm_only') return 'LLM 直接规划'
  if (graph.status === 'fallback_to_llm_only') return '图谱异常，已回退'
  if (graph.applied) return '知识图谱增强已应用'
  return '知识图谱未应用'
}

function modeTone(graph: KnowledgeGraphDisplay): string {
  if (graph.applied) return 'applied'
  if (graph.status === 'fallback_to_llm_only') return 'warning'
  return 'disabled'
}

function originLabel(origins: string[]): string {
  const graph = origins.some((item) => item.startsWith('graph:'))
  const planner = origins.includes('planner')
  if (graph && planner) return '图谱 + Planner'
  if (graph) return '知识图谱'
  return 'Planner'
}

export function knowledgeGraphFromQuality(value: unknown): KnowledgeGraphDisplay | null {
  if (!value || typeof value !== 'object') return null
  const pipeline = (value as { clinical_pipeline?: unknown }).clinical_pipeline
  if (!pipeline || typeof pipeline !== 'object') return null
  const display = (pipeline as { knowledge_graph_display?: unknown }).knowledge_graph_display
  if (display && typeof display === 'object') {
    return display as KnowledgeGraphDisplay
  }

  const legacy = pipeline as Record<string, unknown>
  const mode = typeof legacy.knowledge_graph_mode === 'string'
    ? legacy.knowledge_graph_mode
    : null
  if (!mode) return null
  const topicCount = Number(legacy.knowledge_graph_topic_count || 0)
  return {
    schema_version: 'rehab.knowledge-graph-display.legacy',
    mode,
    status: String(legacy.knowledge_graph_status || 'unknown'),
    scope: String(legacy.knowledge_graph_non_imu_scope || 'not_applied'),
    applied:
      mode === 'graph_enhanced'
      && legacy.knowledge_graph_status === 'matched'
      && legacy.knowledge_graph_non_imu_scope === 'applied',
    prototype_notice: '该历史记录只保存了图谱运行摘要，没有保存逐条路径明细。',
    expert_review_status: 'pending',
    summary: {
      indicator_count: 0,
      path_count: 0,
      displayed_path_count: 0,
      dimension_count: 0,
      final_topic_count: Number.isFinite(topicCount) ? topicCount : 0,
      graph_seeded_topic_count: Number.isFinite(topicCount) ? topicCount : 0,
      matched_rule_count: 0,
      retrieval_status: null,
      retrieval_evidence_count: 0,
    },
    paths: [],
    paths_truncated: false,
    analysis_dimensions: [],
    final_topics: [],
    matched_rules: [],
    data_quality_warnings: [],
    measurement_context_topics: [],
    prediction_results: [],
    data_quality: {
      status: 'unknown',
      trial_count: null,
      short_trial_count: null,
      sync_fallback_count: null,
      sampling_rate_mismatch_count: null,
      warnings: [],
      is_clinical_dimension: false,
      blocked_support: false,
    },
  }
}

function RelationshipGraph({
  paths,
  activePathId,
  onActivePath,
}: {
  paths: KnowledgeGraphDisplayPath[]
  activePathId: string
  onActivePath: (pathId: string) => void
}) {
  const markerSeed = useId().replace(/:/g, '')
  const markerId = `kg-arrow-${markerSeed}`
  const { nodes, positions, height } = useMemo(() => {
    const layerNodes = new Map<LayerKey, Map<string, { id: string; label: string; detail?: string }>>()
    LAYERS.forEach(({ key }) => layerNodes.set(key, new Map()))

    paths.forEach((path) => {
      const sourceKey = nodeKey('source', path.source)
      layerNodes.get('source')?.set(sourceKey, {
        id: sourceKey,
        label: path.source.label,
        detail: `${path.source.value} · ${STATE_LABELS[path.source.state] || path.source.state || '已记录'}`,
      })
      LAYERS.slice(1).forEach(({ key }) => {
        const node = pathNode(path, key)
        const keyValue = nodeKey(key, node)
        layerNodes.get(key)?.set(keyValue, {
          id: keyValue,
          label: node.label,
        })
      })
    })

    const positioned: PositionedNode[] = []
    const positionMap = new Map<string, PositionedNode>()
    let maxCount = 1
    LAYERS.forEach(({ key }, layerIndex) => {
      const entries = Array.from(layerNodes.get(key)?.values() || [])
      maxCount = Math.max(maxCount, entries.length)
      entries.forEach((node, index) => {
        const position: PositionedNode = {
          ...node,
          x: LAYER_X[layerIndex],
          y: 66 + index * NODE_GAP,
          layer: key,
        }
        positioned.push(position)
        positionMap.set(node.id, position)
      })
    })
    return {
      nodes: positioned,
      positions: positionMap,
      height: Math.max(300, 86 + maxCount * NODE_GAP),
    }
  }, [paths])

  const edges = paths.flatMap((path) => (
    LAYERS.slice(0, -1).map((layer, index) => {
      const nextLayer = LAYERS[index + 1]
      const fromNode = pathNode(path, layer.key)
      const toNode = pathNode(path, nextLayer.key)
      return {
        key: `${path.path_id}:${index}`,
        pathId: path.path_id,
        from: positions.get(nodeKey(layer.key, fromNode)),
        to: positions.get(nodeKey(nextLayer.key, toNode)),
      }
    })
  ))

  return (
    <div className="kg-graph-scroll" aria-label="本次知识图谱关系图">
      <svg
        className="kg-graph-svg"
        width={GRAPH_WIDTH}
        height={height}
        viewBox={`0 0 ${GRAPH_WIDTH} ${height}`}
        role="img"
        aria-label="本次数据到RAG主题的五层知识图谱"
      >
        <defs>
          <marker id={markerId} markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
            <path d="M0,0 L8,4 L0,8 z" className="kg-arrow-head" />
          </marker>
        </defs>
        {LAYERS.map((layer, index) => (
          <g key={layer.key}>
            <text x={LAYER_X[index]} y={22} className="kg-layer-title">{layer.label}</text>
            <text x={LAYER_X[index]} y={40} className="kg-layer-caption">{layer.caption}</text>
          </g>
        ))}
        {edges.map((edge) => {
          if (!edge.from || !edge.to) return null
          const active = edge.pathId === activePathId
          return (
            <path
              key={edge.key}
              d={`M ${edge.from.x + NODE_WIDTH} ${edge.from.y + NODE_HEIGHT / 2}
                  C ${edge.from.x + NODE_WIDTH + 34} ${edge.from.y + NODE_HEIGHT / 2},
                    ${edge.to.x - 34} ${edge.to.y + NODE_HEIGHT / 2},
                    ${edge.to.x} ${edge.to.y + NODE_HEIGHT / 2}`}
              className={`kg-edge${active ? ' active' : ''}`}
              markerEnd={`url(#${markerId})`}
              onMouseEnter={() => onActivePath(edge.pathId)}
            />
          )
        })}
        {nodes.map((node) => (
          <foreignObject
            key={node.id}
            x={node.x}
            y={node.y}
            width={NODE_WIDTH}
            height={NODE_HEIGHT}
          >
            <div className={`kg-svg-node kg-node-${node.layer}`}>
              <strong>{node.label}</strong>
              {node.detail && <span>{node.detail}</span>}
            </div>
          </foreignObject>
        ))}
      </svg>
    </div>
  )
}

export default function KnowledgeGraphPanel({ graph }: Props) {
  const [activePathId, setActivePathId] = useState(graph?.paths[0]?.path_id || '')
  if (!graph) return null

  const tone = modeTone(graph)
  const summary = graph.summary
  const paths = Array.isArray(graph.paths) ? graph.paths : []
  const activePath = paths.find((path) => path.path_id === activePathId) || paths[0]

  return (
    <section className={`card kg-panel kg-panel-${tone}`}>
      <div className="kg-panel-head">
        <div>
          <h2>
            本次知识图谱分析
            <span className="h2-suffix">Graph · RAG · Trace</span>
          </h2>
          <p>展示本次数据如何经过图谱关系形成临床分析维度，并进入 Planner 与 RAG。</p>
        </div>
        <span className={`kg-mode-badge ${tone}`}>
          <Network aria-hidden="true" />
          {modeLabel(graph)}
        </span>
      </div>

      {!graph.applied ? (
        <div className={`kg-mode-message ${tone}`}>
          {graph.status === 'fallback_to_llm_only'
            ? '本次图谱运行异常，系统已按既定策略回退到原始 LLM Planner 流程。'
            : graph.mode === 'llm_only'
              ? '本次使用 LLM Planner 直接生成检索计划，没有启用知识图谱增强。'
              : '本次没有形成可展示的知识图谱路径。'}
        </div>
      ) : (
        <>
          <div className="kg-stat-grid">
            <div><Database aria-hidden="true" /><span>匹配指标</span><strong>{numberValue(summary.indicator_count)}</strong></div>
            <div><GitBranch aria-hidden="true" /><span>图谱路径</span><strong>{numberValue(summary.path_count)}</strong></div>
            <div><BrainCircuit aria-hidden="true" /><span>临床维度</span><strong>{numberValue(summary.dimension_count)}</strong></div>
            <div><Search aria-hidden="true" /><span>图谱主题</span><strong>{numberValue(summary.graph_seeded_topic_count)}</strong></div>
            <div><Network aria-hidden="true" /><span>RAG证据</span><strong>{numberValue(summary.retrieval_evidence_count)}</strong></div>
          </div>

          {paths.length > 0 ? (
            <>
              <div className="kg-section-title">
                <span>关系网络概览</span>
                <small>指向或悬停路径可查看对应链路</small>
              </div>
              <RelationshipGraph
                paths={paths}
                activePathId={activePath?.path_id || ''}
                onActivePath={setActivePathId}
              />
              {activePath && (
                <div className="kg-active-path">
                  <strong>当前路径</strong>
                  <span>{activePath.source.label}（{activePath.source.value}）</span>
                  <i>→</i>
                  <span>{activePath.functional_finding.label}</span>
                  <i>→</i>
                  <span>{activePath.clinical_dimension.label}</span>
                  <i>→</i>
                  <span>{activePath.rag_topic.label}</span>
                </div>
              )}
            </>
          ) : (
            <div className="kg-mode-message disabled">
              该记录保存了图谱运行摘要，但没有保存逐条路径；新生成的记录会显示完整关系图。
            </div>
          )}

          {graph.final_topics.length > 0 && (
            <div className="kg-topic-section">
              <div className="kg-section-title">
                <span>进入 Planner / RAG 的最终主题</span>
                <small>保留图谱与 Planner 来源</small>
              </div>
              <div className="kg-topic-list">
                {graph.final_topics.map((topic) => (
                  <span key={topic.topic_id} className="kg-topic-chip">
                    {topic.label}
                    <small>{originLabel(topic.origins)}</small>
                  </span>
                ))}
              </div>
            </div>
          )}

          {graph.prediction_results.length > 0 && (
            <div className="kg-prediction-section">
              <div className="kg-section-title">
                <span>模型预测结果</span>
                <small>模型预测结果，非人工量表实测结果</small>
              </div>
              <div className="kg-prediction-list">
                {graph.prediction_results.map((pred) => (
                  <div key={pred.target_id} className="kg-prediction-card">
                    <div className="kg-prediction-head">
                      <strong>{pred.target_label}</strong>
                      <span className="kg-prediction-badge">模型预测</span>
                    </div>
                    <div className="kg-prediction-value">
                      {pred.value_text}
                      <small>范围：{pred.range}</small>
                    </div>
                    <div className="kg-prediction-note">{pred.range_note}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="kg-quality-section">
            <div className="kg-section-title">
              <span>数据质量与解释约束</span>
              <small>独立于临床维度，仅限定解释</small>
            </div>
            <div className="kg-quality-card">
              <div className="kg-quality-status">
                质量状态：
                <strong className={`kg-quality-${String(graph.data_quality?.status || 'unknown')}`}>
                  {String(graph.data_quality?.status || 'unknown')}
                </strong>
              </div>
              <div className="kg-quality-fields">
                <span>试次数：{String(graph.data_quality?.trial_count ?? '—')}</span>
                <span>短试次：{String(graph.data_quality?.short_trial_count ?? '—')}</span>
                <span>同步回退：{String(graph.data_quality?.sync_fallback_count ?? '—')}</span>
                <span>采样率不一致：{String(graph.data_quality?.sampling_rate_mismatch_count ?? '—')}</span>
              </div>
              {Array.isArray(graph.data_quality?.warnings) && graph.data_quality!.warnings.length > 0 && (
                <ul className="kg-quality-warnings">
                  {graph.data_quality!.warnings.map((w, index) => (
                    <li key={index}>{w.message}</li>
                  ))}
                </ul>
              )}
            </div>
          </div>

          <details className="kg-path-details">
            <summary>查看全部图谱路径（{paths.length} 条）</summary>
            <div className="mini-table-wrap">
              <table className="mini-table kg-path-table">
                <thead>
                  <tr>
                    <th>本次字段和值</th>
                    <th>图谱指标</th>
                    <th>功能表现</th>
                    <th>临床维度</th>
                    <th>RAG主题</th>
                    <th>关系</th>
                  </tr>
                </thead>
                <tbody>
                  {paths.map((path) => (
                    <tr
                      key={path.path_id}
                      className={path.path_id === activePath?.path_id ? 'active' : ''}
                      onMouseEnter={() => setActivePathId(path.path_id)}
                    >
                      <td>
                        <strong>{path.source.label}</strong>
                        <small>{path.source.value} · {STATE_LABELS[path.source.state] || path.source.state}</small>
                      </td>
                      <td>{path.indicator.label}</td>
                      <td>{path.functional_finding.label}</td>
                      <td>{path.clinical_dimension.label}</td>
                      <td>{path.rag_topic.label}</td>
                      <td>{path.relations.map((relation) => RELATION_LABELS[relation] || relation).join(' → ')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>

          {graph.matched_rules.length > 0 && (
            <details className="kg-path-details">
              <summary>本次匹配的组合规则（{graph.matched_rules.length} 条）</summary>
              <ul className="kg-rule-list">
                {graph.matched_rules.map((rule) => (
                  <li key={rule.rule_id}>
                    <strong>{rule.rule_id}</strong>
                    <span>{rule.message}</span>
                  </li>
                ))}
              </ul>
            </details>
          )}
        </>
      )}

      <div className="kg-prototype-note">{graph.prototype_notice}</div>
    </section>
  )
}
