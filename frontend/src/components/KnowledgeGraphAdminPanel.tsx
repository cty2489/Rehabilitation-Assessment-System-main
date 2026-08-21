import { useEffect, useMemo, useState } from 'react'
import {
  BookOpen,
  Database,
  GitBranch,
  Network,
  RefreshCw,
  Search,
  ShieldCheck,
} from 'lucide-react'
import { fetchClinicalKnowledgeGraph } from '../api'
import {
  ClinicalKnowledgeGraphAdmin,
  ClinicalKnowledgeGraphAdminNode,
  ClinicalKnowledgeGraphAdminRelation,
  ClinicalKnowledgeGraphAdminRule,
} from '../types'

const CATEGORY_OPTIONS = [
  { id: 'model', label: '功能量表与分期', seedTypes: ['ModelPrediction'] },
  { id: 'eeg', label: '脑电 EEG', seedTypes: ['EEGBiomarker'] },
  { id: 'emg', label: '肌电 EMG', seedTypes: ['EMGBiomarker'] },
  { id: 'confounder', label: '影响因素', seedTypes: ['Confounder'] },
  { id: 'all', label: '完整图谱', seedTypes: [] },
] as const

const GRAPH_COLUMNS = [
  {
    id: 'source',
    title: '实现来源',
    caption: '项目代码与字段溯源',
    types: ['EvidenceSource'],
  },
  {
    id: 'indicator',
    title: '指标 / 模型与目标',
    caption: 'EEG、EMG、模型、预测目标与混杂因素',
    types: ['PatientContext', 'EEGBiomarker', 'EMGBiomarker', 'ModelPrediction', 'PredictionTarget', 'IndicatorState', 'Confounder'],
  },
  {
    id: 'finding',
    title: '功能表现',
    caption: '指标所测量或描述的表现',
    types: ['FunctionalFinding'],
  },
  {
    id: 'dimension',
    title: '临床维度',
    caption: '用于组织分析方向',
    types: ['ClinicalDimension'],
  },
  {
    id: 'topic',
    title: 'RAG 主题',
    caption: '交给 Planner 和检索的主题',
    types: ['EvidenceTopic'],
  },
] as const

const RELATION_LABELS: Record<string, string> = {
  MEASURES: '测量或反映',
  SUPPORTS: '支持',
  ASSOCIATED_WITH: '相关',
  MAY_CONTRIBUTE_TO: '可能影响',
  BELONGS_TO: '属于',
  DESCRIBES: '描述',
  ASSESSED_BY: '由…评估',
  SUGGESTS_TOPIC: '生成检索主题',
  MAY_CONFOUND: '可能干扰解释',
  IMPLEMENTED_FROM: '项目实现来源',
  SUPPORTED_BY: '医学证据支持',
  PREDICTS_TARGET: '预测目标',
  HAS_INTERPRETATION_TOPIC: '量表解释主题',
}

const NODE_TYPE_LABELS: Record<string, string> = {
  PatientContext: '患者上下文',
  EEGBiomarker: 'EEG 指标',
  EMGBiomarker: 'EMG 指标',
  ModelPrediction: '模型',
  PredictionTarget: '预测目标',
  IndicatorState: '指标状态',
  FunctionalFinding: '功能表征',
  ClinicalDimension: '临床维度',
  EvidenceTopic: 'RAG 主题',
  Confounder: '混杂因素',
  EvidenceSource: '实现来源',
}

const NODE_WIDTH = 194
const NODE_HEIGHT = 42
const GRAPH_WIDTH = 1260
const COLUMN_X = [26, 278, 530, 782, 1034]

function categoryNodeIds(graph: ClinicalKnowledgeGraphAdmin, category: string): Set<string> {
  if (category === 'all') return new Set(graph.nodes.map((node) => node.node_id))
  const option = CATEGORY_OPTIONS.find((item) => item.id === category)
  const ids = new Set(
    graph.nodes
      .filter((node) => option?.seedTypes.includes(node.node_type as never))
      .map((node) => node.node_id),
  )
  // Follow the real directed graph from the selected indicator family through
  // functional findings, dimensions, topics and implementation sources.
  for (let depth = 0; depth < 4; depth += 1) {
    graph.relations.forEach((relation) => {
      if (ids.has(relation.from)) ids.add(relation.to)
    })
  }
  return ids
}

function columnIndex(node: ClinicalKnowledgeGraphAdminNode): number {
  const index = GRAPH_COLUMNS.findIndex((column) => column.types.includes(node.node_type as never))
  return index >= 0 ? index : 1
}

function nodeClass(node: ClinicalKnowledgeGraphAdminNode): string {
  if (node.node_type === 'EvidenceSource') return 'source'
  if (node.node_type === 'FunctionalFinding') return 'finding'
  if (node.node_type === 'ClinicalDimension') return 'dimension'
  if (node.node_type === 'EvidenceTopic') return 'topic'
  if (node.node_type === 'Confounder') return 'confounder'
  return 'indicator'
}

function ruleCondition(rule: ClinicalKnowledgeGraphAdminRule): string {
  const conditions = rule.when_all || (rule.when ? [rule.when] : [])
  return conditions
    .map((condition) => [
      String(condition.field || ''),
      String(condition.operator || ''),
      String(condition.value ?? ''),
    ].filter(Boolean).join(' '))
    .join(' 且 ')
}

function RelationDetail({
  relation,
  selectedNodeId,
  nodesById,
}: {
  relation: ClinicalKnowledgeGraphAdminRelation
  selectedNodeId: string
  nodesById: Map<string, ClinicalKnowledgeGraphAdminNode>
}) {
  const outgoing = relation.from === selectedNodeId
  const neighbour = nodesById.get(outgoing ? relation.to : relation.from)
  return (
    <li>
      <div className="kg-admin-relation-line">
        <span className="kg-admin-direction">{outgoing ? '→' : '←'}</span>
        <strong>{RELATION_LABELS[relation.type] || relation.type}</strong>
        <span>{neighbour?.label || (outgoing ? relation.to : relation.from)}</span>
      </div>
      <div className="kg-admin-relation-meta">
        <code>{relation.relation_id}</code>
        <span className="badge badge-warn">{relation.evidence_level}</span>
        <span className="badge badge-neutral">{relation.expert_review_status}</span>
        <span>{relation.causal_status === 'associative' ? '相关关系，非确定因果' : relation.causal_status}</span>
      </div>
      <p>{relation.notes}</p>
      <small>来源：{relation.source_reference || '待补充'}</small>
    </li>
  )
}

export default function KnowledgeGraphAdminPanel() {
  const [graph, setGraph] = useState<ClinicalKnowledgeGraphAdmin | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [category, setCategory] = useState('model')
  const [relationType, setRelationType] = useState('all')
  const [query, setQuery] = useState('')
  const [showImplementationSources, setShowImplementationSources] = useState(false)
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)

  const load = () => {
    setLoading(true)
    setError(null)
    fetchClinicalKnowledgeGraph()
      .then(setGraph)
      .catch((reason) => setError(String(reason.message || reason)))
      .finally(() => setLoading(false))
  }

  useEffect(load, [])

  const view = useMemo(() => {
    if (!graph) {
      return {
        nodes: [] as ClinicalKnowledgeGraphAdminNode[],
        relations: [] as ClinicalKnowledgeGraphAdminRelation[],
        positions: new Map<string, { x: number; y: number }>(),
        height: 680,
      }
    }
    const categoryIds = categoryNodeIds(graph, category)
    let relations = graph.relations.filter(
      (relation) => categoryIds.has(relation.from) && categoryIds.has(relation.to),
    )
    if (!showImplementationSources) {
      const sourceIds = new Set(
        graph.nodes
          .filter((node) => node.node_type === 'EvidenceSource')
          .map((node) => node.node_id),
      )
      sourceIds.forEach((id) => categoryIds.delete(id))
      relations = relations.filter(
        (relation) => !sourceIds.has(relation.from) && !sourceIds.has(relation.to),
      )
    }
    if (relationType !== 'all') {
      relations = relations.filter((relation) => relation.type === relationType)
      categoryIds.clear()
      relations.forEach((relation) => {
        categoryIds.add(relation.from)
        categoryIds.add(relation.to)
      })
    }
    if (query.trim()) {
      const normalized = query.trim().toLocaleLowerCase()
      const matchedIds = new Set(
        graph.nodes
          .filter(
            (node) => categoryIds.has(node.node_id)
              && `${node.label} ${node.node_id} ${node.source_field || ''}`
                .toLocaleLowerCase()
                .includes(normalized),
          )
          .map((node) => node.node_id),
      )
      const expandedIds = new Set(matchedIds)
      relations.forEach((relation) => {
        if (matchedIds.has(relation.from) || matchedIds.has(relation.to)) {
          expandedIds.add(relation.from)
          expandedIds.add(relation.to)
        }
      })
      categoryIds.clear()
      expandedIds.forEach((id) => categoryIds.add(id))
      relations = relations.filter(
        (relation) => categoryIds.has(relation.from) && categoryIds.has(relation.to),
      )
    }

    const nodes = graph.nodes.filter((node) => categoryIds.has(node.node_id))
    const grouped = GRAPH_COLUMNS.map((_, index) => (
      nodes
        .filter((node) => columnIndex(node) === index)
        .sort((a, b) => a.label.localeCompare(b.label, 'zh-CN'))
    ))
    const maxRows = Math.max(1, ...grouped.map((items) => items.length))
    const height = Math.max(680, 96 + maxRows * 54)
    const positions = new Map<string, { x: number; y: number }>()
    grouped.forEach((items, column) => {
      const usable = height - 100
      const step = usable / Math.max(1, items.length)
      items.forEach((node, index) => {
        positions.set(node.node_id, {
          x: COLUMN_X[column],
          y: 70 + step * index + Math.max(0, (step - NODE_HEIGHT) / 2),
        })
      })
    })
    return { nodes, relations, positions, height }
  }, [category, graph, query, relationType, showImplementationSources])

  useEffect(() => {
    if (selectedNodeId && !view.positions.has(selectedNodeId)) setSelectedNodeId(null)
  }, [selectedNodeId, view.positions])

  const nodesById = useMemo(
    () => new Map((graph?.nodes || []).map((node) => [node.node_id, node])),
    [graph],
  )
  const selectedNode = selectedNodeId ? nodesById.get(selectedNodeId) || null : null
  const selectedRelations = selectedNodeId
    ? view.relations.filter(
      (relation) => relation.from === selectedNodeId || relation.to === selectedNodeId,
    )
    : []
  const connectedIds = new Set<string>()
  if (selectedNodeId) {
    connectedIds.add(selectedNodeId)
    selectedRelations.forEach((relation) => {
      connectedIds.add(relation.from)
      connectedIds.add(relation.to)
    })
  }

  if (loading && !graph) {
    return <section className="card kg-admin-shell"><div className="empty-state">正在读取知识图谱…</div></section>
  }
  if (error && !graph) {
    return (
      <section className="card kg-admin-shell">
        <div className="error-banner">{error}</div>
        <button className="button secondary" onClick={load}><RefreshCw aria-hidden="true" />重试</button>
      </section>
    )
  }
  if (!graph) return null

  const activeRelationTypes = Object.keys(graph.summary.relation_type_counts)

  return (
    <section className="card settings-wide-card kg-admin-shell">
      <div className="kg-admin-head">
        <div>
          <h2>临床知识图谱<span className="h2-suffix">Knowledge Graph</span></h2>
          <p>先按临床内容查看关键关系，再按需展开完整图谱。患者单次评估只会使用与本次数据有关的子图。</p>
        </div>
        <div className="kg-admin-head-actions">
          <span className="kg-mode-badge warning"><ShieldCheck aria-hidden="true" />专家审核待完成</span>
          <button className="button secondary tiny" onClick={load} disabled={loading}>
            <RefreshCw aria-hidden="true" />
            {loading ? '刷新中…' : '刷新'}
          </button>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div className="kg-admin-stats">
        <div><Network aria-hidden="true" /><span>节点</span><strong>{graph.summary.node_count}</strong></div>
        <div><GitBranch aria-hidden="true" /><span>关系</span><strong>{graph.summary.relation_count}</strong></div>
        <div><Database aria-hidden="true" /><span>组合规则</span><strong>{graph.summary.rule_count}</strong></div>
        <div><ShieldCheck aria-hidden="true" /><span>待审核关系</span><strong>{graph.summary.pending_relation_count}</strong></div>
        <div><BookOpen aria-hidden="true" /><span>医学证据关系</span><strong>{graph.summary.supported_by_count}</strong></div>
      </div>

      <div className="kg-admin-toolbar">
        <div className="kg-admin-category-wrap">
          <span>查看内容</span>
          <div className="kg-admin-category" aria-label="知识图谱类别筛选">
          {CATEGORY_OPTIONS.map((option) => (
            <button
              key={option.id}
              className={category === option.id ? 'active' : ''}
              onClick={() => setCategory(option.id)}
            >
              {option.label}
            </button>
          ))}
          </div>
        </div>
        <label className="kg-admin-search">
          <Search aria-hidden="true" />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="搜索节点、字段或 ID"
          />
        </label>
      </div>

      <details className="kg-admin-advanced">
        <summary>高级筛选与工程溯源</summary>
        <div>
          <label className="kg-admin-select">
            <span>关系类型</span>
            <select value={relationType} onChange={(event) => setRelationType(event.target.value)}>
              <option value="all">全部关系类型</option>
              {activeRelationTypes.map((type) => (
                <option key={type} value={type}>
                  {RELATION_LABELS[type] || type}（{graph.summary.relation_type_counts[type]}）
                </option>
              ))}
            </select>
          </label>
          <label className="kg-admin-source-toggle">
            <input
              type="checkbox"
              checked={showImplementationSources}
              onChange={(event) => setShowImplementationSources(event.target.checked)}
            />
            <span>显示项目代码与字段来源</span>
          </label>
        </div>
      </details>

      <div className="kg-admin-view-summary">
        当前显示 <strong>{view.nodes.length}</strong> 个节点、<strong>{view.relations.length}</strong> 条关系。
        点击节点可聚焦并查看相邻关系；“完整图谱”用于高级查看。
      </div>

      <div className="kg-admin-graph-scroll">
        <svg
          className="kg-admin-graph"
          width={GRAPH_WIDTH}
          height={view.height}
          viewBox={`0 0 ${GRAPH_WIDTH} ${view.height}`}
          role="img"
          aria-label="临床知识图谱关系网络"
        >
          <defs>
            <marker id="kg-admin-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto">
              <path d="M0,0 L8,4 L0,8 z" />
            </marker>
          </defs>
          {GRAPH_COLUMNS.map((column, index) => (
            <g key={column.id}>
              <text x={COLUMN_X[index]} y={28} className="kg-admin-column-title">{column.title}</text>
              <text x={COLUMN_X[index]} y={45} className="kg-admin-column-caption">{column.caption}</text>
            </g>
          ))}
          {view.relations.map((relation) => {
            const from = view.positions.get(relation.from)
            const to = view.positions.get(relation.to)
            if (!from || !to) return null
            const startX = from.x + NODE_WIDTH
            const startY = from.y + NODE_HEIGHT / 2
            const endX = to.x
            const endY = to.y + NODE_HEIGHT / 2
            const bend = Math.max(58, Math.abs(endX - startX) * 0.38)
            const active = Boolean(
              selectedNodeId
              && (relation.from === selectedNodeId || relation.to === selectedNodeId),
            )
            return (
              <path
                key={relation.relation_id}
                className={`kg-admin-edge${active ? ' active' : ''}${selectedNodeId && !active ? ' dimmed' : ''}`}
                d={`M ${startX} ${startY} C ${startX + bend} ${startY}, ${endX - bend} ${endY}, ${endX} ${endY}`}
                markerEnd="url(#kg-admin-arrow)"
              >
                <title>{`${RELATION_LABELS[relation.type] || relation.type} · ${relation.relation_id}`}</title>
              </path>
            )
          })}
          {view.nodes.map((node) => {
            const position = view.positions.get(node.node_id)
            if (!position) return null
            const selected = selectedNodeId === node.node_id
            const dimmed = Boolean(selectedNodeId && !connectedIds.has(node.node_id))
            return (
              <g
                key={node.node_id}
                className={`kg-admin-node-group${selected ? ' selected' : ''}${dimmed ? ' dimmed' : ''}`}
                role="button"
                tabIndex={0}
                aria-label={`${NODE_TYPE_LABELS[node.node_type] || node.node_type}：${node.label}`}
                onClick={() => setSelectedNodeId(node.node_id)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault()
                    setSelectedNodeId(node.node_id)
                  }
                }}
              >
                <foreignObject x={position.x} y={position.y} width={NODE_WIDTH} height={NODE_HEIGHT}>
                  <div className={`kg-admin-node kg-admin-node-${nodeClass(node)}`}>
                    <strong>{node.label}</strong>
                    <span>{NODE_TYPE_LABELS[node.node_type] || node.node_type}</span>
                  </div>
                </foreignObject>
                <title>{node.node_id}</title>
              </g>
            )
          })}
        </svg>
      </div>

      <div className="kg-admin-inspector">
        <div className="kg-admin-node-card">
          {selectedNode ? (
            <>
              <div className="kg-admin-node-card-head">
                <div>
                  <span>{NODE_TYPE_LABELS[selectedNode.node_type] || selectedNode.node_type}</span>
                  <h3>{selectedNode.label}</h3>
                </div>
                <button className="button secondary tiny" onClick={() => setSelectedNodeId(null)}>清除选择</button>
              </div>
              <code>{selectedNode.node_id}</code>
              {selectedNode.source_field && <p><strong>项目字段：</strong>{selectedNode.source_field}</p>}
              {selectedNode.reference && <p><strong>实现位置：</strong>{selectedNode.reference}</p>}
              {selectedNode.query_template && <p><strong>检索模板：</strong>{selectedNode.query_template}</p>}
              <p><strong>相邻关系：</strong>{selectedRelations.length} 条</p>
            </>
          ) : (
            <div className="kg-admin-empty-inspector">
              <Network aria-hidden="true" />
              <strong>选择一个节点查看关系详情</strong>
              <span>将显示关系方向、证据等级、代码来源和专家审核状态。</span>
            </div>
          )}
        </div>
        <div className="kg-admin-relation-card">
          <h3>相邻关系</h3>
          {selectedNode ? (
            <ul>
              {selectedRelations.map((relation) => (
                <RelationDetail
                  key={relation.relation_id}
                  relation={relation}
                  selectedNodeId={selectedNode.node_id}
                  nodesById={nodesById}
                />
              ))}
            </ul>
          ) : (
            <div className="kg-admin-relation-legend">
              {Object.entries(graph.summary.relation_type_counts).map(([type, count]) => (
                <span key={type}><strong>{RELATION_LABELS[type] || type}</strong>{count}</span>
              ))}
            </div>
          )}
        </div>
      </div>

      <details className="kg-path-details kg-admin-rules">
        <summary>查看现有 {graph.summary.rule_count} 条组合规则</summary>
        <ul>
          {graph.rules.map((rule) => (
            <li key={rule.rule_id}>
              <div>
                <strong>{rule.rule_id}</strong>
                <code>{ruleCondition(rule)}</code>
              </div>
              <p>{rule.result.message}</p>
              <div>
                <span className="badge badge-warn">{rule.evidence_level}</span>
                <span className="badge badge-neutral">{rule.expert_review_status}</span>
              </div>
            </li>
          ))}
        </ul>
      </details>

      <p className="kg-prototype-note">
        {graph.prototype_notice} 当前 IMPLEMENTED_FROM（项目实现溯源）共 {graph.summary.implemented_from_count} 条；
        SUPPORTED_BY（真实医学证据关系）共 {graph.summary.supported_by_count} 条，两者不会混用。
      </p>
    </section>
  )
}
