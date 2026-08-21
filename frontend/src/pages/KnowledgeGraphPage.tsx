import KnowledgeGraphAdminPanel from '../components/KnowledgeGraphAdminPanel'

export default function KnowledgeGraphPage() {
  return (
    <div>
      <div className="page-head">
        <div>
          <h1 className="page-title">知识图谱</h1>
          <p className="page-sub">临床指标、功能表现、评估维度与 RAG 检索主题的结构化关系</p>
        </div>
      </div>
      <KnowledgeGraphAdminPanel />
    </div>
  )
}
