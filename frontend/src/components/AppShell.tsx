import { useEffect, useState } from 'react'
import Sidebar from './Sidebar'
import TopBar from './TopBar'
import { useRoute } from '../app/AppContext'
import DashboardPage from '../pages/DashboardPage'
import PatientManagementPage from '../pages/PatientManagementPage'
import AssessmentPage from '../pages/AssessmentPage'
import RecordsOverviewPage from '../pages/RecordsOverviewPage'
import StatisticsPage from '../pages/StatisticsPage'
import SystemManagementPage from '../pages/SystemManagementPage'
import ModelSettingsPage from '../pages/ModelSettingsPage'
import TaskInterfacePage from '../pages/TaskInterfacePage'
import KnowledgeGovernancePage from '../pages/KnowledgeGovernancePage'
import KnowledgeGraphPage from '../pages/KnowledgeGraphPage'
import LlmControlTestPage from '../pages/LlmControlTestPage'
import LlmBenchmarkPage from '../pages/LlmBenchmarkPage'
import StrategyReportPage from '../pages/StrategyReportPage'

export default function AppShell() {
  const { route } = useRoute()
  const [taskInterfaceVisited, setTaskInterfaceVisited] = useState(
    route === 'task-interface',
  )

  useEffect(() => {
    if (route === 'task-interface') setTaskInterfaceVisited(true)
  }, [route])

  const renderTaskInterface = taskInterfaceVisited || route === 'task-interface'

  return (
    <div className="layout">
      <Sidebar />
      <div className="layout-main">
        <TopBar />
        <main className="layout-content">
          {route === 'dashboard' && <DashboardPage />}
          {route === 'patients' && <PatientManagementPage />}
          {route === 'assessment' && <AssessmentPage />}
          {route === 'strategy-reports' && <StrategyReportPage />}
          {route === 'records' && <RecordsOverviewPage />}
          {route === 'stats' && <StatisticsPage />}
          {renderTaskInterface && (
            <div hidden={route !== 'task-interface'}>
              <TaskInterfacePage />
            </div>
          )}
          {route === 'knowledge' && <KnowledgeGovernancePage />}
          {route === 'knowledge-graph' && <KnowledgeGraphPage />}
          {(route === 'rag-guidelines' || route === 'rag-guidelines-test') && <KnowledgeGovernancePage initialTab="search" />}
          {route === 'system' && <SystemManagementPage />}
          {route === 'llm-settings' && <ModelSettingsPage />}
          {route === 'llm-control-test' && <LlmControlTestPage />}
          {route === 'llm-benchmark' && <LlmBenchmarkPage />}
        </main>
      </div>
    </div>
  )
}
