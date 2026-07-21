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
import GuidelineTestPage from '../pages/GuidelineTestPage'

export default function AppShell() {
  const { route } = useRoute()

  return (
    <div className="layout">
      <Sidebar />
      <div className="layout-main">
        <TopBar />
        <main className="layout-content">
          {route === 'dashboard' && <DashboardPage />}
          {route === 'patients' && <PatientManagementPage />}
          {route === 'assessment' && <AssessmentPage />}
          {route === 'records' && <RecordsOverviewPage />}
          {route === 'stats' && <StatisticsPage />}
          {route === 'task-interface' && <TaskInterfacePage />}
          {route === 'knowledge' && <KnowledgeGovernancePage />}
          {(route === 'rag-guidelines' || route === 'rag-guidelines-test') && <GuidelineTestPage />}
          {route === 'system' && <SystemManagementPage />}
          {route === 'llm-settings' && <ModelSettingsPage />}
        </main>
      </div>
    </div>
  )
}
