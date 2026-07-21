import {
  AssessmentOverview,
  AuthLoginResponse,
  DeviceCredentialList,
  DeviceCredentialRecord,
  DeviceCredentialSecret,
  DeviceCredentialStatus,
  EnrollmentRequest,
  GuidelineTestSearchResponse,
  GuidelineTestStatus,
  HealthStatus,
  KnowledgeCoverageResponse,
  KnowledgeEntriesResponse,
  KnowledgeEntryDetailResponse,
  KnowledgeSourcesResponse,
  KnowledgeStatusResponse,
  LlmModelSettingsPatch,
  LlmSettings,
  MysqlAssessmentDetail,
  MysqlAssessmentList,
  PatientDetail,
  PatientSummary,
  PatientUpdate,
  StatsSummary,
} from './types'

export function authHeaders(): Record<string, string> {
  return {}
}

export async function parseError(res: Response): Promise<Error> {
  if (res.status === 401 || res.status === 403) {
    window.dispatchEvent(new Event('rehab:unauthorized'))
  }
  const detail = await res.json().catch(() => ({ detail: res.statusText }))
  return new Error(detail.detail || `HTTP ${res.status}`)
}

async function getJSON<T>(url: string): Promise<T> {
  const res = await fetch(url, { headers: authHeaders() })
  if (!res.ok) {
    throw await parseError(res)
  }
  return res.json() as Promise<T>
}

export async function loginUser(username: string, password: string): Promise<AuthLoginResponse> {
  const res = await fetch('/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  })
  if (!res.ok) {
    throw await parseError(res)
  }
  return res.json()
}

export async function logoutUser(): Promise<void> {
  await fetch('/api/auth/logout', { method: 'POST' }).catch(() => undefined)
}

export function fetchAuthSession(): Promise<{ user: string }> {
  return getJSON('/api/auth/session')
}

export async function cancelAssessment(sessionId: string): Promise<void> {
  await fetch(`/api/assess/${encodeURIComponent(sessionId)}`, { method: 'DELETE' }).catch(() => undefined)
}

export function fetchPatients(): Promise<PatientSummary[]> {
  return getJSON('/api/patients')
}

export function fetchPatient(id: number): Promise<PatientDetail> {
  return getJSON(`/api/patients/${id}`)
}

export async function updatePatient(
  id: number,
  payload: PatientUpdate,
): Promise<PatientDetail> {
  const res = await fetch(`/api/patients/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    throw await parseError(res)
  }
  return res.json()
}

export function fetchAssessments(limit = 50, offset = 0): Promise<AssessmentOverview> {
  return getJSON(`/api/assessments?limit=${limit}&offset=${offset}`)
}

export function fetchStats(): Promise<StatsSummary> {
  return getJSON('/api/stats/summary')
}

export function fetchHealth(): Promise<HealthStatus> {
  return getJSON('/api/health')
}

export function fetchKnowledgeStatus(): Promise<KnowledgeStatusResponse> {
  return getJSON('/api/admin/knowledge/status')
}

export function fetchKnowledgeEntries(): Promise<KnowledgeEntriesResponse> {
  return getJSON('/api/admin/knowledge/entries')
}

export function fetchKnowledgeCoverage(): Promise<KnowledgeCoverageResponse> {
  return getJSON('/api/admin/knowledge/coverage')
}

export function fetchKnowledgeSources(): Promise<KnowledgeSourcesResponse> {
  return getJSON('/api/admin/knowledge/sources')
}

export function fetchKnowledgeEntry(knowledgeId: string): Promise<KnowledgeEntryDetailResponse> {
  return getJSON(`/api/admin/knowledge/entries/${encodeURIComponent(knowledgeId)}`)
}

export function fetchLlmSettings(): Promise<LlmSettings> {
  return getJSON('/api/settings/llm')
}

export function fetchDeviceCredentials(): Promise<DeviceCredentialList> {
  return getJSON('/api/admin/device-credentials')
}

export async function createDeviceCredential(
  deviceId: string,
  label: string,
): Promise<DeviceCredentialSecret> {
  const res = await fetch('/api/admin/device-credentials', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ device_id: deviceId, label }),
  })
  if (!res.ok) throw await parseError(res)
  return res.json()
}

export async function updateDeviceCredential(
  id: number,
  patch: { label?: string; status?: Exclude<DeviceCredentialStatus, 'revoked'> },
): Promise<DeviceCredentialRecord> {
  const res = await fetch(`/api/admin/device-credentials/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(patch),
  })
  if (!res.ok) throw await parseError(res)
  return res.json()
}

export async function rotateDeviceCredential(id: number): Promise<DeviceCredentialSecret> {
  const res = await fetch(`/api/admin/device-credentials/${id}/rotate`, {
    method: 'POST',
    headers: authHeaders(),
  })
  if (!res.ok) throw await parseError(res)
  return res.json()
}

export async function revokeDeviceCredential(id: number): Promise<DeviceCredentialRecord> {
  const res = await fetch(`/api/admin/device-credentials/${id}`, {
    method: 'DELETE',
    headers: authHeaders(),
  })
  if (!res.ok) throw await parseError(res)
  return res.json()
}

export async function updateLlmSettings(activeModelId: string): Promise<LlmSettings> {
  const res = await fetch('/api/settings/llm', {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ active_model_id: activeModelId }),
  })
  if (!res.ok) {
    throw await parseError(res)
  }
  return res.json()
}

export async function updateLlmModelSettings(
  modelId: string,
  payload: LlmModelSettingsPatch,
): Promise<LlmSettings> {
  const res = await fetch(`/api/settings/llm/models/${encodeURIComponent(modelId)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    throw await parseError(res)
  }
  return res.json()
}

export type AssessmentExportKind = 'json' | 'pdf' | 'zip'

function filenameFromDisposition(disposition: string | null, fallback: string): string {
  if (!disposition) return fallback
  const utf8Match = disposition.match(/filename\*=UTF-8''([^;]+)/i)
  if (utf8Match) return decodeURIComponent(utf8Match[1])
  const asciiMatch = disposition.match(/filename=\"?([^\";]+)\"?/i)
  return asciiMatch ? asciiMatch[1] : fallback
}

export async function downloadAssessmentExport(
  id: number,
  kind: AssessmentExportKind,
): Promise<void> {
  const suffix = kind === 'json' ? 'export.json' : kind === 'pdf' ? 'report.pdf' : 'export.zip'
  const fallback = `rehab_assessment_${id}.${kind}`
  const res = await fetch(`/api/mysql/assessments/${id}/${suffix}`, {
    headers: authHeaders(),
  })
  if (!res.ok) {
    throw await parseError(res)
  }
  const blob = await res.blob()
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filenameFromDisposition(res.headers.get('Content-Disposition'), fallback)
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

// 任务一与任务三对接接口 ---------------------------------------------------- //
export type Institution = 'hospital' | 'device'

export interface EvalPackagePrefill {
  patient_id: string
  name: string
  sex: string
  age: number | null
  diagnosis: string
  disease_days: number | null
  paralysis_side: string
}

export interface EvalPackageParse {
  institution: Institution
  n_trials: number
  patient_prefill: EvalPackagePrefill
  manifest_summary: Record<string, unknown>
  warnings: string[]
  package_hash: string
  upload_id: string
  enrolled: boolean
}

export interface PackageUploadProgress {
  phase: 'uploading' | 'server_processing'
  loadedBytes: number
  totalBytes: number
  percent: number
}

async function postForm<T>(url: string, form: FormData): Promise<T> {
  const res = await fetch(url, { method: 'POST', headers: authHeaders(), body: form })
  if (!res.ok) {
    throw await parseError(res)
  }
  return res.json() as Promise<T>
}

export function parseEvalPackage(
  institution: Institution,
  file: File,
  onProgress?: (progress: PackageUploadProgress) => void,
): Promise<EvalPackageParse> {
  const form = new FormData()
  form.append('institution', institution)
  form.append('package', file)

  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest()
    request.open('POST', '/api/task-interface/parse')
    request.withCredentials = true
    request.timeout = 30 * 60 * 1000
    Object.entries(authHeaders()).forEach(([name, value]) => request.setRequestHeader(name, value))

    request.upload.onprogress = (event) => {
      const totalBytes = event.lengthComputable && event.total > 0
        ? event.total
        : file.size
      const percent = totalBytes > 0
        ? Math.min(99, Math.round((event.loaded / totalBytes) * 100))
        : 0
      onProgress?.({
        phase: 'uploading',
        loadedBytes: event.loaded,
        totalBytes,
        percent,
      })
    }
    request.upload.onload = () => {
      onProgress?.({
        phase: 'server_processing',
        loadedBytes: file.size,
        totalBytes: file.size,
        percent: 100,
      })
    }

    request.onload = () => {
      let payload: unknown = null
      try {
        payload = request.responseText ? JSON.parse(request.responseText) : null
      } catch {
        payload = null
      }
      if (request.status >= 200 && request.status < 300) {
        resolve(payload as EvalPackageParse)
        return
      }
      if (request.status === 401 || request.status === 403) {
        window.dispatchEvent(new Event('rehab:unauthorized'))
      }
      const detail = payload && typeof payload === 'object'
        ? (payload as { detail?: unknown }).detail
        : null
      reject(new Error(typeof detail === 'string' ? detail : `HTTP ${request.status}`))
    }
    request.onerror = () => reject(new Error('数据包上传失败，请检查网络连接后重试'))
    request.ontimeout = () => reject(new Error('数据包上传或服务器校验超时，请检查网络后重试'))
    request.onabort = () => reject(new Error('数据包上传已取消'))
    request.send(form)
  })
}

export function submitOffline(form: FormData): Promise<{ session_id: string; n_trials: number }> {
  return postForm('/api/task-interface/offline', form)
}

export function fetchOnlineStatus(): Promise<{ status: string; device_url: string; message: string }> {
  return getJSON('/api/task-interface/online/status')
}

// 设备端 MySQL 存储：入组 + 评估记录列表 + 删除 ----------------------------- //
export async function enrollPatient(payload: EnrollmentRequest): Promise<unknown> {
  const res = await fetch('/api/mysql/enroll', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    throw await parseError(res)
  }
  return res.json()
}

export function fetchMysqlAssessments(limit = 50, offset = 0): Promise<MysqlAssessmentList> {
  return getJSON(`/api/mysql/assessments?limit=${limit}&offset=${offset}`)
}

export function fetchMysqlAssessment(id: number): Promise<MysqlAssessmentDetail> {
  return getJSON(`/api/mysql/assessments/${id}`)
}

export async function deleteMysqlAssessment(id: number): Promise<{ deleted: number }> {
  const res = await fetch(`/api/mysql/assessments/${id}`, {
    method: 'DELETE',
    headers: authHeaders(),
  })
  if (!res.ok) {
    throw await parseError(res)
  }
  return res.json()
}

// Knowledge and research evidence retrieval API ----------------------------- //
export function fetchGuidelineStatus(): Promise<GuidelineTestStatus> {
  return getJSON('/api/rag/guidelines/status')
}

export async function searchGuidelines(
  query: string,
  topK = 3,
): Promise<GuidelineTestSearchResponse> {
  const res = await fetch('/api/rag/guidelines/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ query, top_k: topK }),
  })
  if (!res.ok) {
    throw await parseError(res)
  }
  return res.json()
}

// Compatibility aliases for older internal callers. The browser uses the
// neutral endpoints above; these names can be removed after downstream tests
// and integrations migrate.
export const fetchGuidelineTestStatus = fetchGuidelineStatus
export const searchGuidelineTest = searchGuidelines
