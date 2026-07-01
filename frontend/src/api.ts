import {
  AssessmentOverview,
  EnrollmentRequest,
  MysqlAssessmentList,
  PatientDetail,
  PatientSummary,
  PatientUpdate,
  StatsSummary,
} from './types'

async function getJSON<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(detail.detail || `HTTP ${res.status}`)
  }
  return res.json() as Promise<T>
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
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(detail.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export function fetchAssessments(limit = 50, offset = 0): Promise<AssessmentOverview> {
  return getJSON(`/api/assessments?limit=${limit}&offset=${offset}`)
}

export function fetchStats(): Promise<StatsSummary> {
  return getJSON('/api/stats/summary')
}

export function fetchHealth(): Promise<{ status: string; models_loaded: string[] }> {
  return getJSON('/api/health')
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
  enrolled: boolean
}

async function postForm<T>(url: string, form: FormData): Promise<T> {
  const res = await fetch(url, { method: 'POST', body: form })
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(detail.detail || `HTTP ${res.status}`)
  }
  return res.json() as Promise<T>
}

export function parseEvalPackage(institution: Institution, file: File): Promise<EvalPackageParse> {
  const form = new FormData()
  form.append('institution', institution)
  form.append('package', file)
  return postForm('/api/task-interface/parse', form)
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
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(detail.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export function fetchMysqlAssessments(limit = 50, offset = 0): Promise<MysqlAssessmentList> {
  return getJSON(`/api/mysql/assessments?limit=${limit}&offset=${offset}`)
}

export async function deleteMysqlAssessment(id: number): Promise<{ deleted: number }> {
  const res = await fetch(`/api/mysql/assessments/${id}`, { method: 'DELETE' })
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(detail.detail || `HTTP ${res.status}`)
  }
  return res.json()
}
