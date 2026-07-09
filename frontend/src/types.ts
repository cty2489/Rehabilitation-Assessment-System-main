export type Sex = '男' | '女'
export type ParalysisSide = '左' | '右'

// Age / disease_days use `number | ''` so the form fields can be genuinely
// empty (no forced 0, no leading-zero artifacts).
export interface PatientInfo {
  patient_id: string
  name: string
  sex: Sex
  age: number | ''
  diagnosis: string
  disease_days: number | ''
  paralysis_side: ParalysisSide
}

export const DIAGNOSIS_OPTIONS = ['脑外伤', '脑梗死', '脑出血', '其他'] as const

// Navigation routes (no react-router; route enum drives the AppShell). ------ //
export type Route =
  | 'dashboard'
  | 'patients'
  | 'assessment'
  | 'records'
  | 'stats'
  | 'system'
  | 'llm-settings'
  | 'task-interface'

export type ReportStatus = 'generated' | 'failed' | 'manual'

export interface AuthLoginResponse {
  access_token: string
  token_type: 'bearer'
  user: string
}

export interface LlmModelHealth {
  reachable?: boolean
  loaded?: boolean
  status?: string
  error?: string
  detail?: unknown
}

export interface LlmModelOption {
  id: string
  name: string
  vendor?: string
  origin?: string
  provider: 'remote' | 'local' | 'deepseek' | string
  model_id?: string
  remote_url?: string
  weight_path?: string
  adapter_dir?: string
  enabled?: boolean
  description?: string
  is_active?: boolean
  configured?: boolean
  available?: boolean
  status?: string
  weight_exists?: boolean
  report_ready?: boolean
  health?: LlmModelHealth | null
}

export interface LlmSettings {
  schema_version: string
  config_path: string
  active_model_id: string
  active_model: LlmModelOption | null
  models: LlmModelOption[]
}

export interface LlmModelSettingsPatch {
  weight_path?: string
  remote_url?: string
  enabled?: boolean
  adapter_dir?: string
  use_adapter?: boolean
}

export interface HealthStatus {
  status: string
  models_loaded: string[]
  report_provider?: string
  report_model?: string
}

// Backend-mirrored persistence types --------------------------------------- //
export interface AssessmentRecord {
  id: number
  source?: string | null
  assessment_id?: string | null
  session_id: string | null
  package_name?: string | null
  institution?: string | null
  n_trials?: number | null
  package_hash?: string | null
  created_at: string
  assessment_time?: string | null
  fma_ue: number
  hand_tone: string
  hand_function: number
  report: string | null
  report_status: ReportStatus
  biomarkers?: unknown
  parse_warnings?: unknown
  prediction_json?: unknown
  model_version?: string | null
  llm_provider?: string | null
  llm_model?: string | null
  trials?: AssessmentTrial[]
  biomarker_items?: AssessmentBiomarkerItem[]
}

export interface AssessmentTrial {
  id: number
  trial_index: number | null
  assessment_type: string | null
  action_name: string | null
  eeg_file: string | null
  emg_file: string | null
  eeg_name: string | null
  emg_name: string | null
  status: string | null
  note: string | null
  created_at: string
}

export interface AssessmentBiomarkerItem {
  id: number
  group_key: string | null
  group_label: string | null
  marker_key: string
  marker_name: string | null
  value_text: string | null
  value_num: number | null
  unit: string | null
  ref_range: string | null
  n_valid: number | null
  available: boolean
  note: string | null
  created_at: string
}

export interface PatientSummary {
  id: number
  patient_id: string
  name: string
  sex: string
  age: number | null
  diagnosis: string
  disease_days: number | null
  paralysis_side: string
  birth_date: string | null
  id_number: string | null
  phone: string | null
  onset_date: string | null
  created_at: string
  updated_at: string
  assessment_count: number
  last_assessed_at: string | null
}

export interface PatientDetail extends PatientSummary {
  assessments: AssessmentRecord[]
}

export interface PatientUpdate {
  name?: string
  sex?: Sex
  age?: number | null
  diagnosis?: string
  disease_days?: number | null
  paralysis_side?: ParalysisSide
  birth_date?: string | null
  id_number?: string | null
  phone?: string | null
  onset_date?: string | null
}

export interface AssessmentOverviewItem {
  id: number
  created_at: string
  patient_db_id: number
  patient_id: string
  name: string
  fma_ue: number
  hand_tone: string
  hand_function: number
  report_status: ReportStatus
}

export interface AssessmentOverview {
  total: number
  items: AssessmentOverviewItem[]
}

export interface StatsSummary {
  patient_count: number
  assessment_count: number
  report_failed_count: number
  diagnosis_distribution: Record<string, number>
  hand_function_distribution: Record<string, number>
  avg_fma_ue: number | null
  assessments_by_day: { date: string; count: number }[]
}

// Device-end (task-interface) MySQL store ---------------------------------- //
export interface EnrollmentRequest {
  patient_id: string
  name: string
  sex: Sex
  age?: number | null
  diagnosis?: string | null
  paralysis_side?: ParalysisSide | null
  disease_days?: number | null
  // 第一次评估记录（医院手工录入，可全空表示仅入组基本信息）
  fma_ue?: number | null
  hand_tone?: string | null
  hand_function?: number | null
  assessment_time?: string | null
  report?: string | null
}

export interface MysqlAssessmentItem {
  id: number
  created_at: string
  patient_db_id: number
  patient_id: string
  name: string | null
  source: string
  assessment_id: string | null
  session_id: string | null
  package_name: string | null
  institution: string | null
  n_trials: number | null
  package_hash: string | null
  assessment_time: string | null
  fma_ue: number
  hand_tone: string
  hand_function: number
  report_status: string
  model_version: string | null
  llm_provider: string | null
  llm_model: string | null
}

export interface MysqlAssessmentDetail extends MysqlAssessmentItem {
  sex: string | null
  age: number | null
  diagnosis: string | null
  paralysis_side: string | null
  disease_days: number | null
  report: string | null
  biomarkers: unknown
  parse_warnings: unknown
  prediction_json: unknown
  trials?: AssessmentTrial[]
  biomarker_items?: AssessmentBiomarkerItem[]
}

export interface MysqlAssessmentList {
  total: number
  items: MysqlAssessmentItem[]
}

export type StepKey =
  | 'parse'
  | 'preprocess'
  | 'alignment'
  | 'feature_extract'
  | 'graph_fusion'
  | 'inference'
  | 'report'

export type StepStatus = 'pending' | 'running' | 'done'

export interface StepState {
  key: StepKey
  label: string
  status: StepStatus
  details: string[]
}

export type TaskKey = 'FMA_UE' | 'hand_tone' | 'hand_function'

export interface PredictionEntry {
  task: TaskKey
  label: string
  value: number | string
  range?: string
}

// SSE event union ------------------------------------------------------- //
export type SSEEvent =
  | { type: 'step_start'; step: StepKey; label: string }
  | { type: 'step_detail'; step: StepKey; detail: string }
  | { type: 'step_done'; step: StepKey }
  | {
      type: 'prediction'
      task: TaskKey
      value: number | string
      label: string
      range?: string
    }
  | { type: 'report_chunk'; chunk: string }
  | { type: 'report_queued'; ahead: number }
  | {
      type: 'biomarker_coverage'
      available: number
      total: number
      missing_keys: string[]
    }
  | { type: 'done' }
  | { type: 'error'; message: string }

export interface BiomarkerCoverage {
  available: number
  total: number
  missing_keys: string[]
}
