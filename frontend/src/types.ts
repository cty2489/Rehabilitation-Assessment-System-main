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
  | 'knowledge'
  | 'rag-guidelines'
  // Legacy route kept only so an already-open internal link still resolves.
  | 'rag-guidelines-test'
  | 'system'
  | 'llm-settings'
  | 'task-interface'

export type ReportStatus = 'generated' | 'failed' | 'manual'

export interface AuthLoginResponse {
  user: string
  expires_in: number
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

export type DeviceCredentialStatus = 'active' | 'disabled' | 'revoked'

export interface DeviceCredentialRecord {
  id: number
  device_id: string
  label: string | null
  access_scope: 'device' | 'shared'
  token_hint: string
  status: DeviceCredentialStatus
  source: string
  created_by: string | null
  created_at: string
  updated_at: string
  last_used_at: string | null
  rotated_at: string | null
  revoked_at: string | null
  job_count?: number
  last_job_at?: string | null
}

export interface DeviceCredentialList {
  schema_version: 'rehab.device_credentials.v1'
  items: DeviceCredentialRecord[]
}

export interface DeviceCredentialSecret {
  schema_version: 'rehab.device_credential_secret.v1'
  credential: DeviceCredentialRecord
  token: string
}

export interface HealthStatus {
  status: string
  models_loaded: string[]
  report_provider?: string
  report_model?: string
  app_version?: string
  build_commit?: string
}

export interface KnowledgeStatusCount {
  status: string
  label: string
  count: number
  biomarker_count: number
}

export interface KnowledgeEntrySummary {
  knowledge_id: string
  entry_version: string
  title: string
  category: string
  system_key: string
  knowledge_status: string
  knowledge_status_label: string
  clinical_ready: boolean
  demo_ready: boolean
  expert_verified: boolean
  expert_review_status: string
  source_ids: string[]
  issues: string[]
}

export interface KnowledgeSource {
  schema_version: string
  source_id: string
  title: string
  year: number | string | null
  source_type: string
  evidence_tier: string
  url: string
  scope: string
  note: string
  knowledge_ids: string[]
}

export interface KnowledgeStatusResponse {
  schema_version: string
  available: boolean
  error?: string
  versions: {
    application: string
    build_commit: string
    report_model: string
    content_release: string
    source_document: string
    index_collection: string
    index_built_at_utc: string
  }
  rag: {
    mode: string
    assist_approved: boolean
    demo_in_prompt: boolean
    service: {
      reachable: boolean
      status: string
      collection: string
      collection_matches: boolean
    }
  }
  counts: {
    total_entries: number
    mapped_biomarkers: number
    clinical_ready_biomarkers: number
    expert_verified_entries: number
    sources: number
  }
  status_counts: KnowledgeStatusCount[]
  trial_release: {
    release_id?: string
    expert_verified?: boolean
    clinical_ready?: boolean
    warning?: string
    allowed_usage?: string[]
    prohibited_usage?: string[]
  }
  validation: {
    valid: boolean
    issues: string[]
  }
}

export interface KnowledgeEntriesResponse {
  schema_version: string
  total: number
  items: KnowledgeEntrySummary[]
  filters: {
    categories: string[]
    statuses: KnowledgeStatusCount[]
  }
}

export interface KnowledgeCoverageResponse {
  schema_version: string
  expected: number
  mapped: number
  clinical_ready: number
  items: KnowledgeEntrySummary[]
}

export interface KnowledgeSourcesResponse {
  schema_version: string
  total: number
  items: KnowledgeSource[]
}

export interface KnowledgeEntryDetail extends KnowledgeEntrySummary {
  applicable_population: string[]
  content: string
  allowed_interpretation: string
  prohibited_interpretation: string
  acquisition_and_algorithm_requirements: string
  reference_range_policy: string
  implementation_action: string
  review_notes: string[]
  governance: Record<string, unknown>
  source_document: Record<string, unknown>
  sources: KnowledgeSource[]
}

export interface KnowledgeEntryDetailResponse {
  schema_version: string
  entry: KnowledgeEntryDetail
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
  quality_json?: unknown
  validation_status?: string | null
  report_generation?: string | null
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
  hand_function: number | null
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
  hand_function?: number | null
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
  validation_status: string | null
  report_generation: string | null
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
  quality_json: unknown
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
  | { type: 'assessment_queued'; ahead: number }
  | { type: 'report_queued'; ahead: number }
  | {
      type: 'biomarker_coverage'
      available: number
      total: number
      missing_keys: string[]
    }
  | { type: 'done' }
  | { type: 'cancelled'; message?: string }
  | { type: 'error'; message: string }

export interface BiomarkerCoverage {
  available: number
  total: number
  missing_keys: string[]
}

// Isolated guideline RAG test page. These responses are never clinical-ready. //
export interface GuidelineTestStatus {
  mode: 'test_only'
  allowed_rag_mode: 'test_only'
  enabled: boolean
  service_reachable: boolean
  collection: string
  clinical_ready: false
  allow_demo: boolean
  error?: string
}

export interface GuidelineTestReference {
  index: number
  source_id: string
  title: string
  year: string
  doi: string
  page_locator: string
}

export interface GuidelineTestHit {
  rank: number
  score: number
  source_id: string
  title: string
  year: string
  doi: string
  page_locator: string
  text: string
  citation_index: number
  citation_indices: number[]
  chunk_id: string
  references: GuidelineTestReference[]
  source_type: string
  knowledge_type: string
  evidence_scope: string
  research_type: string
  sample_size: string
  applicable_scope: string
  limitations: string[]
  license: string
  non_clinical_statement: string
  research_only: boolean
  expert_verified: boolean
}

export interface GuidelineTestSearchResponse {
  schema_version: string
  mode: 'test_only'
  allowed_rag_mode: 'test_only'
  test_report_banner: string
  query: string
  top_k: number
  dataset: string
  clinical_ready: false
  results: GuidelineTestHit[]
  cached: boolean
  elapsed_ms: number
  citations: GuidelineTestReference[]
  reason_code: string
  blocked_message?: string
}
