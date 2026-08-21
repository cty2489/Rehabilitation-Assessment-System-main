export type Sex = '男' | '女'
export type ParalysisSide = '' | '左' | '右'

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
  | 'knowledge-graph'
  | 'rag-guidelines'
  // Legacy route kept only so an already-open internal link still resolves.
  | 'rag-guidelines-test'
  | 'system'
  | 'llm-settings'
  | 'llm-control-test'
  | 'llm-benchmark'
  | 'task-interface'

export interface BenchmarkPresetConfig {
  assessment_input_mode: 'manual_clinical_scores' | 'dl_prediction'
  clinical_score_source: 'clinician_provided' | 'dl_prediction'
  rag_enabled: boolean
  rag_version: string | null
  knowledge_graph_enabled: boolean
  kg_version: string | null
  prompt_version: string
  temperature?: number | null
  max_new_tokens?: number | null
}

export interface BenchmarkPresetsResponse {
  prompt_version: string
  score_schema: string[]
  presets: Record<string, BenchmarkPresetConfig>
}

export interface BenchmarkBatchPrepareResponse {
  schema_version: 'rehab.llm-benchmark-batch.v1'
  batch_id: string
  preset: string
  config: BenchmarkPresetConfig
  patient_count: number
  patients: Array<{
    patient_id: string
    n_trials: number
    clinical_scores: {
      fma_wrist: number
      fma_hand: number
      hand_mas: string
      brunnstrom_hand: number
    }
    warnings: string[]
  }>
  clinical_scores_file: string
  output_directory: string
  biomarker_counts: Record<string, number>
  model_execution: 'not_started'
}

export interface BenchmarkSinglePrepareResponse {
  schema_version: 'rehab.llm-benchmark-input.v1'
  input: Record<string, unknown>
  model_execution: 'not_started'
}

export interface LlmControlTestSession {
  session_id: string
  input_fingerprint: string
  n_trials: number
  report_model_id: string
  biomarker_source: 'computed_from_uploaded_signals'
  persisted: false
  dl_inference_skipped: true
}

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

export interface ClinicalKnowledgeGraphAdminNode {
  node_id: string
  node_type: string
  label: string
  source_field?: string
  reference?: string
  query_template?: string
}

export interface ClinicalKnowledgeGraphAdminRelation {
  relation_id: string
  from: string
  type: string
  to: string
  source_type: string
  source_reference: string
  evidence_level: string
  applicable_population: string
  applicable_task: string
  causal_status: string
  expert_review_status: string
  notes: string
}

export interface ClinicalKnowledgeGraphAdminRule {
  rule_id: string
  when?: Record<string, unknown>
  when_all?: Array<Record<string, unknown>>
  result: {
    status: string
    evidence_strength: string
    message: string
  }
  evidence_level: string
  expert_review_status: string
}

export interface ClinicalKnowledgeGraphAdmin {
  schema_version: 'rehab.knowledge-graph-admin.v1'
  graph_schema_version: string
  prototype_scope: string
  node_types: string[]
  relation_types: string[]
  summary: {
    node_count: number
    relation_count: number
    rule_count: number
    node_type_counts: Record<string, number>
    relation_type_counts: Record<string, number>
    pending_relation_count: number
    unverified_relation_count: number
    implemented_from_count: number
    supported_by_count: number
    pending_rule_count: number
  }
  nodes: ClinicalKnowledgeGraphAdminNode[]
  relations: ClinicalKnowledgeGraphAdminRelation[]
  rules: ClinicalKnowledgeGraphAdminRule[]
  expert_review_status: string
  prototype_notice: string
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

export interface RagVersionHealth {
  status: string
  enabled?: boolean
  loaded?: boolean
  collection?: string
  backend?: string
  allow_demo?: boolean
  detail?: string
}

export interface RagVersionInfo {
  display_name?: string
  collection?: string
  release_dir?: string
  default?: boolean
  test_only?: boolean
}

export interface RagVersionsResponse {
  schema_version: string
  selected: string
  default: string
  selection_scope: string
  production_report_rag_unchanged: boolean
  versions: Record<string, RagVersionInfo>
  health: Record<string, RagVersionHealth>
  release?: {
    release: string
    knowledge_base: string
    sources: number
    chunks: number
    embeddings: number
    qdrant_points: number
    collection: string
    source_files: number
    embedding_dimensions: number
    distance: string
    role_counts: Record<string, number>
    textbook_ocr?: Record<string, { chunk_count?: number; rag_indexed?: boolean; ocr_pending?: boolean }>
    failed_page_excluded?: Record<string, unknown>
  }
}

export interface RagTestSearchHit {
  rank: number
  score: number
  knowledge_id: string
  chunk_id: string
  title: string
  text: string
  uid?: string
  page_number?: number | null
  source_file_id?: string
  source_page_url?: string
  source_pdf_url?: string
  metadata?: Record<string, unknown>
}

export interface RagTestSearchResponse {
  selected_rag_version: string
  selection_scope: string
  results: Array<{
    key: string
    query: string
    hits: RagTestSearchHit[]
  }>
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

export interface PatientAssessmentSummary {
  id: number
  created_at: string
  assessment_time: string | null
  fma_ue: number
  hand_tone: string
  hand_function: number
  report_status: ReportStatus
}

export interface PatientAssessmentList {
  total: number
  items: PatientAssessmentSummary[]
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
  | { type: 'knowledge_graph'; graph: KnowledgeGraphDisplay }
  | { type: 'done' }
  | { type: 'cancelled'; message?: string }
  | { type: 'error'; message: string }

export interface BiomarkerCoverage {
  available: number
  total: number
  missing_keys: string[]
}

export interface KnowledgeGraphDisplayNode {
  id: string
  label: string
  type?: string
}

export interface KnowledgeGraphDisplaySource extends KnowledgeGraphDisplayNode {
  value: string
  state: string
  modality: string
}

export interface KnowledgeGraphDisplayPath {
  path_id: string
  source_field: string
  source: KnowledgeGraphDisplaySource
  indicator: KnowledgeGraphDisplayNode
  functional_finding: KnowledgeGraphDisplayNode
  clinical_dimension: KnowledgeGraphDisplayNode
  rag_topic: KnowledgeGraphDisplayNode
  relations: string[]
}

export interface KnowledgeGraphDisplayTopic {
  topic_id: string
  label: string
  priority: string
  origins: string[]
}

export interface KnowledgeGraphDisplayRule {
  rule_id: string
  status: string
  message: string
  evidence_level: string
  expert_review_status: string
}

// MVP：预测结果区（运行时对象，不写入静态图谱）
export interface KnowledgeGraphDisplayPrediction {
  target_id: string
  target_label: string
  value: number | string | null
  value_text: string
  range: string
  range_note: string
  model_label: string
  is_model_prediction: boolean
}

// MVP：数据质量区（独立于临床维度）
export interface KnowledgeGraphDisplayQuality {
  status: string
  trial_count: number | null
  short_trial_count: number | null
  sync_fallback_count: number | null
  sampling_rate_mismatch_count: number | null
  warnings: Array<{ code: string; message: string }>
  is_clinical_dimension: boolean
  blocked_support: boolean
}

export interface KnowledgeGraphDisplay {
  schema_version: string
  mode: string
  status: string
  scope: string
  applied: boolean
  prototype_notice: string
  expert_review_status: string
  summary: {
    indicator_count: number
    path_count: number
    displayed_path_count: number
    dimension_count: number
    final_topic_count: number
    graph_seeded_topic_count: number
    matched_rule_count: number
    retrieval_status: string | null
    retrieval_evidence_count: number
  }
  paths: KnowledgeGraphDisplayPath[]
  paths_truncated: boolean
  analysis_dimensions: Array<{ dimension_id: string; label: string }>
  final_topics: KnowledgeGraphDisplayTopic[]
  matched_rules: KnowledgeGraphDisplayRule[]
  data_quality_warnings: unknown[]
  measurement_context_topics: Array<{ topic_id: string; label: string }>
  prediction_results: KnowledgeGraphDisplayPrediction[]
  data_quality: KnowledgeGraphDisplayQuality
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

export interface GuidelineTestSourceDetail {
  source_type: string
  evidence_tier: string
  authority_weight: string
  source_url: string
  access_status: string
  rights_status: string
  local_excerpt: string
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
  source_detail: GuidelineTestSourceDetail
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
