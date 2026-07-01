import { useEffect, useState } from 'react'
import { fetchPatient, fetchPatients, updatePatient } from '../api'
import { useRoute } from '../app/AppContext'
import RecordDetail from '../components/RecordDetail'
import { DIAGNOSIS_OPTIONS, PatientDetail, PatientSummary, PatientUpdate } from '../types'
import { fmtDate, fmtDateTime } from '../util'

export default function PatientManagementPage() {
  const { selectedPatientId, navigate } = useRoute()

  if (selectedPatientId != null) {
    return <PatientDetailView id={selectedPatientId} />
  }
  return <PatientListView onOpen={(id) => navigate('patients', id)} />
}

// --------------------------------------------------------------------------- //
// List                                                                        //
// --------------------------------------------------------------------------- //
function PatientListView({ onOpen }: { onOpen: (id: number) => void }) {
  const [patients, setPatients] = useState<PatientSummary[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchPatients().then(setPatients).catch((e) => setError(String(e.message || e)))
  }, [])

  return (
    <div>
      <div className="page-head">
        <div>
          <h1 className="page-title">患者管理</h1>
          <p className="page-sub">通过智能康复评估的患者档案，按患者聚合多次评估记录</p>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div className="card">
        {patients === null ? (
          <p className="muted">加载中…</p>
        ) : patients.length === 0 ? (
          <p className="muted">暂无患者记录。完成一次康复评估后将自动归档到此处。</p>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>患者编号</th>
                <th>姓名</th>
                <th>性别</th>
                <th>年龄</th>
                <th>诊断</th>
                <th>评估次数</th>
                <th>最近评估</th>
              </tr>
            </thead>
            <tbody>
              {patients.map((p) => (
                <tr key={p.id} className="clickable" onClick={() => onOpen(p.id)}>
                  <td>{p.patient_id}</td>
                  <td>{p.name}</td>
                  <td>{p.sex}</td>
                  <td>{p.age ?? '—'}</td>
                  <td>{p.diagnosis}</td>
                  <td>{p.assessment_count}</td>
                  <td>{fmtDateTime(p.last_assessed_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

// --------------------------------------------------------------------------- //
// Detail                                                                       //
// --------------------------------------------------------------------------- //
function PatientDetailView({ id }: { id: number }) {
  const { navigate } = useRoute()
  const [patient, setPatient] = useState<PatientDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [editing, setEditing] = useState(false)
  const [form, setForm] = useState<PatientUpdate>({})
  const [saving, setSaving] = useState(false)
  const [openRecords, setOpenRecords] = useState<Record<number, boolean>>({})

  const load = () =>
    fetchPatient(id)
      .then((p) => {
        setPatient(p)
        setForm({
          name: p.name,
          sex: p.sex as PatientUpdate['sex'],
          age: p.age,
          diagnosis: p.diagnosis,
          disease_days: p.disease_days,
          paralysis_side: p.paralysis_side as PatientUpdate['paralysis_side'],
          birth_date: p.birth_date ?? '',
          id_number: p.id_number ?? '',
          phone: p.phone ?? '',
          onset_date: p.onset_date ?? '',
        })
      })
      .catch((e) => setError(String(e.message || e)))

  useEffect(() => {
    load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id])

  const save = async () => {
    setSaving(true)
    setError(null)
    try {
      const payload: PatientUpdate = {
        ...form,
        // Normalize empty strings to null for optional text fields.
        birth_date: form.birth_date || null,
        id_number: form.id_number || null,
        phone: form.phone || null,
        onset_date: form.onset_date || null,
        age: form.age === ('' as unknown as number) ? null : form.age,
        disease_days:
          form.disease_days === ('' as unknown as number) ? null : form.disease_days,
      }
      const updated = await updatePatient(id, payload)
      setPatient(updated)
      setEditing(false)
    } catch (e: any) {
      setError(String(e.message || e))
    } finally {
      setSaving(false)
    }
  }

  if (error && !patient) return <div className="error-banner">{error}</div>
  if (!patient) return <p className="muted">加载中…</p>

  const set = (k: keyof PatientUpdate, v: any) => setForm((f) => ({ ...f, [k]: v }))

  return (
    <div>
      <div className="page-head">
        <div>
          <button className="link-back" onClick={() => navigate('patients', null)}>
            ← 返回患者列表
          </button>
          <h1 className="page-title">
            {patient.name}
            <span className="muted-inline">（{patient.patient_id}）</span>
          </h1>
          <p className="page-sub">
            共 {patient.assessment_count} 次评估 · 最近 {fmtDateTime(patient.last_assessed_at)}
          </p>
        </div>
        <button className="button" onClick={() => navigate('assessment')}>
          为该患者新评估
        </button>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {/* Basic + extended info (editable) */}
      <div className="card">
        <h2>
          基本信息
          <span className="h2-suffix">Profile · Editable</span>
          {!editing ? (
            <button className="button secondary inline-btn" onClick={() => setEditing(true)}>
              编辑
            </button>
          ) : (
            <span className="inline-btn-group">
              <button className="button secondary inline-btn" onClick={() => { setEditing(false); load() }}>
                取消
              </button>
              <button className="button inline-btn" onClick={save} disabled={saving}>
                {saving ? '保存中…' : '保存'}
              </button>
            </span>
          )}
        </h2>

        {!editing ? (
          <div className="info-grid">
            <Info label="姓名" value={patient.name} />
            <Info label="性别" value={patient.sex} />
            <Info label="年龄" value={patient.age ?? '—'} />
            <Info label="诊断" value={patient.diagnosis} />
            <Info label="病程（天）" value={patient.disease_days ?? '—'} />
            <Info label="偏瘫侧" value={`${patient.paralysis_side}侧`} />
            <Info label="出生年月日" value={fmtDate(patient.birth_date)} />
            <Info label="身份证号" value={patient.id_number || '—'} />
            <Info label="手机号" value={patient.phone || '—'} />
            <Info label="发病日期" value={fmtDate(patient.onset_date)} />
          </div>
        ) : (
          <div className="grid-2">
            <Field label="姓名">
              <input value={form.name ?? ''} onChange={(e) => set('name', e.target.value)} />
            </Field>
            <Field label="性别">
              <select value={form.sex} onChange={(e) => set('sex', e.target.value)}>
                <option value="男">男</option>
                <option value="女">女</option>
              </select>
            </Field>
            <Field label="年龄">
              <input
                type="number"
                value={form.age ?? ''}
                onChange={(e) => set('age', e.target.value === '' ? '' : Number(e.target.value))}
              />
            </Field>
            <Field label="诊断">
              <select value={form.diagnosis} onChange={(e) => set('diagnosis', e.target.value)}>
                {DIAGNOSIS_OPTIONS.map((o) => (
                  <option key={o} value={o}>{o}</option>
                ))}
              </select>
            </Field>
            <Field label="病程（天）">
              <input
                type="number"
                value={form.disease_days ?? ''}
                onChange={(e) =>
                  set('disease_days', e.target.value === '' ? '' : Number(e.target.value))
                }
              />
            </Field>
            <Field label="偏瘫侧">
              <select value={form.paralysis_side} onChange={(e) => set('paralysis_side', e.target.value)}>
                <option value="左">左</option>
                <option value="右">右</option>
              </select>
            </Field>
            <Field label="出生年月日">
              <input type="date" value={form.birth_date ?? ''} onChange={(e) => set('birth_date', e.target.value)} />
            </Field>
            <Field label="身份证号">
              <input value={form.id_number ?? ''} onChange={(e) => set('id_number', e.target.value)} />
            </Field>
            <Field label="手机号">
              <input value={form.phone ?? ''} onChange={(e) => set('phone', e.target.value)} />
            </Field>
            <Field label="发病日期">
              <input type="date" value={form.onset_date ?? ''} onChange={(e) => set('onset_date', e.target.value)} />
            </Field>
          </div>
        )}
      </div>

      {/* Assessment history */}
      <div className="card">
        <h2>
          评估记录
          <span className="h2-suffix">History · {patient.assessments.length}</span>
        </h2>
        {patient.assessments.length === 0 ? (
          <p className="muted">暂无评估记录。</p>
        ) : (
          <div className="record-list">
            {patient.assessments.map((rec) => {
              const open = !!openRecords[rec.id]
              return (
                <div key={rec.id} className="record-item">
                  <button
                    className="record-head"
                    onClick={() => setOpenRecords((o) => ({ ...o, [rec.id]: !open }))}
                  >
                    <span className="record-time">{fmtDateTime(rec.created_at)}</span>
                    <span className="record-summary">
                      FMA {Math.round(rec.fma_ue)} · BI {Math.round(rec.bi)} · 张力 {rec.hand_tone} 级 · Brunnstrom {rec.hand_function} 期
                    </span>
                    {rec.report_status === 'failed' && (
                      <span className="badge badge-warn">报告未生成</span>
                    )}
                    <span className="record-caret">{open ? '▾' : '▸'}</span>
                  </button>
                  {open && <RecordDetail record={rec} />}
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}

function Info({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="info-item">
      <span className="info-label">{label}</span>
      <span className="info-value">{value}</span>
    </div>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="field">
      <label>{label}</label>
      {children}
    </div>
  )
}
