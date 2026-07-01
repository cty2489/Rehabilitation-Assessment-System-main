import { DIAGNOSIS_OPTIONS, PatientInfo } from '../types'

interface Props {
  value: PatientInfo
  onChange: (info: PatientInfo) => void
  disabled?: boolean
}

export default function PatientForm({ value, onChange, disabled }: Props) {
  const update = <K extends keyof PatientInfo>(key: K, v: PatientInfo[K]) =>
    onChange({ ...value, [key]: v })

  // Empty input stays empty (no forced 0); a real number parses cleanly with no
  // leading-zero artifact because we never prepend '0'.
  const onNumber = (key: 'age' | 'disease_days') => (e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.value
    update(key, raw === '' ? '' : Number(raw))
  }

  return (
    <div className="card">
      <h2>
        患者基本信息
        <span className="h2-suffix">Patient · Profile</span>
      </h2>
      <div className="grid-2">
        <div className="field">
          <label>患者编号</label>
          <input
            type="text"
            value={value.patient_id}
            disabled={disabled}
            onChange={(e) => update('patient_id', e.target.value)}
          />
        </div>
        <div className="field">
          <label>姓名</label>
          <input
            type="text"
            value={value.name}
            disabled={disabled}
            onChange={(e) => update('name', e.target.value)}
          />
        </div>
        <div className="field">
          <label>性别</label>
          <select
            value={value.sex}
            disabled={disabled}
            onChange={(e) => update('sex', e.target.value as PatientInfo['sex'])}
          >
            <option value="男">男</option>
            <option value="女">女</option>
          </select>
        </div>
        <div className="field">
          <label>年龄</label>
          <input
            type="number"
            min={0}
            max={120}
            value={value.age}
            disabled={disabled}
            onChange={onNumber('age')}
          />
        </div>
        <div className="field">
          <label>诊断</label>
          <select
            value={value.diagnosis}
            disabled={disabled}
            onChange={(e) => update('diagnosis', e.target.value)}
          >
            <option value="" disabled>
              请选择
            </option>
            {DIAGNOSIS_OPTIONS.map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>病程（天）</label>
          <input
            type="number"
            min={0}
            value={value.disease_days}
            disabled={disabled}
            onChange={onNumber('disease_days')}
          />
        </div>
        <div className="field">
          <label>偏瘫侧</label>
          <select
            value={value.paralysis_side}
            disabled={disabled}
            onChange={(e) =>
              update('paralysis_side', e.target.value as PatientInfo['paralysis_side'])
            }
          >
            <option value="左">左</option>
            <option value="右">右</option>
          </select>
        </div>
      </div>
    </div>
  )
}
