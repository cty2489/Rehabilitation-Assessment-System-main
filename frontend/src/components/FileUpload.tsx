import { ChangeEvent } from 'react'

interface Props {
  eegFiles: File[]
  emgFiles: File[]
  onChange: (eeg: File[], emg: File[]) => void
  disabled?: boolean
}

export default function FileUpload({ eegFiles, emgFiles, onChange, disabled }: Props) {
  const handleEeg = (e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files ? Array.from(e.target.files) : []
    onChange(files, emgFiles)
  }
  const handleEmg = (e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files ? Array.from(e.target.files) : []
    onChange(eegFiles, files)
  }

  const mismatch = eegFiles.length > 0 && emgFiles.length > 0 && eegFiles.length !== emgFiles.length

  return (
    <div className="card">
      <h2>
        信号文件上传
        <span className="h2-suffix">Biosignal · Ingest</span>
      </h2>
      <div className="file-upload">
        <div className={`file-slot ${eegFiles.length > 0 ? 'has-files' : ''}`}>
          <div className="file-label">
            脑电信号 · EEG
            <span className="tag">32 CH</span>
            <span className="tag">.CSV / .BDF</span>
          </div>
          <input
            type="file"
            accept=".csv,.bdf"
            multiple
            onChange={handleEeg}
            disabled={disabled}
          />
          {eegFiles.length > 0 && (
            <ul>
              {eegFiles.map((f, i) => (
                <li key={i}>
                  {f.name} <span style={{ opacity: 0.6 }}>({(f.size / 1024).toFixed(1)} KB)</span>
                </li>
              ))}
            </ul>
          )}
        </div>
        <div className={`file-slot ${emgFiles.length > 0 ? 'has-files' : ''}`}>
          <div className="file-label">
            肌电 / 惯性信号 · EMG&nbsp;·&nbsp;IMU
            <span className="tag">4 × 6 AXIS</span>
            <span className="tag">.CSV</span>
          </div>
          <input
            type="file"
            accept=".csv"
            multiple
            onChange={handleEmg}
            disabled={disabled}
          />
          {emgFiles.length > 0 && (
            <ul>
              {emgFiles.map((f, i) => (
                <li key={i}>
                  {f.name} <span style={{ opacity: 0.6 }}>({(f.size / 1024).toFixed(1)} KB)</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
      <div className="trial-hint">
        每个 trial 上传一组 EEG / EMG 文件，请保证顺序对应。脑电支持 .csv（仿真）或 .bdf（真实采集，32 通道含 A1/A2），肌电需包含 4 块肌肉 × 6 轴的 CSV。
      </div>
      {mismatch && (
        <div className="error-banner" style={{ marginTop: 12, marginBottom: 0 }}>
          EEG 与 EMG 文件数量不匹配：{eegFiles.length} vs {emgFiles.length}
        </div>
      )}
    </div>
  )
}
