import MarkdownReport from './MarkdownReport'

interface Props {
  text: string
  streaming: boolean
  onExportMarkdown: () => void
  onExportWord: () => void
  onRestart: () => void
  onViewPatient?: () => void
  done: boolean
}

export default function ReportDisplay({
  text,
  streaming,
  onExportMarkdown,
  onExportWord,
  onRestart,
  onViewPatient,
  done,
}: Props) {
  // Hide entirely only while still processing with nothing to show yet. Once
  // done, always render (even with no report text) so the action buttons appear.
  if (!done && !streaming && !text) return null

  const reportFailed = done && !text

  return (
    <div className="card">
      <h2>
        AI 康复评估报告
        <span className="h2-suffix">Generative · Synthesis</span>
      </h2>

      {reportFailed ? (
        <div className="error-banner">
          AI 报告生成失败或未完成，已保留 4 项评估指标。报告状态：未生成。
        </div>
      ) : (
        <div className="report-display">
          {/* The report is Markdown (multi-section + tables) — render it. */}
          <MarkdownReport text={text} />
          {streaming && <span className="cursor" />}
        </div>
      )}

      {done && (
        <div className="actions">
          <button className="button secondary" onClick={onRestart}>
            重新评估
          </button>
          {!reportFailed && (
            <>
              <button className="button" onClick={onExportMarkdown}>
                导出 Markdown
              </button>
              <button className="button secondary" onClick={onExportWord}>
                导出 Word
              </button>
            </>
          )}
          {onViewPatient && (
            <button className="button secondary" onClick={onViewPatient}>
              查看患者档案
            </button>
          )}
        </div>
      )}
    </div>
  )
}
