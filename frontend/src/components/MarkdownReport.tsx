/**
 * Minimal, dependency-free Markdown renderer for the assessment report.
 *
 * The backend (`backend/report_builder.py`) emits a *known, fixed* Markdown
 * subset — headings (#/##/###/####), GitHub-flavoured tables, ordered/unordered
 * lists, blockquotes and **bold** inline spans. Rather than pull in a full
 * markdown library, we parse exactly that subset. This keeps the bundle small,
 * works offline, and renders the multi-table clinical report (总体分期 /
 * 生物标志物 / 手势组合 / 每周计划) the way the template example does.
 *
 * Streaming-safe: it re-parses the accumulated text on every chunk, and an
 * incomplete trailing table/line simply renders as far as it can.
 */
import type { JSX } from 'react'

function renderInline(text: string): (JSX.Element | string)[] {
  // Only **bold** is used in the report; split on it and keep the rest literal.
  const parts: (JSX.Element | string)[] = []
  const re = /\*\*([^*]+)\*\*/g
  let last = 0
  let m: RegExpExecArray | null
  let i = 0
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) parts.push(text.slice(last, m.index))
    parts.push(<strong key={`b${i++}`}>{m[1]}</strong>)
    last = m.index + m[0].length
  }
  if (last < text.length) parts.push(text.slice(last))
  return parts
}

function splitRow(line: string): string[] {
  const t = line.trim().replace(/^\|/, '').replace(/\|$/, '')
  return t.split('|').map((c) => c.trim())
}

const isTableSep = (line: string): boolean =>
  /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$/.test(line)

const LEGACY_BIOMARKER_NOTE =
  '该历史报告采用旧版参考规则，当前不展示单次高低判断；请以同一设备、同一采集流程下的复测趋势为准。'

export default function MarkdownReport({ text }: { text: string }) {
  const lines = text.split('\n')
  const blocks: JSX.Element[] = []
  let i = 0
  let key = 0

  while (i < lines.length) {
    const line = lines[i]

    // Headings
    const h = /^(#{1,6})\s+(.*)$/.exec(line)
    if (h) {
      const level = h[1].length
      const content = renderInline(h[2])
      const Tag = (`h${Math.min(level, 6)}`) as keyof JSX.IntrinsicElements
      blocks.push(<Tag key={key++} className={`md-h md-h${level}`}>{content}</Tag>)
      i++
      continue
    }

    // GFM table: header row followed by a separator row.
    if (line.includes('|') && i + 1 < lines.length && isTableSep(lines[i + 1])) {
      const headers = splitRow(line)
      const rows: string[][] = []
      i += 2
      while (i < lines.length && lines[i].includes('|') && lines[i].trim() !== '') {
        rows.push(splitRow(lines[i]))
        i++
      }
      const legacyBiomarkerTable = headers.includes('标志物') && headers.includes('参考范围')
      const visibleIndexes = legacyBiomarkerTable
        ? [headers.indexOf('标志物'), headers.indexOf('当前值')]
        : headers.map((_, index) => index)
      const visibleHeaders = visibleIndexes.map((index) => headers[index])
      const visibleRows = rows.map((row) => visibleIndexes.map((index) => row[index] || '—'))
      blocks.push(
        <div key={key++} className="md-table-wrap">
          <table className="md-table">
            <thead>
              <tr>{visibleHeaders.map((hd, c) => <th key={c}>{renderInline(hd)}</th>)}</tr>
            </thead>
            <tbody>
              {visibleRows.map((r, ri) => (
                <tr key={ri}>{r.map((cell, c) => <td key={c}>{renderInline(cell)}</td>)}</tr>
              ))}
            </tbody>
          </table>
        </div>,
      )
      if (legacyBiomarkerTable) {
        blocks.push(<blockquote key={key++} className="md-quote">{LEGACY_BIOMARKER_NOTE}</blockquote>)
      }
      continue
    }

    // Blockquote
    if (/^\s*>\s?/.test(line)) {
      const quote = line.replace(/^\s*>\s?/, '')
      blocks.push(<blockquote key={key++} className="md-quote">{renderInline(quote)}</blockquote>)
      i++
      continue
    }

    // Ordered list (group consecutive items)
    if (/^\s*\d+\.\s+/.test(line)) {
      const items: string[] = []
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*\d+\.\s+/, ''))
        i++
      }
      blocks.push(
        <ol key={key++} className="md-ol">
          {items.map((it, n) => <li key={n}>{renderInline(it)}</li>)}
        </ol>,
      )
      continue
    }

    // Unordered list
    if (/^\s*[-*]\s+/.test(line)) {
      const items: string[] = []
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*[-*]\s+/, ''))
        i++
      }
      blocks.push(
        <ul key={key++} className="md-ul">
          {items.map((it, n) => <li key={n}>{renderInline(it)}</li>)}
        </ul>,
      )
      continue
    }

    // Blank line
    if (line.trim() === '') {
      i++
      continue
    }

    // Plain paragraph
    blocks.push(<p key={key++} className="md-p">{renderInline(line)}</p>)
    i++
  }

  return <div className="markdown-report">{blocks}</div>
}
