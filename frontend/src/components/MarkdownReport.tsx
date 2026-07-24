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

function renderCitations(text: string, keyPrefix: string): (JSX.Element | string)[] {
  const parts: (JSX.Element | string)[] = []
  const re = /【(\d+)】/g
  let last = 0
  let match: RegExpExecArray | null
  let index = 0
  while ((match = re.exec(text)) !== null) {
    if (match.index > last) parts.push(text.slice(last, match.index))
    parts.push(
      <span
        key={`${keyPrefix}-c${index++}`}
        className="report-citation"
        title={`对应文末参考文献 ${match[1]}`}
        aria-label={`参考文献 ${match[1]}`}
      >
        {match[0]}
      </span>,
    )
    last = match.index + match[0].length
  }
  if (last < text.length) parts.push(text.slice(last))
  return parts
}

function renderInline(text: string): (JSX.Element | string)[] {
  const parts: (JSX.Element | string)[] = []
  const re = /\*\*([^*]+)\*\*/g
  let last = 0
  let m: RegExpExecArray | null
  let i = 0
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) {
      parts.push(...renderCitations(text.slice(last, m.index), `p${i}`))
    }
    parts.push(<strong key={`b${i}`}>{renderCitations(m[1], `b${i}`)}</strong>)
    i++
    last = m.index + m[0].length
  }
  if (last < text.length) parts.push(...renderCitations(text.slice(last), `t${i}`))
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

const SPECIFIC_METHOD_SEGMENT =
  /(?:^|[；;]\s*)具体方法(?:[（(][^）)]*[）)])?\s*[：:]\s*.*?(?=(?:[；;]\s*)?(?:训练剂量|反馈标准|调整原则|安全注意)\s*[：:]|$)/g


const BRUNNSTROM_ROMAN = ['', 'I', 'II', 'III', 'IV', 'V', 'VI']

function withHistoricalOverallSubtype(text: string): string {
  // planner_rag reports created before 2026-07-24 did not persist a subtype.
  // Their Brunnstrom model-prediction row is sufficient to restore the same
  // test-only, explicitly bounded summary in the existing UI.
  if (/(?:\*\*综合亚型[：:]|综合亚型界定)/.test(text)) return text
  if (!text.includes('## 三、康复策略建议')) return text

  const row = text.match(/\|\s*Brunnstrom手功能分期（模型预测）\s*\|\s*([^|]+)\|/)
  const stage = row?.[1].match(/模型预测值：\s*([1-6])\s*期/)
  if (!row || !stage) return text

  const stageNumber = Number(stage[1])
  const stageText = BRUNNSTROM_ROMAN[stageNumber]
  const detail = (row[1].match(/模型预测结果：([^。|]+)。/)?.[1] || '手功能模型预测结果待确认').trim()
  const subtype = `${stageText}期-手功能综合亚型（测试性归纳）：${detail}；中枢驱动、协同分离和关节活动度仍需结合动作检查确认。`
  const section = `\n### 综合亚型（历史测试结果补全）\n\n**综合亚型：** ${subtype}\n`
  return text.replace('\n## 三、康复策略建议', `${section}\n## 三、康复策略建议`)
}


function historicalMetricPurpose(name: string): string {
  const rules: Array<[RegExp, string]> = [
    [/静息肌电/, '静息时肌肉是否仍有不必要的紧张'],
    [/CCI-腕|腕屈伸肌共收缩/, '腕屈肌和伸肌是否同时用力、动作是否协调'],
    [/CCI-指|指屈伸肌共收缩/, '手指屈肌和伸肌是否同时用力、动作是否协调'],
    [/肌肉激活幅度/, '动作时肌肉募集的总体强弱'],
    [/积分肌电/, '对应肌肉在整个动作中的总用力量'],
    [/IEMG 比/, '屈肌和伸肌出力是否平衡'],
    [/爆发持续时间/, '一次动作中肌肉持续发力的时长'],
    [/中位频率/, '对应肌肉的疲劳或募集变化'],
    [/半球不对称/, '两侧大脑静息活动是否平衡'],
    [/皮层-肌肉相干/, '大脑运动区和肌肉发力是否同步配合'],
    [/前额叶 θ\/β/, '前额叶与注意、任务控制相关的脑电活动比例'],
    [/半球间运动皮层相干/, '左右运动脑区之间的协同活动'],
    [/μ 功率变化/, '动作时运动脑区的μ节律反应'],
    [/β 功率变化/, '动作时运动脑区的β节律反应'],
    [/运动平滑度/, '动作是否连续、流畅，是否频繁停顿或抖动'],
    [/活动度代理/, '本次动作活动范围的大小'],
    [/震颤指数/, '动作中3–6Hz震颤成分的多少'],
    [/峰值角速度/, '对应动作达到的最快速度'],
  ]
  return rules.find(([pattern]) => pattern.test(name))?.[1] || name
}

function historicalResultAndInterpretation(name: string, cell: string): [string, string] {
  const result = cell.match(/^(?:模型预测值|本次记录值)：\s*([^；。]+)/)?.[0] || '—'
  if (name.includes('FMA')) {
    return [result, '反映手部动作完成情况；需结合现场动作检查确认。']
  }
  if (name.includes('肌张力')) {
    const detail = cell.match(/模型预测结果：([^。|]+)。/)?.[1] || '反映肌肉放松和阻力情况'
    return [result, `${detail}；需由治疗师实际检查确认。`]
  }
  if (name.includes('Brunnstrom')) {
    const detail = cell.match(/模型预测结果：([^。|]+)。/)?.[1] || '反映手部动作恢复阶段'
    return [result, `${detail}；以实际抓握和伸指观察为准。`]
  }
  const purpose = historicalMetricPurpose(name)
  if (cell.includes('无通用绝对参考范围') || cell.includes('暂无可用于单次分类')) {
    return [result, `用于观察${purpose}；本次列的数字是设备计算值，目前没有统一好坏范围，后续同条件复测看变化。`]
  }
  if (cell.includes('通常呈上升趋势')) {
    return [result, `用于观察${purpose}；本次列的数字是本次记录，研究通常看同条件下是否升高，本次先作为个人基线。`]
  }
  if (cell.includes('通常呈下降趋势')) {
    return [result, `用于观察${purpose}；本次列的数字是本次记录，研究通常看同条件下是否下降，本次先作为个人基线。`]
  }
  if (cell.includes('高于文献参考范围')) {
    return [result, `用于观察${purpose}；本次数值高于文献参考范围，需结合动作表现和后续复测判断。`]
  }
  if (cell.includes('低于文献参考范围')) {
    return [result, `用于观察${purpose}；本次数值低于文献参考范围，需结合动作表现和后续复测判断。`]
  }
  if (cell.includes('处于文献参考范围')) {
    return [result, `用于观察${purpose}；本次数值在文献参考范围内，仍需结合动作表现判断。`]
  }
  return [result, `用于观察${purpose}；本次列的数字先作为个人基线，后续同条件复测看变化。`]
}

function withHistoricalResultColumns(text: string): string {
  const lines = text.split('\n')
  const output: string[] = []
  for (let index = 0; index < lines.length; index++) {
    if (lines[index].trim() !== '| 指标 | 本次结果与知识解读 | 依据 |') {
      output.push(lines[index])
      continue
    }
    output.push('| 指标 | 本次结果 | 解读 | 依据 |')
    if (index + 1 < lines.length && isTableSep(lines[index + 1])) {
      output.push('|---|---|---|---|')
      index++
    }
    while (index + 1 < lines.length && lines[index + 1].includes('|') && lines[index + 1].trim() !== '') {
      const cells = splitRow(lines[++index])
      if (cells.length < 3) {
        output.push(lines[index])
        continue
      }
      const [result, interpretation] = historicalResultAndInterpretation(cells[0], cells[1])
      output.push(`| ${cells[0]} | ${result} | ${interpretation} | ${cells.slice(2).join(' | ')} |`)
    }
  }
  return output.join('\n')
}

function reportDisplayLines(text: string): string[] {
  const displayText = withHistoricalResultColumns(withHistoricalOverallSubtype(text))
  let inStrategySection = false
  const visible: string[] = []
  // Older saved test results put the overall subtype in one standalone line:
  // "**亚型界定：** ...". Do not hide that line unless this same report
  // also contains the newer, complete overall-subtype section; otherwise the
  // reader loses the only displayed subtype.
  const hasModernOverallSubtype =
    /^##\s+三、综合亚型界定与治疗策略\s*$/m.test(displayText) &&
    /综合状态界定为：\s*\*\*[^*\n]+\*\*/.test(displayText)

  for (const original of displayText.split('\n')) {
    const trimmed = original.trim()
    if (/^##\s+/.test(trimmed)) {
      inStrategySection = trimmed.includes('综合亚型界定与治疗策略')
    }
    if (hasModernOverallSubtype && /^\*\*亚型界定[：:]/.test(trimmed)) continue

    if (inStrategySection && /^\s*\d+\.\s+/.test(original)) {
      const prefix = original.match(/^\s*\d+\.\s+/)?.[0] || ''
      const strategy = original
        .slice(prefix.length)
        .replace(SPECIFIC_METHOD_SEGMENT, '')
        .replace(/^[；;\s]+|[；;\s]+$/g, '')
        .replace(/[；;]\s*[；;]/g, '；')
      if (!strategy) continue
      visible.push(prefix + strategy)
      continue
    }
    visible.push(original)
  }
  return visible
}

export default function MarkdownReport({ text }: { text: string }) {
  const lines = reportDisplayLines(text)
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
