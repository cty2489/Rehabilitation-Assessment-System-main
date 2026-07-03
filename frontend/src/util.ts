// Format an ISO timestamp/date for display in the local timezone.
export function fmtDateTime(iso: string | null | undefined): string {
  if (!iso) return '—'
  const naive = iso.match(/^(\d{4}-\d{2}-\d{2})[ T](\d{2}):(\d{2})(?::\d{2})?$/)
  if (naive) return `${naive[1]} ${naive[2]}:${naive[3]}`
  const d = new Date(iso)
  if (isNaN(d.getTime())) return iso
  const p = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`
}

export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return '—'
  const naive = iso.match(/^(\d{4}-\d{2}-\d{2})(?:[ T]\d{2}:\d{2}(?::\d{2})?)?$/)
  if (naive) return naive[1]
  const d = new Date(iso)
  if (isNaN(d.getTime())) return iso
  const p = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`
}
