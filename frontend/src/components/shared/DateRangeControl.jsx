import { useState } from 'react'

// Presets are expressed in trading-day-ish calendar days; "All" is handled
// specially (no lower bound). Kept here so both History charts (volatility
// and direction) share identical filtering behavior.
const PRESETS = [
  { key: '1m', label: '1M', days: 30 },
  { key: '3m', label: '3M', days: 90 },
  { key: '6m', label: '6M', days: 180 },
  { key: '1y', label: '1Y', days: 365 },
  { key: 'all', label: 'All', days: null },
]

const DEFAULT_PRESET = 'all'

export function useDateRange(rows) {
  const [preset, setPreset] = useState(DEFAULT_PRESET)
  const [customStart, setCustomStart] = useState('')
  const [customEnd, setCustomEnd] = useState('')
  const [showCustom, setShowCustom] = useState(false)

  let filtered = rows
  if (preset === 'custom') {
    if (customStart) filtered = filtered.filter((r) => r.date >= customStart)
    if (customEnd) filtered = filtered.filter((r) => r.date <= customEnd)
  } else {
    const active = PRESETS.find((p) => p.key === preset)
    if (active && active.days && rows.length) {
      const lastDate = new Date(rows[rows.length - 1].date + 'T00:00:00')
      const cutoff = new Date(lastDate)
      cutoff.setDate(cutoff.getDate() - active.days)
      const cutoffStr = cutoff.toISOString().slice(0, 10)
      filtered = rows.filter((r) => r.date >= cutoffStr)
    }
  }

  return {
    filtered: filtered.length ? filtered : rows,
    preset,
    setPreset,
    customStart,
    customEnd,
    setCustomStart,
    setCustomEnd,
    showCustom,
    setShowCustom,
    minDate: rows[0]?.date,
    maxDate: rows[rows.length - 1]?.date,
  }
}

export default function DateRangeControl({ range }) {
  const {
    preset, setPreset, customStart, customEnd,
    setCustomStart, setCustomEnd, showCustom, setShowCustom,
    minDate, maxDate,
  } = range

  return (
    <div className="date-range-control">
      <div className="date-range-presets" role="group" aria-label="Date range">
        {PRESETS.map((p) => (
          <button
            key={p.key}
            type="button"
            className={`date-range-btn ${preset === p.key ? 'active' : ''}`}
            onClick={() => { setPreset(p.key); setShowCustom(false) }}
          >
            {p.label}
          </button>
        ))}
        <button
          type="button"
          className={`date-range-btn ${preset === 'custom' ? 'active' : ''}`}
          onClick={() => { setShowCustom((v) => !v); setPreset('custom') }}
        >
          Custom
        </button>
      </div>

      {showCustom && (
        <div className="date-range-custom">
          <label>
            From
            <input
              type="date"
              value={customStart}
              min={minDate}
              max={customEnd || maxDate}
              onChange={(e) => setCustomStart(e.target.value)}
            />
          </label>
          <label>
            To
            <input
              type="date"
              value={customEnd}
              min={customStart || minDate}
              max={maxDate}
              onChange={(e) => setCustomEnd(e.target.value)}
            />
          </label>
        </div>
      )}
    </div>
  )
}
