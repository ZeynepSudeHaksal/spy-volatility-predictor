const MODES = [
  { id: 'volatility', label: 'Volatility' },
  { id: 'direction', label: 'Direction' },
]

export default function ModeToggle({ mode, onSelect }) {
  return (
    <div className="mode-toggle">
      {MODES.map((m) => (
        <button
          key={m.id}
          className={`mode-toggle-btn ${m.id === mode ? 'active' : ''}`}
          onClick={() => onSelect(m.id)}
        >
          {m.label}
        </button>
      ))}
    </div>
  )
}
