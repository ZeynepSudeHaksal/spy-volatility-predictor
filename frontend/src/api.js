const API_BASE = 'http://localhost:5001/api'

export async function fetchLatestPrediction() {
  const res = await fetch(`${API_BASE}/predict/latest`)
  if (!res.ok) throw new Error('Failed to fetch latest prediction')
  return res.json()
}

export async function fetchHistory() {
  const res = await fetch(`${API_BASE}/history`)
  if (!res.ok) throw new Error('Failed to fetch history')
  return res.json()
}
