const API_BASE = 'http://localhost:5001/api'

export async function fetchTickers() {
  const res = await fetch(`${API_BASE}/tickers`)
  if (!res.ok) throw new Error('Failed to fetch tickers')
  return res.json()
}

export async function fetchLatestPrediction(ticker) {
  const res = await fetch(`${API_BASE}/predict/latest?ticker=${ticker}`)
  if (!res.ok) throw new Error('Failed to fetch latest prediction')
  return res.json()
}

export async function fetchHistory(ticker) {
  const res = await fetch(`${API_BASE}/history?ticker=${ticker}`)
  if (!res.ok) throw new Error('Failed to fetch history')
  return res.json()
}
