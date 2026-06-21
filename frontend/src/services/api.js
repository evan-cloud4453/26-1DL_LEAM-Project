const API_BASE = import.meta.env.VITE_API_URL || ''

export async function startSession() {
  const res = await fetch(`${API_BASE}/api/sessions/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({}),
  })
  return res.json()
}

export async function endSession(sessionId) {
  const res = await fetch(`${API_BASE}/api/sessions/end`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId }),
  })
  return res.json()
}

export async function getReport(sessionId) {
  const res = await fetch(`${API_BASE}/api/reports/${sessionId}`)
  return res.json()
}

export async function getSessions() {
  const res = await fetch(`${API_BASE}/api/sessions/`)
  return res.json()
}

export async function analyzeFrame(sessionId, frameBase64) {
  const res = await fetch(`${API_BASE}/api/analysis/frame`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      frame_base64: frameBase64,
      timestamp: Date.now() / 1000,
    }),
  })
  return res.json()
}

export async function sendFeatures(sessionId, features) {
  const res = await fetch(`${API_BASE}/api/analysis/features`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      timestamp: Date.now() / 1000,
      ...features,
    }),
  })
  return res.json()
}

export function createWebSocket(sessionId) {
  const wsBase = API_BASE.replace(/^http/, 'ws') || `ws://${window.location.host}`
  return new WebSocket(`${wsBase}/api/analysis/ws/${sessionId}`)
}
