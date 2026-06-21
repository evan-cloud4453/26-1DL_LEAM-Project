import { useRef, useState, useCallback } from 'react'
import { createWebSocket } from '../services/api'

export default function useWebSocket(sessionId) {
  const wsRef = useRef(null)
  const [isConnected, setIsConnected] = useState(false)
  const [lastMessage, setLastMessage] = useState(null)
  const callbacksRef = useRef({})

  const connect = useCallback(() => {
    if (!sessionId) return

    const ws = createWebSocket(sessionId)

    ws.onopen = () => {
      setIsConnected(true)
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      setLastMessage(data)

      const handler = callbacksRef.current[data.type]
      if (handler) handler(data)
    }

    ws.onclose = () => {
      setIsConnected(false)
    }

    ws.onerror = (err) => {
      console.error('WebSocket error:', err)
    }

    wsRef.current = ws
  }, [sessionId])

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
  }, [])

  const send = useCallback((data) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data))
    }
  }, [])

  const on = useCallback((type, callback) => {
    callbacksRef.current[type] = callback
  }, [])

  return { isConnected, lastMessage, connect, disconnect, send, on }
}
