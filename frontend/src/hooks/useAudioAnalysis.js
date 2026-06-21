import { useRef, useState, useCallback } from 'react'

export default function useAudioAnalysis() {
  const audioContextRef = useRef(null)
  const analyserRef = useRef(null)
  const mediaStreamRef = useRef(null)
  const recorderRef = useRef(null)
  const [isRecording, setIsRecording] = useState(false)
  const [currentDb, setCurrentDb] = useState(0)

  const startAudio = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: false,  // we want to measure actual noise
        },
      })
      mediaStreamRef.current = stream

      const audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000,
      })
      audioContextRef.current = audioContext

      const source = audioContext.createMediaStreamSource(stream)
      const analyser = audioContext.createAnalyser()
      analyser.fftSize = 2048
      source.connect(analyser)
      analyserRef.current = analyser

      setIsRecording(true)
      return true
    } catch (err) {
      console.error('Microphone access denied:', err)
      return false
    }
  }, [])

  const stopAudio = useCallback(() => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop())
      mediaStreamRef.current = null
    }
    if (audioContextRef.current) {
      audioContextRef.current.close()
      audioContextRef.current = null
    }
    setIsRecording(false)
  }, [])

  const getDecibelLevel = useCallback(() => {
    const analyser = analyserRef.current
    if (!analyser) return 0

    const dataArray = new Float32Array(analyser.fftSize)
    analyser.getFloatTimeDomainData(dataArray)

    let sum = 0
    for (let i = 0; i < dataArray.length; i++) {
      sum += dataArray[i] * dataArray[i]
    }
    const rms = Math.sqrt(sum / dataArray.length)
    const db = rms > 1e-10 ? 20 * Math.log10(rms / 1e-5) : 0
    const clampedDb = Math.max(0, Math.min(120, db))

    setCurrentDb(clampedDb)
    return clampedDb
  }, [])

  const captureAudioChunk = useCallback(() => {
    return new Promise((resolve) => {
      const stream = mediaStreamRef.current
      if (!stream) {
        resolve(null)
        return
      }

      const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' })
      const chunks = []

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data)
      }

      recorder.onstop = async () => {
        const blob = new Blob(chunks, { type: 'audio/webm' })
        const arrayBuffer = await blob.arrayBuffer()
        const base64 = btoa(
          new Uint8Array(arrayBuffer).reduce(
            (data, byte) => data + String.fromCharCode(byte), ''
          )
        )
        resolve(base64)
      }

      recorder.start()
      setTimeout(() => {
        if (recorder.state === 'recording') {
          recorder.stop()
        }
      }, 2000)  // 2 second chunk
    })
  }, [])

  return {
    isRecording,
    currentDb,
    startAudio,
    stopAudio,
    getDecibelLevel,
    captureAudioChunk,
  }
}
