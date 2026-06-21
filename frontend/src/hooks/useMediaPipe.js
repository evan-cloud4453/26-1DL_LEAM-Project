import { useRef, useEffect, useCallback, useState } from 'react'
import {
  computeEAR, computeMAR, computeGaze, computeHeadPose, computeGazePoint,
  LEFT_EYE, RIGHT_EYE,
} from '../utils/landmarks'

export default function useMediaPipe(videoRef, canvasRef, onFeatures) {
  const faceMeshRef = useRef(null)
  const animFrameRef = useRef(null)
  const [isReady, setIsReady] = useState(false)
  const [faceDetected, setFaceDetected] = useState(false)

  const processResults = useCallback((results) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    const video = videoRef.current
    if (!video) return

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
      setFaceDetected(false)
      onFeatures?.({
        face_detected: false,
        ear_left: null,
        ear_right: null,
        mar: null,
        gaze_yaw: null,
        gaze_pitch: null,
        head_yaw: null,
        head_pitch: null,
        head_roll: null,
        gaze_point_x: null,
        gaze_point_y: null,
      })
      return
    }

    setFaceDetected(true)
    const landmarks = results.multiFaceLandmarks[0]

    // Draw face mesh on canvas
    ctx.strokeStyle = 'rgba(79, 195, 247, 0.3)'
    ctx.lineWidth = 0.5
    for (let i = 0; i < landmarks.length; i++) {
      const x = landmarks[i].x * canvas.width
      const y = landmarks[i].y * canvas.height
      ctx.beginPath()
      ctx.arc(x, y, 1, 0, 2 * Math.PI)
      ctx.fillStyle = 'rgba(79, 195, 247, 0.6)'
      ctx.fill()
    }

    // Compute features
    const earLeft = computeEAR(landmarks, LEFT_EYE)
    const earRight = computeEAR(landmarks, RIGHT_EYE)
    const mar = computeMAR(landmarks)
    const gaze = computeGaze(landmarks)
    const headPose = computeHeadPose(landmarks)
    const gazePoint = computeGazePoint(landmarks)

    // Draw gaze point indicator
    ctx.beginPath()
    ctx.arc(gazePoint.x * canvas.width, gazePoint.y * canvas.height, 8, 0, 2 * Math.PI)
    ctx.fillStyle = 'rgba(102, 187, 106, 0.7)'
    ctx.fill()

    onFeatures?.({
      face_detected: true,
      ear_left: earLeft,
      ear_right: earRight,
      mar,
      gaze_yaw: gaze.yaw,
      gaze_pitch: gaze.pitch,
      head_yaw: headPose.yaw,
      head_pitch: headPose.pitch,
      head_roll: headPose.roll,
      gaze_point_x: gazePoint.x,
      gaze_point_y: gazePoint.y,
    })
  }, [canvasRef, videoRef, onFeatures])

  useEffect(() => {
    let cancelled = false

    async function init() {
      const { FaceMesh } = await import('@mediapipe/face_mesh')

      const faceMesh = new FaceMesh({
        locateFile: (file) =>
          `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
      })

      faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,  // enables iris landmarks (468-477)
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      })

      faceMesh.onResults(processResults)

      if (!cancelled) {
        faceMeshRef.current = faceMesh
        setIsReady(true)
      }
    }

    init()

    return () => {
      cancelled = true
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current)
      }
    }
  }, [processResults])

  const sendFrame = useCallback(async () => {
    const video = videoRef.current
    const faceMesh = faceMeshRef.current
    if (!video || !faceMesh || video.readyState < 2) return

    await faceMesh.send({ image: video })
  }, [videoRef])

  // Start continuous processing loop
  const startProcessing = useCallback(() => {
    let lastTime = 0
    const targetInterval = 1000 / 30  // 30 fps

    function loop(timestamp) {
      if (timestamp - lastTime >= targetInterval) {
        lastTime = timestamp
        sendFrame()
      }
      animFrameRef.current = requestAnimationFrame(loop)
    }
    animFrameRef.current = requestAnimationFrame(loop)
  }, [sendFrame])

  const stopProcessing = useCallback(() => {
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current)
      animFrameRef.current = null
    }
  }, [])

  return { isReady, faceDetected, startProcessing, stopProcessing, sendFrame }
}
