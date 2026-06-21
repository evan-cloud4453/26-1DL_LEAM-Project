/**
 * MediaPipe Face Mesh landmark index constants.
 * Total 478 landmarks (468 face + 10 iris).
 */

// Left eye contour (for EAR)
export const LEFT_EYE = {
  top: [159, 145],
  bottom: [23, 130],
  left: 33,
  right: 133,
  upper: [246, 161, 160, 159, 158, 157, 173],
  lower: [33, 7, 163, 144, 145, 153, 154, 155, 133],
}

// Right eye contour (for EAR)
export const RIGHT_EYE = {
  top: [386, 374],
  bottom: [252, 359],
  left: 362,
  right: 263,
  upper: [466, 388, 387, 386, 385, 384, 398],
  lower: [362, 382, 381, 380, 374, 373, 390, 249, 263],
}

// Iris landmarks (468-477)
export const LEFT_IRIS = [468, 469, 470, 471, 472]
export const RIGHT_IRIS = [473, 474, 475, 476, 477]

// Mouth landmarks (for MAR)
export const MOUTH = {
  top: 13,
  bottom: 14,
  left: 78,
  right: 308,
  upperOuter: [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
  lowerOuter: [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
  upperInner: [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
  lowerInner: [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
}

// Head pose reference landmarks
export const HEAD_POSE_LANDMARKS = {
  noseTip: 1,
  chin: 152,
  leftEyeOuter: 33,
  rightEyeOuter: 263,
  leftMouth: 61,
  rightMouth: 291,
}

/**
 * Euclidean distance between two 3D landmark points.
 */
export function dist(a, b) {
  const dx = a.x - b.x
  const dy = a.y - b.y
  const dz = (a.z || 0) - (b.z || 0)
  return Math.sqrt(dx * dx + dy * dy + dz * dz)
}

/**
 * Eye Aspect Ratio (EAR) calculation.
 * EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
 */
export function computeEAR(landmarks, eyeIndices) {
  const p1 = landmarks[eyeIndices.left]
  const p4 = landmarks[eyeIndices.right]
  const p2 = landmarks[eyeIndices.top[0]]
  const p6 = landmarks[eyeIndices.bottom[0]]
  const p3 = landmarks[eyeIndices.top[1]]
  const p5 = landmarks[eyeIndices.bottom[1]]

  const vertical1 = dist(p2, p6)
  const vertical2 = dist(p3, p5)
  const horizontal = dist(p1, p4)

  if (horizontal < 1e-6) return 0.3
  return (vertical1 + vertical2) / (2.0 * horizontal)
}

/**
 * Mouth Aspect Ratio (MAR) calculation.
 */
export function computeMAR(landmarks) {
  const top = landmarks[MOUTH.top]
  const bottom = landmarks[MOUTH.bottom]
  const left = landmarks[MOUTH.left]
  const right = landmarks[MOUTH.right]

  const vertical = dist(top, bottom)
  const horizontal = dist(left, right)

  if (horizontal < 1e-6) return 0
  return vertical / horizontal
}

/**
 * Estimate gaze direction from iris landmarks.
 * Returns { yaw, pitch } in degrees.
 */
export function computeGaze(landmarks) {
  const leftIrisCenter = landmarks[LEFT_IRIS[0]]
  const rightIrisCenter = landmarks[RIGHT_IRIS[0]]
  const leftEyeLeft = landmarks[LEFT_EYE.left]
  const leftEyeRight = landmarks[LEFT_EYE.right]
  const rightEyeLeft = landmarks[RIGHT_EYE.left]
  const rightEyeRight = landmarks[RIGHT_EYE.right]

  // Horizontal ratio for each eye (0=left, 0.5=center, 1=right)
  const leftRatio = (leftIrisCenter.x - leftEyeLeft.x) / (leftEyeRight.x - leftEyeLeft.x + 1e-6)
  const rightRatio = (rightIrisCenter.x - rightEyeLeft.x) / (rightEyeRight.x - rightEyeLeft.x + 1e-6)

  const avgHorizontal = (leftRatio + rightRatio) / 2
  const yaw = (avgHorizontal - 0.5) * 60  // approximate degrees

  // Vertical ratio
  const leftEyeTop = landmarks[LEFT_EYE.top[0]]
  const leftEyeBottom = landmarks[LEFT_EYE.bottom[0]]
  const leftVertRatio = (leftIrisCenter.y - leftEyeTop.y) / (leftEyeBottom.y - leftEyeTop.y + 1e-6)
  const pitch = (leftVertRatio - 0.5) * 40  // approximate degrees

  return { yaw, pitch }
}

/**
 * Estimate head pose (yaw, pitch, roll) from face landmarks.
 */
export function computeHeadPose(landmarks) {
  const nose = landmarks[HEAD_POSE_LANDMARKS.noseTip]
  const chin = landmarks[HEAD_POSE_LANDMARKS.chin]
  const leftEye = landmarks[HEAD_POSE_LANDMARKS.leftEyeOuter]
  const rightEye = landmarks[HEAD_POSE_LANDMARKS.rightEyeOuter]
  const leftMouth = landmarks[HEAD_POSE_LANDMARKS.leftMouth]
  const rightMouth = landmarks[HEAD_POSE_LANDMARKS.rightMouth]

  // Yaw: horizontal asymmetry
  const eyeMidX = (leftEye.x + rightEye.x) / 2
  const yaw = (nose.x - eyeMidX) * 180

  // Pitch: nose vs eye line
  const eyeMidY = (leftEye.y + rightEye.y) / 2
  const pitch = (nose.y - eyeMidY) * 180

  // Roll: eye line angle
  const dx = rightEye.x - leftEye.x
  const dy = rightEye.y - leftEye.y
  const roll = Math.atan2(dy, dx) * (180 / Math.PI)

  return { yaw, pitch, roll }
}

/**
 * Estimate normalized gaze point on screen (0-1 range).
 */
export function computeGazePoint(landmarks) {
  const gaze = computeGaze(landmarks)
  const headPose = computeHeadPose(landmarks)

  // Combine gaze and head pose for screen point estimation
  const x = 0.5 + (gaze.yaw + headPose.yaw * 0.5) / 90
  const y = 0.5 + (gaze.pitch + headPose.pitch * 0.5) / 60

  return {
    x: Math.max(0, Math.min(1, x)),
    y: Math.max(0, Math.min(1, y)),
  }
}
