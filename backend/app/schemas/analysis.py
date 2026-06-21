from typing import Optional

from pydantic import BaseModel


class FrameAnalysisRequest(BaseModel):
    session_id: str
    frame_base64: str
    timestamp: float


class FrameAnalysisResponse(BaseModel):
    phone_detected: bool
    phone_confidence: float
    phone_bbox: Optional[list[float]] = None


class AudioAnalysisRequest(BaseModel):
    session_id: str
    audio_base64: str
    sample_rate: int = 16000
    timestamp: float


class AudioAnalysisResponse(BaseModel):
    environment_class: str
    noise_db: float
    is_disruptive: bool
    detected_sounds: list[str]


class ClientFeatures(BaseModel):
    """Features extracted by MediaPipe on the client side."""
    session_id: str
    timestamp: float
    face_detected: bool
    ear_left: Optional[float] = None
    ear_right: Optional[float] = None
    mar: Optional[float] = None
    gaze_yaw: Optional[float] = None
    gaze_pitch: Optional[float] = None
    head_yaw: Optional[float] = None
    head_pitch: Optional[float] = None
    head_roll: Optional[float] = None
    gaze_point_x: Optional[float] = None
    gaze_point_y: Optional[float] = None


class ConcentrationResult(BaseModel):
    grade: str  # A / B / C / F
    score: float  # 0-100
    triggers: list[str]
    message: Optional[str] = None


class LearningZoneData(BaseModel):
    zones: list[dict]
    environment_type: str
    base_noise_db: float


class CalibrationStatus(BaseModel):
    session_id: str
    phase: str  # calibrating / adapting / active
    progress: float  # 0.0 ~ 1.0
    face_detected: bool
    brightness_ok: bool
    message: str
