"""
Manages the lifecycle of a study session:
calibrating -> adapting -> active -> ended
"""
import time
from dataclasses import dataclass, field

from app.config import (
    CALIBRATION_DURATION_SEC, READAPT_INTERVAL_SEC, CALIBRATION_PHASE_SEC,
)
from app.services.learning_zone import LearningZoneGenerator
from app.services.concentration import HierarchicalConcentrationModel


@dataclass
class ActiveSession:
    session_id: str
    phase: str = "calibrating"  # calibrating / adapting / active / ended
    start_time: float = field(default_factory=time.time)
    phase_start_time: float = field(default_factory=time.time)
    last_readapt_time: float = field(default_factory=time.time)
    zone_generator: LearningZoneGenerator = field(default_factory=LearningZoneGenerator)
    learning_zones: dict | None = None
    environment_type: str = "indoor"
    base_noise_db: float = 40.0
    noise_samples: list[float] = field(default_factory=list)
    calibration_face_ok: bool = False
    calibration_brightness_ok: bool = True


class SessionManager:
    def __init__(self):
        self.sessions: dict[str, ActiveSession] = {}
        self.concentration_model = HierarchicalConcentrationModel()

    def create_session(self, session_id: str) -> ActiveSession:
        session = ActiveSession(session_id=session_id)
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> ActiveSession | None:
        return self.sessions.get(session_id)

    def update_phase(self, session_id: str) -> dict:
        session = self.get_session(session_id)
        if session is None:
            return {"error": "session not found"}

        elapsed = time.time() - session.phase_start_time

        if session.phase == "calibrating":
            if session.calibration_face_ok and elapsed > CALIBRATION_PHASE_SEC:
                session.phase = "adapting"
                session.phase_start_time = time.time()
                return {"phase": "adapting", "message": "학습환경 적응 단계를 시작합니다."}
            progress = min(1.0, elapsed / CALIBRATION_PHASE_SEC)
            return {
                "phase": "calibrating",
                "progress": progress,
                "face_ok": session.calibration_face_ok,
                "brightness_ok": session.calibration_brightness_ok,
            }

        elif session.phase == "adapting":
            adapt_duration = CALIBRATION_DURATION_SEC
            progress = min(1.0, elapsed / adapt_duration)
            if elapsed >= adapt_duration:
                session.learning_zones = session.zone_generator.generate_zones()
                if session.noise_samples:
                    import numpy as np
                    session.base_noise_db = float(np.median(session.noise_samples))
                session.phase = "active"
                session.phase_start_time = time.time()
                session.last_readapt_time = time.time()
                return {
                    "phase": "active",
                    "message": "학습 Zone이 설정되었습니다. 집중도 측정을 시작합니다.",
                    "learning_zones": session.learning_zones,
                }
            return {"phase": "adapting", "progress": progress}

        elif session.phase == "active":
            if time.time() - session.last_readapt_time > READAPT_INTERVAL_SEC:
                session.zone_generator.reset()
                session.noise_samples.clear()
                session.phase = "adapting"
                session.phase_start_time = time.time()
                session.last_readapt_time = time.time()
                return {"phase": "adapting", "message": "재적응 단계를 시작합니다 (30분 경과)."}
            return {"phase": "active"}

        return {"phase": session.phase}

    def trigger_readapt(self, session_id: str) -> dict:
        """사용자 수동 재적응: 측정 점수(집중도 상태)는 유지한 채 적응 단계로 되돌린다."""
        session = self.get_session(session_id)
        if session is None:
            return {"error": "session not found"}
        session.zone_generator.reset()
        session.noise_samples.clear()
        session.phase = "adapting"
        session.phase_start_time = time.time()
        session.last_readapt_time = time.time()
        return {"phase": "adapting", "message": "학습 환경 재적응을 시작합니다. 기존 점수는 누적됩니다."}

    def process_features(self, session_id: str, features: dict, phone_result: dict,
                         audio_result: dict | None) -> dict:
        session = self.get_session(session_id)
        if session is None:
            return {"error": "session not found"}

        face_detected = features.get("face_detected", False)

        if session.phase == "calibrating":
            session.calibration_face_ok = face_detected
            return self.update_phase(session_id)

        if session.phase == "adapting":
            gx = features.get("gaze_point_x")
            gy = features.get("gaze_point_y")
            if gx is not None and gy is not None:
                session.zone_generator.add_gaze_point(gx, gy)
            if audio_result and "noise_db" in audio_result:
                session.noise_samples.append(audio_result["noise_db"])
            return self.update_phase(session_id)

        # Active phase: run hierarchical concentration model
        in_study_zone = True
        if session.learning_zones:
            gx = features.get("gaze_point_x")
            gy = features.get("gaze_point_y")
            if gx is not None and gy is not None:
                zones = session.learning_zones.get("zones", [])
                in_study_zone = session.zone_generator.is_in_study_zone(gx, gy, zones)

        # 소음 방해 판정:
        #  - 학습 이벤트음(키보드·마우스·필기 등)은 dB가 높아도 방해로 보지 않음
        #  - AudioSet 분류가 명확한 방해음(말소리·돌발음)을 잡으면 방해로 판정
        #  - 분류 결과가 없을 때만 dB 임계값(기준+10dB)으로 보조 판정
        noise_disruptive = False
        if audio_result:
            if audio_result.get("is_study_event"):
                noise_disruptive = False
            elif audio_result.get("is_disruptive"):
                noise_disruptive = True
            else:
                noise_db = audio_result.get("noise_db", 0)
                if noise_db > session.base_noise_db + 10:
                    noise_disruptive = True

        result = self.concentration_model.evaluate(
            session_id=session_id,
            face_detected=face_detected,
            ear_left=features.get("ear_left"),
            ear_right=features.get("ear_right"),
            mar=features.get("mar"),
            gaze_yaw=features.get("gaze_yaw"),
            gaze_pitch=features.get("gaze_pitch"),
            phone_detected=phone_result.get("phone_detected", False),
            noise_disruptive=noise_disruptive,
            in_study_zone=in_study_zone,
        )

        result["phase"] = session.phase
        return result

    def end_session(self, session_id: str) -> dict:
        session = self.get_session(session_id)
        if session is None:
            return {"error": "session not found"}

        session.phase = "ended"
        summary = self.concentration_model.get_session_summary(session_id)
        summary["environment_type"] = session.environment_type
        summary["learning_zones"] = session.learning_zones
        summary["base_noise_db"] = session.base_noise_db

        self.concentration_model.remove_session(session_id)
        self.sessions.pop(session_id, None)

        return summary


session_manager = SessionManager()
