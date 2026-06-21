"""
Hierarchical Decision Model for concentration grading.
Layer 1: Face Detection
Layer 2: Drowsiness (EAR/MAR)
Layer 3: Phone Detection
Layer 4-1: Gaze Deviation
Layer 4-2: Noise Analysis
"""
import time
from dataclasses import dataclass, field

from app.config import (
    EAR_THRESHOLD, EAR_CONSEC_FRAMES_DROWSY, EAR_CONSEC_FRAMES_SLEEP,
    MAR_THRESHOLD, MAR_CONSEC_FRAMES_YAWN,
    GRADE_A, GRADE_B, GRADE_C, GRADE_F,
)


@dataclass
class ConcentrationState:
    face_absent_frames: int = 0
    ear_low_frames: int = 0
    mar_high_frames: int = 0
    gaze_deviation_count: int = 0
    gaze_deviation_frames: int = 0
    phone_detected_count: int = 0
    last_grade: str = GRADE_A
    total_frames: int = 0
    focused_frames: int = 0
    penalty_score: float = 0.0
    session_start: float = field(default_factory=time.time)
    grade_history: list[str] = field(default_factory=list)
    trigger_counts: dict = field(default_factory=dict)  # 트리거(졸음/휴대폰/시선이탈 등)별 프레임 수


class HierarchicalConcentrationModel:
    def __init__(self):
        self.states: dict[str, ConcentrationState] = {}

    def get_state(self, session_id: str) -> ConcentrationState:
        if session_id not in self.states:
            self.states[session_id] = ConcentrationState()
        return self.states[session_id]

    def evaluate(
        self,
        session_id: str,
        face_detected: bool,
        ear_left: float | None,
        ear_right: float | None,
        mar: float | None,
        gaze_yaw: float | None,
        gaze_pitch: float | None,
        phone_detected: bool,
        noise_disruptive: bool,
        in_study_zone: bool,
    ) -> dict:
        state = self.get_state(session_id)
        state.total_frames += 1
        triggers: list[str] = []
        grade = GRADE_A

        # Layer 1: Face detection
        if not face_detected:
            state.face_absent_frames += 1
            if state.face_absent_frames > 900:  # ~30s
                grade = GRADE_F
                triggers.append("face_absent_prolonged")
            elif state.face_absent_frames > 150:  # ~5s
                grade = GRADE_F
                triggers.append("face_absent")
            else:
                grade = GRADE_C
                triggers.append("face_momentary_loss")
            return self._build_result(state, grade, triggers)

        state.face_absent_frames = 0

        # Layer 2: Drowsiness
        ear_avg = None
        if ear_left is not None and ear_right is not None:
            ear_avg = (ear_left + ear_right) / 2.0

        if ear_avg is not None and ear_avg < EAR_THRESHOLD:
            state.ear_low_frames += 1
            if state.ear_low_frames >= EAR_CONSEC_FRAMES_SLEEP:
                grade = GRADE_F
                triggers.append("sleeping")
                return self._build_result(state, grade, triggers)
            elif state.ear_low_frames >= EAR_CONSEC_FRAMES_DROWSY:
                grade = GRADE_C
                triggers.append("drowsy")
        else:
            state.ear_low_frames = 0

        if mar is not None and mar > MAR_THRESHOLD:
            state.mar_high_frames += 1
            if state.mar_high_frames >= MAR_CONSEC_FRAMES_YAWN:
                if grade != GRADE_C:
                    grade = GRADE_B
                triggers.append("yawning")
        else:
            state.mar_high_frames = 0

        # Layer 3: Phone detection
        if phone_detected:
            state.phone_detected_count += 1
            grade = GRADE_C
            triggers.append("phone_detected")
            return self._build_result(state, grade, triggers)

        # Layer 4-1: Gaze deviation
        # 모니터 가장자리·키보드 응시 등 정상 학습 행동을 오탐하지 않도록,
        # 학습 Zone 밖 응시가 충분히 '길게 지속'될 때만 감점한다.
        if not in_study_zone and gaze_yaw is not None:
            state.gaze_deviation_frames += 1
            if state.gaze_deviation_frames > 240:  # ~8s sustained → 명백한 이탈
                grade = GRADE_C
                triggers.append("gaze_sustained_deviation")
                state.gaze_deviation_count += 1
            elif state.gaze_deviation_frames > 120:  # ~4s
                if grade == GRADE_A:
                    grade = GRADE_B
                triggers.append("gaze_brief_deviation")
        else:
            # 잠깐 Zone 안으로 돌아오면 누적을 빠르게 회복(점진 감소)
            state.gaze_deviation_frames = max(0, state.gaze_deviation_frames - 3)

        # Layer 4-2: Noise
        if noise_disruptive:
            if grade == GRADE_A:
                grade = GRADE_B
            triggers.append("disruptive_noise")

        if not triggers:
            state.focused_frames += 1

        return self._build_result(state, grade, triggers)

    def _build_result(self, state: ConcentrationState, grade: str, triggers: list[str]) -> dict:
        state.last_grade = grade
        state.grade_history.append(grade)
        for t in triggers:
            state.trigger_counts[t] = state.trigger_counts.get(t, 0) + 1

        penalty = {"A": 0, "B": 1, "C": 3, "F": 10}.get(grade, 0)
        state.penalty_score += penalty

        max_penalty = state.total_frames * 10
        if max_penalty > 0:
            score = max(0, 100 - (state.penalty_score / state.total_frames) * 100)
        else:
            score = 100.0

        message = None
        if grade == GRADE_F and "sleeping" in triggers:
            message = "수면 상태가 감지되었습니다. 30초 이내에 개선되지 않으면 자동 종료됩니다."
        elif grade == GRADE_F and "face_absent" in triggers:
            message = "얼굴이 감지되지 않습니다. 자리를 이탈한 것으로 판단됩니다."
        elif grade == GRADE_C and "phone_detected" in triggers:
            message = "휴대폰이 감지되었습니다."

        return {
            "grade": grade,
            "score": round(score, 1),
            "triggers": triggers,
            "message": message,
        }

    def get_session_summary(self, session_id: str) -> dict:
        state = self.get_state(session_id)
        total = state.total_frames or 1
        grade_counts = {}
        for g in state.grade_history:
            grade_counts[g] = grade_counts.get(g, 0) + 1

        elapsed = time.time() - state.session_start
        focus_ratio = state.focused_frames / total

        # 점수 산출 투명화: 등급별 감점 기여도
        pen = {"A": 0, "B": 1, "C": 3, "F": 10}
        penalty_breakdown = {g: {"frames": grade_counts.get(g, 0),
                                 "penalty_each": pen[g],
                                 "penalty_total": grade_counts.get(g, 0) * pen[g]}
                             for g in ["A", "B", "C", "F"]}
        total_penalty = sum(v["penalty_total"] for v in penalty_breakdown.values())

        return {
            "total_frames": total,
            "focused_frames": state.focused_frames,
            "focus_ratio": round(focus_ratio, 4),
            "elapsed_sec": round(elapsed, 1),
            "focus_time_sec": round(elapsed * focus_ratio, 1),
            "grade_distribution": grade_counts,
            "trigger_counts": state.trigger_counts,
            "penalty_breakdown": penalty_breakdown,
            "total_penalty": total_penalty,
            "phone_detected_count": state.phone_detected_count,
            "gaze_deviation_count": state.gaze_deviation_count,
            "final_score": round(max(0, 100 - (state.penalty_score / total) * 100), 1),
        }

    def remove_session(self, session_id: str):
        self.states.pop(session_id, None)
