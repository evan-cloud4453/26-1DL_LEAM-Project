import datetime
import uuid

from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship

from app.database.db import Base


def gen_uuid():
    return str(uuid.uuid4())


class StudySession(Base):
    __tablename__ = "study_sessions"

    id = Column(String, primary_key=True, default=gen_uuid)
    started_at = Column(DateTime, default=datetime.datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    total_duration_sec = Column(Float, default=0.0)
    focus_duration_sec = Column(Float, default=0.0)
    final_score = Column(Float, nullable=True)
    environment_type = Column(String, nullable=True)  # quiet / cafe / indoor
    learning_zones = Column(JSON, nullable=True)
    base_noise_db = Column(Float, nullable=True)
    gaze_deviation_count = Column(Integer, default=0)
    phone_detected_count = Column(Integer, default=0)
    drowsy_count = Column(Integer, default=0)
    sleep_detected = Column(Integer, default=0)
    status = Column(String, default="calibrating")  # calibrating / adapting / active / ended

    logs = relationship("ConcentrationLog", back_populates="session", cascade="all, delete-orphan")


class ConcentrationLog(Base):
    __tablename__ = "concentration_logs"

    id = Column(String, primary_key=True, default=gen_uuid)
    session_id = Column(String, ForeignKey("study_sessions.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    grade = Column(String, nullable=False)  # A / B / C / F
    trigger = Column(String, nullable=False)  # face_absent / drowsy / phone / gaze / noise
    detail = Column(JSON, nullable=True)

    session = relationship("StudySession", back_populates="logs")
