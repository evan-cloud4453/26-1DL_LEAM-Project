import datetime

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.db import get_db
from app.database.models import StudySession
from app.schemas.session import SessionCreate, SessionResponse, SessionEnd
from app.services.session_manager import session_manager

router = APIRouter()


@router.post("/start", response_model=SessionResponse)
async def start_session(_body: SessionCreate, db: AsyncSession = Depends(get_db)):
    db_session = StudySession()
    db.add(db_session)
    await db.commit()
    await db.refresh(db_session)

    session_manager.create_session(db_session.id)

    return db_session


@router.post("/end")
async def end_session(body: SessionEnd, db: AsyncSession = Depends(get_db)):
    summary = session_manager.end_session(body.session_id)
    if "error" in summary:
        return summary

    result = await db.execute(select(StudySession).where(StudySession.id == body.session_id))
    db_session = result.scalar_one_or_none()
    if db_session:
        db_session.ended_at = datetime.datetime.utcnow()
        db_session.total_duration_sec = summary.get("elapsed_sec", 0)
        db_session.focus_duration_sec = summary.get("focus_time_sec", 0)
        db_session.final_score = summary.get("final_score")
        db_session.environment_type = summary.get("environment_type")
        db_session.learning_zones = summary.get("learning_zones")
        db_session.base_noise_db = summary.get("base_noise_db")
        db_session.gaze_deviation_count = summary.get("gaze_deviation_count", 0)
        db_session.phone_detected_count = summary.get("phone_detected_count", 0)
        db_session.status = "ended"
        await db.commit()

    return summary


@router.get("/", response_model=list[SessionResponse])
async def list_sessions(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(StudySession).order_by(StudySession.started_at.desc()).limit(20)
    )
    return result.scalars().all()


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(StudySession).where(StudySession.id == session_id))
    return result.scalar_one()
