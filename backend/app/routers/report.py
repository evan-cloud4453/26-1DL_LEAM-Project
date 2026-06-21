from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.db import get_db
from app.database.models import StudySession, ConcentrationLog

router = APIRouter()


@router.get("/{session_id}")
async def get_report(session_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(StudySession).where(StudySession.id == session_id))
    session = result.scalar_one_or_none()
    if session is None:
        return {"error": "session not found"}

    log_result = await db.execute(
        select(ConcentrationLog)
        .where(ConcentrationLog.session_id == session_id)
        .order_by(ConcentrationLog.timestamp)
    )
    logs = log_result.scalars().all()

    grade_counts = {"A": 0, "B": 0, "C": 0, "F": 0}
    trigger_counts: dict[str, int] = {}
    for log in logs:
        grade_counts[log.grade] = grade_counts.get(log.grade, 0) + 1
        trigger_counts[log.trigger] = trigger_counts.get(log.trigger, 0) + 1

    total_duration = session.total_duration_sec or 0
    focus_duration = session.focus_duration_sec or 0

    return {
        "session_id": session.id,
        "started_at": session.started_at.isoformat() if session.started_at else None,
        "ended_at": session.ended_at.isoformat() if session.ended_at else None,
        "total_study_time_sec": total_duration,
        "actual_focus_time_sec": focus_duration,
        "focus_ratio": round(focus_duration / total_duration, 4) if total_duration > 0 else 0,
        "final_score": session.final_score,
        "environment_type": session.environment_type,
        "grade_distribution": grade_counts,
        "distraction_summary": {
            "gaze_deviation_count": session.gaze_deviation_count,
            "phone_detected_count": session.phone_detected_count,
            "drowsy_count": session.drowsy_count,
            "sleep_detected": session.sleep_detected,
        },
        "trigger_breakdown": trigger_counts,
        "learning_zones": session.learning_zones,
    }
