from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class SessionCreate(BaseModel):
    pass


class SessionResponse(BaseModel):
    id: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    status: str
    total_duration_sec: float
    focus_duration_sec: float
    final_score: Optional[float] = None

    model_config = {"from_attributes": True}


class SessionEnd(BaseModel):
    session_id: str
