from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import CORS_ORIGINS
from app.database.db import init_db
from app.routers import session, analysis, report

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    from app.models.phone_detector import PhoneDetector
    from app.models.audio_classifier import AudioClassifier
    from app.models.gaze_estimator import GazeEstimator
    app.state.phone_detector = PhoneDetector()
    app.state.audio_classifier = AudioClassifier()
    app.state.gaze_estimator = GazeEstimator()
    yield


app = FastAPI(
    title="LEAM API",
    description="Learning Environment Adaptive Model - Concentration Analysis System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(session.router, prefix="/api/sessions", tags=["sessions"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(report.router, prefix="/api/reports", tags=["reports"])

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok"}
