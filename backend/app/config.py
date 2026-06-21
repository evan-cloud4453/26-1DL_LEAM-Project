import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "weights"
DB_PATH = BASE_DIR / "leam.db"

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite+aiosqlite:///{DB_PATH}")

YOLO_WEIGHTS = WEIGHTS_DIR / "yolov8n_phone.pt"
PANNS_WEIGHTS = WEIGHTS_DIR / "panns_cnn10.pth"
PANNS_PRETRAINED_WEIGHTS = WEIGHTS_DIR / "panns_cnn10_pretrained.pth"  # AudioSet 527-class (이벤트/방해음)
L2CS_WEIGHTS = WEIGHTS_DIR / "l2cs_net.pkl"

# Calibration (화면 중앙 응시로 시선 원점 잡는 단계)
CALIBRATION_PHASE_SEC = 15      # 시선 원점 보정 단계 길이
CALIBRATION_DURATION_SEC = 180  # 환경 적응(adapting) 단계 = 3 minutes
READAPT_INTERVAL_SEC = 1800    # 30 minutes

# Drowsiness thresholds
EAR_THRESHOLD = 0.21
EAR_CONSEC_FRAMES_DROWSY = 9    # ~0.3s at 30fps
EAR_CONSEC_FRAMES_SLEEP = 300   # ~10s at 30fps
MAR_THRESHOLD = 0.6
MAR_CONSEC_FRAMES_YAWN = 45     # ~1.5s at 30fps

# Gaze
GAZE_DEVIATION_THRESHOLD = 15.0  # degrees
GAZE_ABSENT_FRAMES = 900         # 30s at 30fps

# Audio
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_SEC = 2.0
NOISE_DB_THRESHOLD_OFFSET = 10.0

# Concentration grades
GRADE_A = "A"  # Focused
GRADE_B = "B"  # Low Focus
GRADE_C = "C"  # Distracted
GRADE_F = "F"  # Absent

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
