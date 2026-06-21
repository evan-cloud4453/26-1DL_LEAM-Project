"""
개발 서버 실행 스크립트.
    conda run -n leam python run.py
"""
import os
import sys
from pathlib import Path

# Ensure the backend directory is in sys.path and is the working directory
BACKEND_DIR = Path(__file__).resolve().parent
os.chdir(BACKEND_DIR)
sys.path.insert(0, str(BACKEND_DIR))

import uvicorn

if __name__ == "__main__":
    print(f"Working directory: {os.getcwd()}")
    print(f"Starting LEAM server on http://localhost:8000")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
