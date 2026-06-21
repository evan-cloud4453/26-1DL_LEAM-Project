import json
import logging

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

from app.schemas.analysis import (
    FrameAnalysisRequest, FrameAnalysisResponse,
    AudioAnalysisRequest, AudioAnalysisResponse,
    ClientFeatures,
)
from app.services.session_manager import session_manager

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/frame", response_model=FrameAnalysisResponse)
async def analyze_frame(req: FrameAnalysisRequest, request: Request):
    detector = request.app.state.phone_detector
    result = detector.detect_from_base64(req.frame_base64)
    return FrameAnalysisResponse(**result)


@router.post("/audio", response_model=AudioAnalysisResponse)
async def analyze_audio(req: AudioAnalysisRequest, request: Request):
    classifier = request.app.state.audio_classifier
    result = classifier.analyze_from_base64(req.audio_base64, req.sample_rate)
    return AudioAnalysisResponse(**result)


@router.post("/features")
async def process_features(features: ClientFeatures, request: Request):
    """Process client-side extracted features and return concentration grade."""
    session = session_manager.get_session(features.session_id)
    if session is None:
        return {"error": "session not found"}

    phone_result = {"phone_detected": False, "phone_confidence": 0.0}
    audio_result = None

    result = session_manager.process_features(
        session_id=features.session_id,
        features=features.model_dump(),
        phone_result=phone_result,
        audio_result=audio_result,
    )
    return result


@router.websocket("/ws/{session_id}")
async def websocket_analysis(websocket: WebSocket, session_id: str):
    """
    Real-time WebSocket endpoint.
    Client sends JSON messages with type: "features" | "frame" | "audio"
    Server responds with concentration results.
    """
    await websocket.accept()
    logger.info("WebSocket connected for session %s", session_id)

    session = session_manager.get_session(session_id)
    if session is None:
        await websocket.send_json({"error": "session not found"})
        await websocket.close()
        return

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            msg_type = msg.get("type")

            if msg_type == "features":
                phone_result = {"phone_detected": False, "phone_confidence": 0.0}
                audio_result = None

                if "phone" in msg:
                    phone_result = msg["phone"]
                if "audio" in msg:
                    audio_result = msg["audio"]

                result = session_manager.process_features(
                    session_id=session_id,
                    features=msg.get("features", {}),
                    phone_result=phone_result,
                    audio_result=audio_result,
                )
                await websocket.send_json({"type": "concentration", **result})

            elif msg_type == "gaze":
                estimator = websocket.app.state.gaze_estimator
                face_b64 = msg.get("face_base64", "")
                if face_b64:
                    result = estimator.predict_from_base64(face_b64)
                    await websocket.send_json({"type": "gaze_estimation", **result})

            elif msg_type == "frame":
                detector = websocket.app.state.phone_detector
                frame_b64 = msg.get("frame_base64", "")
                if frame_b64:
                    result = detector.detect_from_base64(frame_b64)
                    await websocket.send_json({"type": "phone_detection", **result})

            elif msg_type == "audio":
                classifier = websocket.app.state.audio_classifier
                audio_b64 = msg.get("audio_base64", "")
                if audio_b64:
                    result = classifier.analyze_from_base64(audio_b64, msg.get("sample_rate", 16000))
                    await websocket.send_json({"type": "audio_analysis", **result})

            elif msg_type == "phase_check":
                phase_info = session_manager.update_phase(session_id)
                await websocket.send_json({"type": "phase_update", **phase_info})

            elif msg_type == "readapt":
                info = session_manager.trigger_readapt(session_id)
                await websocket.send_json({"type": "phase_update", **info})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for session %s", session_id)
    except Exception as e:
        logger.error("WebSocket error for session %s: %s", session_id, e)
        await websocket.close()
