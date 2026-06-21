import base64
import io
import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

COCO_PHONE_CLASS_ID = 67  # COCO class id for 'cell phone'


class PhoneDetector:
    """
    듀얼 모델 휴대폰 탐지.
    - fine-tuned 모델: AI Hub small-object 데이터로 학습 → 작고 멀리 있는 휴대폰 특화
    - COCO yolov8n: 다양한 크기로 학습 → 손에 든 크고 가까운 휴대폰 보완
    둘 중 하나라도 탐지하면 phone_detected=True (max confidence 사용).
    """
    def __init__(self, weights_path: str | None = None, confidence: float = 0.40):
        from ultralytics import YOLO
        from app.config import YOLO_WEIGHTS

        # COCO는 오탐이 거의 없어 보통 임계값 사용. fine-tuned 모델은 소형 데이터
        # 과적합으로 얼굴 등을 오탐(최대 ~0.56)하므로 0.65 이상에서만 인정한다.
        self.coco_conf = confidence
        self.ft_conf = 0.65

        # 1) fine-tuned 모델 (전이학습 결과)
        path = weights_path or str(YOLO_WEIGHTS)
        if Path(path).exists():
            self.ft_model = YOLO(path)
            self.ft_phone_ids = self._phone_ids(self.ft_model.names)
            logger.info("Loaded fine-tuned YOLOv8 from %s (phone ids=%s)", path, self.ft_phone_ids)
        else:
            self.ft_model = None
            logger.info("Fine-tuned weights not found at %s", path)

        # 2) COCO 사전학습 모델 (모든 크기의 휴대폰 보완)
        self.coco_model = YOLO("yolov8n.pt")
        self.coco_phone_ids = self._phone_ids(self.coco_model.names)
        logger.info("Loaded COCO yolov8n (phone ids=%s)", self.coco_phone_ids)

    @staticmethod
    def _phone_ids(names: dict) -> set[int]:
        ids = {i for i, n in names.items() if "phone" in str(n).lower()}
        return ids or {COCO_PHONE_CLASS_ID}

    def detect_from_base64(self, frame_b64: str) -> dict:
        from app.image_utils import enhance_lowlight
        img_bytes = base64.b64decode(frame_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        # 저조도 환경 개선(CLAHE) — 어두운 화면에서도 휴대폰 탐지율 보강
        return self.detect(enhance_lowlight(np.array(img)))

    def _detect_one(self, model, phone_ids, frame, best, conf_th):
        results = model(frame, conf=conf_th, verbose=False)
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) in phone_ids:
                    conf = float(box.conf[0])
                    if conf > best["conf"]:
                        best["conf"] = conf
                        best["bbox"] = box.xyxy[0].tolist()
                        best["detected"] = True
        return best

    def detect(self, frame: np.ndarray) -> dict:
        best = {"detected": False, "conf": 0.0, "bbox": None}
        if self.ft_model is not None:
            best = self._detect_one(self.ft_model, self.ft_phone_ids, frame, best, self.ft_conf)
        best = self._detect_one(self.coco_model, self.coco_phone_ids, frame, best, self.coco_conf)

        return {
            "phone_detected": best["detected"],
            "phone_confidence": round(best["conf"], 4),
            "phone_bbox": best["bbox"],
        }
