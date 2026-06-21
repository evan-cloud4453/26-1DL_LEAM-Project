"""저조도 환경 개선용 CLAHE 전처리 유틸 (중간 보고서 STEP2 영상 전처리 반영)."""
import cv2
import numpy as np

# 충분히 밝은 이미지는 보정하지 않는다(워시아웃 방지). L 채널 평균 밝기 기준.
BRIGHTNESS_THRESHOLD = 110
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def enhance_lowlight(rgb: np.ndarray) -> np.ndarray:
    """RGB(HxWx3, uint8) 이미지에 적응형 CLAHE를 적용한다.

    LAB 색공간의 L(명도) 채널에만 CLAHE를 적용하여 색은 보존하고 대비만 개선한다.
    L 채널 평균 밝기가 임계값 이상이면(밝은 환경) 원본을 그대로 반환한다.
    """
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
        return rgb
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    if float(l.mean()) >= BRIGHTNESS_THRESHOLD:
        return rgb  # 저조도가 아니면 보정하지 않음
    l = _clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
