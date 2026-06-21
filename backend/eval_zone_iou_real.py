# -*- coding: utf-8 -*-
"""실측 세션 기반 학습 Zone IoU 평가.

개발자 모드 캡처에서 읽은 시스템 Zone 박스(pred)와, 사용자가 표기한 실제 학습대상
영역(GT)을 같은 정규화 좌표[0,1]에서 비교한다. 합성 평가(eval_zone_iou.py)와
동일한 64x48 격자 / 동일 지표(IoU·Dice·Recall·Specificity)를 사용해 비교 가능하게 한다.

각 샘플은 normalized [x0,x1,y0,y1] 사각형의 리스트로 정의한다.
값은 실제 캡처를 받은 뒤 채운다.
"""
import numpy as np

GRID_W, GRID_H = 64, 48  # learning_zone.py와 동일


def rasterize(rects):
    m = np.zeros((GRID_H, GRID_W), dtype=bool)
    for r in range(GRID_H):
        cy = (r + 0.5) / GRID_H
        for c in range(GRID_W):
            cx = (c + 0.5) / GRID_W
            for (x0, x1, y0, y1) in rects:
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    m[r, c] = True
                    break
    return m


def metrics(pred, gt):
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    union = tp + fp + fn
    iou = tp / union if union else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return iou, dice, recall, spec


# ===== 실측 샘플: (이름, pred 박스들, gt 박스들) — 캡처 받은 뒤 채움 =====
# pred = 화면의 주황 Zone 박스(들), gt = 실제 학습대상 영역(사용자 표기)
SAMPLES = [
    # 샘플1: 정면 모니터 + 키보드 (Zone 2개)
    ("S1 정면 모니터+키보드",
     [(0.19, 0.85, 0.07, 0.59), (0.30, 0.60, 0.79, 0.93)],            # pred: 큰 박스(모니터)+작은 박스(키보드)
     [(0.30, 0.75, 0.15, 0.55), (0.32, 0.62, 0.80, 0.95)]),           # GT: 모니터, 키보드
    # 샘플2: 왼쪽 독서대 + 아래 책 + 중앙 강의화면 (Zone 3개)
    ("S2 독서대+책+강의화면",
     [(0.00, 0.27, 0.46, 1.00), (0.325, 0.595, 0.42, 0.72), (0.325, 0.95, 0.77, 1.00)],
     [(0.02, 0.25, 0.50, 0.95), (0.35, 0.58, 0.45, 0.70), (0.38, 0.90, 0.80, 1.00)]),
    # 샘플3: 단일 모니터(화면만) (Zone 1개)
    ("S3 단일 모니터",
     [(0.29, 0.67, 0.32, 0.73)],
     [(0.33, 0.63, 0.38, 0.68)]),
]

if __name__ == "__main__":
    if not SAMPLES:
        print("아직 실측 샘플이 입력되지 않았습니다. 캡처를 받으면 SAMPLES를 채웁니다.")
        raise SystemExit
    ious, dices, recs, specs = [], [], [], []
    print("=== 실측 세션 학습 Zone IoU ===")
    for name, pred_rects, gt_rects in SAMPLES:
        pred = rasterize(pred_rects)
        gt = rasterize(gt_rects)
        iou, dice, rec, spec = metrics(pred, gt)
        ious.append(iou); dices.append(dice); recs.append(rec); specs.append(spec)
        print(f"[{name}] IoU={iou:.3f} Dice={dice:.3f} Recall={rec:.3f} Spec={spec:.3f}")
    print(f"\n평균 IoU={np.mean(ious):.3f} Dice={np.mean(dices):.3f} "
          f"Recall={np.mean(recs):.3f} Spec={np.mean(specs):.3f}")
