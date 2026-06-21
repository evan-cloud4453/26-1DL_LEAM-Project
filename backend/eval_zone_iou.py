# -*- coding: utf-8 -*-
"""학습 Zone(E_gaze) 정량 평가 — 합성 시선 데이터 기반 IoU / Dice.

시나리오별로 '실제 학습 영역(Ground Truth 사각형)'을 정의하고, 그 영역 안을
사람이 응시하듯 여러 sub-fixation으로 훑는 시선열을 생성한다. 시스템이 생성한
학습 Zone 마스크(64x48)와 GT 마스크의 IoU·Dice를 계산한다.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from app.services.learning_zone import LearningZoneGenerator, GRID_W, GRID_H

FPS = 30.0
DT = 1.0 / FPS


def gt_mask(rects):
    """GT 사각형 목록을 64x48 격자 마스크로 래스터화 (is_in_study_zone과 동일 좌표계)."""
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


def simulate_gaze(gen, rects, rng, cycles=40, dwell_frames=45, jitter=0.015,
                  saccade_noise=0.15):
    """각 GT 사각형 내부를 여러 응시점으로 훑고, 영역 간 이동(saccade)을 섞는다."""
    t = 0.0
    for _ in range(cycles):
        for (x0, x1, y0, y1) in rects:
            # 사각형 내부 임의 지점에 응시(fixation)
            fx = rng.uniform(x0, x1)
            fy = rng.uniform(y0, y1)
            for _ in range(dwell_frames):
                gen.add_gaze_point(np.clip(fx + rng.normal(0, jitter), 0, 1),
                                   np.clip(fy + rng.normal(0, jitter), 0, 1), t)
                t += DT
            # 다음 지점으로 빠른 이동(saccade) — 노이즈로 간주되어야 함
            for k in range(5):
                gen.add_gaze_point(np.clip(fx + (rng.random() - 0.5) * 0.4, 0, 1),
                                   np.clip(fy + (rng.random() - 0.5) * 0.4, 0, 1), t)
                t += DT
        # 가끔 화면 밖 구석을 잠깐 응시(방해/딴짓) — 영역 밖 노이즈
        if rng.random() < saccade_noise:
            ox, oy = rng.choice([0.05, 0.95]), rng.choice([0.05, 0.95])
            for _ in range(8):
                gen.add_gaze_point(ox + rng.normal(0, 0.01), oy + rng.normal(0, 0.01), t)
                t += DT


def metrics(pred, gt):
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    union = tp + fp + fn
    iou = tp / union if union else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0          # 실제 학습 영역 포함률
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0      # 영역 밖 올바른 제외율
    return iou, dice, recall, precision, specificity


SCENARIOS = {
    "노트북(중앙 단일 모니터)": [(0.30, 0.62, 0.30, 0.62)],
    "독서대(모니터+측면 책)": [(0.20, 0.48, 0.28, 0.58), (0.60, 0.86, 0.55, 0.82)],
    "넓은 모니터(가로 와이드)": [(0.18, 0.78, 0.30, 0.60)],
}

if __name__ == "__main__":
    print("=== 학습 Zone 적응 정확도 (E_gaze) — 합성 시선 IoU 평가 ===")
    print(f"격자: {GRID_W}x{GRID_H}, FPS={FPS:.0f}\n")
    ious, dices, recalls, precs, specs = [], [], [], [], []
    for name, rects in SCENARIOS.items():
        rng = np.random.default_rng(42)
        gen = LearningZoneGenerator()
        simulate_gaze(gen, rects, rng)
        res = gen.generate_zones()
        pred = gen._mask
        gt = gt_mask(rects)
        if pred is None:
            print(f"[{name}] Zone 생성 실패(default)")
            continue
        iou, dice, rec, prec, spec = metrics(pred, gt)
        ious.append(iou); dices.append(dice); recalls.append(rec); precs.append(prec); specs.append(spec)
        print(f"[{name}]")
        print(f"  fixations={res['fixation_count']} / points={res['gaze_point_count']} / zones(k)={res['k']}")
        print(f"  IoU={iou:.3f}  Dice={dice:.3f}  Recall(학습영역 포함률)={rec:.3f}  "
              f"Precision={prec:.3f}  Specificity(영역밖 제외율)={spec:.3f}\n")
    if ious:
        print(f"평균 IoU={np.mean(ious):.3f}  Dice={np.mean(dices):.3f}  "
              f"Recall={np.mean(recalls):.3f}  Precision={np.mean(precs):.3f}  "
              f"Specificity={np.mean(specs):.3f}")
        e_gaze = float(np.mean(ious))
        e_audio = 0.795
        print(f"\nE_gaze(평균 IoU) = {e_gaze:.3f}")
        print(f"E_audio(환경음 Accuracy) = {e_audio:.3f}")
        print(f"E_total = 0.7*{e_gaze:.3f} + 0.3*{e_audio:.3f} = {0.7*e_gaze + 0.3*e_audio:.3f}")
