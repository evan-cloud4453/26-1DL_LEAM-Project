"""
Learning Zone generation via gaze fixation density analysis.

Pipeline (replaces the earlier K-Means approach):
1. I-DT fixation detection — separates true fixations from saccades (noise)
2. Duration-weighted kernel density estimation — "where did the user look,
   and for how long" as a continuous gaze heatmap
3. Level-set threshold + connected components — regions covering the top
   MASS_COVERAGE fraction of gaze density become study zones

Rationale: frequent/long gaze at a location implies a study object exists
there. Zone count and shape emerge from the data (no K to guess, arbitrary
zone shapes supported).
"""
import time

import numpy as np
from scipy import ndimage

# Density grid resolution (normalized screen [0,1]x[0,1])
GRID_W, GRID_H = 64, 48

# 넓은 모니터/키보드 응시까지 학습 영역으로 포용하도록 관대하게 설정
KDE_BANDWIDTH = 0.10       # gaussian kernel sigma (클수록 영역이 부드럽고 넓어짐)
MASS_COVERAGE = 0.92       # 전체 응시 밀도의 92%를 학습 영역으로 포함
IDT_DISPERSION = 0.12      # I-DT: 한 fixation으로 인정하는 분산 상한
IDT_MIN_DURATION = 0.10    # I-DT: 키보드 등 짧은 응시도 포착하도록 완화
IDT_MIN_POINTS = 2         # I-DT: 최소 샘플 수
MIN_ZONE_CELLS = 5         # 이보다 작은 연결요소 제거
MASK_DILATION_ITERS = 5    # 학습 영역 경계 여유(모니터 가장자리/키보드 흡수)


class LearningZoneGenerator:
    def __init__(self):
        self.gaze_points: list[tuple[float, float, float]] = []  # (x, y, t)
        self._density: np.ndarray | None = None
        self._mask: np.ndarray | None = None

    def add_gaze_point(self, x: float, y: float, t: float | None = None):
        if 0 <= x <= 1 and 0 <= y <= 1:
            self.gaze_points.append((x, y, t if t is not None else time.time()))

    # ---------- Step 1: I-DT fixation detection ----------
    def _detect_fixations(self) -> list[tuple[float, float, float]]:
        """Returns fixations as (center_x, center_y, duration_sec)."""
        pts = self.gaze_points
        n = len(pts)
        fixations = []
        i = 0
        while i < n - 1:
            j = i + 1
            min_x = max_x = pts[i][0]
            min_y = max_y = pts[i][1]
            while j < n:
                x, y, _ = pts[j]
                nmin_x, nmax_x = min(min_x, x), max(max_x, x)
                nmin_y, nmax_y = min(min_y, y), max(max_y, y)
                if (nmax_x - nmin_x) + (nmax_y - nmin_y) > IDT_DISPERSION:
                    break
                min_x, max_x, min_y, max_y = nmin_x, nmax_x, nmin_y, nmax_y
                j += 1
            duration = pts[j - 1][2] - pts[i][2]
            if duration >= IDT_MIN_DURATION and (j - i) >= IDT_MIN_POINTS:
                xs = [p[0] for p in pts[i:j]]
                ys = [p[1] for p in pts[i:j]]
                fixations.append((float(np.mean(xs)), float(np.mean(ys)),
                                  max(duration, 1e-3)))
                i = j
            else:
                i += 1
        return fixations

    # ---------- Step 2: duration-weighted KDE ----------
    def _compute_density(self, fixations) -> np.ndarray:
        gx, gy = np.meshgrid(np.linspace(0, 1, GRID_W), np.linspace(0, 1, GRID_H))
        density = np.zeros((GRID_H, GRID_W), dtype=np.float64)
        inv_2s2 = 1.0 / (2 * KDE_BANDWIDTH ** 2)
        for x, y, dur in fixations:
            density += dur * np.exp(-((gx - x) ** 2 + (gy - y) ** 2) * inv_2s2)
        return density

    # ---------- Step 3: level set + connected components ----------
    def generate_zones(self) -> dict:
        fixations = self._detect_fixations()
        if len(fixations) < 3:
            return self._default_zones()

        density = self._compute_density(fixations)
        total_mass = density.sum()
        if total_mass <= 0:
            return self._default_zones()

        # Smallest density level whose super-level set covers MASS_COVERAGE
        flat = np.sort(density.ravel())[::-1]
        cum = np.cumsum(flat)
        k = int(np.searchsorted(cum, MASS_COVERAGE * total_mass))
        threshold = float(flat[min(k, len(flat) - 1)])

        mask = density >= threshold
        labeled, ncomp = ndimage.label(mask)

        zones = []
        for c in range(1, ncomp + 1):
            comp = labeled == c
            cells = int(comp.sum())
            if cells < MIN_ZONE_CELLS:
                labeled[comp] = 0
                continue
            cy, cx = ndimage.center_of_mass(density * comp)
            ys, xs = np.where(comp)
            zones.append({
                "type": "study",
                "center_x": float(cx / (GRID_W - 1)),
                "center_y": float(cy / (GRID_H - 1)),
                "x_min": float(xs.min() / (GRID_W - 1)),
                "x_max": float(xs.max() / (GRID_W - 1)),
                "y_min": float(ys.min() / (GRID_H - 1)),
                "y_max": float(ys.max() / (GRID_H - 1)),
                "radius": float(max(xs.max() - xs.min(), ys.max() - ys.min())
                                / 2 / (GRID_W - 1)),
                "area_ratio": cells / (GRID_W * GRID_H),
                "point_count": cells,
            })

        if not zones:
            return self._default_zones()

        # Final mask (small components removed) + margin dilation
        final_mask = labeled > 0
        final_mask = ndimage.binary_dilation(final_mask, iterations=MASK_DILATION_ITERS)
        self._density = density
        self._mask = final_mask

        zones.append({"type": "non_study", "description": "Area outside study zones"})

        # Normalized grid for frontend heatmap rendering
        peak = density.max()
        norm = (density / peak).round(3)
        return {
            "zones": zones,
            "k": len(zones) - 1,
            "method": "fixation_kde",
            "gaze_point_count": len(self.gaze_points),
            "fixation_count": len(fixations),
            "grid": {
                "w": GRID_W,
                "h": GRID_H,
                "threshold": round(threshold / peak, 3),
                "data": norm.tolist(),
            },
        }

    def is_in_study_zone(self, gx: float, gy: float, zones: list[dict]) -> bool:
        if self._mask is not None:
            col = min(GRID_W - 1, max(0, int(gx * GRID_W)))
            row = min(GRID_H - 1, max(0, int(gy * GRID_H)))
            return bool(self._mask[row, col])
        # Fallback for default (circle) zones
        for zone in zones:
            if zone.get("type") != "study":
                continue
            cx = zone.get("center_x", 0.5)
            cy = zone.get("center_y", 0.5)
            r = zone.get("radius", 0.35)
            if (gx - cx) ** 2 + (gy - cy) ** 2 <= r ** 2:
                return True
        return False

    def _default_zones(self) -> dict:
        return {
            "zones": [
                {"type": "study", "center_x": 0.5, "center_y": 0.5, "radius": 0.35,
                 "x_min": 0.15, "x_max": 0.85, "y_min": 0.15, "y_max": 0.85,
                 "point_count": 0},
                {"type": "non_study", "description": "Area outside study zones"},
            ],
            "k": 1,
            "method": "default",
            "gaze_point_count": len(self.gaze_points),
            "fixation_count": 0,
        }

    def reset(self):
        self.gaze_points.clear()
        self._density = None
        self._mask = None
