"""
학습 Zone 적응 정확도 평가.

성능지표:
  - 대표: IoU (Intersection over Union)
  - 보조: Dice Similarity Coefficient

사용법:
    python eval_learning_zone.py --test_cases test_zones.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
from services.learning_zone import LearningZoneGenerator


def compute_iou(pred_zone: dict, gt_zone: dict) -> float:
    """Compute IoU between predicted zone and ground truth zone (both as bounding boxes)."""
    px1, py1 = pred_zone.get("x_min", 0), pred_zone.get("y_min", 0)
    px2, py2 = pred_zone.get("x_max", 1), pred_zone.get("y_max", 1)
    gx1, gy1 = gt_zone.get("x_min", 0), gt_zone.get("y_min", 0)
    gx2, gy2 = gt_zone.get("x_max", 1), gt_zone.get("y_max", 1)

    ix1 = max(px1, gx1)
    iy1 = max(py1, gy1)
    ix2 = min(px2, gx2)
    iy2 = min(py2, gy2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    intersection = inter_w * inter_h

    pred_area = (px2 - px1) * (py2 - py1)
    gt_area = (gx2 - gx1) * (gy2 - gy1)
    union = pred_area + gt_area - intersection

    return intersection / union if union > 0 else 0


def compute_dice(pred_zone: dict, gt_zone: dict) -> float:
    """Compute Dice coefficient between predicted and GT zones."""
    px1, py1 = pred_zone.get("x_min", 0), pred_zone.get("y_min", 0)
    px2, py2 = pred_zone.get("x_max", 1), pred_zone.get("y_max", 1)
    gx1, gy1 = gt_zone.get("x_min", 0), gt_zone.get("y_min", 0)
    gx2, gy2 = gt_zone.get("x_max", 1), gt_zone.get("y_max", 1)

    ix1 = max(px1, gx1)
    iy1 = max(py1, gy1)
    ix2 = min(px2, gx2)
    iy2 = min(py2, gy2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    intersection = inter_w * inter_h

    pred_area = (px2 - px1) * (py2 - py1)
    gt_area = (gx2 - gx1) * (gy2 - gy1)

    return 2 * intersection / (pred_area + gt_area) if (pred_area + gt_area) > 0 else 0


def generate_synthetic_test_case(scenario: str) -> dict:
    """Generate synthetic gaze points for known study zone configurations."""
    np.random.seed(42)

    if scenario == "center_display":
        # User looking at center display
        gaze_points = np.random.multivariate_normal(
            [0.5, 0.45], [[0.01, 0], [0, 0.01]], size=300
        )
        gt_zone = {"x_min": 0.25, "x_max": 0.75, "y_min": 0.2, "y_max": 0.7}

    elif scenario == "left_book":
        # User reading a book placed to the left
        gaze_points = np.random.multivariate_normal(
            [0.3, 0.6], [[0.015, 0], [0, 0.01]], size=300
        )
        gt_zone = {"x_min": 0.1, "x_max": 0.55, "y_min": 0.35, "y_max": 0.85}

    elif scenario == "right_book":
        # Book stand on the right side
        gaze_points = np.random.multivariate_normal(
            [0.7, 0.55], [[0.012, 0], [0, 0.012]], size=300
        )
        gt_zone = {"x_min": 0.45, "x_max": 0.9, "y_min": 0.3, "y_max": 0.8}

    elif scenario == "dual_display_book":
        # Both display (center-top) and book (bottom-left)
        pts1 = np.random.multivariate_normal([0.5, 0.3], [[0.008, 0], [0, 0.005]], size=150)
        pts2 = np.random.multivariate_normal([0.35, 0.7], [[0.01, 0], [0, 0.008]], size=150)
        gaze_points = np.vstack([pts1, pts2])
        gt_zone = {"x_min": 0.15, "x_max": 0.75, "y_min": 0.15, "y_max": 0.85}

    elif scenario == "sparse_gaze":
        # Scattered gaze - difficult case
        gaze_points = np.random.uniform(0, 1, size=(100, 2))
        gt_zone = {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    gaze_points = np.clip(gaze_points, 0, 1)
    return {"gaze_points": gaze_points.tolist(), "gt_zone": gt_zone, "scenario": scenario}


def evaluate_single(test_case: dict) -> dict:
    gen = LearningZoneGenerator()
    for gx, gy in test_case["gaze_points"]:
        gen.add_gaze_point(gx, gy)

    result = gen.generate_zones()
    pred_study = None
    for z in result["zones"]:
        if z.get("type") == "study":
            pred_study = z
            break

    if pred_study is None:
        return {"iou": 0.0, "dice": 0.0, "scenario": test_case.get("scenario", "unknown")}

    gt = test_case["gt_zone"]
    iou = compute_iou(pred_study, gt)
    dice = compute_dice(pred_study, gt)

    return {
        "scenario": test_case.get("scenario", "unknown"),
        "iou": round(iou, 4),
        "dice": round(dice, 4),
        "pred_zone": {k: round(v, 3) for k, v in pred_study.items() if isinstance(v, float)},
        "gt_zone": gt,
    }


def evaluate(test_file: str = None):
    scenarios = ["center_display", "left_book", "right_book", "dual_display_book", "sparse_gaze"]

    if test_file and Path(test_file).exists():
        with open(test_file) as f:
            test_cases = json.load(f)
    else:
        test_cases = [generate_synthetic_test_case(s) for s in scenarios]

    results = [evaluate_single(tc) for tc in test_cases]

    ious = [r["iou"] for r in results]
    dices = [r["dice"] for r in results]

    print("\n" + "=" * 60)
    print("  Learning Zone Adaptation - Evaluation Results")
    print("=" * 60)
    for r in results:
        print(f"  {r['scenario']:>22s}:  IoU={r['iou']:.4f}  Dice={r['dice']:.4f}")
    print("-" * 60)
    print(f"  {'Mean':>22s}:  IoU={np.mean(ious):.4f}  Dice={np.mean(dices):.4f}")
    print("=" * 60)

    output = {"per_scenario": results, "mean_iou": round(np.mean(ious), 4),
              "mean_dice": round(np.mean(dices), 4)}
    output_path = Path(__file__).parent / "results_learning_zone.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_path}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_cases", type=str, default=None)
    args = parser.parse_args()
    evaluate(args.test_cases)
