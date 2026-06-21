"""
YOLOv8n 휴대폰 탐지 모델 성능 평가.

성능지표:
  - 대표: AP (Average Precision) @ IoU=0.5
  - 보조: Precision, Recall

사용법:
    python eval_phone_detector.py --weights ../weights/yolov8n_phone.pt --data_dir ../training/datasets/phone
"""
import argparse
import json
from pathlib import Path

import numpy as np


def evaluate(weights_path: str, data_dir: str, conf: float = 0.4, iou: float = 0.5):
    from ultralytics import YOLO

    model = YOLO(weights_path)
    data_yaml = Path(data_dir) / "data.yaml"

    if not data_yaml.exists():
        # Fallback: use COCO val for phone class
        print("data.yaml not found. Running on COCO val set (class 67: cell phone)...")
        metrics = model.val(conf=conf, iou=iou)
    else:
        metrics = model.val(data=str(data_yaml), conf=conf, iou=iou)

    results = {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }

    print("\n" + "=" * 50)
    print("  YOLOv8n Phone Detection - Evaluation Results")
    print("=" * 50)
    print(f"  AP@0.5 (mAP50):    {results['mAP50']:.4f}")
    print(f"  AP@0.5:0.95:       {results['mAP50_95']:.4f}")
    print(f"  Precision:         {results['precision']:.4f}")
    print(f"  Recall:            {results['recall']:.4f}")
    print("=" * 50)

    output_path = Path(__file__).parent / "results_phone.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

    return results


def evaluate_custom_images(weights_path: str, test_dir: str, conf: float = 0.4):
    """Evaluate on a folder of images with annotation txt files (YOLO format)."""
    from ultralytics import YOLO
    import cv2

    model = YOLO(weights_path)
    PHONE_CLASS = 67

    images_dir = Path(test_dir) / "images"
    labels_dir = Path(test_dir) / "labels"

    if not images_dir.exists():
        print(f"Test images directory not found: {images_dir}")
        return

    tp, fp, fn = 0, 0, 0

    for img_path in sorted(images_dir.glob("*.*")):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        # Ground truth
        label_path = labels_dir / (img_path.stem + ".txt")
        gt_has_phone = False
        if label_path.exists():
            for line in label_path.read_text().strip().split("\n"):
                if line.strip():
                    cls_id = int(line.strip().split()[0])
                    if cls_id == 0:  # in custom dataset, phone = class 0
                        gt_has_phone = True
                        break

        # Prediction
        results = model(str(img_path), conf=conf, verbose=False)
        pred_has_phone = False
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == PHONE_CLASS or int(box.cls[0]) == 0:
                    pred_has_phone = True
                    break

        if gt_has_phone and pred_has_phone:
            tp += 1
        elif not gt_has_phone and pred_has_phone:
            fp += 1
        elif gt_has_phone and not pred_has_phone:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "=" * 50)
    print("  Custom Test Set Results (Image-level)")
    print("=" * 50)
    print(f"  TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="../weights/yolov8n_phone.pt")
    parser.add_argument("--data_dir", type=str, default="../training/datasets/phone")
    parser.add_argument("--test_dir", type=str, default=None, help="Custom test images dir")
    parser.add_argument("--conf", type=float, default=0.4)
    args = parser.parse_args()

    if args.test_dir:
        evaluate_custom_images(args.weights, args.test_dir, args.conf)
    else:
        evaluate(args.weights, args.data_dir, args.conf)
