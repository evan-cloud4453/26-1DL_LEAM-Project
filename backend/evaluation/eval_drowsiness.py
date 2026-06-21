"""
EAR/MAR 기반 졸음 탐지 성능 평가.

성능지표:
  - 대표: Accuracy
  - 보조: F1-score (macro)

EAR/MAR은 학습 기반 모델이 아닌 임계값(threshold) 기반 판단이므로,
다양한 threshold에서 최적값을 탐색하는 방식으로 평가.

사용법:
    python eval_drowsiness.py --video_dir ../training/datasets/drowsiness
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp


# Landmark indices (MediaPipe Face Mesh)
LEFT_EYE = {"left": 33, "right": 133, "top": [159, 145], "bottom": [23, 130]}
RIGHT_EYE = {"left": 362, "right": 263, "top": [386, 374], "bottom": [252, 359]}
MOUTH = {"top": 13, "bottom": 14, "left": 78, "right": 308}


def dist(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def compute_ear(landmarks, eye, w, h):
    def pt(idx):
        lm = landmarks[idx]
        return (lm.x * w, lm.y * h)

    h1 = dist(pt(eye["top"][0]), pt(eye["bottom"][0]))
    h2 = dist(pt(eye["top"][1]), pt(eye["bottom"][1]))
    horiz = dist(pt(eye["left"]), pt(eye["right"]))
    return (h1 + h2) / (2 * horiz) if horiz > 1e-6 else 0.3


def compute_mar(landmarks, w, h):
    def pt(idx):
        lm = landmarks[idx]
        return (lm.x * w, lm.y * h)

    v = dist(pt(MOUTH["top"]), pt(MOUTH["bottom"]))
    hz = dist(pt(MOUTH["left"]), pt(MOUTH["right"]))
    return v / hz if hz > 1e-6 else 0


def process_video(video_path: str, face_mesh):
    """Extract per-frame EAR and MAR from video."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    ear_values = []
    mar_values = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            ear_l = compute_ear(lm, LEFT_EYE, w, h)
            ear_r = compute_ear(lm, RIGHT_EYE, w, h)
            ear = (ear_l + ear_r) / 2
            mar = compute_mar(lm, w, h)
        else:
            ear = None
            mar = None

        ear_values.append(ear)
        mar_values.append(mar)

    cap.release()
    return ear_values, mar_values, fps


def evaluate_threshold(ear_values, labels, fps,
                       ear_threshold=0.21, consec_frames=9):
    """
    Evaluate drowsiness detection at given EAR threshold.

    labels: list of 0 (awake) or 1 (drowsy) per frame
    """
    preds = []
    low_count = 0

    for ear in ear_values:
        if ear is None:
            preds.append(0)
            low_count = 0
            continue
        if ear < ear_threshold:
            low_count += 1
        else:
            low_count = 0
        preds.append(1 if low_count >= consec_frames else 0)

    y_true = np.array(labels[:len(preds)])
    y_pred = np.array(preds[:len(labels)])
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    accuracy = (tp + tn) / n if n > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "ear_threshold": ear_threshold,
        "consec_frames": consec_frames,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def threshold_search(ear_values, labels, fps):
    """Grid search over EAR threshold and consec_frames."""
    best = None
    results = []

    for threshold in np.arange(0.15, 0.30, 0.01):
        for consec in [3, 6, 9, 12, 15, 20]:
            r = evaluate_threshold(ear_values, labels, fps, threshold, consec)
            results.append(r)
            if best is None or r["f1"] > best["f1"]:
                best = r

    print("\n" + "=" * 60)
    print("  EAR Drowsiness Detection - Threshold Search Results")
    print("=" * 60)
    print(f"  Best EAR threshold: {best['ear_threshold']:.2f}")
    print(f"  Best consec frames: {best['consec_frames']}")
    print(f"  Accuracy:           {best['accuracy']:.4f}")
    print(f"  Precision:          {best['precision']:.4f}")
    print(f"  Recall:             {best['recall']:.4f}")
    print(f"  F1-score:           {best['f1']:.4f}")
    print("=" * 60)

    return best, results


def evaluate_from_dir(video_dir: str):
    """
    Expects structure:
        video_dir/
            awake/  *.mp4
            drowsy/ *.mp4
    """
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)

    all_ear, all_labels = [], []

    for label_name, label_val in [("awake", 0), ("drowsy", 1)]:
        d = Path(video_dir) / label_name
        if not d.exists():
            print(f"Directory not found: {d}")
            continue
        for vp in sorted(d.glob("*.mp4")):
            print(f"Processing {vp.name} (label={label_name})...")
            ear_vals, _, fps = process_video(str(vp), face_mesh)
            n = len(ear_vals)
            all_ear.extend(ear_vals)
            all_labels.extend([label_val] * n)

    face_mesh.close()

    if len(all_ear) == 0:
        print("No data found.")
        return

    print(f"\nTotal frames: {len(all_ear)}")
    print(f"  Awake frames:  {all_labels.count(0)}")
    print(f"  Drowsy frames: {all_labels.count(1)}")

    best, all_results = threshold_search(all_ear, all_labels, 30)

    output_path = Path(__file__).parent / "results_drowsiness.json"
    with open(output_path, "w") as f:
        json.dump({"best": best, "all_results": all_results}, f, indent=2)
    print(f"Results saved to {output_path}")

    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, default="../training/datasets/drowsiness")
    args = parser.parse_args()
    evaluate_from_dir(args.video_dir)
