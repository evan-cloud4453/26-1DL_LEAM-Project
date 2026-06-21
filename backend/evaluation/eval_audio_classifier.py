"""
PANNs CNN10 환경음 분류 모델 성능 평가.

성능지표:
  - 대표: Accuracy
  - 보조: Macro F1-score (클래스별 F1 평균)

사용법:
    python eval_audio_classifier.py --weights ../weights/panns_cnn10.pth --data_dir ../training/datasets/audio
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader


LABELS = {"quiet": 0, "cafe": 1, "indoor": 2, "outdoor": 3}
LABEL_NAMES = {v: k for k, v in LABELS.items()}
SAMPLE_RATE = 16000
CLIP_DURATION = 5


class AudioTestDataset(Dataset):
    def __init__(self, data_dir: str):
        self.samples = []
        root = Path(data_dir) / "test"
        if not root.exists():
            root = Path(data_dir) / "val"
        for name, idx in LABELS.items():
            d = root / name
            if d.exists():
                for f in d.glob("*.wav"):
                    self.samples.append((f, idx))
        self.target_length = SAMPLE_RATE * CLIP_DURATION

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        waveform, sr = torchaudio.load(str(path))
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        if waveform.shape[0] > self.target_length:
            waveform = waveform[:self.target_length]
        elif waveform.shape[0] < self.target_length:
            waveform = torch.nn.functional.pad(waveform, (0, self.target_length - waveform.shape[0]))
        return waveform, label


def compute_metrics(y_true, y_pred, num_classes=4):
    """Compute Accuracy, per-class Precision/Recall/F1, Macro F1."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = np.mean(y_true == y_pred)

    per_class = {}
    f1_scores = []
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        name = LABEL_NAMES.get(c, str(c))
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1,
                           "support": int(np.sum(y_true == c))}
        f1_scores.append(f1)

    macro_f1 = np.mean(f1_scores)

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
    }


def build_confusion_matrix(y_true, y_pred, num_classes=4):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def evaluate(weights_path: str, data_dir: str, batch_size: int = 32):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "app" / "models"))
    from panns_model import Cnn10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    panns = Cnn10(sample_rate=SAMPLE_RATE, window_size=512, hop_size=160,
                  mel_bins=64, fmin=50, fmax=8000, classes_num=527)

    # Load fine-tuned environment classifier
    from train_env_classifier import EnvironmentClassifier  # noqa: this is in training/
    model = EnvironmentClassifier(panns, num_classes=len(LABELS))

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    dataset = AudioTestDataset(data_dir)
    if len(dataset) == 0:
        print("ERROR: No test data found.")
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_true, all_pred = [], []
    with torch.no_grad():
        for waveforms, labels in loader:
            waveforms = waveforms.to(device)
            outputs = model(waveforms)
            _, preds = outputs.max(1)
            all_true.extend(labels.numpy())
            all_pred.extend(preds.cpu().numpy())

    metrics = compute_metrics(all_true, all_pred)
    cm = build_confusion_matrix(all_true, all_pred)

    print("\n" + "=" * 50)
    print("  PANNs CNN10 Environment Classification Results")
    print("=" * 50)
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Macro F1:    {metrics['macro_f1']:.4f}")
    print()
    print("  Per-class:")
    for name, m in metrics["per_class"].items():
        print(f"    {name:>8s}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['support']})")
    print()
    print("  Confusion Matrix:")
    header = "          " + "  ".join(f"{LABEL_NAMES[i]:>8s}" for i in range(len(LABELS)))
    print(header)
    for i in range(len(LABELS)):
        row = f"  {LABEL_NAMES[i]:>8s}: " + "  ".join(f"{cm[i][j]:>8d}" for j in range(len(LABELS)))
        print(row)
    print("=" * 50)

    output_path = Path(__file__).parent / "results_audio.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_path}")

    return metrics


def evaluate_simple_db(data_dir: str):
    """Fallback: evaluate dB-based heuristic (no model needed)."""
    dataset = AudioTestDataset(data_dir)
    if len(dataset) == 0:
        print("No test data. Skipping.")
        return

    all_true, all_pred = [], []
    for waveform, label in dataset:
        audio = waveform.numpy()
        rms = np.sqrt(np.mean(audio ** 2))
        db = 20 * np.log10(rms / 1e-5) if rms > 1e-10 else 0

        if db < 35:
            pred = 0  # quiet
        elif db > 60:
            pred = 1  # cafe
        else:
            pred = 2  # indoor

        all_true.append(label)
        all_pred.append(pred)

    metrics = compute_metrics(all_true, all_pred)

    print("\n" + "=" * 50)
    print("  dB Heuristic Baseline Results")
    print("=" * 50)
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Macro F1:    {metrics['macro_f1']:.4f}")
    print("=" * 50)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="../weights/panns_cnn10.pth")
    parser.add_argument("--data_dir", type=str, default="../training/datasets/audio")
    parser.add_argument("--baseline", action="store_true", help="Run dB heuristic baseline")
    args = parser.parse_args()

    if args.baseline:
        evaluate_simple_db(args.data_dir)
    else:
        evaluate(args.weights, args.data_dir)
