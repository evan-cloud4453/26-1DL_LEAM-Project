"""
PANNs CNN10 fine-tuning for study environment sound classification.

Dataset: DCASE 2016 "TUT Acoustic Scenes" (3-class mapping, prepare_dcase.py)

Target classes (학습 맥락 환경 분류 = 적응 단계 축 A):
    0: quiet         (library / home — 조용·집중적합)
    1: social_indoor (cafe / office / grocery — 웅성·대화 실내)
    2: outdoor       (city_center / residential / park — 도시형 야외)

Usage:
    python training/prepare_dcase.py            # 먼저 데이터 변환
    python train_audio_classifier.py --data_dir ./datasets/audio --epochs 30
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio


LABELS = {"quiet": 0, "social_indoor": 1, "outdoor": 2}
NUM_CLASSES = len(LABELS)
SAMPLE_RATE = 16000
CLIP_DURATION = 5  # seconds


class AudioSceneDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train"):
        self.samples: list[tuple[Path, int]] = []
        root = Path(data_dir) / split

        for label_name, label_id in LABELS.items():
            label_dir = root / label_name
            if not label_dir.exists():
                continue
            for f in label_dir.glob("*.wav"):
                self.samples.append((f, label_id))

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
            start = torch.randint(0, waveform.shape[0] - self.target_length, (1,)).item()
            waveform = waveform[start:start + self.target_length]
        elif waveform.shape[0] < self.target_length:
            pad = self.target_length - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        return waveform, label


class EnvironmentClassifier(nn.Module):
    """Fine-tuning head on top of PANNs CNN10 embeddings."""
    def __init__(self, panns_model, num_classes=NUM_CLASSES):
        super().__init__()
        self.panns = panns_model
        for param in self.panns.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        with torch.no_grad():
            out = self.panns(x)
        embedding = out["embedding"]
        return self.classifier(embedding)

    def unfreeze_panns(self, num_layers: int = 2):
        """Unfreeze last N conv blocks for fine-tuning."""
        blocks = [self.panns.conv_block4, self.panns.conv_block3,
                  self.panns.conv_block2, self.panns.conv_block1]
        for block in blocks[:num_layers]:
            for param in block.parameters():
                param.requires_grad = True


def train(data_dir: str, panns_weights: str, epochs: int = 30, batch_size: int = 32,
          lr: float = 1e-3, unfreeze_epoch: int = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained PANNs
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "app" / "models"))
    from panns_model import Cnn10

    panns = Cnn10(sample_rate=SAMPLE_RATE, window_size=512, hop_size=160,
                  mel_bins=64, fmin=50, fmax=8000, classes_num=527)

    panns_path = Path(panns_weights)
    if panns_path.exists():
        checkpoint = torch.load(str(panns_path), map_location=device, weights_only=False)
        state = checkpoint["model"] if "model" in checkpoint else checkpoint
        # strict=False: torchaudio mel 버퍼/원본 추출기 키 차이는 무시,
        # 학습된 conv_block·fc 가중치만 전이 (AudioSet → 환경음 transfer learning)
        res = panns.load_state_dict(state, strict=False)
        loaded = sum(1 for k in state if k in dict(panns.named_parameters()) or "running" in k)
        print(f"Loaded pretrained PANNs weights (transfer): "
              f"{loaded} tensors, missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)}")
    else:
        print("WARNING: No pretrained weights found, training from scratch")

    model = EnvironmentClassifier(panns, NUM_CLASSES).to(device)

    train_ds = AudioSceneDataset(data_dir, "train")
    val_ds = AudioSceneDataset(data_dir, "val")

    if len(train_ds) == 0:
        print("ERROR: No training data found. Expected structure:")
        print(f"  {data_dir}/train/quiet/*.wav")
        print(f"  {data_dir}/train/cafe/*.wav")
        print(f"  {data_dir}/train/indoor/*.wav")
        print(f"  {data_dir}/train/outdoor/*.wav")
        return

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # 클래스 불균형 보정: inverse-frequency 가중치
    class_counts = [0] * NUM_CLASSES
    for _, lbl in train_ds.samples:
        class_counts[lbl] += 1
    total = sum(class_counts)
    weights = torch.tensor(
        [total / (NUM_CLASSES * c) if c > 0 else 0.0 for c in class_counts],
        dtype=torch.float32, device=device,
    )
    print(f"클래스별 학습 샘플 수: {class_counts} → 가중치: {weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    output_path = Path(__file__).parent.parent / "weights" / "panns_cnn10.pth"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        if epoch == unfreeze_epoch:
            model.unfreeze_panns(num_layers=2)
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr * 0.1)
            print(f"Epoch {epoch}: Unfreezing PANNs conv blocks for fine-tuning")

        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for waveforms, labels in train_loader:
            waveforms, labels = waveforms.to(device), labels.to(device)
            outputs = model(waveforms)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = correct / total
        scheduler.step()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for waveforms, labels in val_loader:
                waveforms, labels = waveforms.to(device), labels.to(device)
                outputs = model(waveforms)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = val_correct / val_total if val_total > 0 else 0

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f} "
              f"Train Acc: {train_acc:.4f} Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(output_path))
            print(f"  -> Saved best model (val_acc: {val_acc:.4f})")

    print(f"\nTraining complete. Best Val Acc: {best_val_acc:.4f}")
    print(f"Model saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune PANNs CNN10 for environment classification")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--panns_weights", type=str, default="../weights/panns_cnn10_pretrained.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(args.data_dir, args.panns_weights, args.epochs, args.batch_size, args.lr)
