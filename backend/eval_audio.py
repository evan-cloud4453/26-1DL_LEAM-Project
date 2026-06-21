"""환경음 분류 모델 평가: Accuracy + Macro-F1 + 클래스별 리포트 + 혼동행렬."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from training.train_audio_classifier import (
    AudioSceneDataset, EnvironmentClassifier, LABELS, NUM_CLASSES,
)
from app.models.panns_model import Cnn10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ID2LABEL = {v: k for k, v in LABELS.items()}

panns = Cnn10(sample_rate=16000, window_size=512, hop_size=160,
              mel_bins=64, fmin=50, fmax=8000, classes_num=527)
model = EnvironmentClassifier(panns, NUM_CLASSES).to(device)
state = torch.load("weights/panns_cnn10.pth", map_location=device, weights_only=False)
model.load_state_dict(state)
model.eval()

val_ds = AudioSceneDataset("./datasets/audio", "val")
loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

y_true, y_pred = [], []
with torch.no_grad():
    for wavs, labels in loader:
        out = model(wavs.to(device))
        pred = out.argmax(1).cpu().numpy()
        y_pred.extend(pred.tolist())
        y_true.extend(labels.numpy().tolist())

y_true, y_pred = np.array(y_true), np.array(y_pred)
names = [ID2LABEL[i] for i in range(NUM_CLASSES)]

acc = (y_true == y_pred).mean()
macro_f1 = f1_score(y_true, y_pred, average="macro")
print(f"\n=== 환경음 분류 평가 (val {len(y_true)}개) ===")
print(f"Accuracy : {acc:.4f}")
print(f"Macro-F1 : {macro_f1:.4f}\n")
print(classification_report(y_true, y_pred, target_names=names, digits=4))
print("Confusion Matrix (행=정답, 열=예측):")
print(f"{'':>14}" + "".join(f"{n:>14}" for n in names))
cm = confusion_matrix(y_true, y_pred)
for i, n in enumerate(names):
    print(f"{n:>14}" + "".join(f"{v:>14}" for v in cm[i]))
