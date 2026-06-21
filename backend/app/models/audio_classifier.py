import base64
import io
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio

logger = logging.getLogger(__name__)

# fine-tuned 환경음 분류 클래스 (prepare_dcase.py / train_audio_classifier.py와 일치)
ENVIRONMENT_LABELS = {
    0: "quiet",          # library / home — 조용·집중적합
    1: "social_indoor",  # cafe / office / grocery — 웅성·대화 실내
    2: "outdoor",        # city_center / residential / park — 도시형 야외
}
NUM_ENV_CLASSES = len(ENVIRONMENT_LABELS)
CLIP_SAMPLES = 16000 * 5  # 환경 분류는 학습과 동일하게 5초로 정규화

# ── AudioSet 527 클래스 인덱스 기반 사운드 그룹 (class_labels_indices.csv 순서) ──
# 학습 이벤트 소리(소음이 아니라 학습 행위의 일부 → 방해로 보지 않음)
STUDY_EVENT_IDX = {
    384,  # Typing
    385,  # Typewriter
    386,  # Computer keyboard
    387,  # Writing
    491,  # Clicking (마우스/볼펜 클릭)
    492,  # Clickety-clack
    360,  # Tap
    404,  # Mechanisms
    407,  # Tick
    408,  # Tick-tock
    474,  # Scratch (연필 필기음 근사)
    487,  # Rustle (종이/책장 넘김)
    479,  # Crumpling, crinkling (종이)
    381,  # Scissors
}
# 집중을 방해하는 소리: 의미 있는 말소리 + 돌발/경보음
DISRUPTIVE_IDX = {
    0, 1, 2, 3, 4, 5, 70,          # Speech / Male/Female/Child / Conversation / Narration / Hubbub(babble)
    8, 11, 13, 14,                 # Shout / Yell / Children shouting / Screaming
    75,                            # Bark
    308, 310, 318, 331,            # Vehicle horn / Car alarm / Air horn / Train horn
    323, 324, 325, 396, 397,       # Police/Ambulance/Fire siren / Siren / Civil defense siren
    388, 390, 391, 395, 399, 400,  # Alarm / Telephone bell / Ringtone / Alarm clock / Smoke/Fire alarm
    359,                           # Knock
    441, 443,                      # Glass / Shatter
}
SOUND_PROB_TH = 0.10   # 해당 사운드로 인정하는 최소 확률


class EnvironmentClassifier(nn.Module):
    """PANNs CNN10 백본 + 환경음 분류 헤드 (train_audio_classifier.py와 동일 구조)."""
    def __init__(self, panns_model, num_classes=NUM_ENV_CLASSES):
        super().__init__()
        self.panns = panns_model
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        out = self.panns(x)
        return self.classifier(out["embedding"])


class AudioClassifier:
    """
    두 갈래 오디오 분석:
      (1) 환경 분류(축 A): fine-tuned 3-class EnvironmentClassifier → quiet/social_indoor/outdoor
      (2) 실시간 방해 판정(축 B): AudioSet 527-class PANNs로 학습 이벤트음(키보드 등)과
          방해음(말소리/돌발음)을 식별. 학습 이벤트음은 dB가 높아도 방해로 보지 않는다.
    """
    def __init__(self, weights_path: str | None = None):
        from app.config import PANNS_WEIGHTS, PANNS_PRETRAINED_WEIGHTS

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        path = weights_path or str(PANNS_WEIGHTS)

        # (1) 환경 분류기 (전이학습 결과)
        try:
            self.model = self._load_env_model(path)
            self.use_panns = True
            logger.info("Loaded fine-tuned PANNs environment classifier from %s", path)
        except Exception as e:
            self.model = None
            self.use_panns = False
            logger.warning("환경음 모델 로드 실패(%s). dB 기반 fallback 사용.", e)

        # (2) AudioSet 527-class 모델 (이벤트/방해음 식별)
        self.audioset_labels = self._load_audioset_labels()
        try:
            self.event_model = self._load_audioset_model(str(PANNS_PRETRAINED_WEIGHTS))
            logger.info("Loaded PANNs AudioSet(527) model for event/disruptive detection")
        except Exception as e:
            self.event_model = None
            logger.warning("AudioSet(527) 모델 로드 실패(%s). 이벤트음 판정 비활성.", e)

    def _load_env_model(self, path: str):
        if not Path(path).exists():
            raise FileNotFoundError(path)
        from app.models.panns_model import Cnn10
        panns = Cnn10(sample_rate=16000, window_size=512, hop_size=160,
                      mel_bins=64, fmin=50, fmax=8000, classes_num=527)
        model = EnvironmentClassifier(panns, NUM_ENV_CLASSES)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state = checkpoint["model"] if "model" in checkpoint else checkpoint
        model.load_state_dict(state, strict=False)
        model.to(self.device); model.eval()
        return model

    def _load_audioset_model(self, path: str):
        if not Path(path).exists():
            raise FileNotFoundError(path)
        from app.models.panns_model import Cnn10
        model = Cnn10(sample_rate=16000, window_size=512, hop_size=160,
                      mel_bins=64, fmin=50, fmax=8000, classes_num=527)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state = checkpoint["model"] if "model" in checkpoint else checkpoint
        model.load_state_dict(state, strict=False)
        model.to(self.device); model.eval()
        return model

    def _load_audioset_labels(self) -> list[str]:
        p = Path(__file__).parent / "audioset_labels.txt"
        if p.exists():
            return p.read_text(encoding="utf-8").strip().split("\n")
        return []

    def analyze_from_base64(self, audio_b64: str, sample_rate: int = 16000) -> dict:
        audio_bytes = base64.b64decode(audio_b64)
        waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
        if sr != sample_rate:
            waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return self.analyze(waveform.squeeze().numpy(), sample_rate)

    def _fit_length(self, audio: np.ndarray) -> np.ndarray:
        n = len(audio)
        if n > CLIP_SAMPLES:
            start = (n - CLIP_SAMPLES) // 2
            return audio[start:start + CLIP_SAMPLES]
        if n < CLIP_SAMPLES:
            return np.pad(audio, (0, CLIP_SAMPLES - n))
        return audio

    def analyze(self, audio: np.ndarray, sample_rate: int = 16000) -> dict:
        noise_db = self._compute_db(audio)
        clip = self._fit_length(audio.astype(np.float32))
        tensor = torch.from_numpy(clip).float().unsqueeze(0).to(self.device)

        # (1) 환경 분류
        environment_class, env_confidence = "quiet", 0.0
        if self.use_panns and self.model is not None:
            with torch.no_grad():
                probs = torch.softmax(self.model(tensor), dim=1).squeeze().cpu().numpy()
            cls = int(np.argmax(probs))
            environment_class = ENVIRONMENT_LABELS.get(cls, "quiet")
            env_confidence = float(probs[cls])
        else:
            environment_class = "social_indoor" if noise_db > 60 else ("quiet" if noise_db < 35 else "social_indoor")

        # (2) AudioSet 이벤트/방해음 식별
        study_events, disruptive_sounds = [], []
        is_study_event = is_disruptive = False
        if self.event_model is not None and self.audioset_labels:
            with torch.no_grad():
                out = self.event_model(tensor)
                clip_probs = out["clipwise_output"].squeeze().cpu().numpy()
            study_p = max((float(clip_probs[i]) for i in STUDY_EVENT_IDX if i < len(clip_probs)), default=0.0)
            disr_p = max((float(clip_probs[i]) for i in DISRUPTIVE_IDX if i < len(clip_probs)), default=0.0)
            for i in STUDY_EVENT_IDX:
                if i < len(clip_probs) and clip_probs[i] >= SOUND_PROB_TH:
                    study_events.append(self.audioset_labels[i])
            for i in DISRUPTIVE_IDX:
                if i < len(clip_probs) and clip_probs[i] >= SOUND_PROB_TH:
                    disruptive_sounds.append(self.audioset_labels[i])
            is_study_event = study_p >= SOUND_PROB_TH
            # 방해 판정: 방해음이 명확하고(임계 이상) 학습 이벤트보다 우세할 때만
            is_disruptive = disr_p >= SOUND_PROB_TH and disr_p >= study_p

        return {
            "environment_class": environment_class,
            "environment_confidence": round(env_confidence, 4),
            "noise_db": round(noise_db, 2),
            "is_study_event": is_study_event,
            "is_disruptive": is_disruptive,
            "study_events": study_events[:5],
            "detected_sounds": disruptive_sounds[:5],
        }

    def _compute_db(self, audio: np.ndarray) -> float:
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-10:
            return 0.0
        db = 20 * np.log10(rms / 1e-5)
        # NumPy scalar types (for example np.float32) are not JSON serializable.
        # WebSocket responses must contain native Python numbers.
        return float(max(0.0, min(120.0, db)))
