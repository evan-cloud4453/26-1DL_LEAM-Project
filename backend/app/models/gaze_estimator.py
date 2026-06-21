"""
L2CS-Net gaze estimation wrapper.
ResNet50 backbone with dual yaw/pitch classification heads (90 bins each).
"""
import base64
import io
import logging

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models.resnet as resnet_module
from PIL import Image

from app.config import L2CS_WEIGHTS

logger = logging.getLogger(__name__)


class L2CS(nn.Module):
    def __init__(self, block, layers, num_bins):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_yaw_gaze = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch_gaze = nn.Linear(512 * block.expansion, num_bins)
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc_yaw_gaze(x), self.fc_pitch_gaze(x)


NUM_BINS = 90

_transform = transforms.Compose([
    transforms.Resize(448),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class GazeEstimator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.softmax = nn.Softmax(dim=1)
        self.idx_tensor = torch.arange(NUM_BINS, dtype=torch.float32)
        self._load_model()

    def _load_model(self):
        weights_path = L2CS_WEIGHTS
        if not weights_path.exists():
            logger.warning("L2CS weights not found at %s — gaze estimation disabled", weights_path)
            return

        self.model = L2CS(resnet_module.Bottleneck, [3, 4, 6, 3], NUM_BINS)
        state_dict = torch.load(str(weights_path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.idx_tensor = self.idx_tensor.to(self.device)
        logger.info("L2CS-Net loaded on %s", self.device)

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        if self.model is None:
            return {"error": "model not loaded"}

        img_tensor = _transform(image).unsqueeze(0).to(self.device)
        yaw_logits, pitch_logits = self.model(img_tensor)

        yaw_probs = self.softmax(yaw_logits)
        pitch_probs = self.softmax(pitch_logits)

        # 90 bins * 4 degrees each = 360 range, offset by -180
        yaw_deg = float((torch.sum(yaw_probs * self.idx_tensor, dim=1) * 4 - 180).item())
        pitch_deg = float((torch.sum(pitch_probs * self.idx_tensor, dim=1) * 4 - 180).item())

        return {"yaw": yaw_deg, "pitch": pitch_deg}

    def predict_from_base64(self, b64: str) -> dict:
        try:
            from app.image_utils import enhance_lowlight
            img_bytes = base64.b64decode(b64)
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            # 저조도 환경 개선(CLAHE) — 어두운 얼굴에서도 시선 추정 안정화
            image = Image.fromarray(enhance_lowlight(np.array(image)))
            return self.predict(image)
        except Exception as e:
            logger.error("Gaze prediction failed: %s", e)
            return {"error": str(e)}
