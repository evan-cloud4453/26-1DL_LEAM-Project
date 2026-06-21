"""
PANNs CNN10 model architecture.
Reference: https://github.com/qiuqiangkong/panns_inference
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias") and layer.bias is not None:
        layer.bias.data.fill_(0.0)


def init_bn(bn):
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, x, pool_size=(2, 2), pool_type="avg"):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "avg":
            x = F.avg_pool2d(x, pool_size)
        elif pool_type == "max":
            x = F.max_pool2d(x, pool_size)
        return x


class Cnn10(nn.Module):
    def __init__(self, sample_rate=16000, window_size=512, hop_size=160,
                 mel_bins=64, fmin=50, fmax=8000, classes_num=527):
        super().__init__()

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=window_size,
            hop_length=hop_size,
            n_mels=mel_bins,
            f_min=fmin,
            f_max=fmax,
        )

        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, x):
        # x: (batch, samples)
        mel = self.mel_spec(x)  # (batch, mel_bins, time)
        mel = (mel + 1e-10).log()
        mel = mel.unsqueeze(1)  # (batch, 1, mel_bins, time)
        mel = mel.transpose(2, 3)  # (batch, 1, time, mel_bins)

        x = mel.transpose(1, 3)  # (batch, mel_bins, time, 1)
        x = self.bn0(x)
        x = x.transpose(1, 3)  # (batch, 1, time, mel_bins)

        x = self.conv_block1(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)

        x = torch.mean(x, dim=3)  # (batch, 512, time')
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(embedding))

        return {"clipwise_output": clipwise_output, "embedding": embedding}
