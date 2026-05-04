import torch
import torch.nn as nn


def _make_divisible(ch, divisor, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # make sure that round down does not go down by more than 10%
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        # Input: [B, in_channels, H, W]
        hidden_channels = int(round(in_channels * expand_ratio))
        self.use_res_connect = (stride == 1 and in_channels == out_channels)

        layers = []
        if expand_ratio != 1:
            # 1x1 pw: [B, hidden_channels, H, W]
            layers.append(ConvBlock(in_channels, hidden_channels, kernel_size=1))

        # 3x3 dw: [B, hidden_channels, H/stride, W/stride]
        layers.append(ConvBlock(hidden_channels, hidden_channels, kernel_size=3, stride=stride, groups=hidden_channels))

        # 1x1 pw-linear: [B, out_channels, H/stride, W/stride]
        layers.append(nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))

        layers.append(nn.BatchNorm2d(out_channels))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=100, alpha=1.0, round_nearest=8, dropout=0.3):
        super().__init__()
        in_channels = _make_divisible(32 * alpha, round_nearest)
        last_channels = _make_divisible(1280 * alpha, round_nearest)

        # Input: [B, 3, 32, 32]
        self.stem = ConvBlock(3, in_channels, kernel_size=3, stride=1)  # [B, 32, 32, 32]

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],  # [B, 16, 32, 32]
            [6, 24, 2, 1],  # [B, 24, 32, 32]
            [6, 32, 3, 2],  # [B, 32, 16, 16]
            [6, 64, 4, 2],  # [B, 64, 8, 8]
            [6, 96, 3, 1],  # [B, 96, 8, 8]
            [6, 160, 3, 2],  # [B, 160, 4, 4]
            [6, 320, 1, 1],  # [B, 320, 4, 4]
        ]

        features = []
        for t, c, n, s in inverted_residual_setting:
            out_channels = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(in_channels, out_channels, stride, expand_ratio=t))
                in_channels = out_channels

        # [B, 1280, 4, 4]
        features.append(ConvBlock(in_channels, last_channels, kernel_size=1))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # [B, 1280, 1, 1]
            nn.Flatten(),  # [B, 1280]
            nn.Dropout(dropout),
            nn.Linear(last_channels, num_classes)  # [B, 100]
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.classifier(x)
        return x
