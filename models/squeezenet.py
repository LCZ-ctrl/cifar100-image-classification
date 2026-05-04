import torch
import torch.nn as nn


class Fire(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super().__init__()

        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels, squeeze_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(squeeze_channels),
            nn.ReLU(inplace=True)
        )

        self.expand1x1 = nn.Sequential(
            nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(expand1x1_channels),
            nn.ReLU(inplace=True)
        )

        self.expand3x3 = nn.Sequential(
            nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(expand3x3_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.squeeze(x)
        x1 = self.expand1x1(x)
        x3 = self.expand3x3(x)
        return torch.cat([x1, x3], dim=1)


class SqueezeNet(nn.Module):
    def __init__(self, num_classes=100, dropout=0.3):
        super().__init__()

        # Input: [B, 3, 32, 32]
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),  # [B, 64, 32, 32]

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [B, 64, 16, 16]

            Fire(64, 16, 64, 64),  # [B, 128, 16, 16]
            Fire(128, 16, 64, 64),  # [B, 128, 16, 16]

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [B, 128, 8, 8]

            Fire(128, 32, 128, 128),  # [B, 256, 8, 8]
            Fire(256, 32, 128, 128),  # [B, 256, 8, 8]

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [B, 256, 4, 4]

            Fire(256, 48, 192, 192),  # [B, 384, 4, 4]
            Fire(384, 48, 192, 192),  # [B, 384, 4, 4]
            Fire(384, 64, 256, 256),  # [B, 512, 4, 4]
            Fire(512, 64, 256, 256)  # [B, 512, 4, 4]
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(512, num_classes, kernel_size=1),  # [B, 100, 4, 4]
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # [B, 100, 1, 1]
            nn.Flatten()  # [B, 100]
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
