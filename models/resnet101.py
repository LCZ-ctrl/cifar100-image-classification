import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x += self.shortcut(identity)
        x = self.relu3(x)

        return x


class ResNet101(nn.Module):
    def __init__(self, num_classes=100, dropout=0.3):
        super().__init__()

        # Input: [B, 3, 32, 32]
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)  # [B, 64, 32, 32]
        )

        self.layer1 = self._make_layer(64, 64, num_blocks=3, stride=1)  # [B, 256, 32, 32]
        self.layer2 = self._make_layer(256, 128, num_blocks=4, stride=2)  # [B, 512, 16, 16]
        self.layer3 = self._make_layer(512, 256, num_blocks=23, stride=2)  # [B, 1024, 8, 8]
        self.layer4 = self._make_layer(1024, 512, num_blocks=3, stride=2)  # [B, 2048, 4, 4]

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # [B, 2048, 1, 1]
            nn.Flatten(),  # [B, 2048]
            nn.Dropout(dropout),
            nn.Linear(2048, num_classes)  # [B, 100]
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [Bottleneck(in_channels, out_channels, stride)]
        new_in_channels = out_channels * Bottleneck.expansion
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(new_in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.classifier(x)

        return x
