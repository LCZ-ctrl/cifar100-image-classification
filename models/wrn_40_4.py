import torch
import torch.nn as nn


class WideBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dropout):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                )
            )

    def forward(self, x):
        identity = x

        out = self.relu1(self.bn1(x))
        out = self.conv1(out)

        out = self.relu2(self.bn2(out))
        out = self.dropout(out)
        out = self.conv2(out)

        out += self.shortcut(identity)
        return out


class WideResNet_40_4(nn.Module):
    def __init__(self, depth=40, width_factor=4, num_classes=100, dropout=0.3):
        super().__init__()

        n = (depth - 4) // 6
        k = width_factor
        channels = [16, 16 * k, 32 * k, 64 * k]

        # Input: [B, 3, 32, 32]
        self.stem = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False)  # [B, 16, 32, 32]

        self.layer1 = self._make_layer(channels[0], channels[1], n, stride=1, dropout=dropout)  # [B, 64, 32, 32]
        self.layer2 = self._make_layer(channels[1], channels[2], n, stride=2, dropout=dropout)  # [B, 128, 16, 16]
        self.layer3 = self._make_layer(channels[2], channels[3], n, stride=2, dropout=dropout)  # [B, 256, 8, 8]

        self.bn = nn.BatchNorm2d(channels[3])
        self.relu = nn.ReLU(inplace=True)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # [B, 256, 1, 1]
            nn.Flatten(),  # [B, 256]
            nn.Linear(channels[3], num_classes)  # [B, 100]
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, dropout):
        layers = [WideBasicBlock(in_channels, out_channels, stride, dropout)]
        for _ in range(1, num_blocks):
            layers.append(WideBasicBlock(out_channels, out_channels, 1, dropout))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.bn(x))
        x = self.classifier(x)
        return x
