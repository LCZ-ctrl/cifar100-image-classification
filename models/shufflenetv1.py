import torch
import torch.nn as nn


def channel_shuffle(x, groups):
    b, c, h, w = x.size()

    x = x.view(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    x = x.view(b, c, h, w)
    return x


class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups, stride):
        super().__init__()
        self.stride = stride
        self.groups = groups

        hidden_channels = out_channels // 4

        if self.stride == 2:
            out_channels -= in_channels

        # 1x1 GConv
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        # 3x3 DWConv
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=stride, padding=1, groups=hidden_channels,
                      bias=False),
            nn.BatchNorm2d(hidden_channels)
        )

        # 1x1 GConv
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        if self.stride == 2:
            self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = channel_shuffle(out, self.groups)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.stride == 2:
            residual = self.pool(residual)
            out = torch.cat([out, residual], dim=1)
        else:
            out = out + residual

        return torch.relu(out)


class ShuffleNetV1(nn.Module):
    def __init__(self, groups=3, num_classes=100, dropout=0.3):
        super().__init__()

        cfg = {
            1: [144, 288, 576],
            2: [200, 400, 800],
            3: [240, 480, 960],
            4: [272, 544, 1088],
            8: [384, 768, 1536]
        }
        out_channels = cfg[groups]

        # Input: [B, 3, 32, 32]
        # groups = 3
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)  # [B, 24, 32, 32]
        )

        self.stage2 = self._make_stage(24, out_channels[0], groups, 3)  # [B, 240, 16, 16]
        self.stage3 = self._make_stage(out_channels[0], out_channels[1], groups, 7)  # [B, 480, 8, 8]
        self.stage4 = self._make_stage(out_channels[1], out_channels[2], groups, 3)  # [B, 960, 4, 4]

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # [B, 960, 1, 1]
            nn.Flatten(),  # [B, 960]
            nn.Dropout(dropout),
            nn.Linear(out_channels[2], num_classes)  # [B, 100]
        )

    def _make_stage(self, in_channels, out_channels, groups, repeat):
        layers = [ShuffleUnit(in_channels, out_channels, groups, stride=2)]
        for _ in range(repeat):
            layers.append(ShuffleUnit(out_channels, out_channels, groups, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.classifier(x)
        return x
