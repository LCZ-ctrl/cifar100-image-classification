import torch
import torch.nn as nn


def channel_shuffle(x, groups):
    b, c, h, w = x.size()

    x = x.view(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    x = x.view(b, c, h, w)
    return x


class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.stride = stride
        hidden_channels = out_channels // 2

        if self.stride == 2:
            # 3x3 DWConv -> 1x1 Conv
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels,
                          bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential()

        in_channels = in_channels if stride == 2 else hidden_channels

        self.branch2 = nn.Sequential(
            # 1x1 Conv
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            # 3x3 DWConv
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=stride, padding=1, groups=hidden_channels,
                      bias=False),
            nn.BatchNorm2d(hidden_channels),
            # 1x1 Conv
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if self.stride == 1:
            # channel split
            x1, x2 = torch.chunk(x, 2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        return channel_shuffle(out, 2)


class ShuffleNetV2(nn.Module):
    def __init__(self, ratio=1.0, num_classes=100, dropout=0.3):
        super().__init__()

        cfg = {
            0.5: [48, 96, 192, 1024],
            1.0: [116, 232, 464, 1024],
            1.5: [176, 352, 704, 1024],
            2.0: [244, 488, 976, 2048]
        }
        out_channels = cfg[ratio]

        # Input: [B, 3, 32, 32]
        # ratio = 1.0
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)  # [B, 24, 32, 32]
        )

        self.stage2 = self._make_stage(24, out_channels[0], 3)  # [B, 116, 16, 16]
        self.stage3 = self._make_stage(out_channels[0], out_channels[1], 7)  # [B, 232, 8, 8]
        self.stage4 = self._make_stage(out_channels[1], out_channels[2], 3)  # [B, 464, 4, 4]

        self.conv5 = nn.Sequential(
            nn.Conv2d(out_channels[2], out_channels[3], kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels[3]),
            nn.ReLU(inplace=True)  # [B, 1024, 4, 4]
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # [B, 1024, 1, 1]
            nn.Flatten(),  # [B, 1024]
            nn.Dropout(dropout),
            nn.Linear(out_channels[3], num_classes)  # [B, 100]
        )

    def _make_stage(self, in_channels, out_channels, repeat):
        layers = [ShuffleUnit(in_channels, out_channels, stride=2)]
        for _ in range(repeat):
            layers.append(ShuffleUnit(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.conv5(x)
        x = self.classifier(x)
        return x
