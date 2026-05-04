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
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, act='RE'):
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

        if act == 'RE':
            self.act = nn.ReLU6(inplace=True)
        elif act == 'HS':
            self.act = nn.Hardswish(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        hidden_channels = _make_divisible(in_channels // reduction, 8)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, use_se, act):
        super().__init__()
        # Input: [B, in_channels, H, W]
        hidden_channels = int(round(in_channels * expand_ratio))
        self.use_res_connect = (stride == 1 and in_channels == out_channels)

        layers = []
        if expand_ratio != 1:
            # 1x1 pw: [B, hidden_channels, H, W]
            layers.append(ConvBlock(in_channels, hidden_channels, kernel_size=1, act=act))

        # dw: [B, hidden_channels, H/stride, W/stride]
        layers.append(
            ConvBlock(hidden_channels, hidden_channels, kernel_size=kernel_size, stride=stride, groups=hidden_channels,
                      act=act)
        )

        if use_se:
            layers.append(SqueezeExcitation(hidden_channels))

        # 1x1 pw-linear: [B, out_channels, H/stride, W/stride]
        layers.append(nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))

        layers.append(nn.BatchNorm2d(out_channels))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, num_classes=100, alpha=1.0, round_nearest=8, dropout=0.3):
        super().__init__()
        in_channels = _make_divisible(16 * alpha, round_nearest)
        last_channels = _make_divisible(960 * alpha, round_nearest)
        classifier_channels = _make_divisible(1280 * alpha, round_nearest)

        # Input: [B, 3, 32, 32]
        self.stem = ConvBlock(3, in_channels, kernel_size=3, stride=1, act='HS')  # [B, 16, 32, 32]

        # k, t, c, SE, NL, s
        inverted_residual_setting = [
            [3, 1, 16, False, 'RE', 1],  # stage 1: [B, 16, 32, 32]

            [3, 4, 24, False, 'RE', 1],  # stage 2: [B, 24, 32, 32]
            [3, 3, 24, False, 'RE', 1],

            [5, 3, 40, True, 'RE', 2],  # stage 3: [B, 40, 16, 16]
            [5, 3, 40, True, 'RE', 1],
            [5, 3, 40, True, 'RE', 1],

            [3, 6, 80, False, 'HS', 2],  # stage 4: [B, 80, 8, 8]
            [3, 2.5, 80, False, 'HS', 1],
            [3, 2.3, 80, False, 'HS', 1],
            [3, 2.3, 80, False, 'HS', 1],

            [3, 6, 112, True, 'HS', 1],  # stage 5: [B, 112, 8, 8]
            [3, 6, 112, True, 'HS', 1],

            [5, 6, 160, True, 'HS', 2],  # stage 6: [B, 160, 4, 4]
            [5, 6, 160, True, 'HS', 1],
            [5, 6, 160, True, 'HS', 1],
        ]

        features = []
        for k, t, c, se, nl, s in inverted_residual_setting:
            out_channels = _make_divisible(c * alpha, round_nearest)
            features.append(
                InvertedResidual(in_channels, out_channels, kernel_size=k, stride=s, expand_ratio=t, use_se=se, act=nl)
            )
            in_channels = out_channels

        # [B, 960, 4, 4]
        features.append(ConvBlock(in_channels, last_channels, kernel_size=1, act='HS'))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # [B, 960, 1, 1]
            nn.Flatten(),  # [B, 960]
            nn.Linear(last_channels, classifier_channels),  # [B, 1280]
            nn.Hardswish(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_channels, num_classes)  # [B, 100]
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.classifier(x)
        return x
