import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super().__init__()

        self.p1_1 = ConvBlock(in_channels, c1, kernel_size=1)

        self.p2_1 = ConvBlock(in_channels, c2[0], kernel_size=1)
        self.p2_2 = ConvBlock(c2[0], c2[1], kernel_size=3, padding=1)

        self.p3_1 = ConvBlock(in_channels, c3[0], kernel_size=1)
        self.p3_2 = ConvBlock(c3[0], c3[1], kernel_size=5, padding=2)

        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = ConvBlock(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))

        return torch.cat((p1, p2, p3, p4), dim=1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=100, dropout=0.3):
        super().__init__()

        # Input: [B, 3, 32, 32]
        self.b1 = ConvBlock(3, 64, kernel_size=3, padding=1)  # [B, 64, 32, 32]

        self.b2 = nn.Sequential(
            ConvBlock(64, 64, kernel_size=1),
            ConvBlock(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # [B, 192, 16, 16]
        )

        self.b3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),  # [B, 256, 16, 16]
            Inception(256, 128, (128, 192), (32, 96), 64),  # [B, 480, 16, 16]
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # [B, 480, 8, 8]
        )

        self.b4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),  # [B, 512, 8, 8]
            Inception(512, 160, (112, 224), (24, 64), 64),  # [B, 512, 8, 8]
            Inception(512, 128, (128, 256), (24, 64), 64),  # [B, 512, 8, 8]
            Inception(512, 112, (144, 288), (32, 64), 64),  # [B, 512, 8, 8]
            Inception(528, 256, (160, 320), (32, 128), 128),  # [B, 832, 8, 8]
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # [B, 832, 4, 4]
        )

        self.b5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),  # [B, 832, 4, 4]
            Inception(832, 384, (192, 384), (48, 128), 128),  # [B, 1024, 4, 4]
            nn.AdaptiveAvgPool2d((1, 1)),  # [B, 1024, 1, 1]
            nn.Flatten(),  # [B, 1024]
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes)  # [B, 100]
        )

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)

        return x
