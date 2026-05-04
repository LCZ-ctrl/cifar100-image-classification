import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, stride=1, bias=False)

        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        # Input: [B, C, H, W]
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)  # [B, 4 * growth_rate, H, W]

        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)  # [B, growth_rate, H, W]

        return x


class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate):
        super().__init__()
        self.layers = nn.ModuleList()
        curr_channels = in_channels

        for i in range(num_layers):
            self.layers.append(DenseLayer(curr_channels, growth_rate))
            curr_channels += growth_rate

        self.out_channels = curr_channels

    def forward(self, x):
        # Input: [B, C, H, W]
        features = [x]

        for layer in self.layers:
            new_feature = layer(x if len(features) == 1 else torch.cat(features, dim=1))
            features.append(new_feature)

        return torch.cat(features, dim=1)  # [B, C + num_layers * growth_rate, H, W]


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Input: [B, C, H, W]
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)  # [B, out_c, H, W]
        x = self.pool(x)  # [B, out_c, H/2, W/2]
        return x


class DenseNet121(nn.Module):
    def __init__(self, num_classes=100, growth_rate=32, dropout=0.3):
        super().__init__()

        # Input: [B, 3, 32, 32]
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),  # [B, 64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.block1 = DenseBlock(64, num_layers=6, growth_rate=growth_rate)  # [B, 256, 32, 32]
        self.trans1 = Transition(in_channels=64 + 6 * growth_rate, out_channels=128)  # [B, 128, 16, 16]

        self.block2 = DenseBlock(128, num_layers=12, growth_rate=growth_rate)  # [B, 512, 16, 16]
        self.trans2 = Transition(in_channels=128 + 12 * growth_rate, out_channels=256)  # [B, 256, 8, 8]

        self.block3 = DenseBlock(256, num_layers=24, growth_rate=growth_rate)  # [B, 1024, 8, 8]
        self.trans3 = Transition(in_channels=256 + 24 * growth_rate, out_channels=512)  # [B, 512, 4, 4]

        self.block4 = DenseBlock(512, num_layers=16, growth_rate=growth_rate)  # [B, 1024, 4, 4]

        self.classifier = nn.Sequential(
            nn.BatchNorm2d(512 + 16 * growth_rate),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # [B, 1024, 1, 1]
            nn.Flatten(),  # [B, 1024]
            nn.Dropout(dropout),
            nn.Linear(512 + 16 * growth_rate, num_classes)  # [B, 100]
        )

    def forward(self, x):
        x = self.stem(x)

        x = self.block1(x)
        x = self.trans1(x)

        x = self.block2(x)
        x = self.trans2(x)

        x = self.block3(x)
        x = self.trans3(x)

        x = self.block4(x)

        x = self.classifier(x)
        return x
