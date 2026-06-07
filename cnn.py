# This file defines custom CNN (Convolutional Neural Network).

import torch
import torch.nn as nn


def make_conv_block(in_channels, out_channels, kernel_size=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.stage1 = nn.Sequential(
            make_conv_block(in_channels=3,  out_channels=32),
            make_conv_block(in_channels=32, out_channels=32),
        )
        self.stage2 = nn.Sequential(
            make_conv_block(in_channels=32, out_channels=64),
            make_conv_block(in_channels=64, out_channels=64),
        )
        self.stage3 = nn.Sequential(
            make_conv_block(in_channels=64,  out_channels=128),
            make_conv_block(in_channels=128, out_channels=128),
        )
        self.stage4 = nn.Sequential(
            make_conv_block(in_channels=128, out_channels=256),
            make_conv_block(in_channels=256, out_channels=256),
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        features = {}

        out = self.stage1(x)
        features["stage1"] = out
        out = self.pool(out)

        out = self.stage2(out)
        features["stage2"] = out
        out = self.pool(out)

        out = self.stage3(out)
        features["stage3"] = out
        out = self.pool(out)

        out = self.stage4(out)
        features["stage4"] = out

        return features


class Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = FeatureExtractor()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.head = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        deep = features["stage4"]

        pooled = self.pool(deep)
        flat   = pooled.view(pooled.size(0), -1)

        logits = self.head(flat)
        return logits