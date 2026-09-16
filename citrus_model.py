"""
CitrusNet architecture.

Everything is trained from scratch - no pretrained weights, no backbones,
no transfer learning.
"""

import torch
import torch.nn as nn


class CitrusNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        # Block 1 - plain conv stack
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Block 2 - depthwise separable conv
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),  # depthwise
            nn.Conv2d(32, 64, kernel_size=1),  # pointwise
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Block 3 - dilated conv, padding=2 keeps spatial size unchanged
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Block 4 - no pooling, feeds straight into GAP
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


def init_weights(model):
    """
    Conv2d: He/Kaiming normal (relu nonlinearity), bias 0.
    BatchNorm2d: weight 1, bias 0.
    Linear(256, 128): He/Kaiming normal (followed by ReLU).
    Linear(128, num_classes): Xavier/Glorot uniform (feeds straight into the loss).
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.zeros_(m.bias)

    # classifier[0] = Linear(256, 128), followed by ReLU
    # classifier[-1] = Linear(128, num_classes), feeds into CrossEntropyLoss
    nn.init.kaiming_normal_(model.classifier[0].weight, nonlinearity="relu")
    nn.init.xavier_uniform_(model.classifier[-1].weight)


class CitrusNetArchAblation(nn.Module):
    """
    E2 architecture ablation: block2 toggles depthwise-separable vs a standard
    conv, block3 toggles dilated vs a standard conv. Everything else matches
    CitrusNet, including the classifier attribute name/structure, so the
    existing init_weights() works on this unmodified.
    """

    def __init__(self, num_classes=4, use_depthwise_separable=True, use_dilated=True):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        if use_depthwise_separable:
            block2_conv = [
                nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
                nn.Conv2d(32, 64, kernel_size=1),
            ]
        else:
            block2_conv = [nn.Conv2d(32, 64, kernel_size=3, padding=1)]

        self.block2 = nn.Sequential(
            *block2_conv,
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        if use_dilated:
            block3_conv = nn.Conv2d(64, 128, kernel_size=3, padding=2, dilation=2)
        else:
            block3_conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.block3 = nn.Sequential(
            block3_conv,
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class CitrusNetRegNormAblation(nn.Module):
    """
    E3 regularization/normalization ablation: normalization_type toggles the
    norm layer used after every conv between batchnorm, layernorm (via
    GroupNorm(1, C), the standard stand-in for LayerNorm on NCHW conv feature
    maps, since nn.LayerNorm doesn't operate naturally on that layout), and
    none (nn.Identity() - no normalization at all). dropout_p controls the
    dropout rate in the classifier head. block2 stays depthwise-separable and
    block3 stays dilated in every config here - this ablation only touches
    regularization/normalization, not architecture. Attribute names match
    CitrusNet, so the existing init_weights() works on this unmodified.
    """

    def __init__(self, num_classes=4, dropout_p=0.5, normalization_type="batchnorm"):
        super().__init__()

        def make_norm(channels):
            if normalization_type == "batchnorm":
                return nn.BatchNorm2d(channels)
            elif normalization_type == "layernorm":
                return nn.GroupNorm(1, channels)
            elif normalization_type == "none":
                return nn.Identity()
            else:
                raise ValueError(f"unknown normalization_type: {normalization_type}")

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            make_norm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            make_norm(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),  # depthwise
            nn.Conv2d(32, 64, kernel_size=1),  # pointwise
            make_norm(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=2, dilation=2),
            make_norm(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            make_norm(256),
            nn.ReLU(inplace=True),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x