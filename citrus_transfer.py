"""
citrus_transfer.py

Helpers for the transfer-learning experiments: building pretrained backbones
(ResNet50, EfficientNet-B0, DenseNet121) with a CitrusNet-matching classifier
head, and freezing/unfreezing utilities for two-phase fine-tuning.
"""

import torch.nn as nn
from torchvision.models import (
    resnet50, ResNet50_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    densenet121, DenseNet121_Weights,
)

# Pretrained backbones expect ImageNet normalization, not this project's
# dataset-specific stats (see notebook 13 for the reasoning).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_SUPPORTED_ARCHITECTURES = ("resnet50", "efficientnet_b0", "densenet121")


def _make_head(in_features: int, num_classes: int) -> nn.Sequential:
    """Same head design used by CitrusNet's classifier, for a fair comparison."""
    return nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes),
    )


def build_model(architecture: str, num_classes: int = 4) -> nn.Module:
    """Load a pretrained backbone and swap in the CitrusNet-style head."""
    if architecture == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = _make_head(model.fc.in_features, num_classes)
    elif architecture == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        # classifier is Sequential(Dropout, Linear) -- the Linear holds in_features
        in_features = model.classifier[-1].in_features
        model.classifier = _make_head(in_features, num_classes)
    elif architecture == "densenet121":
        model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        model.classifier = _make_head(model.classifier.in_features, num_classes)
    else:
        raise ValueError(
            f"Unsupported architecture {architecture!r}; expected one of "
            f"{_SUPPORTED_ARCHITECTURES}."
        )
    return model


def _get_head(model: nn.Module, architecture: str) -> nn.Module:
    if architecture == "resnet50":
        return model.fc
    elif architecture in ("efficientnet_b0", "densenet121"):
        return model.classifier
    else:
        raise ValueError(
            f"Unsupported architecture {architecture!r}; expected one of "
            f"{_SUPPORTED_ARCHITECTURES}."
        )


def freeze_backbone(model: nn.Module, architecture: str) -> None:
    """Freeze every parameter except the new classifier head."""
    head = _get_head(model, architecture)
    head_param_ids = {id(p) for p in head.parameters()}
    for p in model.parameters():
        p.requires_grad = id(p) in head_param_ids


def unfreeze_all(model: nn.Module) -> None:
    """Make every parameter trainable again, for the full fine-tune phase."""
    for p in model.parameters():
        p.requires_grad = True


def count_trainable_params(model: nn.Module):
    """Return (trainable_count, total_count), computed from requires_grad."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
