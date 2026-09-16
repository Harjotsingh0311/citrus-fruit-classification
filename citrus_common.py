"""
Shared transforms and dataset class for the citrus leaf project.
Used by the augmentation notebook and later by the training notebook.
"""

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def load_normalization_stats(path="normalization_stats.json"):
    """Read the per-channel mean/std saved during preprocessing.

    Returns (mean, std), each a tuple of 3 floats in RGB order.
    """
    with open(path, "r") as f:
        stats = json.load(f)
    mean = tuple(stats["mean"])
    std = tuple(stats["std"])
    return mean, std


class AddGaussianNoise:
    """Adds Gaussian noise to a tensor image, then clamps back to [0, 1].

    Meant to go after ToTensor() and before Normalize() in the pipeline,
    since it assumes pixel values are still in [0, 1].
    """

    def __init__(self, std_dev=0.02):
        self.std_dev = std_dev

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std_dev
        return torch.clamp(tensor + noise, 0.0, 1.0)

    def __repr__(self):
        return f"{self.__class__.__name__}(std_dev={self.std_dev})"


def get_transforms(augment: bool, mean, std, size: int = 224):
    """Build the transform pipeline for training or for val/test.

    size is the target resolution and is exposed as a parameter so the
    same function works for both the 224 and 128 processed variants.
    """
    if augment:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            AddGaussianNoise(std_dev=0.02),
            transforms.Normalize(mean, std),
        ])
    else:
        # no random ops here - just make sure size matches and normalize
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


class CitrusLeafDataset(Dataset):
    """Loads images from root_dir/split/<class_label>/*.jpg.

    class_to_idx is built from the sorted class folder names, so it
    comes out the same regardless of which split is loaded first -
    train/val/test are scanned independently but must agree.
    """

    def __init__(self, root_dir, split, transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        split_dir = self.root_dir / split
        class_names = sorted(d.name for d in split_dir.iterdir() if d.is_dir())
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        self.samples = []
        for class_name in class_names:
            label = self.class_to_idx[class_name]
            for img_path in sorted((split_dir / class_name).glob("*.jpg")):
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label
