"""Dataset loader for the Mixed_Shadow_Dataset.

Structure:
    Mixed_Shadow_Dataset/
    ├── train/
    │   ├── input/    (shadow images, named by New_Name e.g. 00001.jpg)
    │   ├── mask/     (shadow masks)
    │   ├── target/   (shadow-free images)
    │   └── train_metadata.csv
    └── test/
        ├── input/
        ├── mask/
        ├── target/
        └── test_metadata.csv

The CSV maps New_Name -> (Source_Dataset, Original_Name, Original_Index).
"""
import os
import csv
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import ToTensor, Resize, RandomHorizontalFlip


class ShadowRemovalDataset(Dataset):
    """Loads (input, target) shadow/shadow-free pairs, optionally with mask.

    Args:
        root: path to the dataset folder (e.g. Mixed_Shadow_Dataset_1024x768).
        split: 'train' or 'test'.
        csv_path: path to the metadata CSV.
        size: (H, W) to resize images to. None = keep original.
        use_mask: whether to also load the shadow mask.
        augment: whether to apply random horizontal flip (train only).
    """

    def __init__(self, root, split="train", csv_path=None, size=None,
                 use_mask=False, augment=False):
        self.root = root
        self.split = split
        self.size = size
        self.use_mask = use_mask
        self.augment = augment

        split_dir = os.path.join(root, split)
        self.input_dir = os.path.join(split_dir, "input")
        self.target_dir = os.path.join(split_dir, "target")
        self.mask_dir = os.path.join(split_dir, "mask")

        # Load image names from CSV (New_Name column).
        if csv_path is None:
            csv_path = os.path.join(root, f"{split}_metadata.csv")
        self.names = self._load_names(csv_path)

        self.to_tensor = ToTensor()
        self.resize = Resize(size) if size is not None else None
        self.flip = RandomHorizontalFlip(0.5)

    def _load_names(self, csv_path):
        names = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                names.append(row["New_Name"])
        return names

    def __len__(self):
        return len(self.names)

    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        return np.array(img)

    def __getitem__(self, idx):
        name = self.names[idx]
        input_img = self._load_image(os.path.join(self.input_dir, name))
        target_img = self._load_image(os.path.join(self.target_dir, name))

        input_t = self.to_tensor(input_img)
        target_t = self.to_tensor(target_img)

        if self.resize is not None:
            input_t = self.resize(input_t)
            target_t = self.resize(target_t)

        if self.augment:
            input_t, target_t = self.flip(input_t, target_t)

        if self.use_mask:
            mask_img = self._load_image(os.path.join(self.mask_dir, name))
            mask_t = self.to_tensor(mask_img)
            if self.resize is not None:
                mask_t = self.resize(mask_t)
            if self.augment:
                input_t, target_t, mask_t = self.flip(input_t, target_t, mask_t)
            return input_t, target_t, mask_t

        return input_t, target_t
