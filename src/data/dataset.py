"""Dataset loader for the Mixed_Shadow_Dataset.

Structure:
    <root>/
    ├── train/
    │   ├── input/    (shadow images)
    │   ├── mask/     (shadow masks)
    │   └── target/   (shadow-free images)
    └── test/
        ├── input/
        ├── mask/
        └── target/

Image names are loaded from the input/ folder. If a metadata CSV is provided,
it is used to filter/order the names (New_Name column); otherwise all files in
input/ are used.
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
        root: path to the dataset folder (e.g. master_mix_192x256).
        split: 'train' or 'test'.
        csv_path: optional path to a metadata CSV (New_Name column).
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

        # Load image names: from CSV if provided, else list input/ folder.
        if csv_path is not None and os.path.exists(csv_path):
            self.names = self._load_names_from_csv(csv_path)
        else:
            self.names = self._load_names_from_dir(self.input_dir)

        self.to_tensor = ToTensor()
        self.resize = Resize(size) if size is not None else None
        self.flip = RandomHorizontalFlip(0.5)

    def _load_names_from_csv(self, csv_path):
        names = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                names.append(row["New_Name"])
        return names

    def _load_names_from_dir(self, input_dir):
        """List all image files in the input/ folder (no CSV needed)."""
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        names = sorted(
            f for f in os.listdir(input_dir)
            if f.lower().endswith(exts)
        )
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
