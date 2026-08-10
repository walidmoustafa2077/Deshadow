"""Debug visualization for LP-IOANet training.

Saves side-by-side sample images (Input / Target / Output) during training
so you can visually monitor progress, matching the reference DocShadow-Lite
train.py debug feature.
"""
import os
import numpy as np
import torch
import cv2


def _to_numpy(tensor, batch_idx=0):
    """Convert a (B, C, H, W) tensor in [0,1] to an HxWx3 uint8 numpy array."""
    arr = tensor[batch_idx].detach().cpu().permute(1, 2, 0).numpy()
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def _add_label(img, text):
    """Add a text label above an image."""
    h, w = img.shape[:2]
    label_height = 30
    labeled = np.zeros((h + label_height, w, 3), dtype=np.uint8)
    labeled[label_height:, :, :] = img
    cv2.putText(labeled, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)
    return labeled


def save_sample(input_img, target_img, output, save_dir, epoch, sample_num=1,
                extra_images=None, extra_labels=None):
    """Save a side-by-side debug sample.

    Args:
        input_img, target_img, output: (B, C, H, W) tensors in [0,1].
        save_dir: directory to save the PNG.
        epoch: current epoch (for filename).
        sample_num: sample index within the batch.
        extra_images: optional list of (B, C, H, W) tensors to include.
        extra_labels: optional list of label strings for extra_images.
    Returns:
        Path to the saved file, or None if save_dir is None.
    """
    if save_dir is None:
        return None
    os.makedirs(save_dir, exist_ok=True)

    images = [_add_label(_to_numpy(input_img, sample_num), "Input (Shadow)")]
    if extra_images:
        for img, label in zip(extra_images, extra_labels or []):
            images.append(_add_label(_to_numpy(img, sample_num), label))
    images.append(_add_label(_to_numpy(output, sample_num), "Output (Predicted)"))
    images.append(_add_label(_to_numpy(target_img, sample_num), "Target (Clean)"))

    combined = np.concatenate(images, axis=1)
    save_path = os.path.join(save_dir, f"epoch_{epoch:04d}_sample{sample_num}.png")
    cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    return save_path
