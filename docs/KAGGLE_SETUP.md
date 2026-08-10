# Kaggle Training Setup — LP-IOANet

This guide covers training LP-IOANet on Kaggle using the **Mixed_Shadow_Dataset**.

## Dataset structure (on Kaggle)

```
Mixed_Shadow_Dataset_1024x768/        # high-res version (Stage 2)
├── train/
│   ├── input/     (shadow images, e.g. 00001.jpg)
│   ├── mask/      (shadow masks)
│   ├── target/    (shadow-free images)
│   └── train_metadata.csv
└── test/
    ├── input/
    ├── mask/
    ├── target/
    └── test_metadata.csv

Mixed_Shadow_Dataset_256x192/         # low-res version (Stage 1)
└── (same structure)
```

- **22,400** training samples, **640** test samples.
- Portrait **4:3** aspect ratio: low-res `256×192`, high-res `1024×768`.
- Source datasets: `FSDSRD`, `SD7K`, `SynDoc_Wild_3D`, `OSR`, `Jung`, `RDD`.

## How the CSV is used
The metadata CSV maps `New_Name` (e.g. `00001.jpg`) to the source dataset.
The dataset loader reads `New_Name` to find the matching `input/`, `mask/`,
and `target/` images.

## Two-stage training

### Stage 1 — Train IOANet (low-res core)
```bash
python src/train_stage1.py \
    --data_root Mixed_Shadow_Dataset_256x192 \
    --epochs 1000 \
    --batch_size 8 \
    --device cuda
```
- Trains `IOANet` at `256×192` with `L1*10 + LPIPS*5`.
- Saves `checkpoints/ioanet_final.pth`.

### Stage 2 — Train upsampler (high-res)
```bash
python src/train_stage2.py \
    --data_root Mixed_Shadow_Dataset_1024x768 \
    --ioanet_ckpt checkpoints/ioanet_final.pth \
    --epochs 200 \
    --batch_size 4 \
    --device cuda
```
- Freezes IOANet (`.eval()` + `requires_grad=False`).
- Trains the Laplacian pyramid upsampler with L1 loss.
- Saves `checkpoints/upsampler_final.pth`.

## Kaggle notebook setup
1. Add the dataset to your notebook.
2. Install dependencies:
   ```python
   !pip install lpips
   ```
3. Upload the `src/` folder (or copy the files into the notebook).
4. Run Stage 1, then Stage 2.

## Notes
- **Batch size:** high-res (1024×768) is memory-heavy; use `batch_size=4` or lower on Kaggle GPUs.
- **LPIPS** requires the `lpips` package (downloads VGG weights on first use).
- The `mask/` folder is available but the current training uses only `input`/`target` pairs (IOANet implicitly localizes shadows via IOA). Masks can be added later for explicit shadow supervision.
