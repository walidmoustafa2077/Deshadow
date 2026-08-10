# Kaggle Training Setup — LP-IOANet

This guide covers training LP-IOANet on Kaggle using the **Mixed_Shadow_Dataset**.

## Dataset structure (on Kaggle)

Two separate Kaggle datasets:

```
master_mix_192x256/          # low-res version (Stage 1)
├── train/
│   ├── input/     (shadow images)
│   ├── mask/      (shadow masks)
│   └── target/    (shadow-free images)
└── test/
    ├── input/
    ├── mask/
    └── target/

master_mix_768x1024/         # high-res version (Stage 2)
└── (same structure)
```

- Portrait **4:3** aspect ratio: low-res `256x192`, high-res `1024x768`.
- Source datasets: `FSDSRD`, `SD7K`, `SynDoc_Wild_3D`, `OSR`, `Jung`, `RDD`.

## How the CSV is used
The metadata CSV maps `New_Name` (e.g. `00001.jpg`) to the source dataset.
The dataset loader reads `New_Name` to find the matching `input/`, `mask/`,
and `target/` images.

## Two-stage training

### Stage 1 — Train IOANet (low-res core)
```bash
python src/train_stage1.py \
    --data_root /kaggle/input/master_mix_192x256 \
    --epochs 250 \
    --batch_size 32 \
    --debug
```
- Trains `IOANet` at `256x192` with `L1*10 + LPIPS*5`.
- Saves `checkpoints/stage1/best_model.pth`.

### Stage 2 — Train upsampler (high-res)
```bash
python src/train_stage2.py \
    --data_root /kaggle/input/master_mix_768x1024 \
    --ioanet_ckpt checkpoints/stage1/best_model.pth \
    --epochs 200 \
    --batch_size 4 \
    --debug
```
- Freezes IOANet (`.eval()` + `requires_grad=False`).
- Trains the Laplacian pyramid upsampler with L1 loss.
- Saves `checkpoints/stage2/best_model.pth`.

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
