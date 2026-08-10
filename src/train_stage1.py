"""Stage 1 training: train IOANet (low-res shadow removal core).

From the paper: train IOANet at low resolution (256x192) for 1000 epochs
using Adam with L1*10 + LPIPS*5 loss.

Features (matching reference DocShadow-Lite train.py):
  - tqdm progress bar with live MAE/PSNR/SSIM
  - Debug sample images saved during validation
  - TensorBoard logging
  - Early stopping + best-model checkpointing
  - Resume / fine-tune from checkpoint
  - CosineAnnealingWarmRestarts LR scheduler

Usage (Kaggle):
    python train_stage1.py --data_root Mixed_Shadow_Dataset_256x192 \
        --epochs 1000 --batch_size 32 --debug
"""
import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict

# Ensure the project root is on sys.path so `from src.*` imports work
# when running as `python src/train_stage1.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import LOW_RES, STAGE1_LR, L1_WEIGHT, LPIPS_WEIGHT
from src.data.dataset import ShadowRemovalDataset
from src.losses.losses import CombinedLoss
from src.models.ioanet import IOANet
from src.utils.metrics import MetricsCalculator
from src.utils.visualization import save_sample


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="Mixed_Shadow_Dataset_256x192")
    p.add_argument("--csv", default=None,
                   help="Optional metadata CSV (New_Name column). If omitted, "
                        "image names are listed from the input/ folder.")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=STAGE1_LR)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument("--log_dir", default="logs")
    p.add_argument("--sample_dir", default="samples")
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--debug", action="store_true", default=True,
                   help="Save debug sample images")
    p.add_argument("--no-debug", action="store_false", dest="debug")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume training from checkpoint")
    p.add_argument("--finetune", action="store_true", default=False,
                   help="Fine-tune from checkpoint (weights only, reset optimizer)")
    p.add_argument("--val_interval", type=int, default=5,
                   help="Validate every N epochs")
    p.add_argument("--ckpt_interval", type=int, default=50,
                   help="Save periodic checkpoint every N epochs")
    p.add_argument("--patience", type=int, default=50,
                   help="Early stopping patience")
    return p.parse_args()


def main():
    args = parse_args()

    # --- Directories ---
    checkpoint_dir = Path(args.save_dir) / "stage1"
    log_dir = Path(args.log_dir) / "stage1"
    sample_dir = Path(args.sample_dir) / "stage1"
    debug_dir = sample_dir / "debug"
    for d in [checkpoint_dir, log_dir, sample_dir, debug_dir]:
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    writer = SummaryWriter(str(log_dir))

    # --- Header ---
    print("\n" + "=" * 100)
    print(f"{'IOANet TRAINING - STAGE 1 (256x192)':^100}")
    print("=" * 100)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[OK] GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("[!] Running on CPU (training will be slow)")
    print(f"[OK] Device: {device}")
    print(f"[OK] Debug: {args.debug}")
    if args.resume:
        mode = "FINE-TUNE" if args.finetune else "RESUME"
        print(f"[OK] Checkpoint: {args.resume} ({mode} mode)")
    print("=" * 100)

    # --- Dataset ---
    # If a CSV is provided, derive the test CSV from it; else use folder listing.
    val_csv = args.csv.replace("train", "test") if args.csv else None
    train_ds = ShadowRemovalDataset(
        root=args.data_root, split="train", csv_path=args.csv,
        size=LOW_RES, use_mask=True, augment=True,
    )
    val_ds = ShadowRemovalDataset(
        root=args.data_root, split="test", csv_path=val_csv,
        size=LOW_RES, use_mask=True, augment=False,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    print(f"[stage1] train={len(train_ds)} val={len(val_ds)} low-res {LOW_RES}")

    # --- Model ---
    model = IOANet(pretrained=args.pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=150, T_mult=1, eta_min=1e-6
    )
    criterion = CombinedLoss(w_l1=L1_WEIGHT, w_lpips=LPIPS_WEIGHT, device=device)

    # --- Resume / finetune ---
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        print(f"[OK] Loaded weights from {args.resume}")
        if not args.finetune and isinstance(ckpt, dict):
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt.get("epoch", 0)
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            patience_counter = ckpt.get("patience_counter", 0)
            print(f"[OK] Resumed: epoch={start_epoch}, best={best_val_loss:.4f}")

    # --- Metrics guide ---
    print("\n" + "-" * 100)
    print("METRICS GUIDE")
    print("-" * 100)
    print("  MAE  (target < 0.02): Mean absolute error")
    print("  PSNR (target > 28 dB): Peak signal-to-noise ratio")
    print("  SSIM (target > 0.95):  Structural similarity")
    print("  [+] = Validation improved > model saved")
    print("  [-] = No improvement > patience counter +1")
    print("-" * 100 + "\n")

    # --- Training loop ---
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_losses = defaultdict(float)
        train_metrics = defaultdict(float)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}", leave=True,
                    dynamic_ncols=True,
                    bar_format='{l_bar}{bar}| [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
        for batch_idx, (input_img, target_img, mask) in enumerate(pbar):
            input_img = input_img.to(device)
            target_img = target_img.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            pred = model(input_img)
            loss, loss_components = criterion(pred, target_img)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses["total"] += loss.item()
            train_losses["l1"] += loss_components["l1"]
            train_losses["lpips"] += loss_components["lpips"]

            with torch.no_grad():
                metrics = MetricsCalculator.compute_all(pred, target_img, mask)
                for k, v in metrics.items():
                    if k != "regions":
                        train_metrics[k] += v

            pbar.set_postfix({
                "Loss": f"{loss.item():.3f}",
                "MAE": f"{metrics['mae']:.4f}",
                "PSNR": f"{metrics['psnr']:.1f}",
                "SSIM": f"{metrics['ssim']:.3f}",
            })

        num_batches = max(len(train_loader), 1)
        for k in train_losses:
            train_losses[k] /= num_batches
        for k in train_metrics:
            train_metrics[k] /= num_batches

        for k, v in train_losses.items():
            writer.add_scalar(f"train/loss_{k}", v, epoch)
        for k, v in train_metrics.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        print(f"[Train] Epoch {epoch+1:3d} | Loss: {train_losses['total']:.4f} "
              f"(l1={train_losses['l1']:.4f}, lpips={train_losses['lpips']:.4f}) | "
              f"MAE: {train_metrics['mae']:.4f} | PSNR: {train_metrics['psnr']:.2f} dB | "
              f"SSIM: {train_metrics['ssim']:.3f}")

        # --- Validation ---
        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            val_losses = defaultdict(float)
            val_metrics = defaultdict(float)
            val_samples = []

            with torch.no_grad():
                for batch_idx, (input_img, target_img, mask) in enumerate(val_loader):
                    input_img = input_img.to(device)
                    target_img = target_img.to(device)
                    mask = mask.to(device)

                    pred = model(input_img)
                    loss, _ = criterion(pred, target_img)
                    val_losses["total"] += loss.item()

                    metrics = MetricsCalculator.compute_all(pred, target_img, mask)
                    for k, v in metrics.items():
                        if k != "regions":
                            val_metrics[k] += v

                    if batch_idx < 8:
                        val_samples.append((input_img, target_img, pred))

            num_val = max(len(val_loader), 1)
            for k in val_losses:
                val_losses[k] /= num_val
            for k in val_metrics:
                val_metrics[k] /= num_val

            for k, v in val_losses.items():
                writer.add_scalar(f"val/loss_{k}", v, epoch)
            for k, v in val_metrics.items():
                writer.add_scalar(f"val/{k}", v, epoch)

            val_loss = val_losses["total"]
            improved = val_loss < best_val_loss
            status = "[+]" if improved else "[-]"
            print(f"{status} Epoch {epoch+1:3d} | Loss: {val_loss:.4f} | "
                  f"MAE: {val_metrics['mae']:.4f} | PSNR: {val_metrics['psnr']:.2f} dB | "
                  f"SSIM: {val_metrics['ssim']:.3f}")

            # Save debug samples
            if args.debug and val_samples:
                for i, (inp, tgt, out) in enumerate(val_samples[:8]):
                    path = save_sample(inp, tgt, out, debug_dir, epoch + 1, sample_num=i + 1)
                    if path:
                        print(f"    ► Sample {i+1} saved: {os.path.basename(path)}")

            # Best model + early stopping
            if improved:
                best_val_loss = val_loss
                patience_counter = 0
                save_path = checkpoint_dir / "best_model.pth"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "patience_counter": patience_counter,
                }, save_path)
                print(f"    ► Best model saved: {save_path.name}")
            else:
                patience_counter += 1
                print(f"    (No improvement - patience: {patience_counter}/{args.patience})")
                if patience_counter >= args.patience:
                    print(f"\n[!] Early stopping triggered at epoch {epoch + 1}")
                    break

        # Periodic checkpoint
        if (epoch + 1) % args.ckpt_interval == 0:
            ckpt_path = checkpoint_dir / f"checkpoint_epoch{epoch+1}.pth"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch + 1,
                "val_loss": best_val_loss,
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
            }, ckpt_path)

        scheduler.step()

    # --- Final save ---
    torch.save(model.state_dict(), checkpoint_dir / "ioanet_final.pth")
    print("\n" + "=" * 100)
    print(f"[OK] Training Complete!")
    print(f"[OK] Best Model: {checkpoint_dir / 'best_model.pth'}")
    print(f"[OK] Best Val Loss: {best_val_loss:.4f}")
    print(f"[OK] TensorBoard Logs: {log_dir}")
    print("=" * 100 + "\n")
    writer.close()


if __name__ == "__main__":
    main()
