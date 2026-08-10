"""Stage 2 training: train the Laplacian pyramid upsampler.

From the paper: freeze IOANet and train the upsampler using L1 loss on
high-resolution data (1024x768) for 200 epochs.

Important: the frozen IOANet core must be in .eval() mode so BatchNorm
running statistics do not update during Stage 2.

Features (matching reference DocShadow-Lite train.py):
  - tqdm progress bar with live MAE/PSNR/SSIM
  - Debug sample images saved during validation
  - TensorBoard logging
  - Early stopping + best-model checkpointing
  - Resume from checkpoint
  - CosineAnnealingLR scheduler

Usage (Kaggle):
    python train_stage2.py --data_root Mixed_Shadow_Dataset_1024x768 \
        --ioanet_ckpt checkpoints/stage1/best_model.pth --epochs 200 --debug
"""
import argparse
import os
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import LOW_RES, HIGH_RES, STAGE2_LR
from src.data.dataset import ShadowRemovalDataset
from src.losses.losses import L1Loss
from src.models.ioanet import IOANet
from src.models.upsampler import LaplacianPyramidUpsampler
from src.utils.metrics import MetricsCalculator
from src.utils.visualization import save_sample


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="Mixed_Shadow_Dataset_1024x768")
    p.add_argument("--csv", default="train_metadata.csv")
    p.add_argument("--ioanet_ckpt", required=True,
                   help="path to Stage 1 IOANet checkpoint")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=STAGE2_LR)
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
    p.add_argument("--val_interval", type=int, default=5,
                   help="Validate every N epochs")
    p.add_argument("--ckpt_interval", type=int, default=50,
                   help="Save periodic checkpoint every N epochs")
    p.add_argument("--patience", type=int, default=30,
                   help="Early stopping patience")
    return p.parse_args()


def main():
    args = parse_args()

    # --- Directories ---
    checkpoint_dir = Path(args.save_dir) / "stage2"
    log_dir = Path(args.log_dir) / "stage2"
    sample_dir = Path(args.sample_dir) / "stage2"
    debug_dir = sample_dir / "debug"
    for d in [checkpoint_dir, log_dir, sample_dir, debug_dir]:
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    writer = SummaryWriter(str(log_dir))

    # --- Header ---
    print("\n" + "=" * 100)
    print(f"{'LPTN-LITE TRAINING - STAGE 2 (1024x768)':^100}")
    print("=" * 100)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[OK] GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("[!] Running on CPU (training will be VERY slow)")
    print(f"[OK] Device: {device}")
    print(f"[OK] Stage 1 Checkpoint: {args.ioanet_ckpt}")
    print(f"[OK] Debug: {args.debug}")
    if args.resume:
        print(f"[OK] Resume Checkpoint: {args.resume}")
    print("=" * 100)

    # --- Dataset (high-res) ---
    train_ds = ShadowRemovalDataset(
        root=args.data_root, split="train", csv_path=args.csv,
        size=HIGH_RES, use_mask=True, augment=True,
    )
    val_ds = ShadowRemovalDataset(
        root=args.data_root, split="test", csv_path=args.csv.replace("train", "test"),
        size=HIGH_RES, use_mask=True, augment=False,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    print(f"[stage2] train={len(train_ds)} val={len(val_ds)} high-res {HIGH_RES}")

    # --- Load frozen IOANet core ---
    ioanet = IOANet(pretrained=args.pretrained).to(device)
    ckpt = torch.load(args.ioanet_ckpt, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ioanet.load_state_dict(ckpt["model_state_dict"])
    else:
        ioanet.load_state_dict(ckpt)
    # Freeze BN statistics AND gradients.
    ioanet.eval()
    for param in ioanet.parameters():
        param.requires_grad = False
    print(f"[OK] IOANet loaded & frozen (eval mode)")

    # --- Trainable upsampler ---
    upsampler = LaplacianPyramidUpsampler(levels=2).to(device)
    optimizer = torch.optim.Adam(upsampler.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    criterion = L1Loss()

    # --- Resume ---
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    if args.resume and os.path.exists(args.resume):
        rckpt = torch.load(args.resume, map_location=device)
        if isinstance(rckpt, dict) and "model_state_dict" in rckpt:
            upsampler.load_state_dict(rckpt["model_state_dict"])
            if "optimizer_state_dict" in rckpt:
                optimizer.load_state_dict(rckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in rckpt:
                scheduler.load_state_dict(rckpt["scheduler_state_dict"])
            start_epoch = rckpt.get("epoch", 0)
            best_val_loss = rckpt.get("best_val_loss", float("inf"))
            patience_counter = rckpt.get("patience_counter", 0)
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
        upsampler.train()
        train_losses = []
        train_metrics = defaultdict(float)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}", leave=True,
                    dynamic_ncols=True,
                    bar_format='{l_bar}{bar}| [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
        for batch_idx, (input_img, target_img, mask) in enumerate(pbar):
            input_img = input_img.to(device)
            target_img = target_img.to(device)
            mask = mask.to(device)

            # Low-res shadow-free output from frozen IOANet.
            input_low = nn.functional.interpolate(
                input_img, size=LOW_RES, mode="bilinear", align_corners=False)
            with torch.no_grad():
                out_low = ioanet(input_low)

            optimizer.zero_grad()
            out_high = upsampler(out_low, input_img)
            loss = criterion(out_high, target_img)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(upsampler.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

            with torch.no_grad():
                metrics = MetricsCalculator.compute_all(out_high, target_img, mask)
                for k, v in metrics.items():
                    if k != "regions":
                        train_metrics[k] += v

            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "MAE": f"{metrics['mae']:.4f}",
                "PSNR": f"{metrics['psnr']:.1f}",
                "SSIM": f"{metrics['ssim']:.3f}",
            })

        avg_train_loss = sum(train_losses) / max(len(train_loader), 1)
        num_batches = max(len(train_loader), 1)
        for k in train_metrics:
            train_metrics[k] /= num_batches

        writer.add_scalar("train/loss", avg_train_loss, epoch)
        for k, v in train_metrics.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        print(f"[Train] Epoch {epoch+1:3d} | Loss: {avg_train_loss:.4f} | "
              f"MAE: {train_metrics['mae']:.4f} | PSNR: {train_metrics['psnr']:.2f} dB | "
              f"SSIM: {train_metrics['ssim']:.3f}")

        # --- Validation ---
        if (epoch + 1) % args.val_interval == 0:
            upsampler.eval()
            val_losses = []
            val_metrics = defaultdict(float)
            val_samples = []

            with torch.no_grad():
                for batch_idx, (input_img, target_img, mask) in enumerate(val_loader):
                    input_img = input_img.to(device)
                    target_img = target_img.to(device)
                    mask = mask.to(device)

                    input_low = nn.functional.interpolate(
                        input_img, size=LOW_RES, mode="bilinear", align_corners=False)
                    out_low = ioanet(input_low)
                    out_high = upsampler(out_low, input_img)
                    loss = criterion(out_high, target_img)
                    val_losses.append(loss.item())

                    metrics = MetricsCalculator.compute_all(out_high, target_img, mask)
                    for k, v in metrics.items():
                        if k != "regions":
                            val_metrics[k] += v

                    if batch_idx < 4:
                        val_samples.append((input_img, target_img, out_high))

            avg_val_loss = sum(val_losses) / max(len(val_loader), 1)
            num_val = max(len(val_loader), 1)
            for k in val_metrics:
                val_metrics[k] /= num_val

            writer.add_scalar("val/loss", avg_val_loss, epoch)
            for k, v in val_metrics.items():
                writer.add_scalar(f"val/{k}", v, epoch)

            improved = avg_val_loss < best_val_loss
            status = "[+]" if improved else "[-]"
            print(f"{status} Epoch {epoch+1:3d} | Loss: {avg_val_loss:.4f} | "
                  f"MAE: {val_metrics['mae']:.4f} | PSNR: {val_metrics['psnr']:.2f} dB | "
                  f"SSIM: {val_metrics['ssim']:.3f}")

            # Save debug samples
            if args.debug and val_samples:
                for i, (inp, tgt, out) in enumerate(val_samples[:4]):
                    path = save_sample(inp, tgt, out, debug_dir, epoch + 1, sample_num=i + 1)
                    if path:
                        print(f"    ► Sample {i+1} saved: {os.path.basename(path)}")

            # Best model + early stopping
            if improved:
                best_val_loss = avg_val_loss
                patience_counter = 0
                save_path = checkpoint_dir / "best_model.pth"
                torch.save({
                    "model_state_dict": upsampler.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": avg_val_loss,
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
                "model_state_dict": upsampler.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch + 1,
                "val_loss": best_val_loss,
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
            }, ckpt_path)

        scheduler.step()

    # --- Final save ---
    torch.save(upsampler.state_dict(), checkpoint_dir / "upsampler_final.pth")
    print("\n" + "=" * 100)
    print(f"[OK] Stage 2 Training Complete!")
    print(f"[OK] Best Model: {checkpoint_dir / 'best_model.pth'}")
    print(f"[OK] Best Val Loss: {best_val_loss:.4f}")
    print(f"[OK] TensorBoard Logs: {log_dir}")
    print("=" * 100 + "\n")
    writer.close()


if __name__ == "__main__":
    main()
