"""
train.py — full training pipeline for skeleton-based presentation scoring

Usage:
    python train.py --data_dir ./data --labels ./data/labels.csv

    # Use LSTM baseline first (faster, easier to debug):
    python train.py --data_dir ./data --labels ./data/labels.csv --model lstm

    # Load pretrained ST-GCN weights:
    python train.py --data_dir ./data --labels ./data/labels.csv --pretrained stgcn_ntu.pth
"""

import os
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import make_loaders, SCORE_COLUMNS
from model import STGCN, SkeletonLSTM, load_pretrained, body25_edges


# ------------------------------------------------------------------ #
#  Config                                                             #
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(description="Train skeleton presentation scorer")
    p.add_argument("--data_dir",    default="./data",       help="Folder with .pkl files")
    p.add_argument("--labels",      default="./data/labels.csv", help="Labels CSV path")
    p.add_argument("--model",       default="stgcn",        choices=["stgcn", "lstm"])
    p.add_argument("--pretrained",  default=None,           help="Path to pretrained .pth")
    p.add_argument("--epochs",      default=50,  type=int)
    p.add_argument("--batch_size",  default=8,   type=int)
    p.add_argument("--lr",          default=1e-3, type=float)
    p.add_argument("--frames",      default=150, type=int,  help="Frames per clip")
    p.add_argument("--n_joints",    default=25,  type=int,  help="Number of joints")
    p.add_argument("--dropout",     default=0.5, type=float)
    p.add_argument("--freeze_epochs", default=10, type=int,
                   help="Epochs to train head only before unfreezing backbone")
    p.add_argument("--checkpoint_dir", default="./checkpoints")
    p.add_argument("--workers",     default=0,   type=int)
    return p.parse_args()


# ------------------------------------------------------------------ #
#  Training utilities                                                 #
# ------------------------------------------------------------------ #

class EarlyStopping:
    """Stop training if val loss doesn't improve for `patience` epochs."""

    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    per_dim_loss = np.zeros(len(SCORE_COLUMNS))

    for skeletons, scores in loader:
        skeletons = skeletons.to(device)
        scores = scores.to(device)

        optimizer.zero_grad()
        preds = model(skeletons)
        loss = criterion(preds, scores)
        loss.backward()

        # Gradient clipping prevents exploding gradients with LSTMs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

        # Track per-dimension loss for diagnostics
        with torch.no_grad():
            dim_losses = ((preds - scores) ** 2).mean(dim=0).cpu().numpy()
            per_dim_loss += dim_losses

    n = len(loader)
    return total_loss / n, per_dim_loss / n


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    per_dim_loss = np.zeros(len(SCORE_COLUMNS))
    all_preds = []
    all_targets = []

    for skeletons, scores in loader:
        skeletons = skeletons.to(device)
        scores = scores.to(device)

        preds = model(skeletons)
        loss = criterion(preds, scores)
        total_loss += loss.item()

        dim_losses = ((preds - scores) ** 2).mean(dim=0).cpu().numpy()
        per_dim_loss += dim_losses

        all_preds.append(preds.cpu())
        all_targets.append(scores.cpu())

    # Compute R² score per dimension (higher = better fit, 1.0 = perfect)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    ss_res = ((all_preds - all_targets) ** 2).sum(dim=0)
    ss_tot = ((all_targets - all_targets.mean(dim=0)) ** 2).sum(dim=0)
    r2 = (1 - ss_res / (ss_tot + 1e-8)).numpy()

    n = len(loader)
    return total_loss / n, per_dim_loss / n, r2


def freeze_backbone(model):
    """Freeze all layers except the head."""
    for name, param in model.named_parameters():
        if "head" not in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Froze backbone. Trainable params: {trainable:,}")


def unfreeze_all(model, lr_reduce=0.1):
    """Unfreeze all parameters and return a new lower-LR optimizer."""
    for param in model.parameters():
        param.requires_grad = True
    total = sum(p.numel() for p in model.parameters())
    print(f"Unfroze all layers. Total params: {total:,}")


# ------------------------------------------------------------------ #
#  Main training loop                                                 #
# ------------------------------------------------------------------ #

def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- Data ---
    print("\nLoading data...")
    train_loader, val_loader, n_joints, n_coords = make_loaders(
        data_dir=args.data_dir,
        labels_csv=args.labels,
        batch_size=args.batch_size,
        target_frames=args.frames,
        num_workers=args.workers,
    )
    # Use detected joints unless overridden
    n_joints = args.n_joints or n_joints

    # --- Model ---
    print(f"\nBuilding {args.model.upper()} model...")
    if args.model == "stgcn":
        model = STGCN(
            in_channels=n_coords,
            n_joints=n_joints,
            n_outputs=len(SCORE_COLUMNS),
            edges=body25_edges(),
            dropout=args.dropout,
        )
        if args.pretrained:
            print(f"Loading pretrained weights from {args.pretrained}")
            model = load_pretrained(model, args.pretrained)
            freeze_backbone(model)
    else:
        model = SkeletonLSTM(
            n_joints=n_joints,
            n_coords=n_coords,
            n_outputs=len(SCORE_COLUMNS),
            dropout=args.dropout,
        )
        args.freeze_epochs = 0  # LSTM has no backbone to freeze

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # --- Loss, optimizer, scheduler ---
    criterion = nn.MSELoss()
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=5, verbose=True)
    early_stop = EarlyStopping(patience=15)

    # --- Training state ---
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "r2": [], "lr": []}
    backbone_unfrozen = (args.freeze_epochs == 0)

    print(f"\n{'='*60}")
    print(f"Training for up to {args.epochs} epochs")
    print(f"Scoring dimensions: {SCORE_COLUMNS}")
    print(f"Freeze backbone for first {args.freeze_epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Unfreeze backbone after freeze_epochs
        if not backbone_unfrozen and epoch > args.freeze_epochs:
            print(f"\nEpoch {epoch}: Unfreezing backbone with lr={args.lr * 0.1:.2e}")
            unfreeze_all(model)
            optimizer = Adam(model.parameters(),
                             lr=args.lr * 0.1, weight_decay=1e-4)
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                          patience=5, verbose=True)
            backbone_unfrozen = True

        # Train
        train_loss, train_dim = train_epoch(model, train_loader,
                                            optimizer, criterion, device)

        # Validate
        val_loss, val_dim, r2 = val_epoch(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Log
        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"LR: {current_lr:.2e} | {elapsed:.1f}s")

        # Per-dimension breakdown every 5 epochs
        if epoch % 5 == 0:
            print("  Per-dimension val MSE and R²:")
            for i, col in enumerate(SCORE_COLUMNS):
                print(f"    {col:<22} MSE={val_dim[i]:.4f}  R²={r2[i]:.3f}")

        # History
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["r2"].append(r2.tolist())
        history["lr"].append(current_lr)

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "r2": r2.tolist(),
                "config": {
                    "model": args.model,
                    "n_joints": n_joints,
                    "n_coords": n_coords,
                    "n_outputs": len(SCORE_COLUMNS),
                    "score_columns": SCORE_COLUMNS,
                    "target_frames": args.frames,
                    "dropout": args.dropout,
                },
            }, ckpt_path)
            print(f"  ✓ New best model saved (val_loss={val_loss:.4f})")

        # Save training history
        with open(os.path.join(args.checkpoint_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        # Early stopping
        early_stop(val_loss)
        if early_stop.should_stop:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {os.path.join(args.checkpoint_dir, 'best_model.pth')}")


if __name__ == "__main__":
    main()
