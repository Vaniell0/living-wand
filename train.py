"""Pre-training script for SpellFormer.

Generates dataset from algebraic rules, trains the Transformer,
saves weights to data/spellformer.pt.

Usage:
    python train.py [--samples N] [--epochs N] [--batch N] [--lr F]
"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.wand.datagen import SpellDataGenerator, samples_to_tensors
from src.wand.loss import SpellLoss
from src.wand.transformer import SpellFormer


def cosine_lr(step: int, warmup: int, total: int, lr_max: float, lr_min: float) -> float:
    """Cosine learning rate with linear warmup."""
    if step < warmup:
        return lr_max * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def train(
    n_samples: int = 180_000,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 3e-4,
    lr_min: float = 1e-5,
    warmup_steps: int = 500,
    save_path: str = "data/spellformer.pt",
    seed: int = 42,
):
    torch.manual_seed(seed)

    # --- Generate data ---
    print(f"Generating {n_samples:,} training samples...")
    gen = SpellDataGenerator(seed=seed)

    # Scale phases proportionally
    scale = n_samples / 180_000
    samples = gen.generate_dataset(
        n_random=int(100_000 * scale),
        n_synergy=int(50_000 * scale),
        n_perturbed=int(20_000 * scale),
        n_edge=int(10_000 * scale),
    )
    print(f"Generated {len(samples):,} valid samples")

    # Split: 95% train, 5% val
    n_val = max(100, len(samples) // 20)
    n_train = len(samples) - n_val
    train_samples = samples[:n_train]
    val_samples = samples[n_train:]

    tok_train, role_train, deg_train, tgt_train = samples_to_tensors(train_samples)
    tok_val, role_val, deg_val, tgt_val = samples_to_tensors(val_samples)

    train_ds = TensorDataset(tok_train, role_train, deg_train, tgt_train)
    val_ds = TensorDataset(tok_val, role_val, deg_val, tgt_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    print(f"Train: {n_train:,}, Val: {n_val:,}")

    # --- Model ---
    model = SpellFormer()
    print(f"SpellFormer: {model.param_count():,} parameters")

    # --- Training ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = SpellLoss()

    total_steps = epochs * len(train_loader)
    step = 0
    best_val_loss = float("inf")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_components = {"pole": 0, "scalar": 0, "extent": 0, "intensity": 0, "semantic": 0}
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            tok, roles, deg, targets = batch

            # LR schedule
            current_lr = cosine_lr(step, warmup_steps, total_steps, lr, lr_min)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            output = model(tok, roles, deg)
            losses = loss_fn(output["raw"], targets)

            optimizer.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += losses["total"].item()
            for k in epoch_components:
                epoch_components[k] += losses[k].item()
            n_batches += 1
            step += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        dt = time.time() - t0

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                tok, roles, deg, targets = batch
                output = model(tok, roles, deg)
                losses = loss_fn(output["raw"], targets)
                val_loss += losses["total"].item()
                n_val_batches += 1

        avg_val = val_loss / max(n_val_batches, 1)

        # Print progress
        comp_str = " | ".join(
            f"{k}={v / n_batches:.4f}" for k, v in epoch_components.items()
        )
        print(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"train={avg_loss:.4f} val={avg_val:.4f} | "
            f"lr={current_lr:.2e} | {dt:.1f}s | {comp_str}"
        )

        # Save best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model (val={avg_val:.4f})")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SpellFormer")
    parser.add_argument("--samples", type=int, default=180_000, help="Total training samples")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--save", type=str, default="data/spellformer.pt", help="Save path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train(
        n_samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        save_path=args.save,
        seed=args.seed,
    )
