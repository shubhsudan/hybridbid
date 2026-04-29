"""
Forecaster training loop.

Smoke phase:  500 steps (~validation check, fast exit) → sys.exit(0)
Full phase:  30 000 steps with val-MSE early stopping (patience 3) → sys.exit(0) on stop

Loss: MSE in log-transformed price space (robust to Fern-scale spikes).
Optimizer: Adam, lr=5e-4, weight_decay=1e-5, cosine LR schedule.
"""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from .forecaster import PriceTransformer

SMOKE_STEPS  = 500
FULL_STEPS   = 30_000
VAL_INTERVAL = 1_000   # validate every N steps
PATIENCE     = 3       # patience in val-interval units (3 × 1000 = 3000 steps no improvement)
LOG_INTERVAL = 200


def _val_loss(model, val_loader, criterion, device, max_batches=50):
    """Evaluate on up to max_batches val batches (fast in-training check)."""
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for i, (hist, tgt) in enumerate(val_loader):
            if i >= max_batches:
                break
            pred = model(hist.to(device))
            loss = criterion(pred, tgt.to(device))
            total += loss.item()
            n += 1
    return total / max(n, 1)


def run_training(
    model:          PriceTransformer,
    train_loader,
    val_loader,
    checkpoint_dir: str,
    smoke_only:     bool = False,
    device:         str  = "cpu",
    lr:             float = 5e-4,
    weight_decay:   float = 1e-5,
) -> list[dict]:
    """
    Train the price forecaster.

    smoke_only=True  → 500 steps, val check, sys.exit(0)
    smoke_only=False → 30k steps with early stopping, sys.exit(0) on stop
    """
    ckpt_dir  = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = str(ckpt_dir / "forecaster_best.pt")
    last_ckpt = str(ckpt_dir / "forecaster_last.pt")
    log_path  = str(ckpt_dir / "forecaster_training_log.json")

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    max_steps = SMOKE_STEPS if smoke_only else FULL_STEPS
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=lr / 20)

    print(f"[forecaster/train] {'Smoke' if smoke_only else 'Full'} training: "
          f"{max_steps} steps on {device}")
    print(f"[forecaster/train] Train batches available: {len(train_loader)}")

    best_val_loss    = float("inf")
    patience_counter = 0
    step             = 0
    log: list[dict]  = []
    running_loss     = 0.0
    t0               = time.time()

    train_iter = iter(train_loader)

    while step < max_steps:
        model.train()
        try:
            hist, tgt = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            hist, tgt = next(train_iter)

        hist = hist.to(device)
        tgt  = tgt.to(device)

        pred = model(hist)
        loss = criterion(pred, tgt)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        step += 1

        if step % LOG_INTERVAL == 0:
            avg_loss = running_loss / LOG_INTERVAL
            elapsed  = time.time() - t0
            print(f"  step={step:6d}/{max_steps}  train_loss={avg_loss:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  t={elapsed:.0f}s")
            running_loss = 0.0

        if step % VAL_INTERVAL == 0 or step == max_steps:
            val_loss = _val_loss(model, val_loader, criterion, device)
            print(f"  [val] step={step}  val_loss={val_loss:.4f}" +
                  ("  *" if val_loss < best_val_loss else ""))

            entry = {"step": step, "val_loss": round(val_loss, 6)}

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(
                    {"step": step, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "val_loss": val_loss},
                    best_ckpt,
                )
            else:
                patience_counter += 1

            log.append(entry)

            if not smoke_only and patience_counter >= PATIENCE:
                print(
                    f"[forecaster/train] Early stopping at step {step} "
                    f"(no val improvement for {PATIENCE} × {VAL_INTERVAL} steps). "
                    f"Best val_loss={best_val_loss:.4f}"
                )
                torch.save(
                    {"step": step, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "val_loss": val_loss},
                    last_ckpt,
                )
                with open(log_path, "w") as f:
                    json.dump(log, f, indent=2)
                print("[forecaster/train] sys.exit(0) per sprint discipline.")
                sys.exit(0)

    torch.save(
        {"step": step, "model": model.state_dict(),
         "optimizer": optimizer.state_dict(), "val_loss": best_val_loss},
        last_ckpt,
    )
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    if smoke_only:
        print(
            f"[forecaster/train] Smoke complete ({SMOKE_STEPS} steps). "
            f"Best val_loss={best_val_loss:.4f}. sys.exit(0) per sprint discipline."
        )
        sys.exit(0)

    print(f"[forecaster/train] Training complete at step {step}. Best val_loss={best_val_loss:.4f}.")
    return log
