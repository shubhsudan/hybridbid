"""
BC training loop.

Probes:
  - action-distribution probe every 5 epochs (mean/std/zero-fraction per dim,
    100 random val states, no reward computation)
  - Fern slice eval at epochs 5 and 25 (in-loop; SoC reset to 50%)

Sprint discipline gates:
  - smoke phase (5 epochs): sys.exit(0) after probe prints
  - full phase (50 epochs): early stopping if val MSE doesn't improve for
    PATIENCE epochs, then sys.exit(0)
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .model import BCNet

P_MAX   = 10.0
PATIENCE = 5   # early stopping patience in epochs


def probe_action_distribution(
    net: BCNet,
    val_probe_obs: np.ndarray,   # (N_PROBE, 398)
    device: str = "cpu",
    epoch: int = 0,
) -> dict:
    """
    Compute action-distribution statistics on N_PROBE val states.
    No reward computation. Catches mean-collapse failure mode.

    Prints one line per action dimension: mean, std, frac_near_zero.
    Returns dict for training log.
    """
    net.eval()
    DIM_NAMES = ["p_energy", "c_regup", "c_regdn", "c_rrs", "c_ecrs", "c_nsrs"]
    ZERO_TOL  = 0.1   # MW — "near-zero" threshold

    with torch.no_grad():
        x = torch.from_numpy(val_probe_obs).to(device)   # (N, 398)
        a = net(x).cpu().numpy()                          # (N, 6) physical MW

    stats = {}
    parts = []
    for j, name in enumerate(DIM_NAMES):
        dim = a[:, j]
        mean = float(np.mean(dim))
        std  = float(np.std(dim))
        frac_zero = float(np.mean(np.abs(dim) < ZERO_TOL))
        stats[name] = {"mean": round(mean, 3), "std": round(std, 3), "frac_zero": round(frac_zero, 3)}
        parts.append(f"{name}: mean={mean:+.2f} std={std:.2f} zero={frac_zero:.0%}")

    print(f"  [probe e{epoch:02d}] " + "  |  ".join(parts))
    return stats


def train_epoch(
    net: BCNet,
    loader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    net.train()
    total_loss = 0.0
    n_batches  = 0
    for obs_b, act_b in loader:
        obs_b = obs_b.to(device)
        act_b = act_b.to(device)
        pred  = net(obs_b)
        loss  = criterion(pred, act_b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1
    return total_loss / max(n_batches, 1)


def eval_loss(
    net: BCNet,
    loader,
    criterion: nn.Module,
    device: str,
) -> float:
    net.eval()
    total_loss = 0.0
    n_batches  = 0
    with torch.no_grad():
        for obs_b, act_b in loader:
            obs_b = obs_b.to(device)
            act_b = act_b.to(device)
            loss  = criterion(net(obs_b), act_b)
            total_loss += loss.item()
            n_batches  += 1
    return total_loss / max(n_batches, 1)


def save_checkpoint(
    net: BCNet,
    optimizer: optim.Optimizer,
    epoch: int,
    val_loss: float,
    ckpt_path: str,
) -> None:
    torch.save(
        {
            "epoch":    epoch,
            "model":    net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss,
        },
        ckpt_path,
    )


def run_training(
    net:            BCNet,
    train_loader,
    val_loader,
    val_probe_obs:  np.ndarray,
    checkpoint_dir: str,
    fern_probe_fn,             # callable(net) → dict  (or None to skip)
    lr:     float = 3e-4,
    weight_decay: float = 1e-4,
    max_epochs: int = 50,
    smoke_only: bool = False,
    device: str = "cpu",
) -> list[dict]:
    """
    Main training loop.

    smoke_only=True → run 5 epochs, save checkpoint, sys.exit(0).
    smoke_only=False → run max_epochs with early stopping, sys.exit(0) on stop.

    Returns list of per-epoch log dicts (also written to training_log.json).
    """
    ckpt_dir  = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = str(ckpt_dir / "best.pt")
    last_ckpt = str(ckpt_dir / "last.pt")
    log_path  = str(ckpt_dir.parent / "training_log.json")

    net = net.to(device)
    optimizer  = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    criterion  = nn.MSELoss()

    best_val_loss    = float("inf")
    patience_counter = 0
    n_epochs         = 5 if smoke_only else max_epochs
    log: list[dict]  = []

    print(f"[bc/train] Starting {'smoke (5 epochs)' if smoke_only else f'full ({max_epochs} epochs)'} training on {device}")
    print(f"[bc/train] batches/epoch={len(train_loader)}")

    for epoch in range(1, n_epochs + 1):
        tr_loss = train_epoch(net, train_loader, optimizer, criterion, device)
        vl_loss = eval_loss(net, val_loader, criterion, device)

        entry = {"epoch": epoch, "train_loss": round(tr_loss, 6), "val_loss": round(vl_loss, 6)}

        # Action-distribution probe every 5 epochs
        if epoch % 5 == 0:
            probe_stats = probe_action_distribution(net, val_probe_obs, device, epoch)
            entry["probe"] = probe_stats
            # Collapse detection: flag if ALL dims have std < 0.5 MW
            stds = [probe_stats[k]["std"] for k in probe_stats]
            if max(stds) < 0.5:
                print(f"  [probe e{epoch:02d}] *** COLLAPSE WARNING: all dim stds < 0.5 MW ***")

        # Fern slice in-loop eval at epochs 5 and 25
        if fern_probe_fn is not None and epoch in {5, 25}:
            print(f"  [bc/train e{epoch:02d}] Running Fern slice eval...")
            fern_result = fern_probe_fn(net)
            entry["fern_slice"] = fern_result
            print(
                f"  [bc/train e{epoch:02d}] Fern slice: "
                f"${fern_result['total_revenue_usd']:,.0f} total "
                f"({fern_result['annualized_kw_yr']:.2f} $/kW-yr)  "
                f"Fern day: ${fern_result['fern_day_rev_usd']:,.0f}"
            )

        print(
            f"[bc/train] epoch={epoch:02d}/{n_epochs}  "
            f"train_loss={tr_loss:.4f}  val_loss={vl_loss:.4f}"
            + ("  *" if vl_loss < best_val_loss else "")
        )

        # Checkpoint if improved
        if vl_loss < best_val_loss:
            best_val_loss    = vl_loss
            patience_counter = 0
            save_checkpoint(net, optimizer, epoch, vl_loss, best_ckpt)
        else:
            patience_counter += 1

        log.append(entry)

        # Early stopping (full training only)
        if not smoke_only and patience_counter >= PATIENCE:
            print(
                f"[bc/train] Early stopping at epoch {epoch} "
                f"(no improvement for {PATIENCE} epochs). "
                f"Best val_loss={best_val_loss:.4f}"
            )
            save_checkpoint(net, optimizer, epoch, vl_loss, last_ckpt)
            with open(log_path, "w") as f:
                json.dump(log, f, indent=2)
            print("[bc/train] Checkpoints saved. sys.exit(0) per sprint discipline.")
            sys.exit(0)

    save_checkpoint(net, optimizer, epoch, vl_loss, last_ckpt)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    if smoke_only:
        print(
            f"[bc/train] Smoke complete (5 epochs). "
            f"best_val_loss={best_val_loss:.4f}. "
            "sys.exit(0) per sprint discipline."
        )
        sys.exit(0)

    print(f"[bc/train] Training complete. Best val_loss={best_val_loss:.4f}.")
    return log
