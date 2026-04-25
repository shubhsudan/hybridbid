"""
BC entry point.

Usage:
  python -m methods.bc.run_bc --phase smoke
      → 5 epochs, action probe, Fern slice eval, sys.exit(0). Review before continuing.

  python -m methods.bc.run_bc --phase full
      → 50 epochs with early stopping. sys.exit(0) on stop or completion.
      → On completion, runs full T-60 harness eval and writes data/results/eval_bc/.

Sprint discipline:
  Smoke gate MUST be reviewed before full training.
  No skipping --phase smoke.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.prepare_postbreak import evaluate
from methods.bc.data_loader import load_datasets
from methods.bc.model import BCNet
from methods.bc.policy import BCPolicy
from methods.bc.train import run_training
from methods.eval_utils import (
    add_ceiling_metrics,
    enrich_summary_file,
    prepare_fern_slice_data,
    run_fern_slice,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_NPZ   = str(ROOT / "data/expert_trajectories/receding_horizon_postbreak_train.npz")
VAL_NPZ     = str(ROOT / "data/expert_trajectories/receding_horizon_postbreak_val.npz")
DATA_DIR    = str(ROOT / "data/processed")
RESULTS_DIR = str(ROOT / "data/results")
CKPT_DIR    = str(ROOT / "methods/bc/checkpoints")


def make_fern_probe(net: BCNet, price_data, system_data, ts_ct, fern_start, fern_end, device):
    """Build the Fern slice probe callable for the training loop."""
    from methods.bc.policy import BCPolicy as _BCPolicy

    class _InlinePolicy:
        def __init__(self, n):
            self.net = n
        def reset(self): pass
        def __call__(self, obs):
            ph = obs["price_history"].flatten()
            sf = obs["static_features"]
            x  = np.concatenate([ph, sf]).astype(np.float32)
            with torch.no_grad():
                t = torch.from_numpy(x).unsqueeze(0).to(device)
                a = self.net(t).squeeze(0).cpu().numpy()
            return a

    def _probe(n: BCNet) -> dict:
        policy = _InlinePolicy(n)
        n.eval()
        return run_fern_slice(
            policy, price_data, system_data, ts_ct,
            fern_start, fern_end, soc_init=10.0,
        )

    return _probe


def main():
    parser = argparse.ArgumentParser(description="BC training — behavior cloning from MILP expert")
    parser.add_argument("--phase", choices=["smoke", "full"], required=True,
                        help="smoke = 5 epochs (MUST run first); full = 50 epochs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    if args.phase == "full" and not (Path(CKPT_DIR) / "best.pt").exists():
        print("[bc/run] ERROR: no smoke checkpoint found. Run --phase smoke first.")
        sys.exit(1)

    print(f"[bc/run] Phase: {args.phase.upper()}  device={args.device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("[bc/run] Loading datasets...")
    train_loader, val_loader, val_probe_obs = load_datasets(
        TRAIN_NPZ, VAL_NPZ, batch_size=args.batch_size,
    )
    print(
        f"[bc/run] Train: {len(train_loader.dataset)} transitions  "
        f"Val: {len(val_loader.dataset)} transitions  "
        f"Batches/epoch: {len(train_loader)}"
    )

    # ── Fern slice data (load once, no re-reads during training) ──────────────
    print("[bc/run] Loading Fern slice data...")
    price_data, system_data, ts_ct, fern_start, fern_end = prepare_fern_slice_data(DATA_DIR)
    n_fern = fern_end - fern_start
    print(f"[bc/run] Fern slice: [{fern_start}:{fern_end}] = {n_fern} steps ({n_fern//288} days)")

    # ── Model ─────────────────────────────────────────────────────────────────
    net = BCNet()
    if args.phase == "full":
        # Resume from smoke checkpoint
        smoke_ckpt = str(Path(CKPT_DIR) / "best.pt")
        state = torch.load(smoke_ckpt, map_location=args.device, weights_only=True)
        net.load_state_dict(state["model"])
        print(f"[bc/run] Loaded smoke checkpoint (epoch={state['epoch']}, val_loss={state['val_loss']:.4f})")

    # ── Fern probe callable ────────────────────────────────────────────────────
    fern_probe = make_fern_probe(
        net, price_data, system_data, ts_ct, fern_start, fern_end, args.device
    )

    # ── Training ───────────────────────────────────────────────────────────────
    run_training(
        net=net,
        train_loader=train_loader,
        val_loader=val_loader,
        val_probe_obs=val_probe_obs,
        checkpoint_dir=CKPT_DIR,
        fern_probe_fn=fern_probe,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=50,
        smoke_only=(args.phase == "smoke"),
        device=args.device,
    )

    # ── Full T-60 eval (only reached if full training completes without early stop) ──
    print("\n[bc/run] Training complete. Running full T-60 harness eval...")
    best_ckpt = str(Path(CKPT_DIR) / "best.pt")
    policy = BCPolicy(checkpoint_path=best_ckpt, device=args.device)

    summary = evaluate(
        policy,
        method_name="bc",
        data_dir=DATA_DIR,
        results_dir=RESULTS_DIR,
    )
    # Enrich summary with ceiling metrics
    summary_path = os.path.join(RESULTS_DIR, "eval_bc", "summary.json")
    enrich_summary_file(summary_path)

    print(f"\n[bc/run] BC eval complete: {summary['all_days']['annualized_kw_yr']:.2f} $/kW-yr")
    print("[bc/run] STOP: Awaiting Karthik's green-light for Phase 3 (MILP+forecaster).")


if __name__ == "__main__":
    main()
