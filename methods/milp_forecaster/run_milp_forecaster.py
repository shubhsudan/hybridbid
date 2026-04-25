"""
MILP+forecaster entry point (Method 1).

Usage:
  python -m methods.milp_forecaster.run_milp_forecaster --phase smoke
      → 500 forecaster training steps, val loss print, sys.exit(0). Review first.

  python -m methods.milp_forecaster.run_milp_forecaster --phase full
      → 30k steps with early stopping. sys.exit(0) on stop or completion.
      → On completion runs full T-60 harness eval and writes data/results/eval_milpf/.

Sprint discipline:
  Smoke gate MUST be reviewed before full training.
  No skipping --phase smoke.
"""

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.prepare_postbreak import evaluate
from methods.eval_utils import enrich_summary_file, add_soc_diagnostics
from methods.milp_forecaster.forecast_dataset import make_datasets
from methods.milp_forecaster.forecaster import PriceTransformer
from methods.milp_forecaster.policy import MILPForecasterPolicy
from methods.milp_forecaster.train_forecaster import run_training

DATA_DIR    = str(ROOT / "data/processed")
RESULTS_DIR = str(ROOT / "data/results")
CKPT_DIR    = str(ROOT / "methods/milp_forecaster/checkpoints")


def main():
    parser = argparse.ArgumentParser(description="MILP+forecaster training and eval")
    parser.add_argument("--phase", choices=["smoke", "full"], required=True)
    parser.add_argument("--device", default="cpu",
                        help="cpu / mps / cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    # MPS availability check
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("[milpf/run] MPS not available, falling back to CPU.")
        args.device = "cpu"

    best_ckpt = Path(CKPT_DIR) / "forecaster_best.pt"
    if args.phase == "full" and not best_ckpt.exists():
        print("[milpf/run] ERROR: no smoke checkpoint found. Run --phase smoke first.")
        sys.exit(1)

    print(f"[milpf/run] Phase: {args.phase.upper()}  device={args.device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("[milpf/run] Building forecast datasets...")
    train_loader, val_loader = make_datasets(DATA_DIR, batch_size=args.batch_size)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = PriceTransformer()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[milpf/run] PriceTransformer: {n_params:,} parameters")

    if args.phase == "full":
        state = torch.load(str(best_ckpt), map_location=args.device, weights_only=True)
        model.load_state_dict(state["model"])
        print(f"[milpf/run] Loaded smoke checkpoint (step={state['step']}, "
              f"val_loss={state['val_loss']:.4f})")

    # ── Training ───────────────────────────────────────────────────────────────
    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=CKPT_DIR,
        smoke_only=(args.phase == "smoke"),
        device=args.device,
        lr=args.lr,
    )

    # ── Full T-60 eval (only reached if full training completes without early stop) ──
    print("\n[milpf/run] Training complete. Loading best checkpoint for eval...")
    state = torch.load(str(best_ckpt), map_location=args.device, weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()

    policy = MILPForecasterPolicy(model=model, device=args.device, verbose=True)

    print("[milpf/run] Running full T-60 harness eval (54 daily MILP solves)...")
    summary = evaluate(
        policy,
        method_name="milpf",
        data_dir=DATA_DIR,
        results_dir=RESULTS_DIR,
    )

    summary_path = os.path.join(RESULTS_DIR, "eval_milpf", "summary.json")
    traj_path    = os.path.join(RESULTS_DIR, "eval_milpf", "trajectory.parquet")
    enrich_summary_file(summary_path)
    add_soc_diagnostics(summary_path, traj_path)

    print(
        f"\n[milpf/run] MILP+forecaster eval complete: "
        f"{summary['all_days']['annualized_kw_yr']:.2f} $/kW-yr"
    )
    print("[milpf/run] STOP: Awaiting Karthik's green-light for Phase 3 complete / RL methods.")


if __name__ == "__main__":
    main()
