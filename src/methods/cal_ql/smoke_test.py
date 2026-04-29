"""
Cal-QL 5k smoke test.

Thin wrapper around train_offline.py --mode smoke.
Writes SMOKE_RESULTS.md and exits via sys.exit(0).

Usage (from repo root):
  python -m methods.cal_ql.smoke_test [--gpu 13]

Narnia-specific: GPUs 13 and 16 are available (A16). Default GPU 13.

Smoke gate criteria:
  PASS: Q_max bounded, Q_mean > 0, CQL term bounded, actions non-degenerate
  FAIL: any NaN, Q_max > 50k, Q_mean < 0, CQL term unbounded, or all-zero actions

After smoke PASS, Karthik reviews SMOKE_RESULTS.md and confirms before launching
the 50k full run.
"""

import sys
import argparse
from pathlib import Path

ROOT = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, ROOT)

from src.methods.cal_ql.train_offline import train


def main():
    ap = argparse.ArgumentParser(description="Cal-QL 5k smoke test")
    ap.add_argument("--gpu",          type=int, default=13,
                    help="GPU index (Narnia: 13 or 16)")
    ap.add_argument("--train-path",
                    default="data/expert_trajectories/receding_horizon_postbreak_train.npz")
    ap.add_argument("--val-path",
                    default="data/expert_trajectories/receding_horizon_postbreak_val.npz")
    ap.add_argument("--v-beh-cache", default="data/cal_ql/V_behavior.npy")
    args = ap.parse_args()

    train(
        mode="smoke",
        gpu=args.gpu,
        train_path=args.train_path,
        val_path=args.val_path,
        v_beh_cache=args.v_beh_cache,
        resume_ckpt="",
    )


if __name__ == "__main__":
    main()
