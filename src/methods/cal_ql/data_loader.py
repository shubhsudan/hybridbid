"""
Data loader for Cal-QL offline phase.

Loads post-break MILP expert trajectories, recomputes rewards to physical-$,
builds SARSA-done mask from truncateds, and pre-computes per-state behavior
policy returns (V_behavior) used for CQL calibration.

Dataset tuple: (obs, act, rew, next_obs, sarsa_done, v_beh)

  obs / next_obs : (398,)  flattened price_history + static_features
  act            : (6,)    p.u. normalized actions from dataset
  rew            : ()      physical-$ reward (recomputed, NOT stored field)
  sarsa_done     : ()      1.0 at CT-midnight episode boundaries (truncateds)
  v_beh          : ()      discounted return under behavior policy from this state
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.methods._shared.reward_recompute import recompute_rewards

OBS_DIM = 32 * 12 + 14   # 398
ACT_DIM = 6


def compute_v_behavior(rewards: np.ndarray, truncateds: np.ndarray,
                       gamma: float = 0.99) -> np.ndarray:
    """
    Compute per-state Monte Carlo discounted return under the behavior policy.

    V_behavior[t] = r[t] + γ·r[t+1] + γ²·r[t+2] + … (within the same CT-day episode)

    Episode boundaries are defined by truncateds[t]=True (last step of a CT-day).
    Discount does NOT cross CT-midnight boundaries.

    Args:
        rewards   : (N,) physical-$ rewards (recomputed)
        truncateds: (N,) bool; True at the last step of each CT-day episode (68 in train)
        gamma     : discount factor

    Returns:
        V_behavior: (N,) float32
    """
    N = len(rewards)
    V = np.zeros(N, dtype=np.float64)
    future = 0.0
    for t in range(N - 1, -1, -1):
        if truncateds[t]:
            # Last step of this episode: no within-episode future to bootstrap
            V[t] = float(rewards[t])
        else:
            V[t] = float(rewards[t]) + gamma * future
        future = V[t]
    return V.astype(np.float32)


def load_or_compute_v_behavior(npz_path: str, cache_path: str,
                               gamma: float = 0.99) -> np.ndarray:
    """Load cached V_behavior or compute and save it."""
    if cache_path and os.path.exists(cache_path):
        v_beh = np.load(cache_path)
        print(f"[V_behavior] Loaded from cache {cache_path}  "
              f"mean={v_beh.mean():.3f}  max={v_beh.max():.3f}")
        return v_beh

    data    = np.load(npz_path, allow_pickle=False)
    rewards = recompute_rewards(dict(data))
    v_beh   = compute_v_behavior(rewards, data["truncateds"], gamma)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, v_beh)
        print(f"[V_behavior] Computed and cached to {cache_path}  "
              f"mean={v_beh.mean():.3f}  max={v_beh.max():.3f}")
    else:
        print(f"[V_behavior] Computed (no cache path)  "
              f"mean={v_beh.mean():.3f}  max={v_beh.max():.3f}")
    return v_beh


class PostbreakDatasetCalQL(Dataset):
    """
    PyTorch Dataset over post-break MILP expert trajectories for Cal-QL.

    Differences from DQL/QDT PostbreakDataset:
      - Includes v_beh (V_behavior) per transition for CQL calibration.
      - sarsa_done = truncateds (zeros Q-bootstrap at CT-midnight boundaries).
        dones are always 0 (no terminal states in continuous BESS operation).
    """

    # Physical-$ baselines from CLAUDE.md (68 CT-day train, 62 CT-day val).
    # Derived from MILP trajectory totals: train $116,669 / 19,584 intervals,
    # val $76,525 / 17,856 intervals. Tolerance 1% per sprint spec.
    _REWARD_BASELINES = {
        19584: 116_669.0 / 19_584,   # train: $5.957/interval
        17856:  76_525.0 / 17_856,   # val:   $4.285/interval
    }
    _REWARD_TOL = 0.01   # 1%

    def __init__(self, npz_path: str, v_behavior_cache: str = "",
                 gamma: float = 0.99):
        data    = np.load(npz_path, allow_pickle=False)
        rewards = recompute_rewards(dict(data))

        ph   = data["price_history"]        # (N, 32, 12)
        sf   = data["static_features"]      # (N, 14)
        nph  = data["next_price_history"]   # (N, 32, 12)
        nsf  = data["next_static_features"] # (N, 14)
        acts = data["actions"].astype(np.float32)
        N    = len(acts)

        # ── Reward recompute assertion (CLAUDE.md §SPRINT DISCIPLINE) ─────────
        # Recomputed physical-$ mean must be within 1% of the cc-baselines
        # physical-$ baseline. Fires on known splits (train 19584, val 17856).
        recomp_mean = float(rewards.mean())
        if N in self._REWARD_BASELINES:
            baseline = self._REWARD_BASELINES[N]
            pct_diff = abs(recomp_mean - baseline) / baseline
            status   = "PASS" if pct_diff < self._REWARD_TOL else "FAIL"
            print(f"  [REWARD ASSERT] recomputed={recomp_mean:.4f}  "
                  f"baseline={baseline:.4f}  delta={pct_diff*100:.3f}%  [{status}]")
            if status == "FAIL":
                raise RuntimeError(
                    f"Reward recompute assertion FAILED: recomp mean={recomp_mean:.4f}, "
                    f"baseline={baseline:.4f}, delta={pct_diff*100:.2f}% > 1% tolerance. "
                    f"Check reward_recompute.py or trajectory generation."
                )
        # ─────────────────────────────────────────────────────────────────────

        obs      = np.concatenate([ph.reshape(N, -1),  sf],  axis=1).astype(np.float32)
        next_obs = np.concatenate([nph.reshape(N, -1), nsf], axis=1).astype(np.float32)

        # SARSA done: zero Q-bootstrap at episode boundaries (CT-midnight truncations)
        sarsa_done = data["truncateds"].astype(np.float32)   # (N,); 68 ones in train split

        # V_behavior for CQL calibration
        v_beh = load_or_compute_v_behavior(npz_path, v_behavior_cache, gamma)
        assert len(v_beh) == N, f"V_behavior length {len(v_beh)} != N {N}"

        self.obs        = torch.from_numpy(obs)
        self.act        = torch.from_numpy(acts)
        self.rew        = torch.from_numpy(rewards)
        self.next_obs   = torch.from_numpy(next_obs)
        self.sarsa_done = torch.from_numpy(sarsa_done)
        self.v_beh      = torch.from_numpy(v_beh)

        print(f"[CalQLDataset] {N:,} transitions from {npz_path}")
        print(f"  rewards  — mean: {rewards.mean():.3f}  std: {rewards.std():.3f}  "
              f"min: {rewards.min():.3f}  max: {rewards.max():.3f}")
        print(f"  V_beh    — mean: {v_beh.mean():.3f}  P50: {np.median(v_beh):.3f}  "
              f"max: {v_beh.max():.3f}")
        print(f"  sarsa_done boundaries: {int(sarsa_done.sum())}")

    def __len__(self) -> int:
        return len(self.rew)

    def __getitem__(self, idx):
        return (self.obs[idx], self.act[idx], self.rew[idx],
                self.next_obs[idx], self.sarsa_done[idx], self.v_beh[idx])


def make_infinite_loader(npz_path: str, v_behavior_cache: str,
                         batch_size: int = 256, gamma: float = 0.99):
    """Infinite shuffled loader for offline training."""
    dataset = PostbreakDatasetCalQL(npz_path, v_behavior_cache, gamma)
    while True:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=True, drop_last=True)
        yield from loader
