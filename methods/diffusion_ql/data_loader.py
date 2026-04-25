"""
Data loader for Diffusion-QL.

Loads post-break MILP expert trajectories, replaces stored rewards with
physical-$ recomputed values (see methods/_shared/reward_recompute.py),
and exposes a PyTorch Dataset for offline RL training.

Action convention (training space, p.u.):
  actions[:, 0]  = p_energy ∈ [-1, 1]    (discharge positive, charge negative)
  actions[:, 1:] = c_as    ∈ [0, 1] × 5

Rewards (recomputed, physical $):
  r[t] = p_energy_mw × rt_lmp × DT + sum(c_as_mw × rt_mcpc) × DT

Observation:
  price_history: (32, 12) float32
  static_features: (14,) float32
  → flattened to (398,) for MLP input
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys, os

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from methods._shared.reward_recompute import recompute_rewards

OBS_DIM = 32 * 12 + 14  # 398
ACT_DIM = 6


def flatten_obs(price_history: np.ndarray, static_features: np.ndarray) -> np.ndarray:
    """Flatten observation dict arrays to (398,) vector."""
    return np.concatenate([price_history.reshape(-1), static_features], axis=-1)


class PostbreakDataset(Dataset):
    """
    PyTorch Dataset over post-break MILP trajectories.

    Returns (obs, act, rew, next_obs, done) tuples where:
      - obs / next_obs: (398,) float32 flattened observation
      - act: (6,) float32 normalized p.u. action
      - rew: scalar float32 physical-$ reward (recomputed)
      - done: scalar float32 (always 0.0 — no terminal states; truncateds handled
              by NOT zeroing Q-bootstrap at CT-midnight boundaries per sprint spec)
    """

    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=False)

        # Recompute rewards: discard stored mixed-unit rewards
        rewards = recompute_rewards(dict(data))

        # Flatten observations
        ph  = data["price_history"]       # (N, 32, 12)
        sf  = data["static_features"]     # (N, 14)
        nph = data["next_price_history"]  # (N, 32, 12)
        nsf = data["next_static_features"]  # (N, 14)

        obs      = np.concatenate([ph.reshape(len(ph), -1), sf],   axis=1).astype(np.float32)
        next_obs = np.concatenate([nph.reshape(len(nph), -1), nsf], axis=1).astype(np.float32)

        self.obs      = torch.from_numpy(obs)
        self.act      = torch.from_numpy(data["actions"].astype(np.float32))
        self.rew      = torch.from_numpy(rewards)
        self.next_obs = torch.from_numpy(next_obs)
        # No terminal states (dones all-False); CT-midnight truncations do NOT zero Q-bootstrap
        self.done = torch.zeros(len(rewards), dtype=torch.float32)

        print(f"[PostbreakDataset] Loaded {len(rewards):,} transitions from {npz_path}")
        print(f"  Rewards — mean: {rewards.mean():.3f}  std: {rewards.std():.3f}  "
              f"min: {rewards.min():.3f}  max: {rewards.max():.3f}")
        print(f"  (stored-rewards sum was {float(data['rewards'].sum()):.0f}; "
              f"recomputed sum: {float(rewards.sum()):.0f})")

    def __len__(self) -> int:
        return len(self.rew)

    def __getitem__(self, idx):
        return (self.obs[idx], self.act[idx], self.rew[idx],
                self.next_obs[idx], self.done[idx])


def make_dataloader(npz_path: str, batch_size: int = 256,
                    shuffle: bool = True) -> DataLoader:
    dataset = PostbreakDataset(npz_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True)
