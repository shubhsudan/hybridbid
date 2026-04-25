"""
BC data loader — loads (obs, action_mw) pairs from trajectory NPZs.

The stored `rewards` field is NEVER loaded or used. BC loss is MSE on actions only.
Obs is flattened: price_history (32×12=384) + static_features (14) = 398 dims.
Actions are converted from p.u. to physical MW at load time (× P_MAX).
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

P_MAX = 10.0
N_PROBE = 100   # val states for action-distribution probe


class BCDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path)

        # Obs: flatten price_history + static_features
        ph = data["price_history"].astype(np.float32)   # (N, 32, 12)
        sf = data["static_features"].astype(np.float32) # (N, 14)
        self.obs = np.concatenate(
            [ph.reshape(len(ph), -1), sf], axis=1
        ).astype(np.float32)                             # (N, 398)

        # Actions: p.u. → physical MW
        self.actions_mw = data["actions"].astype(np.float32) * P_MAX  # (N, 6)

        assert self.obs.shape[1] == 398, f"Unexpected obs dim: {self.obs.shape[1]}"
        assert self.actions_mw.shape[1] == 6

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.actions_mw[idx]),
        )


def load_datasets(
    train_npz: str,
    val_npz: str,
    batch_size: int = 256,
    rng_seed: int = 42,
) -> tuple[DataLoader, DataLoader, np.ndarray]:
    """
    Load train/val datasets and sample probe states.

    Returns
    -------
    train_loader : DataLoader (shuffled)
    val_loader   : DataLoader (ordered, no drop)
    val_probe_obs: (N_PROBE, 398) float32 np.ndarray for action-distribution probe
    """
    train_ds = BCDataset(train_npz)
    val_ds   = BCDataset(val_npz)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=512, shuffle=False, drop_last=False,
    )

    # Sample N_PROBE random val states for the action-distribution probe
    rng = np.random.default_rng(rng_seed)
    probe_indices = rng.choice(len(val_ds), size=N_PROBE, replace=False)
    val_probe_obs = val_ds.obs[probe_indices]   # (N_PROBE, 398)

    return train_loader, val_loader, val_probe_obs
