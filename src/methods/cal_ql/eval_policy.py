"""
Eval harness wrapper for Cal-QL.

Implements the PolicyInterface expected by experiments/prepare_postbreak.py:
  policy.reset()          → None
  policy(obs: dict)       → np.ndarray (6,) physical MW

The actor operates in p.u. space; de-normalized here by × P_MAX=10.

Eval uses deterministic action (squashed mean), not sampled.
"""

import numpy as np
import torch
from pathlib import Path
import sys

ROOT = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, ROOT)

from src.methods.cal_ql.networks import Actor

P_MAX = 10.0   # MW; hardcoded per sprint spec (do NOT read configs/battery.yaml)


class CalQLPolicy:
    """
    Wraps a trained Cal-QL Actor for the eval harness.

    Parameters
    ----------
    checkpoint_path : str
        Path to checkpoint saved by train_offline.py (contains 'actor' key).
    device : torch.device or str, optional
        Defaults to CPU for eval harness compatibility.
    """

    def __init__(self, checkpoint_path: str, device=None):
        if device is None:
            device = torch.device("cpu")
        self.device = torch.device(device) if isinstance(device, str) else device

        self.actor = Actor().to(self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        # Support both raw state_dict and wrapped checkpoint
        if "actor" in ckpt:
            self.actor.load_state_dict(ckpt["actor"])
        else:
            self.actor.load_state_dict(ckpt)

        self.actor.eval()

    def reset(self) -> None:
        """Stateless policy; no episode state to reset."""
        pass

    def __call__(self, obs: dict) -> np.ndarray:
        """
        Parameters
        ----------
        obs : dict with keys:
          'price_history'   : (32, 12) float32
          'static_features' : (14,) float32

        Returns
        -------
        np.ndarray shape (6,) physical MW
          [p_energy_mw, c_regup_mw, c_regdn_mw, c_rrs_mw, c_ecrs_mw, c_nsrs_mw]
          p_energy signed: positive = discharge, negative = charge
        """
        ph  = obs["price_history"]    # (32, 12)
        sf  = obs["static_features"]  # (14,)

        obs_flat = np.concatenate([ph.reshape(-1), sf]).astype(np.float32)  # (398,)
        obs_t    = torch.from_numpy(obs_flat).unsqueeze(0).to(self.device)   # (1, 398)

        with torch.no_grad():
            action_pu = self.actor.deterministic(obs_t)   # (1, 6) p.u.

        action_pu_np = action_pu.squeeze(0).cpu().numpy()  # (6,)

        # De-normalize to physical MW
        # p_energy: p.u. ∈ (−1,1) → MW ∈ (−10, 10)
        # c_as:     p.u. ∈ (0, 1)  → MW ∈ (0, 10)
        action_mw = action_pu_np * P_MAX

        return action_mw.astype(np.float32)
