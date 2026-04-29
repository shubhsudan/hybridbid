"""
Eval harness wrapper for Diffusion-QL.

Implements the PolicyInterface expected by experiments/prepare_postbreak.py:
  policy.reset()                          → None
  policy(obs: dict)                       → np.ndarray (6,) physical MW

The model operates in p.u. space; this wrapper de-normalizes to physical MW.
"""

import numpy as np
import torch

P_MAX = 10.0  # MW


class DiffusionQLPolicy:
    """
    Wraps DiffusionQL model for use with the eval harness.

    Parameters
    ----------
    model : DiffusionQL
        Trained model (or checkpoint) on correct device.
    device : torch.device
    """

    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()

    def reset(self) -> None:
        """Called at the start of each eval episode. Stateless policy."""
        pass

    def __call__(self, obs: dict) -> np.ndarray:
        """
        Parameters
        ----------
        obs : dict with keys 'price_history' (32, 12) and 'static_features' (14,)

        Returns
        -------
        np.ndarray shape (6,) physical MW
          [p_energy_mw, c_regup_mw, c_regdn_mw, c_rrs_mw, c_ecrs_mw, c_nsrs_mw]
          p_energy signed: positive = discharge, negative = charge
        """
        price_history   = obs["price_history"]    # (32, 12)
        static_features = obs["static_features"]  # (14,)

        obs_flat = np.concatenate(
            [price_history.reshape(-1), static_features], axis=0
        ).astype(np.float32)  # (398,)

        obs_t = torch.from_numpy(obs_flat).unsqueeze(0).to(self.device)  # (1, 398)

        with torch.no_grad():
            action_pu = self.model.sample_action(obs_t)  # (1, 6) p.u.

        action_pu_np = action_pu.squeeze(0).cpu().numpy()  # (6,)

        # De-normalize: p_energy in [-1,1] × P_MAX, c_as in [0,1] × P_MAX
        action_mw = action_pu_np * P_MAX  # (6,) physical MW

        return action_mw.astype(np.float32)
