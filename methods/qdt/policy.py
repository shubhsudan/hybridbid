"""
Eval harness wrapper for QDT (Decision Transformer stage).

Implements PolicyInterface for experiments/prepare_postbreak.py.
Maintains a sliding K=20 step context window over the 54-day continuous eval.
"""

import numpy as np
import torch

from methods.qdt.data_loader import K, OBS_DIM, ACT_DIM

P_MAX = 10.0  # MW


class QDTPolicy:
    """
    Wraps DecisionTransformer for eval harness.

    Parameters
    ----------
    model : DecisionTransformer
    device : torch.device
    target_rtg : float
        Target return-to-go (Q-value scale). Set to P90 of training Q-values,
        roughly corresponding to "ask the DT to perform well."
    """

    def __init__(self, model, device: torch.device, target_rtg: float):
        self.model      = model
        self.device     = device
        self.target_rtg = target_rtg
        self.model.eval()

        # Sliding context buffers (filled as eval progresses)
        self._rtg_ctx : list[float]            = []
        self._obs_ctx : list[np.ndarray]       = []  # each (398,)
        self._act_ctx : list[np.ndarray]       = []  # each (6,)

    def reset(self) -> None:
        """Called at start of eval episode. Clear context."""
        self._rtg_ctx = []
        self._obs_ctx = []
        self._act_ctx = []

    def __call__(self, obs: dict) -> np.ndarray:
        """
        Parameters
        ----------
        obs : dict with 'price_history' (32,12) and 'static_features' (14,)

        Returns
        -------
        np.ndarray (6,) physical MW
        """
        obs_flat = np.concatenate([
            obs["price_history"].reshape(-1),
            obs["static_features"],
        ], axis=0).astype(np.float32)  # (398,)

        # Append current observation, use target RTG for this step
        self._rtg_ctx.append(self.target_rtg)
        self._obs_ctx.append(obs_flat)

        # Trim to K most recent steps
        ctx_len = min(len(self._rtg_ctx), K)
        rtg_win = self._rtg_ctx[-ctx_len:]  # list of floats
        obs_win = self._obs_ctx[-ctx_len:]  # list of (398,)
        act_win = self._act_ctx[-(ctx_len - 1):] if len(self._act_ctx) >= 1 else []
        # Pad act_win to ctx_len with zeros if needed (first step has no prior action)
        while len(act_win) < ctx_len:
            act_win = [np.zeros(ACT_DIM, dtype=np.float32)] + act_win

        # Convert to tensors
        rtg_t = torch.tensor(rtg_win, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, T)
        obs_t = torch.from_numpy(np.stack(obs_win)).unsqueeze(0).to(self.device)          # (1, T, 398)
        act_t = torch.from_numpy(np.stack(act_win)).unsqueeze(0).to(self.device)          # (1, T, 6)

        with torch.no_grad():
            action_pu = self.model.predict_action(rtg_t, obs_t, act_t)  # (1, 6)

        action_pu_np = action_pu.squeeze(0).cpu().numpy()  # (6,)

        # Store for next step context
        self._act_ctx.append(action_pu_np.copy())

        # De-normalize to physical MW
        return (action_pu_np * P_MAX).astype(np.float32)
