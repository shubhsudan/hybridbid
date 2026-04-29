"""
PolicyInterface wrapper for a trained BCNet.

Wraps a BCNet checkpoint for use with the eval harness and Fern slice evaluator.
Obs dict → flatten → forward pass → physical MW action.
"""

import numpy as np
import torch

from .model import BCNet

P_MAX = 10.0


class BCPolicy:
    """
    Eval harness PolicyInterface wrapping a trained BCNet.

    policy.reset()             → no-op (BC is stateless)
    policy(obs: dict) → (6,)  physical MW
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.net = BCNet()
        self.net.load_state_dict(state["model"])
        self.net.eval()
        self.device = device

    def reset(self) -> None:
        pass

    def __call__(self, obs: dict) -> np.ndarray:
        ph = obs["price_history"].flatten()   # (384,)
        sf = obs["static_features"]           # (14,)
        x  = np.concatenate([ph, sf]).astype(np.float32)   # (398,)
        with torch.no_grad():
            t = torch.from_numpy(x).unsqueeze(0).to(self.device)  # (1, 398)
            a = self.net(t).squeeze(0).cpu().numpy()               # (6,)
        return a
