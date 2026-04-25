"""
PolicyInterface for MILP+forecaster method.

At each CT day boundary (every 288 eval steps), the policy:
  1. Reads current SoC from obs["static_features"][-1]
  2. Reads price_history (32, 12) from obs["price_history"]
  3. Calls PriceTransformer → forecasted RT prices (288, 6)
  4. Solves 24h MILP with forecasted prices and current SoC
  5. Buffers the 288-step action sequence

At all other steps, returns the next buffered action.

This means 54 MILP solves for the full T-60 window.
"""

from __future__ import annotations

import numpy as np
import torch

from .forecaster import PriceTransformer, predict_prices
from .milp_solve import solve_daily_milp

P_MAX        = 10.0
E_MAX        = 20.0
STEPS_PER_DAY = 288


class MILPForecasterPolicy:
    """
    Eval harness PolicyInterface for the MILP+forecaster method.

    policy.reset()           → clears buffer, resets step counter
    policy(obs: dict) → (6,) physical MW
    """

    def __init__(
        self,
        model:      PriceTransformer,
        device:     str = "cpu",
        verbose:    bool = True,
    ):
        self.model   = model
        self.model.eval()
        self.device  = device
        self.verbose = verbose

        self._action_buffer: np.ndarray = np.empty((0, 6), dtype=np.float32)
        self._buf_idx  = 0
        self._eval_step = 0
        self._solve_log: list[dict] = []

    def reset(self) -> None:
        self._action_buffer = np.empty((0, 6), dtype=np.float32)
        self._buf_idx   = 0
        self._eval_step = 0
        self._solve_log = []

    def __call__(self, obs: dict) -> np.ndarray:
        if self._buf_idx >= len(self._action_buffer):
            self._refill(obs)

        action = self._action_buffer[self._buf_idx].copy()
        self._buf_idx   += 1
        self._eval_step += 1
        return action

    def _refill(self, obs: dict) -> None:
        """Forecast next 24h prices, solve MILP, buffer 288 actions."""
        soc = float(obs["static_features"][-1]) * E_MAX

        # Forecast prices from obs price_history
        price_history = obs["price_history"]   # (32, 12) float32 raw
        forecast_rt   = predict_prices(self.model, price_history, self.device)  # (288, 6)

        rt_lmp  = forecast_rt[:, 0]    # (288,)
        rt_mcpc = forecast_rt[:, 1:]   # (288, 5)

        day_num = self._eval_step // STEPS_PER_DAY + 1
        actions, meta = solve_daily_milp(rt_lmp, rt_mcpc, soc_init=soc)

        if self.verbose:
            print(
                f"  [milpf day {day_num:02d}] soc_in={soc:.1f} MWh  "
                f"status={meta.get('status','?')}  "
                f"forecast_rev=${meta.get('revenue',0):,.0f}  "
                f"t={meta.get('solve_time',0):.2f}s"
            )

        self._solve_log.append({"day": day_num, **meta})
        self._action_buffer = actions   # (288, 6)
        self._buf_idx = 0
