"""
PolicyInterface for MILP+forecaster method.

Two variants:

MILPForecasterPolicy — original: transformer for all 6 price dimensions.
  (Archived; has AS price collapse bug — see DIAGNOSIS.md)

HybridMILPForecasterPolicy — production: transformer for LMP, climatological
  AS for the 5 MCPC dimensions.
  - LMP: transformer output (captures diurnal patterns and event signals)
  - AS MCPC: lookup table as_climate_hod[hour_of_day, product] from post-break
    training data (Dec 5 2025 – Feb 10 2026). Avoids persistence spike
    propagation on a sparse bursty distribution.

At each CT day boundary (every 288 eval steps), both policies:
  1. Read SoC from obs["static_features"][-1]
  2. Read price_history (32, 12) from obs["price_history"]
  3. Compose 24h price forecast
  4. Solve 24h MILP, buffer 288 actions
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import torch

from .forecaster import PriceTransformer, predict_prices
from .milp_solve import solve_daily_milp

P_MAX         = 10.0
E_MAX         = 20.0
STEPS_PER_DAY = 288

# Jan 1, 2026 is a Thursday (Python weekday 3)
# Eval starts Jan 1 CT midnight → hour-of-week = 3 * 24 = 72
_EVAL_START_HOW = 72


def _load_as_climate(ckpt_dir: str | None = None) -> np.ndarray:
    """Load (24, 5) hour-of-day AS climatology from checkpoint directory."""
    if ckpt_dir is None:
        ckpt_dir = str(Path(__file__).parent / "checkpoints")
    path = Path(ckpt_dir) / "as_climate_hod.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"AS climate table not found at {path}. "
            "Run the precomputation step first."
        )
    return np.load(str(path)).astype(np.float32)   # (24, 5)


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


class HybridMILPForecasterPolicy:
    """
    Eval harness PolicyInterface for the hybrid MILP+forecaster method.

    LMP:     transformer output (obs price_history → PriceTransformer)
    AS MCPC: climatological lookup table (hour-of-day mean from post-break training data)

    The climatological AS forecast avoids persistence spike propagation on the
    sparse bursty AS price distribution where yesterday's spikes ≠ today's spikes.
    """

    def __init__(
        self,
        model:      PriceTransformer,
        device:     str = "cpu",
        verbose:    bool = True,
        ckpt_dir:   str | None = None,
    ):
        self.model      = model
        self.model.eval()
        self.device     = device
        self.verbose    = verbose
        self.as_climate = _load_as_climate(ckpt_dir)   # (24, 5)

        self._action_buffer: np.ndarray = np.empty((0, 6), dtype=np.float32)
        self._buf_idx   = 0
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
        """Hybrid forecast: transformer LMP + climatological AS → MILP."""
        soc     = float(obs["static_features"][-1]) * E_MAX
        day_num = self._eval_step // STEPS_PER_DAY + 1

        # Transformer LMP forecast (288,)
        price_history = obs["price_history"]   # (32, 12)
        forecast_rt   = predict_prices(self.model, price_history, self.device)  # (288, 6)
        rt_lmp        = forecast_rt[:, 0]       # transformer LMP

        # Climatological AS MCPC (288, 5) — hour-of-day lookup
        # For step i (0-287), hour_of_day = i // 12
        hours  = np.arange(STEPS_PER_DAY) // 12          # (288,) in 0-23
        rt_mcpc = self.as_climate[hours]                  # (288, 5)

        actions, meta = solve_daily_milp(rt_lmp, rt_mcpc, soc_init=soc)

        if self.verbose:
            lmp_mean = rt_lmp.mean()
            as_mean  = rt_mcpc[:, 0].mean()   # regup as representative
            print(
                f"  [hybrid day {day_num:02d}] soc_in={soc:.1f} MWh  "
                f"status={meta.get('status','?')}  "
                f"lmp_fcst_mean={lmp_mean:.1f}  as_regup_clim={as_mean:.3f}  "
                f"forecast_rev=${meta.get('revenue',0):,.0f}  "
                f"t={meta.get('solve_time',0):.2f}s"
            )

        self._solve_log.append({"day": day_num, **meta})
        self._action_buffer = actions
        self._buf_idx = 0
