"""
Shared reward recomputation utility for offline RL training.

The stored `rewards` field in trajectory NPZs uses Li et al. Eq.26 mixed convention:
  - energy term: per-unit (energy_mag_pu × rt_lmp × DT), missing ×P_MAX=10
  - timing bonus: BETA_ARB=10 × energy_mag_pu × price_dev × DT, also p.u.
  - AS revenue: physical $ (c_as_mw × rt_mcpc × DT)

This causes two problems for Q-learning:
  1. Gradient bias: 1 MW of AS earns ~10× larger gradient signal than 1 MW of energy
     at the same price level.
  2. Sign flips: timing bonus can dominate and give stored_reward > 0 when physical
     revenue is negative (e.g., charging below EMA earns positive stored reward but
     costs physical dollars). Q-targets trained on this are directionally wrong.

Methodology note (paper write-up): "Stored trajectory rewards used a mixed-unit
convention from the Li et al. formulation; we recompute rewards at training-time
using consistent physical-dollar units to avoid gradient bias and sign-flip artifacts
in Q-learning." (See REWARD_CONVENTION.md for full investigation.)

This formula matches experiments/prepare_postbreak.py lines 329-330 exactly.
"""

import numpy as np

P_MAX: float = 10.0     # MW
DT: float = 5.0 / 60.0  # hours per 5-min interval


def recompute_rewards(trajectory: dict) -> np.ndarray:
    """
    Recompute per-interval physical-$ rewards from stored actions and price_history.

    Discards stored trajectory['rewards'] entirely. The returned array is on the
    same scale as the eval harness: energy_rev + as_rev in physical dollars per
    5-min interval.

    Parameters
    ----------
    trajectory : dict
        NPZ trajectory dict with keys:
          'actions'        : (N, 6) float32  normalized p.u. [p_energy∈[-1,1], c_as∈[0,1]×5]
          'price_history'  : (N, 32, 12) float32  rolling window; last row = current step

    Returns
    -------
    np.ndarray, shape (N,), dtype float32
        Physical-$ reward per interval:
          reward[t] = p_energy_mw[t] * rt_lmp[t] * DT
                    + sum(c_as_mw[t] * rt_mcpc[t]) * DT
    """
    actions: np.ndarray = trajectory["actions"]           # (N, 6)
    price_history: np.ndarray = trajectory["price_history"]  # (N, 32, 12)

    p_energy_mw: np.ndarray = actions[:, 0] * P_MAX        # (N,) signed MW
    c_as_mw: np.ndarray = actions[:, 1:] * P_MAX           # (N, 5) MW

    rt_lmp: np.ndarray = price_history[:, -1, 0].astype(np.float64)    # (N,) $/MWh
    rt_mcpc: np.ndarray = price_history[:, -1, 1:6].astype(np.float64)  # (N, 5) $/MWh

    energy_rev: np.ndarray = p_energy_mw.astype(np.float64) * rt_lmp * DT
    as_rev: np.ndarray = (c_as_mw.astype(np.float64) * rt_mcpc).sum(axis=1) * DT

    return (energy_rev + as_rev).astype(np.float32)
