"""
TBx policies for the T-60 eval window.

Two variants, both implementing the harness PolicyInterface:
  TBxEnergyOnlyPolicy  — energy arbitrage only, no AS (apples-to-apples with
                         the pre-break v5.1 TBx number)
  TBxWithASPolicy      — energy arbitrage + AS when idle (correct post-RTC+B
                         comparable)

Battery constants hardcoded per CLAUDE.md (do NOT read configs/battery.yaml).
"""

import numpy as np

P_MAX = 10.0
E_MAX = 20.0
ETA   = 0.95
SOC_MIN = 2.0
SOC_MAX = 18.0

# ERCOT AS sustain durations [h]: [regup, regdn, rrs, ecrs, nsrs]
# Source: postbreak_milp.py AS_SUSTAIN_H; matches prepare_postbreak.py AS_SUSTAIN_H
AS_SUSTAIN_H = np.array([1.0, 1.0, 1.0 / 6.0, 1.0 / 4.0, 0.5], dtype=np.float64)


def calibrate_thresholds(
    rt_lmp_train: np.ndarray,
    p_low_pct:  float = 25.0,
    p_high_pct: float = 75.0,
) -> tuple[float, float]:
    """
    Compute TBx thresholds from training-window RT LMP distribution.

    Parameters
    ----------
    rt_lmp_train : (N,) RT LMP values for the training window (Dec 5 – Feb 10)
    p_low_pct    : percentile below which to charge (default 25th)
    p_high_pct   : percentile above which to discharge (default 75th)

    Returns
    -------
    (p_low, p_high) in $/MWh
    """
    p_low  = float(np.percentile(rt_lmp_train, p_low_pct))
    p_high = float(np.percentile(rt_lmp_train, p_high_pct))
    return p_low, p_high


class TBxEnergyOnlyPolicy:
    """
    Threshold-based arbitrage — energy only, no AS participation.

    Discharges at P_max when rt_lmp > p_high.
    Charges at P_max when rt_lmp < p_low.
    Idles otherwise.

    Current RT LMP is obs["price_history"][-1, 0] (last row of rolling window).
    """

    def __init__(self, p_low: float, p_high: float):
        self.p_low  = p_low
        self.p_high = p_high

    def reset(self) -> None:
        pass

    def __call__(self, obs: dict) -> np.ndarray:
        rt_lmp = float(obs["price_history"][-1, 0])
        if rt_lmp > self.p_high:
            p_energy = P_MAX    # discharge
        elif rt_lmp < self.p_low:
            p_energy = -P_MAX   # charge
        else:
            p_energy = 0.0      # idle
        return np.array([p_energy, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


class TBxWithASPolicy:
    """
    Threshold-based arbitrage + AS when idle.

    Energy dispatch: same binary threshold logic as TBxEnergyOnly.
    AS when idle: bid RegUp + RegDown + NSRS at SoC-headroom-feasible capacity.
    RRS and ECRS excluded — near-zero in MILP due to short sustain vs. SoC limits.

    The harness project_action enforces:
      (a) individual AS SoC sustain limits
      (b) joint shared capacity |p_energy| + sum(c_as) ≤ P_max
    so this policy bids aggressively and lets the harness clip.
    """

    def __init__(self, p_low: float, p_high: float):
        self.p_low  = p_low
        self.p_high = p_high

    def reset(self) -> None:
        pass

    def __call__(self, obs: dict) -> np.ndarray:
        rt_lmp = float(obs["price_history"][-1, 0])
        # soc_frac is static_features[-1], convert to MWh
        soc = float(obs["static_features"][-1]) * E_MAX

        if rt_lmp > self.p_high:
            return np.array([P_MAX, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        elif rt_lmp < self.p_low:
            return np.array([-P_MAX, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Idle: bid max SoC-feasible AS capacity.
        # Harness project_action scales to joint P_max budget.
        soc_headroom_dch = max(0.0, soc - SOC_MIN)
        soc_headroom_chg = max(0.0, SOC_MAX - soc)

        # RegUp (discharge direction): c * sustain_h ≤ soc_headroom_dch * ETA
        c_regup = min(P_MAX, soc_headroom_dch * ETA / AS_SUSTAIN_H[0])
        # RegDn (charge direction): c * sustain_h * ETA ≤ soc_headroom_chg
        #   → c ≤ soc_headroom_chg / (ETA * sustain_h)  [harness project_action convention]
        c_regdn = min(P_MAX, soc_headroom_chg / (ETA * AS_SUSTAIN_H[1]))
        # RRS, ECRS excluded
        c_rrs   = 0.0
        c_ecrs  = 0.0
        # NSRS (discharge direction)
        c_nsrs  = min(P_MAX, soc_headroom_dch * ETA / AS_SUSTAIN_H[4])

        return np.array([0.0, c_regup, c_regdn, c_rrs, c_ecrs, c_nsrs], dtype=np.float32)
