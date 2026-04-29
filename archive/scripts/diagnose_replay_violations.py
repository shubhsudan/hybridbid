"""
Diagnostic: why do 99.5% of MILP-replay trajectories terminate on SoC violation?

Part 1: MILP SoC distribution (raw traces, no replay) — is MILP itself running at the floor?
Part 2: Replay first 100 days — pre-violation SoC, triggering atoms, step-in-day.
Part 3: Case study — first terminating day, SoC trajectory vs MILP's stored SoC.

Adapted to the project's actual env API (ERCOTBatteryEnv, dict obs, 4-D action,
5-tuple step return, info["soc_violated"]).
"""
import numpy as np
import pandas as pd

from src.env.ercot_env import ERCOTBatteryEnv, MODE_CHARGE, MODE_DISCHARGE, MODE_IDLE
from src.models.networks import TIER2A_ACTION_LEVELS, tier2a_idx_to_env_action

INPUT_NPZ = "data/expert_trajectories/receding_horizon_train.npz"
DATA_DIR = "data/processed"
DATE_RANGE = ("2020-01-01", "2023-12-31")
P_MAX = 10.0
STEPS_PER_DAY = 288

# Atoms in MW (same as TIER2A_ACTION_LEVELS × P_MAX)
ATOMS = np.array(TIER2A_ACTION_LEVELS, dtype=np.float64) * P_MAX   # [-10, -20/3, -10/3, 0, 10/3, 20/3, 10]
N_ATOMS = len(ATOMS)


def raw_to_signed_mw(mode: int, magnitude: float) -> float:
    """Map (mode ∈ {0=charge, 1=discharge, 2=idle}, magnitude ∈ [0,1]) → signed MW."""
    if mode == MODE_CHARGE:
        return -float(magnitude) * P_MAX
    elif mode == MODE_DISCHARGE:
        return +float(magnitude) * P_MAX
    else:
        return 0.0


def snap_to_atom_mw(signed_mw: float):
    idx = int(np.argmin(np.abs(ATOMS - signed_mw)))
    return idx, float(ATOMS[idx])


# ─────────────────────────────────────────────────────────────────────
# Load MILP raw trace
# ─────────────────────────────────────────────────────────────────────
raw = np.load(INPUT_NPZ)
milp_ts   = pd.to_datetime(raw["timestamps"], utc=True)
milp_mode = raw["modes"]
milp_mag  = raw["magnitudes"]
milp_socs = raw["socs"]
N_milp = len(milp_socs)

# ─────────────────────────────────────────────────────────────────────
# PART 1: MILP SoC distribution (raw traces, no replay)
# ─────────────────────────────────────────────────────────────────────
print("=== PART 1: MILP SoC distribution (raw trajectory, no replay) ===")
print(f"  N steps: {N_milp:,}")
print(f"  Mean: {milp_socs.mean():.2f}  Std: {milp_socs.std():.2f}")
print(f"  Min:  {milp_socs.min():.2f}   Max: {milp_socs.max():.2f}")
print(f"  Time at floor   (SoC ≤ 3.0):  {(milp_socs <= 3.0).mean():.1%}")
print(f"  Time at ceiling (SoC ≥ 17.0): {(milp_socs >= 17.0).mean():.1%}")
print(f"  Time mid-range  (4.0 ≤ SoC ≤ 16.0): {((milp_socs >= 4.0) & (milp_socs <= 16.0)).mean():.1%}")

hist, bin_edges = np.histogram(milp_socs, bins=np.arange(2, 19, 1))
print(f"\n  SoC histogram (1-MWh bins):")
for i, count in enumerate(hist):
    pct = count / N_milp * 100
    bar = "#" * int(pct)
    print(f"    [{bin_edges[i]:2.0f}, {bin_edges[i+1]:2.0f}): {count:7d} ({pct:5.1f}%) {bar}")

# ─────────────────────────────────────────────────────────────────────
# Build env and day-start lookup
# ─────────────────────────────────────────────────────────────────────
env = ERCOTBatteryEnv(data_dir=DATA_DIR, mode="energy_only", seq_len=32, date_range=DATE_RANGE)
env_ts_series = pd.Series(np.arange(len(env.timestamps)), index=env.timestamps)

# MILP timestamp → MILP index
milp_ts_series = pd.Series(np.arange(N_milp), index=milp_ts)

# Pre-compute MILP EMA (matches env + MILP formula)
EMA_TAU = 0.9
milp_ema = np.empty(N_milp, dtype=np.float64)
ema = float(raw["rt_lmp"][0])
milp_ema[0] = ema
for i in range(1, N_milp):
    ema = EMA_TAU * ema + (1.0 - EMA_TAU) * float(raw["rt_lmp"][i])
    milp_ema[i] = ema

# ─────────────────────────────────────────────────────────────────────
# PART 2: Replay first 100 days, record pre-violation state
# ─────────────────────────────────────────────────────────────────────
N_DAYS_SAMPLE = 100
pre_violation_socs = []
pre_violation_actions = []
pre_violation_steps = []
days_completed = 0
days_terminated = 0
days_skipped = 0

for k in range(min(N_DAYS_SAMPLE, len(env.day_starts))):
    day_start_env_idx = env.day_starts[k]
    day_env_indices = np.arange(day_start_env_idx, day_start_env_idx + STEPS_PER_DAY)
    if day_env_indices[-1] >= len(env.timestamps):
        days_skipped += 1
        continue
    day_ts = env.timestamps[day_env_indices]

    milp_indices_raw = milp_ts_series.reindex(day_ts).to_numpy()
    if np.any(pd.isna(milp_indices_raw)):
        days_skipped += 1
        continue
    milp_indices = milp_indices_raw.astype(np.int64)

    env.current_day_idx = k
    obs, _ = env.reset(options={"day_idx": k})
    env.soc       = float(milp_socs[milp_indices[0]])
    env.ema_price = float(milp_ema[milp_indices[0]])

    prev_soc = env.soc
    terminated_here = False

    for step_in_day in range(STEPS_PER_DAY):
        milp_i = milp_indices[step_in_day]
        signed_mw = raw_to_signed_mw(int(milp_mode[milp_i]), float(milp_mag[milp_i]))
        atom_idx, _snapped_mw = snap_to_atom_mw(signed_mw)
        env_action = tier2a_idx_to_env_action(atom_idx)

        _next_obs, _reward, terminated, truncated, info = env.step(env_action)

        if terminated and info.get("soc_violated", False):
            pre_violation_socs.append(prev_soc)
            pre_violation_actions.append(atom_idx)
            pre_violation_steps.append(step_in_day)
            days_terminated += 1
            terminated_here = True
            break

        prev_soc = env.soc

    if not terminated_here:
        days_completed += 1

print(f"\n=== PART 2: Replay violation analysis (first {N_DAYS_SAMPLE} days) ===")
print(f"  Terminated on violation: {days_terminated}/{N_DAYS_SAMPLE - days_skipped}")
print(f"  Completed full day:      {days_completed}/{N_DAYS_SAMPLE - days_skipped}")
print(f"  Skipped (coverage gap):  {days_skipped}")

if pre_violation_socs:
    arr = np.array(pre_violation_socs)
    print(f"\n  Pre-violation SoC (step before violation fired):")
    print(f"    Mean: {arr.mean():.2f}  Std: {arr.std():.2f}")
    print(f"    Min:  {arr.min():.2f}   Max: {arr.max():.2f}")
    print(f"    %iles (10/25/50/75/90): {np.percentile(arr, [10, 25, 50, 75, 90]).round(2).tolist()}")
    print(f"    %% near floor   (≤ 3.0):        {(arr <= 3.0).mean():.1%}")
    print(f"    %% near ceiling (≥ 17.0):       {(arr >= 17.0).mean():.1%}")
    print(f"    %% mid-range    (4.0 ≤ ≤16.0):  {((arr >= 4.0) & (arr <= 16.0)).mean():.1%}")

    action_counts = np.bincount(pre_violation_actions, minlength=N_ATOMS)
    print(f"\n  Violation-triggering atom distribution:")
    for i, count in enumerate(action_counts):
        if count > 0:
            atom_val = ATOMS[i]
            direction = "discharge" if atom_val > 0 else ("charge" if atom_val < 0 else "idle")
            print(f"    atom {i} ({atom_val:+5.2f} MW, {direction}): {count} ({count/days_terminated:.1%})")

    step_arr = np.array(pre_violation_steps)
    print(f"\n  Step-within-day when violation fired:")
    print(f"    Mean: {step_arr.mean():.1f}  Median: {np.median(step_arr):.0f}")
    print(f"    %iles (10/25/50/75/90): {np.percentile(step_arr, [10, 25, 50, 75, 90]).tolist()}")

# ─────────────────────────────────────────────────────────────────────
# PART 3: Case study — first terminating day, full trajectory
# ─────────────────────────────────────────────────────────────────────
if days_terminated > 0:
    print(f"\n=== PART 3: Case study — first terminating day in 100-sample ===")
    for k in range(min(N_DAYS_SAMPLE, len(env.day_starts))):
        day_start_env_idx = env.day_starts[k]
        day_env_indices = np.arange(day_start_env_idx, day_start_env_idx + STEPS_PER_DAY)
        if day_env_indices[-1] >= len(env.timestamps):
            continue
        day_ts = env.timestamps[day_env_indices]
        milp_indices_raw = milp_ts_series.reindex(day_ts).to_numpy()
        if np.any(pd.isna(milp_indices_raw)):
            continue
        milp_indices = milp_indices_raw.astype(np.int64)

        env.current_day_idx = k
        obs, _ = env.reset(options={"day_idx": k})
        env.soc       = float(milp_socs[milp_indices[0]])
        env.ema_price = float(milp_ema[milp_indices[0]])

        soc_trajectory = [env.soc]
        action_trajectory = []
        terminated = False
        for step_in_day in range(STEPS_PER_DAY):
            milp_i = milp_indices[step_in_day]
            signed_mw = raw_to_signed_mw(int(milp_mode[milp_i]), float(milp_mag[milp_i]))
            atom_idx, snapped_mw = snap_to_atom_mw(signed_mw)
            env_action = tier2a_idx_to_env_action(atom_idx)
            _obs, _r, terminated, _trunc, info = env.step(env_action)
            soc_trajectory.append(env.soc)
            action_trajectory.append((signed_mw, snapped_mw, atom_idx, int(milp_mode[milp_i]), float(milp_mag[milp_i])))
            if terminated:
                print(f"  Day k={k} (env_idx={day_start_env_idx}, date={day_ts[0].date()}) terminated at step {step_in_day}")
                print(f"    info.soc_violated = {info.get('soc_violated')}")
                print(f"    info.p_net = {info.get('p_net'):+.3f}, info.mode = {info.get('mode')}")
                lo = max(0, step_in_day - 6)
                hi = step_in_day + 1
                print(f"\n  Last {hi - lo} steps before termination:")
                print(f"    {'step':>4}  {'milp_mode':>9} {'milp_mag':>8}  {'milp_mw':>8}  {'snap_mw':>8} {'atom':>4}  {'env_soc_bef':>11}  {'env_soc_aft':>11}  {'milp_soc':>9}")
                for s in range(lo, hi):
                    mc, sp, ai, mm, mg = action_trajectory[s]
                    milp_soc_here = float(milp_socs[milp_indices[s]])
                    print(f"    {s:>4d}  {mm:>9d} {mg:>8.3f}  {mc:>+8.3f}  {sp:>+8.3f} {ai:>4d}  {soc_trajectory[s]:>11.3f}  {soc_trajectory[s+1]:>11.3f}  {milp_soc_here:>9.3f}")
                break
        if terminated:
            # Also compare SoC on the *first* 5 steps of this day (did divergence start pre-violation?)
            print(f"\n  First 5 steps of this day (did replay SoC track MILP SoC before the violation?):")
            print(f"    {'step':>4}  {'atom_mw':>8}  {'env_soc_aft':>11}  {'milp_soc':>9}  {'diff':>8}")
            for s in range(min(5, len(action_trajectory))):
                _, sp, ai, _, _ = action_trajectory[s]
                milp_soc_here = float(milp_socs[milp_indices[s]])
                diff = soc_trajectory[s + 1] - milp_soc_here
                print(f"    {s:>4d}  {sp:>+8.3f}  {soc_trajectory[s+1]:>11.3f}  {milp_soc_here:>9.3f}  {diff:>+8.3f}")
            break
