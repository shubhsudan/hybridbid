# Reward Convention Investigation
**Session:** cc-rl-narnia  
**Date:** 2026-04-25  
**Branch:** sprint-offline-rl  
**Triggered by:** Phase 0 recon flag — stored rewards ≠ physical revenue, ~2× gap

---

## Step 1: Reward computation in MILP (`src/data/postbreak_milp.py`)

### 1a. Energy reward (lines 331–335)

```python
# src/data/postbreak_milp.py:331–335
energy_term = energy_mag_pu * rt_lmp * (v_dch * ETA - v_ch / ETA) * DT
timing_bonus = (
    BETA_ARB * energy_mag_pu * price_dev
    * (I_dch * v_dch * ETA + I_ch * v_ch / ETA) * DT
)
```

Where (lines 315–329):
```python
# src/data/postbreak_milp.py:315–319
def compute_step_reward(
    energy_mag_pu: float,   # |p_energy| / P_max ∈ [0,1]
    mode:          int,     # 0=charge, 1=discharge, 2=idle
    rt_lmp:        float,
    ema:           float,   # UPDATED EMA (already includes current rt_lmp)
    c_as_mw:       np.ndarray,  # (5,) physical MW
```

```python
# src/data/postbreak_milp.py:90–93
EMA_TAU   = 0.9
BETA_ARB  = 10.0
DT        = 5.0 / 60.0
```

`energy_mag_pu = |p_energy_mw| / P_MAX` (lines 414 or 418). **The MW denominator is never removed.** The energy term is per-unit (missing `× P_MAX = 10`). Physical energy revenue would be `p_energy_mw × rt_lmp × DT = energy_mag_pu × P_MAX × rt_lmp × DT`.

`BETA_ARB = 10.0` amplifies the timing bonus by ×10. Both `energy_term` and `timing_bonus` are in p.u. scale.

### 1b. AS reward (line 336)

```python
# src/data/postbreak_milp.py:336
as_rev = float(np.sum(c_as_mw * rt_mcpc) * DT)
```

Where `c_as_mw` is passed as `c_as_t = c_as_arr[t]` (line 408), and `c_as_arr` is the raw MILP solution in **physical MW** (line 376: `c_as_arr = np.clip(result["c_as"], 0, P_MAX)`).

AS revenue is in **physical $**: `[MW] × [$/MWh] × [h] = [$]`.

### 1c. Stored per-interval reward (line 435–438)

```python
# src/data/postbreak_milp.py:435–438
rew_buf[t] = compute_step_reward(
    energy_mag_pu, mode, rt_lmp_t, ema,
    c_as_mw=c_as_t, rt_mcpc=rt_mcpc_t,
)
```

**Return at line 338:**
```python
return float(energy_term + timing_bonus + as_rev)
```

**Stored reward = energy_term(p.u.) + timing_bonus(p.u.) + as_rev(physical $).**

### 1d. Physical revenue for diagnostics (lines 462–464 — NOT stored in NPZ)

```python
# src/data/postbreak_milp.py:462–464
energy_rev_day = float(np.sum((p_dch_arr - p_ch_arr) * day_p[:, 0]) * DT)
as_rev_day     = float(np.sum(c_as_arr * day_p[:, 1:6]) * DT)
total_rev_day  = energy_rev_day + as_rev_day
```

This is the separate physical-$ computation used only for logging/diagnostics. **It does not flow into the NPZ.**

---

## Step 2: Numerical verification (train split, 19,584 transitions)

### 2A: Sum of stored rewards as-is

```
sum(stored_rewards) = 63,327.41
N = 19,584, mean = 3.23/step
```

### 2B: Physical reconstruction from actions + price_history

For every transition: `p_energy_mw = actions[:,0] × P_MAX`, `c_as_mw = actions[:,1:] × P_MAX`, prices = `price_history[:,-1,:]` (last row of rolling window = current step).

```
sum(physical_energy_rev) = 96,212.93
sum(physical_as_rev)     = 20,298.31
sum(physical_total)      = 116,511.25   ← matches $116,669 MILP reference (rounding)

sum(stored_rewards)      = 63,327.41
stored minus physical_AS =  43,029.10   ← this is energy_pu + timing_bonus_pu
physical_energy / P_MAX  =   9,621.29   ← would be energy_pu alone
implied timing_bonus_pu  =  33,407.81   ← timing bonus is DOMINANT (53% of stored total)
```

**Stored reward breakdown (train split):**

| Component | Sum | % of stored total |
|---|---|---|
| Energy p.u. (no timing bonus) | ~9,621 | 15% |
| Timing bonus p.u. (BETA_ARB×10) | ~33,408 | **53%** |
| AS revenue (physical $) | ~20,298 | 32% |
| **Total stored** | **63,327** | 100% |

Compare: physical revenue = $116,511 (energy $96,213 + AS $20,298).

### 2C: Per-interval cross-check (10 random intervals)

```
idx      stored_r   phy_direct    ratio     mode
------------------------------------------------------------
1682       8.5833       8.5833    1.000     idle    ← idle: pure AS, exact match
1747       0.9167       0.9167    1.000     idle    ← idle: exact match
1844       0.5417       0.5417    1.000     idle    ← idle: exact match
3945       0.6083       0.6083    1.000     idle    ← idle: exact match
8478      -1.6905      -9.1750    0.184      chg    ← charge: ratio ≠ 1
8592      21.0488      54.7583    0.384      dch    ← discharge: ratio ≠ 1
12814      0.1750       0.1750    1.000     idle    ← idle: exact match
13655      0.4417       0.4417    1.000     idle    ← idle: exact match
15150      4.1128       3.6514    1.126      chg    ← timing bonus dominates
16811     15.0605     -20.4833   -0.735      chg    ← SIGN FLIP: stored +15 vs physical -$20
```

**Critical observation — sign flips:** At idx 16811, the MILP is charging (negative physical revenue: spending $20 to buy energy), but stored reward = +15.06. The timing bonus overwhelmed the energy cost: the MILP charged when current price was below the long-run EMA, which Li et al. Eq.26 rewards heavily. The stored reward says "this was good"; physical revenue says "this cost $20."

**Idle-step match:** For all 4 idle steps (zero or near-zero energy action), stored = physical exactly. These are pure-AS intervals: AS revenue is in physical $ on both sides.

---

## Step 3: Eval harness reward computation (`experiments/prepare_postbreak.py`)

The harness computes revenue at lines 329–330:

```python
# experiments/prepare_postbreak.py:329–330
energy_rev = p_energy * rt_lmp * DT          # p_energy in physical MW
as_rev = c_as * rt_mcpc * DT                 # c_as in physical MW, element-wise
```

**Confirmed:**
1. Revenue is computed from policy-returned physical MW actions × market prices — NOT from stored NPZ rewards.
2. Revenue is in physical $ throughout.
3. The stored `rewards` field of the NPZ is never read by the eval harness.

The `MILP_T60_USD = 90_814.0` constant in the harness is a reference value from a prior validation run (noted as "MILP replay validation target"). It is physical $, computed from MILP actions × T-60 prices. It does not come from summing NPZ rewards.

---

## Step 4: $96,169 / $90,814 reconciliation

| Number | Source | Method |
|---|---|---|
| $96,169 | `MILP_REPLAY_GAP_VERIFIED.md` "MILP reference" | `sum(diag["total_rev"])` = `sum(energy_rev_day + as_rev_day)` for 54 T-60 days — physical $ from lines 462–464 |
| $96,001 | Same report "sum of planned revenues (no projection)" | physical $ from actions before clipping |
| $86,394 | Same report "harness replay" | physical $ from eval harness, post-projection |
| $90,814 | `MILP_T60_USD` constant in harness | prior CT-aligned validation run, physical $ |

**None of these derive from summing stored NPZ rewards.** All are physical $. There is no inconsistency in yesterday's validation. The $63,327 stored-rewards sum for the train split is a completely separate quantity that was never used in the MILP-replay gap calculation.

---

## Diagnosis

**Diagnosis: (b) Intentional mixed convention with two compounding problems.**

The stored reward is Li et al. Eq.26 faithfully implemented:
- Energy and timing bonus use `energy_mag_pu` (p.u.) as the scale — intentional per the paper's formulation for grid-scale normalization.
- AS revenue is added as physical $ — an inconsistency introduced because Li et al.'s reward formula doesn't include AS products (ERCOT-specific extension).

This is intentional design, not a code bug. The eval harness is unaffected (computes its own physical $ independently).

**However, "consistent signal" is not sufficient for Q-learning.** Two problems compound:

### Problem 1: AS gradient bias (factor ×10 per MW)

For the same MW of capacity:
- 1 MW discharge at $100/MWh: stored energy contribution ≈ `(1/P_MAX) × 100 × ETA × DT = 0.079 per p.u.`
- 1 MW regup at $10/MWh: stored AS contribution = `1.0 × 10 × DT = 0.833 physical $`

The AS gradient is **10.5× larger** for the same MW at a 10:1 price ratio. Q-learning will systematically over-value AS actions relative to energy actions.

### Problem 2: Timing bonus causes reward sign flips

The timing bonus (BETA_ARB=10) can exceed the energy cost and flip the reward sign. As shown at idx 16811: physical revenue = -$20 (charging costs money), stored reward = +15 (timing bonus says good timing). Q-targets computed from stored rewards will have the wrong sign for a non-trivial fraction of charge transitions.

In aggregate: timing bonus accounts for 53% of total stored reward signal and is the dominant component. Q-values will primarily encode Li et al.'s timing-optimality signal rather than physical revenue.

---

## Recommendation

**Do NOT regen trajectories (too costly for sprint timeline).** Use data-load-time reward recomputation.

**At training time, discard stored `rewards` and recompute physical revenue:**

```python
# In data_loader.py for both methods:
P_MAX = 10.0
DT = 5.0 / 60.0

p_energy_mw = data['actions'][:, 0:1] * P_MAX       # (N, 1) signed MW
c_as_mw     = data['actions'][:, 1:]  * P_MAX       # (N, 5) MW

rt_lmp  = data['price_history'][:, -1, 0:1]         # (N, 1) current step LMP
rt_mcpc = data['price_history'][:, -1, 1:6]         # (N, 5) current step AS MCPC

energy_rev = p_energy_mw * rt_lmp * DT              # (N, 1)
as_rev     = (c_as_mw * rt_mcpc).sum(axis=1, keepdims=True) * DT  # (N, 1)
rewards    = (energy_rev + as_rev).squeeze()         # (N,) physical $
```

**Why this works:**
- `price_history[:, -1, :]` is the current step's prices (rolling window, last row = current)
- Gives true physical revenue matching the eval harness formula exactly
- Energy and AS on the same scale → Q-gradient correctly weighted
- No trajectory regen needed
- Drops timing bonus (reward shaping artifact; not needed for offline RL on expert demonstrations — timing info is already in state via price_history)

**Mark in CLOSEOUT docs:** Trajectory regen with corrected reward formula is the correct long-term fix. Sprint expedient: data-load-time recomputation.

**Action needed from Karthik:**
1. Approve the data-load-time recomputation approach (or direct regen if preferred)
2. Confirm bc-baselines session uses physical $ rewards if it has a BC training loop on stored data (harness itself is unaffected; only applies to any offline training)
3. Green-light Phase 1 after this decision

**Do NOT proceed to Phase 1 until reward convention is resolved.**
