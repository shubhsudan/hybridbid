# Cal-QL Smoke Results
**Steps:** 5000
**Mode:** offline, smoke
**Wall time:** 162.3s (32.5ms/step)
**Smoke result:** PASS

## Smoke pass criteria
| Criterion | Value | Status |
|-----------|-------|--------|
| Q_max bounded (<50k) | 3235.96 | ✓ |
| Q_mean > 0 | 288.71 | ✓ |
| CQL term bounded | [-648.08, -68.99] | ✓ |
| Action dist. non-degenerate | — | ✓ |

## Q-value statistics (val NPZ, deterministic policy)
- **Q_max_smoke: 3235.96**
- Q_mean:  288.71
- Q_min:   -2169.19
- Q_p50:   288.92
- Q_p90:   458.85

## Kill threshold recommendation for 25k–50k run
*(V_beh max = $11,952; correctly-learning Q must reach spike-state range)*
- Q_max_smoke = **3235.96**
- Recommended kill guard: **4× = 12944** (borderline — inspect 4× breach but also watch absolute 50k threshold)
- Mode: `4x_inspect`
- To apply: update `kill_q_max_multiplier` in config.yaml (4×) or treat 50k absolute as hard guard and use 4× as alert-only.

## Calibration term (CQL term, last 500 steps avg)
- Mean: -202.300
- Range: [-648.080, -68.989]

## Action distribution — global (val NPZ, p.u.)
- **p_energy** [ok]: mean=0.844  std=0.493  p5=-0.942  p95=0.997  frac_zero=0.000
- **c_regup** [ok]: mean=0.877  std=0.210  p5=0.249  p95=0.981  frac_zero=0.000
- **c_regdn** [ok]: mean=0.193  std=0.274  p5=0.009  p95=0.893  frac_zero=0.000
- **c_rrs** [ok]: mean=0.092  std=0.191  p5=0.012  p95=0.619  frac_zero=0.005
- **c_ecrs** [ok]: mean=0.069  std=0.176  p5=0.009  p95=0.551  frac_zero=0.006
- **c_nsrs** [ok]: mean=0.904  std=0.193  p5=0.369  p95=0.985  frac_zero=0.002

## Action distribution — negative-V_beh slice (2123 states, 11.9% of val)
*(States with V_behavior < 0 — no calibration floor; QDT's failure zone in miniature)*
- **p_energy** [ok]: mean=0.977 (global 0.844, drift=+0.133)  std=0.078  frac_zero=0.000
- **c_regup** [ok]: mean=0.890 (global 0.877, drift=+0.013)  std=0.161  frac_zero=0.000
- **c_regdn** [DRIFT]: mean=0.386 (global 0.193, drift=+0.193)  std=0.338  frac_zero=0.000
- **c_rrs** [ok]: mean=0.214 (global 0.092, drift=+0.122)  std=0.292  frac_zero=0.000
- **c_ecrs** [ok]: mean=0.187 (global 0.069, drift=+0.118)  std=0.288  frac_zero=0.000
- **c_nsrs** [ok]: mean=0.939 (global 0.904, drift=+0.035)  std=0.100  frac_zero=0.000

## Training loss (last 500 steps avg)
- TD loss: 9186.8763
- CQL loss: -121.3799
- Actor loss: -367.8347
- log_pi: 13.789

## Checkpoint
- `/home/stu9/s11/km5503/hybridbid/checkpoints/sprint/cal_ql/calql_step5000.pt`

## Next step
Karthik reviews. If PASS, set kill threshold per recommendation above, then:
```
CUDA_VISIBLE_DEVICES=<gpu> python -m methods.cal_ql.train_offline --mode full --gpu 0
```
Halts at step 25k (sys.exit 0). Resume with `--resume checkpoints/sprint/cal_ql/calql_step25000.pt`.
