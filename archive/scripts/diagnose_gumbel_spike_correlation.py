"""
Diagnostic: Do actor gradient mega-spikes correlate with Gumbel-Softmax
mode-distribution shifts or sharp annealing of tau_g?

Hypothesis: as tau_g anneals toward 0.1, Gumbel-STE produces near-one-hot
samples — small logit perturbations cause discrete mode flips, producing
large ∂Q/∂logit gradients that feed back into the actor.

Reads the Narnia tier1_seed42 training log and produces:
  1. Mega-spike list with surrounding mode distributions + tau_g
  2. Mode flip-rate correlation with grad_a_pre
  3. tau_g at each mega-spike (is annealing the trigger?)
  4. Plot: experiments/diagnose_gumbel_spike_correlation.png
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG_PATH = Path("checkpoints/tier1_seed42_preserved/tier1_seed42.log")
MEGA_THRESHOLD = 100.0

# Parse training log
rows = []
with open(LOG_PATH) as f:
    for line in f:
        if not line.startswith("Step"):
            continue
        try:
            step = int(re.search(r"Step\s+(\d+)", line).group(1))
            grad_a_pre = float(re.search(r"grad_a=([0-9.]+)→", line).group(1))
            grad_a_post = float(re.search(r"grad_a=[0-9.]+→([0-9.]+)", line).group(1))
            tau_g = float(re.search(r"tau_g=([0-9.]+)", line).group(1))
            # mode_env=[ch=30% dc=36% id=35%]
            env_ch = float(re.search(r"mode_env=\[ch=([0-9.]+)%", line).group(1))
            env_dc = float(re.search(r"mode_env=\[ch=[0-9.]+%\s+dc=([0-9.]+)%", line).group(1))
            env_id = float(re.search(r"mode_env=\[ch=[0-9.]+%\s+dc=[0-9.]+%\s+id=([0-9.]+)%", line).group(1))
            bat_ch = float(re.search(r"mode_batch=\[ch=([0-9.]+)%", line).group(1))
            bat_dc = float(re.search(r"mode_batch=\[ch=[0-9.]+%\s+dc=([0-9.]+)%", line).group(1))
            bat_id = float(re.search(r"mode_batch=\[ch=[0-9.]+%\s+dc=[0-9.]+%\s+id=([0-9.]+)%", line).group(1))
            q_maxabs = float(re.search(r"q_exp_maxabs=([0-9.]+)", line).group(1))
            q_mean = float(re.search(r"q_exp_mean=(-?[0-9.]+)", line).group(1))
            bin_argmax = float(re.search(r"bin_argmax=(-?[0-9.]+)", line).group(1))
            critic = float(re.search(r"critic=([0-9.]+)", line).group(1))
            rows.append({
                "step": step,
                "grad_a_pre": grad_a_pre,
                "grad_a_post": grad_a_post,
                "tau_g": tau_g,
                "env_ch": env_ch, "env_dc": env_dc, "env_id": env_id,
                "bat_ch": bat_ch, "bat_dc": bat_dc, "bat_id": bat_id,
                "q_maxabs": q_maxabs, "q_mean": q_mean,
                "bin_argmax": bin_argmax, "critic": critic,
            })
        except (AttributeError, ValueError):
            continue

steps = np.array([r["step"] for r in rows])
grad_a = np.array([r["grad_a_pre"] for r in rows])
tau_g = np.array([r["tau_g"] for r in rows])
env_dist = np.array([[r["env_ch"], r["env_dc"], r["env_id"]] for r in rows])
bat_dist = np.array([[r["bat_ch"], r["bat_dc"], r["bat_id"]] for r in rows])
q_maxabs = np.array([r["q_maxabs"] for r in rows])
q_mean = np.array([r["q_mean"] for r in rows])
bin_argmax = np.array([r["bin_argmax"] for r in rows])

# Mode shift = L2 diff to previous step's distribution (batch side — used in training)
bat_shift = np.zeros(len(bat_dist))
bat_shift[1:] = np.linalg.norm(bat_dist[1:] - bat_dist[:-1], axis=1)
env_shift = np.zeros(len(env_dist))
env_shift[1:] = np.linalg.norm(env_dist[1:] - env_dist[:-1], axis=1)

# Mode entropy (lower = more one-hot = Gumbel-STE edge effect)
def mode_entropy(d):
    p = d / 100.0
    p = np.clip(p, 1e-8, 1.0)
    return -np.sum(p * np.log(p), axis=1)

bat_ent = mode_entropy(bat_dist)
env_ent = mode_entropy(env_dist)

# === Analysis ===
print("=" * 72)
print("DIAGNOSTIC: Gumbel-STE mode-flip correlation with actor mega-spikes")
print("=" * 72)

mega_mask = grad_a > MEGA_THRESHOLD
normal_mask = grad_a <= MEGA_THRESHOLD
n_mega = int(mega_mask.sum())

print(f"\n1. Mega-spike inventory (grad_a_pre > {MEGA_THRESHOLD})")
print(f"   Count: {n_mega} / {len(grad_a)} steps ({n_mega/len(grad_a)*100:.1f}%)")

print(f"\n2. tau_g at mega-spikes vs normal steps:")
if n_mega > 0:
    print(f"   Mean tau_g at mega-spikes: {tau_g[mega_mask].mean():.3f}")
    print(f"   Mean tau_g at normal steps: {tau_g[normal_mask].mean():.3f}")
    print(f"   Min tau_g at mega-spikes:  {tau_g[mega_mask].min():.3f}")
    print(f"   (tau_g anneals 1.0 → 0.1; lower = more one-hot Gumbel samples)")

print(f"\n3. Mode-distribution shifts (L2 diff to previous step):")
print(f"   Batch mode-shift at mega-spikes: mean={bat_shift[mega_mask].mean():.2f}, max={bat_shift[mega_mask].max():.2f}")
print(f"   Batch mode-shift at normal steps: mean={bat_shift[normal_mask].mean():.2f}")
print(f"   Env   mode-shift at mega-spikes: mean={env_shift[mega_mask].mean():.2f}, max={env_shift[mega_mask].max():.2f}")
print(f"   Env   mode-shift at normal steps: mean={env_shift[normal_mask].mean():.2f}")

print(f"\n4. Mode entropy at mega-spikes (lower → more one-hot, closer to STE edge):")
print(f"   Batch entropy at mega-spikes: mean={bat_ent[mega_mask].mean():.3f}")
print(f"   Batch entropy at normal steps: mean={bat_ent[normal_mask].mean():.3f}")
print(f"   (uniform entropy = ln(3) = {np.log(3):.3f}; one-hot = 0)")

# Pearson correlations
def _corr(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    return np.corrcoef(x[mask], y[mask])[0, 1]

log_grad = np.log(np.clip(grad_a, 1e-3, None))
print(f"\n5. Pearson correlations with log(grad_a_pre):")
print(f"   tau_g         : r = {_corr(log_grad, tau_g):+.3f}  (neg = more spikes at low tau)")
print(f"   bat_mode_shift: r = {_corr(log_grad, bat_shift):+.3f}  (pos = spikes track mode flips)")
print(f"   env_mode_shift: r = {_corr(log_grad, env_shift):+.3f}")
print(f"   bat_entropy   : r = {_corr(log_grad, bat_ent):+.3f}  (neg = spikes when one-hot)")
print(f"   q_exp_maxabs  : r = {_corr(log_grad, q_maxabs):+.3f}")
print(f"   |bin_argmax|  : r = {_corr(log_grad, np.abs(bin_argmax)):+.3f}")

if n_mega > 0:
    print(f"\n6. Mega-spike details (surrounding mode distributions):")
    print(f"   {'Step':>7} | {'grad_a':>9} | {'tau_g':>5} | "
          f"{'bat_ch':>6} {'bat_dc':>6} {'bat_id':>6} | "
          f"{'shift':>5} | {'q_max':>6} | {'bin_am':>7}")
    print(f"   {'-'*7}-+-{'-'*9}-+-{'-'*5}-+-{'-'*20}-+-{'-'*5}-+-{'-'*6}-+-{'-'*7}")
    for i in np.where(mega_mask)[0]:
        print(f"   {steps[i]:>7} | {grad_a[i]:>9.1f} | {tau_g[i]:>5.2f} | "
              f"{bat_dist[i,0]:>5.1f}% {bat_dist[i,1]:>5.1f}% {bat_dist[i,2]:>5.1f}% | "
              f"{bat_shift[i]:>5.2f} | {q_maxabs[i]:>6.2f} | {bin_argmax[i]:>+7.2f}")

# === Plot ===
fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)

axes[0].semilogy(steps, np.clip(grad_a, 0.01, None), "b-", alpha=0.6, label="grad_a_pre")
axes[0].axhline(MEGA_THRESHOLD, color="red", linestyle="--", label=f"mega threshold ({MEGA_THRESHOLD})")
axes[0].scatter(steps[mega_mask], grad_a[mega_mask], color="red", s=20, zorder=5, label="mega-spike")
axes[0].set_ylabel("grad_a_pre (log)")
axes[0].legend(fontsize=8)
axes[0].set_title("Gumbel-STE / mega-spike correlation")

axes[1].plot(steps, tau_g, "g-", alpha=0.7, label="tau_g (Gumbel temperature)")
for s in steps[mega_mask]:
    axes[1].axvline(s, color="red", alpha=0.3, linewidth=0.5)
axes[1].set_ylabel("tau_g")
axes[1].legend(fontsize=8)

axes[2].plot(steps, bat_dist[:, 0], "b-", alpha=0.6, label="ch%")
axes[2].plot(steps, bat_dist[:, 1], "r-", alpha=0.6, label="dc%")
axes[2].plot(steps, bat_dist[:, 2], "g-", alpha=0.6, label="id%")
for s in steps[mega_mask]:
    axes[2].axvline(s, color="red", alpha=0.3, linewidth=0.5)
axes[2].set_ylabel("batch mode %")
axes[2].legend(fontsize=8)

axes[3].plot(steps, bat_shift, "m-", alpha=0.6, label="batch mode L2 shift")
axes[3].plot(steps, env_shift, "c-", alpha=0.4, label="env mode L2 shift")
for s in steps[mega_mask]:
    axes[3].axvline(s, color="red", alpha=0.3, linewidth=0.5)
axes[3].set_ylabel("mode shift (L2)")
axes[3].legend(fontsize=8)

axes[4].plot(steps, bat_ent, "b-", alpha=0.6, label="batch mode entropy")
axes[4].axhline(np.log(3), color="gray", linestyle="--", label=f"uniform (ln 3 = {np.log(3):.2f})")
axes[4].axhline(0.0, color="black", linestyle=":", label="one-hot (0)")
for s in steps[mega_mask]:
    axes[4].axvline(s, color="red", alpha=0.3, linewidth=0.5)
axes[4].set_ylabel("entropy")
axes[4].set_xlabel("Training step")
axes[4].legend(fontsize=8)

plt.tight_layout()
out_path = Path("experiments/diagnose_gumbel_spike_correlation.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=100)
print(f"\nPlot saved: {out_path}")

# === Verdict ===
print("\n" + "=" * 72)
print("VERDICT")
print("=" * 72)
r_tau = _corr(log_grad, tau_g)
r_shift = _corr(log_grad, bat_shift)
r_ent = _corr(log_grad, bat_ent)

if n_mega == 0:
    print("  No mega-spikes in log — nothing to correlate.")
else:
    print(f"  tau_g at mega-spikes:    {tau_g[mega_mask].mean():.3f} vs {tau_g[normal_mask].mean():.3f} normal")
    print(f"  bat_shift at mega-spikes: {bat_shift[mega_mask].mean():.2f} vs {bat_shift[normal_mask].mean():.2f} normal")
    print(f"  bat_entropy at mega-spikes: {bat_ent[mega_mask].mean():.3f} vs {bat_ent[normal_mask].mean():.3f} normal")
    print()
    if r_shift > 0.3 or bat_shift[mega_mask].mean() > 2 * bat_shift[normal_mask].mean():
        print("  Gumbel mode flips COVARY with mega-spikes — Gumbel-STE is a live suspect.")
        print("  Consider raising tau_g floor (e.g., 0.3) or switching to reparam-only softmax.")
    elif r_tau < -0.3 and tau_g[mega_mask].mean() < 0.5:
        print("  Mega-spikes cluster at LOW tau_g — annealing drives instability.")
        print("  Consider slowing anneal schedule or raising the tau_g floor.")
    else:
        print("  Gumbel-STE does NOT obviously drive mega-spikes.")
        print("  Primary mechanism is likely the actor→TTFE feedback loop through symexp.")
        print("  The stop-gradient fix (obs_encoded.detach() on actor path) should address it.")
