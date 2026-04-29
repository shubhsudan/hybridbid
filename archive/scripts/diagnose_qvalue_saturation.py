"""
Diagnostic: Is q_exp_maxabs saturating against the HL-Gauss support bounds?

HL-Gauss support is [-20, 20] with 101 bins. If Q-values are pushing against
the support boundary, the critic loses gradient signal for high-reward states,
creating degenerate Q-surfaces that trigger actor gradient explosions.

Reads the Narnia tier1_seed42 training log and produces:
  1. Peak q_exp_maxabs and % of HL-Gauss support
  2. Critic loss trajectory analysis
  3. Mega-spike correlation with Q-value peaks
  4. Plot: experiments/diagnose_qvalue_saturation.png
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG_PATH = Path("checkpoints/tier1_seed42_preserved/tier1_seed42.log")
HLGAUSS_MAX = 20.0
HLGAUSS_BINS = 101

# Parse training log
rows = []
with open(LOG_PATH) as f:
    for line in f:
        if not line.startswith("Step"):
            continue
        try:
            step = int(re.search(r"Step\s+(\d+)", line).group(1))
            q_maxabs = float(re.search(r"q_exp_maxabs=([0-9.]+)", line).group(1))
            q_mean = float(re.search(r"q_exp_mean=(-?[0-9.]+)", line).group(1))
            # grad_a is logged as "grad_a=PRE→POST"
            grad_a_pre = float(re.search(r"grad_a=([0-9.]+)→", line).group(1))
            grad_t_attn = float(re.search(r"attn=([0-9.]+)\]", line).group(1))
            critic_loss = float(re.search(r"critic=([0-9.]+)", line).group(1))
            bin_ent = float(re.search(r"bin_ent=([0-9.]+)", line).group(1))
            bin_argmax = float(re.search(r"bin_argmax=(-?[0-9.]+)", line).group(1))
            rows.append({
                "step": step,
                "q_exp_maxabs": q_maxabs,
                "q_exp_mean": q_mean,
                "grad_a_pre": grad_a_pre,
                "grad_t_attn": grad_t_attn,
                "critic_loss": critic_loss,
                "bin_ent": bin_ent,
                "bin_argmax": bin_argmax,
            })
        except (AttributeError, ValueError):
            continue

steps = np.array([r["step"] for r in rows])
q_maxabs = np.array([r["q_exp_maxabs"] for r in rows])
q_mean = np.array([r["q_exp_mean"] for r in rows])
grad_a = np.array([r["grad_a_pre"] for r in rows])
grad_t = np.array([r["grad_t_attn"] for r in rows])
critic = np.array([r["critic_loss"] for r in rows])
bin_ent = np.array([r["bin_ent"] for r in rows])
bin_argmax = np.array([r["bin_argmax"] for r in rows])

# --- Analysis ---

print("=" * 70)
print("DIAGNOSTIC: Q-value saturation against HL-Gauss support")
print("=" * 70)

# 1. Peak q_exp_maxabs
peak_idx = np.argmax(q_maxabs)
q_peak = q_maxabs[peak_idx]
q_peak_step = steps[peak_idx]
saturation_ratio = q_peak / HLGAUSS_MAX

print(f"\n1. Peak q_exp_maxabs: {q_peak:.2f} at step {q_peak_step}")
print(f"   HL-Gauss support: [-{HLGAUSS_MAX}, {HLGAUSS_MAX}]")
print(f"   Saturation ratio: {saturation_ratio:.1%} of support")
print(f"   q_exp_maxabs > 10.0 count: {np.sum(q_maxabs > 10.0)} / {len(q_maxabs)} steps")
print(f"   q_exp_maxabs > 15.0 count: {np.sum(q_maxabs > 15.0)} / {len(q_maxabs)} steps")

# NOTE: q_exp_maxabs is in RAW (post-symexp) scale. The HL-Gauss support
# is in SYMLOG scale [-20, 20]. So we need to check what q_maxabs=14.9
# means in symlog space: symlog(14.9) = sign(14.9) * log(1 + |14.9|) = log(15.9) ≈ 2.77.
# That's well within [-20, 20]. The support is NOT saturating in symlog space.
print(f"\n   CRITICAL: q_exp_maxabs is in RAW (post-symexp) scale.")
print(f"   HL-Gauss support is in SYMLOG scale [-20, 20].")
print(f"   symlog({q_peak:.1f}) = {np.sign(q_peak) * np.log1p(abs(q_peak)):.2f}")
print(f"   q_exp_mean at peak step: {q_mean[peak_idx]:.2f} (symlog scale)")
print(f"   bin_argmax at peak step: {bin_argmax[peak_idx]:.2f} (symlog scale)")
print(f"   → Support utilization: {abs(bin_argmax[peak_idx]) / HLGAUSS_MAX:.1%} of support")

# 2. Critic loss trajectory
print(f"\n2. Critic loss trajectory:")
print(f"   Mean: {critic.mean():.4f}")
print(f"   Std:  {critic.std():.4f}")
print(f"   Min:  {critic.min():.4f} at step {steps[np.argmin(critic)]}")
print(f"   Max:  {critic.max():.4f} at step {steps[np.argmax(critic)]}")
# Check if flat
first_half = critic[:len(critic)//2].mean()
second_half = critic[len(critic)//2:].mean()
print(f"   First half mean: {first_half:.4f}")
print(f"   Second half mean: {second_half:.4f}")
print(f"   Drift: {second_half - first_half:+.4f}")

# 3. Mega-spike correlation
mega_mask = grad_a > 100
normal_mask = grad_a <= 100
n_mega = mega_mask.sum()
print(f"\n3. Mega-spike analysis (grad_a_pre > 100):")
print(f"   Count: {n_mega} / {len(grad_a)} steps ({n_mega/len(grad_a)*100:.1f}%)")
if n_mega > 0:
    print(f"   Mean q_exp_maxabs at mega-spikes: {q_maxabs[mega_mask].mean():.2f}")
    print(f"   Mean q_exp_maxabs at normal steps: {q_maxabs[normal_mask].mean():.2f}")
    print(f"   Mean q_exp_mean at mega-spikes:    {q_mean[mega_mask].mean():.2f} (symlog)")
    print(f"   Mean q_exp_mean at normal steps:   {q_mean[normal_mask].mean():.2f} (symlog)")
    print(f"\n   Mega-spike details:")
    print(f"   {'Step':>7} | {'q_maxabs':>8} | {'q_mean':>6} | {'grad_a':>10} | {'grad_t_attn':>11} | {'bin_argmax':>10}")
    print(f"   {'-'*7}-+-{'-'*8}-+-{'-'*6}-+-{'-'*10}-+-{'-'*11}-+-{'-'*10}")
    for i in np.where(mega_mask)[0]:
        print(f"   {steps[i]:>7} | {q_maxabs[i]:>8.1f} | {q_mean[i]:>6.2f} | {grad_a[i]:>10.1f} | {grad_t[i]:>11.1f} | {bin_argmax[i]:>10.2f}")

# 4. bin_argmax trajectory — where is the critic's predicted Q?
print(f"\n4. bin_argmax trajectory (symlog scale):")
print(f"   Start (steps 2-5k): {bin_argmax[:4].mean():.2f}")
print(f"   Mid (steps 20-30k): {bin_argmax[18:28].mean():.2f}")
print(f"   Late (steps 90-100k): mean={bin_argmax[-20:-10].mean():.2f}, max={bin_argmax[-20:-10].max():.2f}")
print(f"   Peak bin_argmax: {bin_argmax.max():.2f} at step {steps[np.argmax(bin_argmax)]}")
print(f"   → {bin_argmax.max():.2f} / {HLGAUSS_MAX:.0f} = {bin_argmax.max()/HLGAUSS_MAX:.1%} of support")

# 5. symexp gradient amplification at observed Q values
print(f"\n5. Symexp gradient amplification at observed Q values:")
# ∂symexp/∂q_symlog = exp(|q_symlog|) for |q_symlog| > 1
for q_sym in [1.0, 1.5, 2.0, 2.5, 3.0]:
    amp = np.exp(q_sym)
    raw = np.sign(q_sym) * (np.exp(abs(q_sym)) - 1)
    print(f"   q_symlog={q_sym:.1f} → raw={raw:.1f}, gradient amp={amp:.1f}×")

print(f"\n   Observed q_exp_mean range: {q_mean.min():.2f} to {q_mean.max():.2f} (symlog)")
print(f"   → gradient amp range: {np.exp(abs(q_mean.min())):.1f}× to {np.exp(abs(q_mean.max())):.1f}×")

# --- Plot ---
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

# Panel 1: Q-value trajectory
axes[0].plot(steps, q_maxabs, "b-", alpha=0.7, label="q_exp_maxabs (raw scale)")
axes[0].plot(steps, q_mean, "g-", alpha=0.7, label="q_exp_mean (symlog scale)")
axes[0].plot(steps, bin_argmax, "m-", alpha=0.7, label="bin_argmax (symlog scale)")
axes[0].axhline(HLGAUSS_MAX, color="red", linestyle="--", linewidth=2, label=f"HL-Gauss support ±{HLGAUSS_MAX}")
axes[0].axhline(HLGAUSS_MAX * 0.5, color="orange", linestyle=":", label="50% support")
axes[0].set_ylabel("Q-value")
axes[0].legend(fontsize=8)
axes[0].set_title(f"Q-value saturation check — peak raw {q_peak:.1f} @ step {q_peak_step} "
                   f"(symlog: {np.sign(q_peak)*np.log1p(abs(q_peak)):.2f}, "
                   f"support util: {abs(bin_argmax[peak_idx])/HLGAUSS_MAX:.1%})")

# Panel 2: Actor gradient (log scale)
axes[1].semilogy(steps, np.clip(grad_a, 0.01, None), "b-", alpha=0.7, label="grad_a (pre-clip)")
axes[1].semilogy(steps, np.clip(grad_t, 0.001, None), "r-", alpha=0.5, label="grad_t_attn")
axes[1].axhline(100, color="red", linestyle="--", label="mega-spike threshold")
axes[1].set_ylabel("Gradient (log)")
axes[1].legend(fontsize=8)

# Panel 3: Critic loss
axes[2].plot(steps, critic, "b-", alpha=0.7, label="critic_loss")
axes[2].set_ylabel("Critic loss")
axes[2].legend(fontsize=8)

# Panel 4: bin_ent
axes[3].plot(steps, bin_ent, "b-", alpha=0.7, label="bin_entropy")
axes[3].axhline(np.log(HLGAUSS_BINS), color="gray", linestyle="--", label=f"uniform entropy (log {HLGAUSS_BINS}={np.log(HLGAUSS_BINS):.2f})")
axes[3].set_ylabel("Bin entropy")
axes[3].set_xlabel("Training step")
axes[3].legend(fontsize=8)

plt.tight_layout()
out_path = Path("experiments/diagnose_qvalue_saturation.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=100)
print(f"\nPlot saved: {out_path}")

# --- Summary verdict ---
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
if bin_argmax.max() / HLGAUSS_MAX > 0.75:
    print("  HL-Gauss support is likely saturating. Widen support.")
elif bin_argmax.max() / HLGAUSS_MAX > 0.50:
    print("  HL-Gauss support is borderline. Check bin distribution at extremes.")
else:
    print("  HL-Gauss support is NOT saturating.")
    print("  Root cause is elsewhere — likely actor/TTFE gradient feedback loop.")
    print("  Symexp gradient amplification at q_symlog ≈ 2.5 is ~12×,")
    print("  which combined with ERCOT price spikes creates the mega-spike events.")
