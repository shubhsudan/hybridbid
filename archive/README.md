# Archive

Historical artifacts from v1–v5.9.2 development and the offline RL sprint (Apr 2026). None of these are part of the final reported results. Preserved for reproducibility and audit; not needed to run the project.

## Contents

| Directory | What's in it |
|---|---|
| `sprint_docs/` | Sprint investigation markdown files: timezone audit, price reconciliation, eval harness validation, MILP gap analysis, reward convention, per-machine recon docs. Written during the Apr 25-29, 2026 offline RL sprint. |
| `eval_sweeps/` | Shell scripts used to launch evaluation sweeps across Stage 1/2 SAC checkpoints (v5.8–v5.9.2 era). Superseded by the T-60 eval harness in `src/evaluation/eval_t60.py`. |
| `scripts/` | Diagnostic Python scripts from the Stage 2 SAC / Tier 2c training period: Gumbel spike correlation, Q-value saturation, replay violation analysis, phase2c publication charts, and LinkedIn visualization code. |
| `experiments/` | `prepare_v1.py` — the original eval harness before the CT-realignment rewrite. Superseded by `src/evaluation/eval_t60.py`. |
| `checkpoints/` | All intermediate checkpoint generations from Stage 1 (v4, v5.9.1, v5.9.2), Stage 2 (v2, v3a), Tier 2c (seed42 full run + smoke), ablation (pricenorm-only), and preserved peak records. Filesystem only — not git-tracked (large binaries). Active finals are in `models/`. |
| `CLAUDE_hybridbid_v51.md` | Claude Code instructions written for the HybridBid v5.1 work on `main`. Not relevant to this branch. |
