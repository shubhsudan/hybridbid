
# AI Usage Documentation

This document records the use of AI tools for code generation, code architecture decisions, and code debugging across the HybridBid project. Each entry follows the required format: tool used, request, what was generated, modifications made, and what was learned. A reflection section appears at the end.

## Tools Used

- **Claude Code (CLI agent)** code generation, refactoring, debugging assistance, repo restructuring.
- **Claude.ai (web interface, Opus 4.x)** code architecture decisions, root-cause diagnosis of training failures, methodology decisions that shaped code structure.

## Code Generation

### 1. Stage 1 Training Loop (SAC + TTFE)

- **Tool:** Claude Code
- **Request:** Implement the Stage 1 training loop adapted from Li et al. (2024, TempDRL), using SAC v2 with automatic entropy tuning and a Transformer Temporal Feature Extractor.
- **Generated:** Initial training script with SAC actor/critic networks, TTFE encoder, replay buffer, and training loop.
- **Modifications:** Multiple iterations across v1–v5.9.2. Significant rewrite at v5 (paper-spec reset) as deviations from Li et al. were diagnosed as the root cause of SoC drift. Kept gradient clipping (max_norm=1.0) as a non-paper addition. Spotted and flagged various issues with early implementations of the code that were leading to SoC drift and required significant debugging to identify and resolve.
- **Learned:** Every deviation from the paper specification required a major fix. Cross check specs at each stage and smoke test periodically

### 2. MILP Baselines (TBx, Perfect Foresight, Receding-Horizon Demonstrations)

- **Tool:** Claude Code
- **Request:** Implement three MILP baselines: (1) TBx energy-only with perfect price foresight, (2) Perfect Foresight oracle with full-horizon look-ahead, (3) receding-horizon MILP demonstration generator for offline RL training data.
- **Generated:** 3 baseline scripts.
- **Modifications:** No modifications needed, these were used as is. Only verified that they were correct.
- **Learned:** [FILL IN: e.g., HiGHS performance vs. ECOS, sensitivity to horizon length, how the demonstration dataset reward bug surfaced]

### 3. Offline RL Method Implementations (Cal-QL, Diffusion-QL, QDT)

- **Tool:** Claude Code
- **Request:** Implement Cal-QL, Diffusion-QL, and QDT for offline training on the MILP demonstration dataset, with shared evaluation harness integration.
- **Generated:** Three method implementations with twin-Q critics, calibration anchor (Cal-QL), diffusion policy parameterization (Diffusion-QL), and three-stage CQL→RTG-relabel→DT pipeline (QDT).
- **Modifications:** [FILL IN: hyperparameter tuning, custom logging, integration with eval harness, any deviations from reference papers]
- **Learned:** Implementation details that aren't in the original papers (calibration term binding behavior, diffusion β_Q sensitivity, RTG distribution properties under CQL relabeling) drove the failure modes identified in our paper. AI-generated reference implementations got us to working code fast, but the diagnostic work that revealed why each method failed required reading training-time logs by hand.

### 4. Evaluation Harness

- **Tool:** Claude Code
- **Request:** Build a frozen evaluation harness over a 54-day post-RTC+B test window enforcing continuous battery state-of-charge, applying silent feasibility projection to proposed actions, and computing physical-dollar revenue separated by all-days, ex-Fern, and Fern-only.
- **Generated:** Harness with eval-harness validation canary (MILP-replay → \$58.40/kW-yr) confirming consistency across runs.
- **Modifications:** [FILL IN]
- **Learned:** The silent feasibility projection — necessary at deployment — turned out to be the central diagnostic for the action-space training-deployment mismatch failure mode in offline RL. A design choice for one purpose surfaced a structural failure for another. Building diagnostic invariants (the MILP-replay canary returning \$58.40/kW-yr in every consistent eval) into the harness from the start was the single most useful pattern in the codebase.

### 5. ERCOT Data Pipeline

- **Tool:** Claude Code (initial scrapers); manual debugging
- **Request:** Build data ingestion for ERCOT public API across RT LMP, RT MCPC (5 products), DAM SPP, DAM AS clearing prices, and system variables (load, wind, solar forecasts).
- **Generated:** Initial scraper using the gridstatus library.
- **Modifications:** Substantial. The gridstatus scraper was largely broken following ERCOT's CSV→XML migration. Replaced with direct ErcotAPI calls. RT LMP required `get_lmp_by_settlement_point` (NP6-788-CD) for true 5-minute resolution rather than 15-minute bulk files. RT SCED MCPC required `NP6-332-CD` via the data API endpoint, not the archive. ECRS data has NaN before June 2023 and required handling. Wind/solar forecasts required deduplication by latest publish time.
- **Learned:** AI-generated scrapers based on outdated library documentation can produce silently broken pipelines. A 429 rate-limit error was initially misread as "no data exists" — would have silently dropped ~170 days had it not been spot-checked. Validating output against a known reference (ERCOT's web UI, an independently-fetched single day) was the only reliable way to catch this class of bug.

### 6. Repository Restructuring

- **Tool:** Claude Code
- **Request:** Restructure the repository to meet FOML submission requirements (`src/`, `notebooks/`, `models/`, `configs/`, `data/`, `requirements.txt`, `README.md`), preserving git history.
- **Generated:** [FILL IN after Claude Code session: inspection report, proposed move plan, executed moves on `repo-restructure` branch]
- **Modifications:** [FILL IN]
- **Learned:** [FILL IN]

## Code Architecture & Design Decisions

These entries cover Claude.ai conversations that shaped what code got written, even where the conversation itself was about architecture rather than line-by-line code generation.

### 7. Two-Stage Architecture Design

- **Tool:** Claude.ai
- **Request:** Adapt TempDRL (SAC + TTFE) to ERCOT's post-RTC+B market break, designing a pretrain→finetune system that retains pre-RTC+B knowledge while adapting to the new joint-clearing structure.
- **Generated:** Two-stage design that informed code structure Stage 1 energy-only pretrain (1D action space, replay buffer 1M, batch size 256); Stage 2 fine-tune with 6D action space (replay buffer 30–50k, batch size 128); progressive TTFE unfreezing at 10× lower LR per ULMFiT; fresh critic re-initialization; partial actor initialization (energy from Stage 1, AS dimensions near-zero). Each decision translated directly into code architecture (separate config files per stage, weight-loading utilities, frozen-layer parameter groups in the optimizer).
- **Modifications:** Refined through multiple iterations as Stage 1 instability emerged. Eventually pivoted to offline RL on post-RTC+B data when Stage 1 didn't produce a deployable checkpoint — the existing code structure (separate Stage 2 entry point, MILP demonstration data loader) made the pivot mechanical rather than requiring a rewrite.
- **Learned:** Architectural decisions made in conversation benefit from explicit go/no-go checkpoints before code is written. Designing for the pivot (separate stage entry points, decoupled data loaders) before knowing whether the pivot would happen kept the codebase flexible at low cost.

### 8. Paper-Spec Reset (v5)

- **Tool:** Claude.ai
- **Request:** Diagnose why v1–v4 implementations were producing SoC-pinning and mode collapse despite multiple compensatory fixes in code.
- **Generated:** Root-cause analysis identifying that the cascade of code-level deviations from Li et al. (continuous-only action space, reward scaling, price normalization, alpha floor) were each masking symptoms of an upstream issue. Recommendation: code reset to paper specification (Gumbel-Softmax 3-class mode + continuous magnitude in the actor head, EMA arbitrage bonus τ=0.9 β=10 in the reward function, episode termination penalty), keeping only gradient clipping as a non-paper addition. This translated into a substantial code rewrite, not just a config change.
- **Modifications:** Implemented the reset (v5) and verified 89 tests passing. Subsequently identified mode collapse in v5.1 traceable to SAC v2's learned alpha being pulled down by the continuous magnitude component — a separate structural issue requiring a different code change (fixed alpha) rather than tuning.
- **Learned:** When patches accumulate in code, the right move is often to revert to the reference and re-derive each addition. The diagnostic work of separating "compensatory" from "necessary" additions is the only way to escape the patch cascade.

### 9. Stage 2 AS Revenue Decoupling

- **Tool:** Claude.ai
- **Request:** Reconcile Li et al.'s binary mode formulation (which ties AS revenue to active mode in code) with ERCOT's ADER/ESR framework (which allows AS availability payments while idle).
- **Generated:** Analysis showing the paper's binary mode could not be applied directly in our reward function code; AS revenue computation must be decoupled from the action mode in Stage 2.
- **Modifications:** [FILL IN: was this implemented in the reward function, or deferred to future work? If implemented, what did the code change look like?]
- **Learned:** Adapting a published method to a different market structure requires reading the paper's assumptions carefully — not just transcribing the algorithm into code. The mismatch between Li et al.'s binary mode and ERCOT's AS framework was a code-level concern (how the reward function computes AS revenue) but only surfaceable through careful reading.

## Code Debugging & Diagnosis

### 10. Reward Formula Bug Diagnosis

- **Tool:** Claude.ai
- **Request:** Investigate ~120× reward inflation in v5.1 smoke test (critic loss spiked to 135M).
- **Generated:** Diagnosis identifying two bugs in the reward function code: (1) missing Δt = 5/60 time-step scaling factor, and (2) physical MW values used instead of per-unit (p.u.) in reward computation.
- **Modifications:** Applied both fixes in the reward function; verified critic loss returned to expected magnitude (80.16). Added a unit test asserting the reward magnitude against the MILP-replay canary.
- **Learned:** Reward formula precision matters disproportionately. Verifying reward magnitude against an independent calculation (the MILP-replay canary at \$58.40/kW-yr) caught this before training had progressed. Build the invariant into the test suite the first time you write the reward function.

### 11. Stage 1 Failure Mode Analysis

- **Tool:** Claude.ai
- **Request:** Characterize Stage 1 training failures across implementations and identify whether the difficulty was implementation-specific or structural in the code.
- **Generated:** Analysis of two distinct failure mechanisms — alpha collapse under SAC v2 (Implementation A v1–v5, Implementation B Plans A–B) and critic-instability cascade under fixed-α SAC v1 (Implementation B Plan C). Onset comparable across both (~115k–120k steps), suggesting the failure was not specific to a particular implementation choice.
- **Modifications:** [FILL IN: how this analysis informed the v5.9.2 code attempt — alpha cap, idle-action-logit penalty — and the eventual decision to pivot to offline RL]
- **Learned:** When two independent implementations fail at comparable scales via mechanistically distinct paths, the failure is more likely structural than implementation-specific. This framing changed how the code investigation was scoped — from "find the bug" to "characterize the regime."

### 12. Offline RL Failure Mode Analysis

- **Tool:** Claude.ai
- **Request:** Diagnose why Cal-QL, Diffusion-QL, and QDT each failed on the MILP demonstration dataset, and identify common vs. method-specific mechanisms in the code paths.
- **Generated:** Three-mechanism analysis — Cal-QL calibration deactivation at step ~13k followed by action-space training-deployment mismatch (Q-extrapolation into the infeasible region of the action space); Diffusion-QL Q-divergence at steps 29–31k surfacing the same mismatch without the calibration anchor's delaying effect; QDT Stage 2 RTG bimodality (a different mechanism rooted in the dataset's reward distribution rather than the offline RL algorithm).
- **Modifications:** [FILL IN: any code-level investigation, e.g., the bootstrap-target trace for Cal-QL that ended up in the paper's appendix, instrumentation added to log infeasibility ratios, AS scaling factors]
- **Learned:** The action-space mechanism — invisible to standard offline RL diagnostics that focus on Q-magnitudes — became the central methodological finding. Identifying it required reading deployment-time projection logs, not just training-time loss curves. The instrumentation that surfaced it was added to the eval harness specifically because the diagnostic conversation suggested where to look.

### 13. Gumbel Temperature Annealing Misread

- **Tool:** Claude.ai
- **Request:** Investigate whether Gumbel temperature of 0.282 at step 400 in a 500-step smoke test indicated an annealing schedule bug in the actor code.
- **Generated:** Resolution showing the production annealing references `config.total_steps` (500k); 0.282 at step 400 of 500k is the expected schedule output and not a bug.
- **Modifications:** None needed — the smoke test parameters were correct. Added a comment in the actor code clarifying that smoke test temperature trajectories will not match production trajectories.
- **Learned:** Don't misread annealing schedules. Smoke tests use shortened total_steps for speed; the temperature trajectory in a smoke run does not match the production trajectory. Worth a code comment if it might confuse a future reader.

## Reflection

### Where AI tools were most helpful for code

- **First-draft scaffolding.** Implementations of Cal-QL, Diffusion-QL, QDT, the TTFE, and the MILP baselines all started from AI-generated scaffolds. The first 60–70% of each was generated; the remaining 30–40% was the meaningful work — integration with our environment and reward, tuning to our data scale, fixing subtle bugs that AI-generated code tends to introduce.
- **Diagnostic conversations.** When training failed (alpha collapse, critic-instability cascade, Cal-QL calibration deactivation, Diffusion-QL Q-divergence), having a conversation partner to walk through training logs, propose hypotheses, and rule out compensatory fixes was substantially faster than working alone. The Stage 1 failure mode analysis and the action-space training-deployment mismatch are both products of this kind of dialogue.
- **Refactoring under structure.** Repository restructuring, separating Stage 1 and Stage 2 entry points, extracting hyperparameter blocks into configs — mechanical work where AI tools execute reliably given clear instructions and inspection-first protocols.
- **Resisting reward hacking.** Several times the AI suggestion was to add a compensatory mechanism (reward scaling, alpha floor, action penalty) when the right move was to revert and find the root cause. Pushing back on these suggestions — and having the AI then reason about why the deeper issue was real — was useful, but required me to recognize the pattern.

### Where AI tools were not helpful for code

- **Out-of-distribution data work.** ERCOT's API documentation is stale in many places; AI suggestions based on outdated library docs produced silently broken pipelines. Validating each scraper output against the source was the only reliable approach.
- **Subtle bugs in generated code.** The Δt scaling factor and MW vs. p.u. confusion both came from AI-generated code and survived multiple AI-assisted reviews. AI tools were poor at flagging when their own generated code had subtle bugs. External validation — via the MILP-replay canary, by spot-checking against known references, by running smoke tests — caught what the AI didn't.
- **Definitive method recommendations.** When asked "should we use Cal-QL or Diffusion-QL," AI tools could enumerate trade-offs but couldn't replace the empirical work of running both. The value was in framing the comparison and structuring the experiments, not in answering the comparison.
- **Domain-specific market knowledge.** ERCOT's ADER/ESR framework, AS sustain duration requirements, and the structural change introduced by RTC+B required reading source documents directly. AI summaries often glossed over the implementation details that turned out to matter for the reward function and constraint code.

### How AI-generated code was verified

- **Smoke tests.** Every significant code change was run through smoke tests before a full training run. Smoke tests caught most reward function and dimension mismatches early.
- **Eval-harness canary.** The MILP-replay baseline returned \$58.40/kW-yr in every consistent eval. A different number meant the harness was broken before any policy result could be trusted. This invariant caught several silent regressions in the eval code.
- **Independent reward computation.** Recomputed rewards verified against an independently-computed physical-dollar baseline within 1% tolerance.
- **Cross-codebase agreement.** Stage 1 failure analysis used two independent implementations (Implementation A and B by different team members). Mechanism-level agreement across both was treated as stronger evidence than agreement within either alone.
- **Paper-spec reference.** When in doubt, the paper-spec (Li et al. 2024, arXiv:2402.19110) was the source of truth, not the AI-generated implementation. Several bugs were caught by re-reading the paper rather than re-reading the code.
- **Manual log inspection.** Training logs (alpha trajectories, critic loss curves, action distributions, infeasibility ratios) were read by hand. AI tools were good at proposing what to look for but did not reliably catch anomalies on their own.

### What I'd do differently

[FILL IN: a few prompts to consider:
- Earlier paper-spec reset rather than accumulating compensatory patches in v2–v4?
- More aggressive validation of AI-generated data pipelines from day one?
- Different choice of when to delegate to Claude Code vs. discuss in Claude.ai?
- Building eval-harness invariants (like the MILP-replay canary) into the test suite before writing the training code rather than after?
- Anything about the team workflow with two independent codebases?]
