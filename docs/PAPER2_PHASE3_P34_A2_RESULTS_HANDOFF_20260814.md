# Handoff: P3.4 A2 Completion — Replicated DEV Gain Below the Confirmation Trigger

**Date:** 2026-08-14
**Program:** Paper Two, Phase 3, P3.4 A2 repaired campaign
**Status:** both registered main seeds complete at step 4,000; DEV-only analysis banked; CONFIRM and EVAL-E remain sealed
**Bottom line:** the causal sidecar improved the registered 1,024-row DEV panel in both seeds, by `+5` and `+10` rows. The mean `+7.5/1,024` (`+0.732` points) is a replicated positive exploratory signal, but it is below the locked Trigger-B threshold of mean `+10/1,024`. P3.6 confirmation is therefore not eligible. Under the ratified rule, effect growth returns to the P3.5 lever queue.

## 0. Decision summary

1. **The repaired training campaign completed.** Both seeds reached all 4,000 steps. Neither stopped on task safety, loss allocation, non-finite values, lineage drift, or infrastructure failure.
2. **The task effect replicated in sign but not at the required magnitude.** Seed 0 finished `507/1,024` versus `502/1,024` for the registered base (`+5`). Seed 1 finished `512/1,024` (`+10`). The two-seed mean was `+7.5`, below Trigger B (`+10`) and far below Trigger A (`+22`).
3. **The registered endpoint, not the best intermediate look, governs.** Seed 0 briefly reached `+16` at step 3,800 and returned to `+5` at step 4,000. Seed 1 peaked at `+11` at steps 3,400–3,600 and finished `+10`. This non-monotonicity is why no best-checkpoint substitution is allowed.
4. **The mechanism remained causal and selective.** Final forced-direction capture `pi_dir` was `14.75%` and `15.50%`; deployed capture `pi_dep` was `15.70%` and `27.59%`. Collateral `chi` was exactly zero in all registered audits.
5. **The repaired loss-share controller succeeded.** Seed 0 recorded 39 pass windows and one isolated KL warning. Seed 1 recorded 34 passes, three one-window warnings, and three reversible demotions; it recovered and its final five windows all passed. There were no four-window stop events.
6. **CONFIRM remains sealed.** `confirm_scored=false` and `eval_e_scored=false` in both final summaries. The next scientific decision is a P3.5 lever, not a post-hoc confirmation attempt.

## 1. Question and rationale

P3.4 asks whether the direction-supervised latent sidecar can convert token-level causal correction into better answers on real tasks under the registered task-inference graph. Earlier P3.4 attempts showed a small positive signal but stopped because the original loss-allocation controller could not sustain the preregistered per-loss shares. A2 repaired the controller without changing the task panel, targets, estimator, inference graph, or scientific thresholds.

The campaign was deliberately sized as exploration. Its purpose was to estimate task conversion and decide whether the sealed confirmation exam had been earned. The ratified eligibility rule required:

- both seed endpoints positive;
- each seed's 512-row target-half endpoint non-negative;
- Trigger A: mean two-seed pooled gain at least `+22/1,024`; or
- Trigger B: mean gain at least `+10/1,024`, followed by a separately preregistered graded confirmation claim.

Below `+10/1,024`, the ratified amendment explicitly keeps CONFIRM sealed and returns the program to P3.5.

## 2. Locked design

### 2.1 Lineage and initialization

- Campaign of record: full P3.4 sidecar, main arm, seeds 0 and 1.
- Seed 0 continued from the last healthy registered P3.4 checkpoint at step 400, SHA `56dfa30d19166dfd3a788e2e6f68e0613f366e55601b5d690b087e1a3edb9230`.
- Seed 1 continued from the last healthy checkpoint at step 1,000, SHA `2ff122cdc1d3c3208c9eb367345f360a31676f0f821c311ed98f6cc690c8e66f`.
- Frozen lineage digest before and after: `e8b287fcf1eba10ad02b7f1659b349fc0d437b9ef53c52b7d14aaf34a11eb3ed7`, equal in both seeds.
- Lock SHA: `4dc4cf82856552375e976bf1b4813c29f6ca8d1eb0b7c1084c6843b40ccc19c3`.
- A2 amendment SHA: `d697c752cd7100d50cada00161c9e25c92489f389db94a1b691c0245f2114945`.
- Ratification r2: Drive `1PrUIh20K9ZQf37PWQuDOZ68lZq12eFoc`, SHA `b713ae8540e8d83a82f6e7363eccbf27a5fb348162e427b78fef1c2880413ee5`.
- Implementation commit: `d8ba75a2` (`Ratify and implement P3.4 dynamic share recovery`).

### 2.2 Training and task inference

- AdamW, learning rate `3e-4`, weight decay `0.01`, betas `(0.9, 0.999)`, 100 warmup steps.
- Batch size 128; sampled flow depth distribution `[0.10, 0.20, 0.30, 0.40]` over depths 1–4.
- 4,000 total steps and exactly 20 registered looks at 200-step cadence, counting each inherited continuation endpoint as its original look.
- Task graph: fresh scratch state per emitted token, four flow loops, greedy decoding, position zero closed, no cross-token persistence, draft head inactive for scoring.
- DEV panel: 1,024 rows, panel SHA `3e6c62ac4ef36a22eeba961e5d4d84c3403fd55aa837d32746eb8f35d8fe3163`.
- Target half: ARC-Challenge, GSM8K, MBPP, 512 rows. Floor half: ARC-Easy, MMLU, Tier-1, 512 rows.
- Schedule SHA: `c44ac8d66a8e51a445c4b0fc8e3b5e044ccf0445f75afb044858333b06e9a4eef`.

### 2.3 Dynamic loss allocation and safety

The A2 controller adjusted scalar objective weights every 100 steps against the actual registered depth-mixture estimator. Lower share floors were KL `35%`, aim `15%`, CE `10%`, and gate `3%`; preservation had an upper bound of `25%`. One miss was observed, two consecutive misses demoted the annealing rung, and four consecutive misses stopped the run. Rung demotion was reversible.

Task Tier-S/W rules were evaluated at each look. Tier-2 causal audits ran at looks 5, 10, 15, and 20, with the inherited source audit representing the look already spent before continuation where applicable.

## 3. Primary results

| Read | Seed 0 | Seed 1 | Joint reading |
|---|---:|---:|---|
| Registered base | 502/1,024 (49.02%) | 502/1,024 (49.02%) | fixed panel reference |
| A2 endpoint | 507/1,024 (49.51%) | 512/1,024 (50.00%) | mean 509.5/1,024 |
| Net endpoint gain | +5 rows (+0.488 points) | +10 rows (+0.977 points) | **mean +7.5 rows (+0.732 points)** |
| Fixes / regressions | 50 / 45 | 54 / 44 | positive in both seeds |
| Paired one-sided sign test | p = 0.341 | p = 0.182 | exploratory, individually unresolved |
| Best observed look | +16 at step 3,800 | +11 at steps 3,400–3,600 | not used for eligibility |
| Trigger A (+22 mean) | no | no | **not met** |
| Trigger B (+10 mean) | seed endpoint equals +10 | — | **not met: mean +7.5** |

Using the previously banked total teacher–base gap of 292 rows, the endpoint closes about `1.71%` of the pooled gap in seed 0 and `3.42%` in seed 1; the two-seed mean is `2.57%`. These are descriptive DEV ratios, not confirmation estimates.

### 3.1 Seed-0 target/floor decomposition

The final seed-0 private row receipt was recovered before runtime shutdown:

| Group | Rows | Base | A2 | Delta |
|---|---:|---:|---:|---:|
| Target half | 512 | 176 | 182 | **+6** |
| Floor half | 512 | 326 | 325 | **−1** |

Battery deltas were ARC-Challenge `+1`, GSM8K `−2`, MBPP `+7`, ARC-Easy `+2`, MMLU `−1`, and Tier-1 `−2`. The target effect is therefore concentrated in MBPP, with a small GSM8K regression.

The seed-1 pooled endpoint and all mechanism/controller receipts are complete. Its private final row file was not copied by the runtime's post-run Drive sync because that runtime token lacked Drive write scope. The missing file prevents an exact seed-1 target/floor decomposition in this handoff. This does **not** change the registered decision because the pooled mean independently misses Trigger B. If P3.6 is reconsidered later, seed 1's target-half receipt must first be reconstructed from the saved step-4,000 evaluation checkpoint.

## 4. Mechanism and controller results

### 4.1 Causal capture

| Seed | Final pi_dir | Final pi_dep | Collateral chi | Gate recall | Gate precision |
|---|---:|---:|---:|---:|---:|
| 0 | 14.75% | 15.70% | 0.000% | 89.67% | 67.91% |
| 1 | 15.50% | 27.59% | 0.000% | 95.53% | 46.26% |

Seed 0's `pi_dir` stayed near 15% across all four audits and `pi_dep` stayed near 16%. Seed 1's `pi_dir` also remained near 15–16%, while deployed capture remained 27–28%. The important asymmetry is therefore selector/deployment behavior rather than a larger learned direction space. The seed-1 gate opens more broadly and less precisely, but preferentially enough to produce substantially higher deployed capture.

The final task reads were made at the rung active before the look-20 audit: seed 0 at rung 1 (`0.08` gate ceiling), seed 1 at rung 0 (`0.02`). Both post-audit resumable checkpoints end at rung 1 after the seed-1 controller re-earned advancement. This distinction matters for any P3.5 initialization.

### 4.2 Loss-share controller

| Read | Seed 0 | Seed 1 |
|---|---:|---:|
| 100-step windows | 40 | 40 |
| Pass | 39 | 34 |
| One-window warning | 1 | 3 |
| Reversible demotion | 0 | 3 |
| Stop | 0 | 0 |
| Failed loss labels | KL once | aim and gate, six reads each |
| Final five windows | all pass | all pass |

Seed 0 remained at rung 1 and had one isolated KL miss. Seed 1 demoted from rung 1 to 0 at steps 200, 2,200, and 3,200, then re-earned rung 1 at audit steps 2,000, 3,000, and 4,000. This is the reversible-tier behavior the amendment intended. The final seed-0 share vector was KL 47.23%, aim 22.28%, CE 14.92%, gate 4.23%, preserve 11.35%. Seed 1 finished at 53.80%, 24.46%, 15.83%, 4.04%, and 1.87%. Both satisfy every bound.

### 4.3 Guardrails and lineage

- No Tier-S event in either seed.
- No Tier-W event in either seed.
- No non-finite loss or gradient.
- No frozen-lineage change.
- No collateral audit flip (`chi=0`) at any registered audit.
- No sealed CONFIRM or EVAL-E contact.
- The post-run Drive archive command retried with `ACCESS_TOKEN_SCOPE_INSUFFICIENT`; this occurred after training and scoring were complete. Authoritative checkpoints and receipts were exported directly through the Colab CLI and verified locally before both A100 sessions were released.

## 5. Interpretation

### 5.1 Supported

1. **Token-level causal correction converts into real-task answer gains at exploration grade.** Both independently trained seeds finish positive on the same frozen panel.
2. **The A2 optimization repair worked.** The campaign completed the budget, the controller recovered from transient allocation misses, and final shares satisfy all registered contracts.
3. **The sidecar remains causally live and highly selective.** Direction capture is stable near 15%, deployment can enrich it substantially, and registered collateral remains zero.
4. **The current task effect is small and noisy.** Endpoint gains are under one percentage point, individual paired tests are unresolved, and seed 0 varies from +1 to +16 late in training.
5. **More duration alone is not the obvious answer.** Both aim capture and task gain oscillate or plateau rather than showing a clean rising endpoint trend. The ratified amendment already states that effects below Trigger B move to P3.5 rather than reopening confirmation.

### 5.2 Not supported

- No confirmed capability improvement.
- No claim that the model is broadly better.
- No claim that Trigger B was approximately met; the registered rule uses the exact two-seed mean.
- No claim that seed 1 satisfies the target-half coherence gate until its private row receipt is reconstructed.
- No claim that zero collateral generalizes beyond the registered audit population and magnitudes.
- No claim that the slot-supervision or capacity alternatives have been falsified; they remain P3.5 candidates.

## 6. Recommended next decision

The locked reading is `below_trigger_b_confirm_remains_sealed`. Strategy should select one bounded P3.5 lever, with a cheap discriminating receipt before training.

My recommended order is:

1. **Oracle-direction refresh on the current endpoints.** This is the cheapest direct test of the charter's known staleness limitation. Recompute oracle directions on both final models and measure whether `pi_dir` rises under the same audit. If refresh materially raises available capture, train against refreshed directions before changing architecture.
2. **Cross-token persistence probe.** The current inference graph reinitializes scratch state for every emitted token. A read-only or minimally trained persistence probe can test whether carrying bounded state across generation improves GSM8K without sacrificing the zero-collateral pattern. This targets the sequence-level conversion bottleneck directly.
3. **Revisit the shelved slot arm if information remains thin.** The A_r pricing audit previously favored slot supervision over width. The repaired controller removes the confound that stopped its first 400-step run, so a matched repaired seed-0 continuation is now a cleaner test than the archived attempt.
4. **Flow unfreeze at 0.1× only after the above.** It expands adaptation risk and should follow a measured failure of refresh/persistence, not precede them.
5. **Capacity or multi-point injection last.** The prior A_r result found only 25–26% oracle energy in the learned readout span but indicated diffuse residual energy; widening without a localization result risks buying parameters rather than usable information.

The creativity-slot proposal is a **paired endpoint/rung probe**: evaluate both final seed checkpoints at fixed gate ceilings `0.02` and `0.08` on DEV, score-only and checkpoint-selection-barred. Seed 1 achieved the larger task gain under the lower active ceiling while seed 0 briefly peaked under the higher one. A fixed-ceiling cross-over read would separate learned-state quality from controller exposure before choosing persistence, slot supervision, or gate scheduling.

## 7. Questions for strategy

1. Authorize the fixed-ceiling paired DEV probe before selecting the P3.5 lever?
2. If seed 1 remains better at both ceilings, should oracle refresh precede persistence as recommended?
3. If seed 0 improves at `0.02` while seed 1 degrades at `0.08`, should per-seed or learned gate scheduling become the first P3.5 lever?
4. Is the missing seed-1 target-half decomposition worth one reconstruction pass now, or should it remain deferred because Trigger B already failed independently?
5. Should the repaired slot arm return immediately after the fixed-ceiling probe, or only if oracle refresh fails to raise `pi_dir`?

## 8. Plain-language summary

We trained two independent copies of the system to make small, targeted internal corrections while answering real questions. Both copies ended better than the same baseline: one fixed five more answers than it broke, and the other fixed ten more. That shows the mechanism is real and can help outside the token-level laboratory.

The improvement is still too small for the sealed confirmation exam. Our rule required an average gain of at least ten questions across the two runs; the observed average was seven and a half. We therefore do not spend the exam and do not claim a confirmed better model.

The training repair itself was successful. The optimizer kept all five learning objectives funded, recovered automatically when one run drifted, completed every step, preserved the frozen model, and caused no measured collateral failures. The next problem is not making training run. It is increasing how much useful correction reaches the answer. The most economical next tests are to refresh the correction directions on the newly trained models and to test whether bounded reasoning state should persist across generated tokens.

## 9. Canonical artifacts

### 9.1 Final checkpoints

- Seed 0 scored step-4,000 checkpoint SHA: `381955ec5b78d0a00883c29e9f940feac8cfc8665f7a3a4446c79734532f4ed7`.
- Seed 0 post-audit resumable checkpoint SHA: `56d4f0340eb6e291628756e506528382fbc205ace5a7f4ac1e850e7bf940705b`.
- Seed 1 scored step-4,000 checkpoint SHA: `97ad532a5bffd72b2563799047b517e531e00115793bf4808f060148dfffc1ec`.
- Seed 1 post-audit resumable checkpoint SHA: `bf29ca6945dffbef0517144b33eff1076cfb0c3ad568b1515416831e072a61f4`.

### 9.2 Local durable receipts

- `durable/p34_a2_seed0/final_manifest.json`
- `durable/p34_a2_seed0/final_receipts.tar.gz`, SHA `7d8c89debd959d97325a72589394fbc13d700ffcb9bd0ef27dcd58a04d03b376`
- `durable/p34_a2_seed0/final_resume.pt`
- `durable/p34_a2_seed1/final_manifest.json`
- `durable/p34_a2_seed1/final_receipts.tar.gz`, SHA `3c2d2694f0fb0c12fb47fe1dfa1d3177080032b547eb2ad80e40e58a41c27ea2`
- `durable/p34_a2_seed1/final_resume.pt`
- `artifacts/p34_a2_20260814/p34_a2_results_summary.json`
- `artifacts/p34_a2_20260814/p34_a2_results_figure.svg`
- `artifacts/p34_a2_20260814/p34_a2_results_figure.png`

Both Colab A100 sessions were released only after final checkpoint and receipt hashes matched their manifests. `colab sessions` returned no active sessions.
