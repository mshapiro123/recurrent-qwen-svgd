# Paper Two Phase 3 Handoff: P3.3 i1 Aim-Only Re-Read

**Date:** 2026-08-12  
**Status:** both registered seeds complete; A100 released; strategy review required  
**Result:** safe, instrument-valid, replicated plateau; P3.4 threshold not met

## 0. Executive verdict

P3.3 i1 did not convert the P3.3 middle-band result into a P3.4-qualifying result. The registered all-row BF16 baseline was `pi_dir = 14.9010%`. After 1,000 aim-focused steps, seed 0 reached 15.0883% (+0.1873 percentage points) and seed 1 reached 14.8208% (-0.0803 points). Their mean was 14.9545%, only +0.0535 points over baseline and 10.0455 points below the 25% P3.4 threshold.

This is a clean negative for the narrow intervention tested, not a broken run. Both seeds completed all 20 looks without warnings or hard stops. The frozen selector was bit-identical at every audit, token retention was 100%, measured collateral was zero, instrumentation was non-perturbing, and zero-loop identity was exact. The aim objective received more than 99.998% of post-clip gradient norm at every reported endpoint, yet fixed-audit aim loss moved only 0.05% in seed 0 and 0.20% in seed 1. Both receipts independently classify the curve as `capacity_next_tail_plateau`.

The immediate implication is: do not buy more duration under this trainable-set contract. If the program continues, the next intervention should change aim-path capacity or representation while preserving the now-validated frozen selector and safety instrumentation.

## 1. Question and rationale

P3.3 established a live, selective causal channel, but its canonical all-row direction-capture rate remained in the middle band at 14.901%. The gate had already converged to high recall, and the deployed gate preferentially opened where the trained direction worked. The remaining uncertainty was whether the direction itself had been underfunded during joint training.

i1 isolated that question. It froze the selector and trained only `bridge.output_projection.weight`, forcing the aim objective to receive at least 70% of post-clip gradient share. If direction quality was merely starved, `pi_dir` should rise toward or through 25%. If it remained flat while the loss was fully funded, the limitation would move from duration or objective competition toward representational capacity or objective alignment.

## 2. Locked design

- Two inherited P3.3 lineages, seeds 0 and 1.
- Exactly 1,000 additional steps and 20 scheduled looks, every 50 steps.
- Trainable set: `bridge.output_projection.weight` only, 114,688 parameters.
- Frozen selector invariant: gate logits, hidden/control/scratch gate weights, gate probabilities, and open-set membership must remain bit-identical to the inherited P3.3 endpoint.
- Optimizer: AdamW, learning rate `3e-4`, betas `(0.9, 0.999)`, weight decay `0.01`, 100 warmup steps.
- Batch size 128, with 32 positives and 96 negatives.
- Aim gradient-share floor: 70% post-clip. Preservation remains present but cannot dominate.
- Operating gate ceiling: 0.02. The clamp bound at initialization on loop 1 and not on loops 2-4, as disclosed rather than failed.
- Canonical read: all 4,096 positive-audit rows, BF16-consistent reader, prompt-level bootstrap over 746 documents.
- Registered comparison: P3.3 all-row `pi_dir = 0.14901016586409846`.
- P3.4 launch threshold: `pi_dir >= 0.25`.
- No task-level capability scoring; this remains a token-level causal falsifier.

## 3. Results

| Metric | Seed 0 | Seed 1 | Combined reading |
|---|---:|---:|---:|
| Final `pi_dir` | 15.0883% | 14.8208% | Mean 14.9545% |
| Change from 14.9010% baseline | +0.1873 pp | -0.0803 pp | +0.0535 pp mean |
| Final 95% prompt-bootstrap CI | 13.4529-16.7218% | 13.1453-16.4889% | Both include baseline |
| Final `pi_dep` | 19.4553% | 19.7674% | Gated headroom remains |
| Initial to final aim loss | 0.928329 to 0.927869 | 0.931652 to 0.929784 | -0.05% and -0.20% |
| Convergence classification | capacity-next plateau | capacity-next plateau | Replicated |
| Final aim gradient share | 99.9987% | 99.9984% | Objective was not starved |
| Selector invariant | bit-identical, 21 audits | bit-identical, 21 audits | Passed |
| Retention | 1,024/1,024 at every look | 1,024/1,024 at every look | 100% throughout |
| Collateral `chi` | 0 at every look | 0 at every look | 0/42 audits |
| Warnings / stops | none / none | none / none | Clean completion |
| Result band | middle-band strategy review | middle-band strategy review | P3.4 not authorized |

The trajectory figure is `docs/figures/p33_i1_re_read_curves_20260812.svg` (PNG companion beside it). It shows `pi_dir` oscillating tightly around the registered baseline while aim loss changes only marginally.

## 4. Contract and instrument checks

All checks needed to interpret the negative passed:

- The preflight summary, guardrail calibration, positive audit, negative audit, and retention panel matched their locked hashes.
- Zero-loop hidden states and logits were bit-exact.
- Observatory instrumentation preserved RNG, loss, and metrics bit-exactly.
- The selector's deployed and unclamped values, as well as open-set membership, were bit-identical across all 21 audits per seed.
- Both seeds completed exactly 20 post-initialization looks.
- No optimizer state was inherited from the source endpoints.
- Final bridge gradient norms were below the 0.5 clip ceiling, so clipping did not create the plateau.
- The canonical reader retained all 4,096 rows with a 100% reader match rate.

The result therefore cannot reasonably be attributed to a dead write path, selector drift, a reader mismatch, a safety stop, preservation loss competition, or clipping.

## 5. Interpretation

### What the run supports

1. **The narrow aim-only update is insufficient.** Moving only the 114,688-parameter output projection for 1,000 steps does not materially improve direction capture.
2. **Duration is not the next lever under this contract.** Both fixed-audit aim-loss curves are flat enough to trigger the preregistered `capacity_next_tail_plateau` classification.
3. **The selector and preservation story remain strong.** High selector recall, zero measured collateral, and perfect retention survive across two seeds and all looks.
4. **Useful gated headroom remains uncaptured.** Final `pi_dep` exceeds final `pi_dir` by 4.37 points in seed 0 and 4.95 points in seed 1. The channel remains selective, but this output projection cannot consistently turn that selection into the desired token change.
5. **The optimizer did optimize the intended loss.** Aim held virtually all gradient share. The negative is therefore more informative than the original joint-training ambiguity.

### What the run does not support

- No claim that the broader sidecar architecture cannot work.
- No claim that 14.95% is a universal upper bound.
- No claim of improved `pi_dir`; the tiny mean increase is not seed-consistent and both confidence intervals cover the baseline.
- No task-level quality or capability claim.
- No claim that preservation is universally perfect; it is perfect on the registered 1,024-position high-confidence panel at the tested 0.02 operating ceiling.

## 6. Recommended decision

Do not open P3.4 under the current charter because the 25% threshold was not met. Bank i1 as `SAFE_PLATEAU` for the narrow output-projection trainable set.

If strategy wants one further development cycle, change capacity rather than duration:

1. Run a zero-training checkpoint analysis first: singular-value movement of the output projection, update rank, update-to-weight norm, per-horizon and teachability-decile gains/losses, and whether improvements concentrate in a stable row subset across seeds.
2. If that analysis shows the projection is rank- or subspace-limited, authorize a bounded aim-path capacity arm: for example, a small low-rank residual aim adapter or the immediately preceding aim transformation, while retaining the frozen selector and 0.02 gate ceiling.
3. Keep the same all-row BF16 `pi_dir` reader, retention panel, and causal controls so the only changed variable is aim-path capacity.
4. Do not extend the current output-projection arm beyond 1,000 steps. Its own registered convergence classifier says duration is not the next explanation.

## 7. Questions for strategy review

1. Does the replicated `capacity_next_tail_plateau` authorize a bounded capacity arm, or should the 25% miss close this interface entirely?
2. If capacity work continues, should the next trainable set be a low-rank residual aim adapter, the preceding aim MLP/projection, or both as separately attributable arms?
3. Should the remaining `pi_dep - pi_dir` gap be treated as evidence of exploitable selector enrichment, or only as a diagnostic until a capacity arm converts it?
4. Is a paired row-level overlap analysis across seeds required before choosing the next capacity location? Recommendation: yes; it is free and may distinguish a shared representational ceiling from seed-specific noise.
5. Should the 100% retention and zero-collateral result be elevated as a positive bounded finding in Paper Two, even though the efficacy threshold failed? Recommendation: yes, with the population and gate-ceiling scope explicit.

## 8. Execution and cost notes

- Code and lock lineage: lock commit `195bb41c`; executed runner lineage through `02fbb79d`.
- A named 40 GB A100 was sufficient after removing unused teacher-cache transport.
- The 693 MB preflight JSONL package was transported as two hash-verified compressed parts totaling about 116 MB, then restored byte-identically to Pharma Drive.
- Seed 1 reused seed 0's staged lattice and model caches on the same VM, avoiding a second large Drive transfer.
- The named A100 ran for approximately 38 minutes and was explicitly terminated after both seeds completed.
- A short CPU-only session pulled the receipt summaries locally and was also explicitly terminated.
- Final CLI state: no active Colab sessions.

## 9. Canonical artifacts

Drive root:

`/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/stage5_paper2_phase3_p33_i1_20260812/`

Primary receipts:

- `receipts/seed_0/summary.json`
- `receipts/seed_1/summary.json`
- `receipts/seed_0/status.json`
- `receipts/seed_1/status.json`

Final checkpoints:

- Seed 0: `private/seed_0/resume.pt`, SHA-256 `01c804bc69d35a01730fff236cf5a8d974899d2e4de7e15b92a227b2a9ce5d88`
- Seed 1: `private/seed_1/resume.pt`, SHA-256 `2ed3296f510a6c3a66c451051ecbe2284de03b35dde4052827174a66a10c1d4a`

Local derived artifacts:

- `outputs/stage5/stage5_paper2_phase3_p33_i1_20260812/combined_analysis.json`
- `docs/figures/p33_i1_re_read_curves_20260812.svg`
- `docs/figures/p33_i1_re_read_curves_20260812.png`

## 10. Plain-language summary

We gave the direction learner the cleanest possible chance: its own training run, nearly all of the gradient, a frozen gate that already knew where to act, and two independent seeds. It trained safely, but it did not get better in a reproducible way. One seed rose by two-tenths of a point, the other fell by one-tenth, and both stayed near 15%, far below the 25% threshold. The useful conclusion is not that the whole architecture is dead. It is that this single output projection has reached its practical ceiling under the tested objective. More of the same training is not justified; a next attempt would need more or different aim-path capacity while preserving the selector and safety machinery that worked.
