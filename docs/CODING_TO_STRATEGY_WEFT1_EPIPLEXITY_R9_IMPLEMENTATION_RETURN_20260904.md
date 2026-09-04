# CODING → STRATEGY — WEFT-1 Epiplexity R9 implementation return

**Date:** 2026-09-04
**Status:** R9 verified and implemented where authority is complete; ECA-LATENT registered CPU campaign launched; SEAM-AUDIT pass held on one mathematical defect. Corpus P-A continues concurrently. No P-B, G-TOK, target-training, checkpoint, or sealed-data gate is minted here.

## 0. Outcome

The R9 authority chain is local, byte-exact, committed, and pushed. D-EP-1 now has an executable final-to-earlier visit-KL diagnostic and a unit-safe prequential-area reducer. D-EP-2 now has a restartable, four-way-sharded ECA-LATENT runner covering all 72 registered cells, with atomic self-hashed receipts, deterministic replay fingerprints, independent probe fit/eval sets, and a separate aggregate verifier that remains `analysis_pending` until the unbound executor/compiler interpretation is supplied.

The full repository gate ran every test twice. After correcting one stale C2 config identity introduced by the earlier sampled-`L_stage` scaffold, the raw result is **1 failed, 4,218 passed, 20 warnings**; the sole failure is the unchanged governed Paper Two evidence-ledger node. The strict exact-node wrapper passed. The repository is therefore still explicitly red, not globally green.

One R9 item cannot be run as written. A row-shuffled sender has exactly the same covariance as the original sender, so its registered `alpha_r` is identical. The required trained value cannot exceed that null by 3×. No SEAM-AUDIT pass is minted; the exact ruling needed is in §6.

## 1. Authority verification

| artifact | bytes | SHA-256 | Drive |
|---|---:|---|---|
| `STRATEGY_EPIPLEXITY_ADJUDICATION_20260903.md` | 24,723 | `7b5786185aebdd2388fa01a8db5522a2ce58e09a5c027947807ac0a2aebd3730` | `1XnZrYr3NOlfiaf8bW1Ewh-WVfAzfo6Ik` |
| `STRATEGY_EP_RATIFICATION_20260903.md` | 3,848 | `c9ecfa58f57c6904380f3edd895e92aada5efb3da77735c4d4e3d288197730ee` | `16KuNlJL5JyK9HatcxZyqJ73VzuP3jQGW` |
| `eca_latent_loop_20260903.py` | 4,137 | `457045eef6c5eaabde7f9d0571e62caed4cb1b0dc944ac57ece0343a512e9ee1` | local registered companion |

The three artifacts were ingested unchanged at `047c6287`. The adjudication and ratification extend the 2026-09-03 packet as R9; D-HD-1 remains R8. No earlier authority file was edited.

## 2. D-EP-1 implementation

### VISIT-KL

`AblationLM.forward` now emits `KL(p_final || p_sampled_earlier)` whenever D-MC-1 actually performs its second decode. It is absent—not zero—at K=1 and whenever the sampled decode is structurally OFF.

The implementation makes the previously implicit engineering literals explicit:

- units: bits per valid prediction token;
- direction: final visit to sampled earlier visit;
- support: shifted next-token positions for which both attention positions are valid, source and target are inside the same nonnegative packed-document segment, and the target label is not `-100`;
- arithmetic: detached FP32 under `no_grad`, with non-finite or materially negative KL failing closed;
- join fields: sampled visit index, valid-token count, direction, units, and support are emitted beside the value. The existing composition receipt retains the O-9 sampled-visit RNG coordinate.

This adds no coda decode, parameter, or gradient path. It is not literally zero arithmetic: two full-vocabulary log-softmax operations and the KL reduction remain. At the target width that should be small relative to the coda/readout already paid for, but it must be measured rather than described as free.

### Prequential area

`analysis/weft1_epiplexity.py` computes the ratified signed expression without the demonstrator's `max(..., 0)` clamp. Every interval supplies an explicit positive count of consumed prediction tokens, a loss in bits per prediction token, and exact `executed_k == scored_k`. It refuses to infer a step-zero interval. The result's unit is therefore bits. BPB may be reported next to it, but BPB cannot be multiplied by token counts and called bits.

This is a report-time primitive. It is intentionally not wired into G-TOK, whose governed metric is BPB and whose run sequence remains unchanged. Production Step 6 still needs its trainer/report surface before a language-model `preq_area` can be minted.

## 3. ECA-LATENT runner and launch

`analysis/weft1_eca_latent.py` implements:

- rules 30, 54, and 110;
- horizons 4, 8, and 16;
- K in `{1, tau/2, tau, 2*tau}`;
- replicas 0 and 1;
- the `1 → 2 → 4 → K` curriculum only when K is at least 4;
- evaluation at the executed `K_t` on one fixed held-out population;
- signed prequential area in bits;
- a full K-by-tau ridge-probe matrix, trained and scored on independent O-9 streams;
- no executor/compiler label, because no categorical threshold is ratified.

An adversarial review caught two pre-launch defects and both are now regression-tested:

1. selected-cell jobs originally changed the campaign identity and could not compose into a 72-cell result;
2. an orphan valid-JSON cell could originally be altered and accepted after a manifest-loss crash.

The final runner uses one shard-count-independent 72-cell identity; deterministic stride assignments in isolated shard directories; nonblocking process-lifetime writer locks; canonical per-cell self-hashes; complete semantic reconstruction of schedules, K values, metrics, datasets, prequential arithmetic, and probe ranges; initial-model and minibatch-order SHA fingerprints; and an aggregate verifier that accepts only all 72 exact cells. A shard can only report `shard_complete`; the instrument remains `analysis_pending`.

The registered campaign launched locally at 2026-09-04 18:08 ET in four workers under:

```text
.runlogs/weft1-eca-r9-8f6138e3-20260904
```

Each worker owns one of four disjoint shards. If the laptop sleeps or a process exits, completed cells survive and the exact shard resumes without overwriting them; only the in-flight cell is repeated. The first four cells (the two K=1 and two K=2 replicas for rule 30 / tau 4) were atomically present within three minutes. The first registered measurement, rule 30 / tau 4 / K 1 / replica 0, reached terminal BPC `0.0002341433` and accuracy `1.0`; this is an early cell, not a campaign conclusion. A visit- and kernel-weighted projection from that calibration puts the slowest shards at roughly seven hours, to be replaced by an empirical ETA after deeper cells land.

## 4. Concurrent corpus P-A continuity

The Pharma Initiatives Colab job remains isolated from this local CPU campaign and pinned at repository commit `28fd7614b8aca3f0a323420f0f518a5c0a6a93d9`, parsed-asset code identity `605dbfb85f4c3bc4a1d4bac98c31d95f7c704a27df441053eb8bdbb7353fd226`.

At the 2026-09-04 18:09 ET read-only snapshot:

- supervisor PID 86432 and parent PID 86494 were alive after 3 h 19 m;
- the current isolated cache-fill child PID 131842 was running at 94.8% CPU;
- the parent had read 22.23 GB and written 22.27 GB at the block-I/O layer;
- the earlier child had completed and the parent had advanced to a fresh child, evidence of phase progress rather than a restart;
- no blocked, failed, or complete receipt existed, which is correct while the cache-fill phase is active;
- Colab remained connected with 64.10 / 225.83 GB disk used.

The live checkout was not synced to the newer build commits, restarted, or otherwise perturbed.

## 5. Verification and commits

| check | result |
|---|---|
| R9 / ECA / seam focused tests | **34 passed** before hardening; final ECA + epiplexity suite **24 passed** |
| all `test_ablation_lm_*` plus R9 tests | **385 passed**, 18 known Torch JIT deprecation warnings |
| C2 exact replay after config-identity repair | **7 passed** |
| quarantine-schema and R9/C2 focused replay | **41 passed**, 1 known bf16 CPU warning |
| raw full repository suite | **1 failed, 4,218 passed, 20 warnings in 243.35 s** |
| strict exact-node wrapper | **PASS**; same one failure, 4,218 passes, 20 warnings in 243.01 s |
| Python compilation | PASS |
| `git diff --check` | PASS; informational Windows LF→CRLF notices only |
| lint | Ruff unavailable; no lint result claimed |

Commits pushed to `origin/codex/bicameral-stage0`:

- `047c6287` — ingest byte-exact R9 sources;
- `54642489` — implement D-EP-1, ECA-LATENT, seam fail-closed guard, and C2 identity repair;
- `8f6138e3` — roll forward the exact-node engineering quarantine.

The live quarantine is `training/ablation_lm_engineering_quarantine_20260904_ep_r9.json`. It preserves the one governed Paper Two failure, forbids a repository-wide green claim, authorizes no training or sealed-data contact, and is next due 2026-09-11 or before the next repository-wide receipt.

## 6. Exact strategy return: C-EP9-1

For sender samples `X` and any row-permutation matrix `Pi`:

```text
C(Pi X) = (Pi X)^T (Pi X) / N
        = X^T Pi^T Pi X / N
        = X^T X / N
        = C(X).
```

Centering does not help because the sample mean is also permutation-invariant. Therefore

```text
alpha_r(Pi X) = tr(P_r C(Pi X)) / tr(C(Pi X)) = alpha_r(X).
```

The row-shuffled sender null is identical to the trained statistic, so a positive trained `alpha_16` cannot be at least 3× that null. The repository now has an executable guard proving the invariance for centered and uncentered covariance and refusing to mint the comparison.

Please bind a non-invariant second null or a different sample-conditioned statistic. Two coherent options are:

1. apply a registered seeded Haar feature rotation to the sender covariance before measuring overlap with the fixed receiver projector; or
2. define a paired sample-conditioned sender/receiver statistic for which row shuffling actually destroys the alignment.

The same ruling should state whether `C_s` is centered, the exact input-GGN construction, null seed/count, and checkpoint pairing. Until then the random-init-receiver null and matrix-capture scaffolding may proceed, but the SEAM-AUDIT pass cannot.

## 7. Boundary and next work

R9 does not change the build queue, K/V policy, objective stack, curriculum, or no-injection rules. Corpus sequence remains P-A → attribution → P-B → G-TOK. The scaffold continues in parallel.

The next architecture promotion remains Step 3 and is still held on the pre-existing carrier/bridge literals returned in `CODING_TO_STRATEGY_WEFT1_PACKET_R8_IMPLEMENTATION_RETURN_20260904.md`: rotor initialization versus T2, rank-8 write inputs/shapes/init/gate and optimizer classes, `bridge_out` fusion/residual scaling, retention-gauge construction, and exact-versus-approximate plane orthogonality. R9 introduced no additional block on that path.
