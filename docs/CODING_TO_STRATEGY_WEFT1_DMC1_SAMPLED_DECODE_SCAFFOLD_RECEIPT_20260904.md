# CODING TO STRATEGY — WEFT-1 D-MC-1 Sampled-Decode Scaffold Receipt

**Date:** 2026-09-04  
**Status:** build-axis mechanism and accounting ready; structural OFF by default  
**Implementation commit:** `677c706c94a26532bed42a3ce4c6b8fbbbd7eafe`  
**Run-axis effect:** none; no corpus, checkpoint, sealed data, or GPU cell consumed

## 1. Authority used

| authority | identity | use |
|---|---|---|
| `STRATEGY_MATH_CHECK_RATIFICATION_20260903.md` | 2,868 bytes; SHA-256 `9c5822daef5dbb0609bc3e46019cc4b1e332991c30e8a42c1b4432800a747ab1` | D-MC-1: final visit plus one uniformly sampled earlier visit |
| `STRATEGY_HANDOFF_PACKET_20260903.md` | 8,679 bytes; SHA-256 `07687b98a37be271318294848338c1588aee435e4ba3f891e1c158c72edd2b1f` | consolidated precedence and build-step routing |
| `WEFT1_STEP6_OBJECTIVE_AND_SAMPLED_DECODE_SPEC_20260903.md` | repository executable specification | sampling, serial decode, receipt, and test contract |
| `CODING_TO_STRATEGY_WEFT1_DMC1_ALLOCATION_CORRECTION_20260904.md` | repository correction receipt | exact `N_active_train` accounting and current allocation posture |

The source records govern wherever the packet compresses their language.

## 2. Implemented scaffold

The implementation is behind `use_lstage_sampled_decode`, whose production-safe
default is `False` and which requires structural recurrence.

When the model is in training mode and `K_exec > 1`:

1. it draws one `j` uniformly from `[0, K_exec - 2]` before recurrent execution;
2. the draw comes from the isolated, checkpointable O-9 stream
   `weft.lstage.sample` at coordinate zero;
3. it retains only the selected earlier state with autograd, while ordinary
   diagnostic trajectory states remain detached;
4. it decodes the final state first and the sampled state second through two
   serial calls to the same coda, final normalization, and tied readout;
5. on the bicameral graph, both final and sampled states use the same S-2
   combiner.

At `K_exec = 1` and in evaluation mode, the model performs one decode, consumes
no L-stage RNG draw, emits a null sampled visit, and exposes no sampled logits.
The ordinary `logits` and `loss` path is bit-identical to the structural-OFF
control in the registered CPU BF16 fixture.

The per-forward composition receipt now records:

- `coda_decodes_per_step` and `lstage_sampled_visit`;
- exact source key, coordinate, draw index, and derived seed;
- `n_coda`, `n_active_train`, and `active_train_exact`, without changing the
  meaning of `n_active_eval`.

The accounting identity is

```text
N_active_train = N_fixed
               + K_exec * N_recurrent
               + (coda_decodes_per_step - 1) * N_coda
```

`N_coda` is the exact unique non-vocabulary parameter union executed by the
second decode: coda blocks, final normalization, and the bicameral S-2 combiner
when present. The tied readout remains in the vocabulary partition.

## 3. Fail-closed receipt checks added during review

Independent review found two direct-mint cases that the integrated forward path
would not produce but the receipt constructor had accepted. Both are closed:

- two coda decodes with no sampled visit are rejected;
- a seed/visit pair must both derive from the configured O-9 root and replay to
  the recorded `j`. A forged but self-consistent seed/visit pair is rejected.

The receipt also rejects fractional execution statistics with a representative
sample, samples outside the actually executed visit range, incomplete RNG
metadata, and sampled execution metadata on a graph where D-MC-1 is OFF.

## 4. Verification

| check | result |
|---|---:|
| Python compilation of the six changed implementation/test files | pass |
| focused D-MC-1, contract, and RNG suite | **47 passed** |
| all `test_ablation_lm_*.py` tests | **355 passed** |
| existing TorchScript deprecation warnings | 18, unchanged |
| `git diff --check` | pass; only the pre-existing checkout LF-to-CRLF notice appeared |
| independent implementation audit | no remaining code blocker |

This is not a claim that the whole repository's governed suite was rerun or
reminted at this commit.

## 5. Deliberate nonclaims and remaining gates

This receipt does **not** promote or complete Step 6.

- `output.loss` remains the ordinary final-state language-model loss. The
  curriculum target and `lambda_stage` are still unbound, so the sampled logits
  are exposed but not silently added to training loss.
- The source specification requires a registered uniformity decision test, but
  supplies no threshold. Deterministic replay and coverage are tested; the
  formal uniformity gate remains unminted rather than inventing a band.
- `LSTAGE-FULL` is a registered contrast and is not implemented as a fallback.
- Steps 3–5 and their integrated modules are not claimed complete by this
  standalone, structural-OFF scaffold.

## 6. Packet review notes preserved for future work

Three compressed statements in the handoff packet must continue to defer to
their cited source records:

1. `DELAY-1` and `MEM-SYN-STATIC` are registered future MEM-OP controls, not
   merely named alternatives.
2. `TwoLaneBirkhoffMixer` retires when the per-band callosum lands at Step 4,
   not at Step 2.
3. Serving-cache multipliers are live `2K`, static `1`, and midpoint `2`.

No code action was taken on those future controls in this commit.

## 7. Corpus-run isolation

The active P-A corpus supervisor remains pinned to
`28fd7614b8aca3f0a323420f0f518a5c0a6a93d9` and parsed-asset identity
`605dbfb85f4c3bc4a1d4bac98c31d95f7c704a27df441053eb8bdbb7353fd226`.
This newer build-axis commit must not be synced into, or used to restart, that
already verified run. The two workstreams remain independent.
