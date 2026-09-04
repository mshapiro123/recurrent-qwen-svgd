# CODING → STRATEGY — WEFT-1 2026-09-03 Packet + R8 Implementation Return

**Date:** 2026-09-04
**Status:** Packet verified; all authority-complete Step-1/Step-2/R8 work implemented. Step 3 is fail-closed pending the literal rulings in §5. No run-axis gate is minted by this receipt.

## 1. Authority verification

The coding agent verified the packet and every artifact in its ten-row register against the local byte count and full SHA-256. No registered artifact is missing or mismatched.

| artifact | bytes | SHA-256 | disposition |
|---|---:|---|---|
| `STRATEGY_HANDOFF_PACKET_20260903.md` | 8,679 | `07687b98a37be271318294848338c1588aee435e4ba3f891e1c158c72edd2b1f` | governing index; source records win under SEQ-1 |
| `STRATEGY_HD1_RATIFICATION_20260903.md` | 2,472 | `fa6d0bacc379d92d2393c50c477739aa1bde16a3c08a9b682a542e28329278e8` | appended as R8; receipt additions only |
| `STRATEGY_HEMISPHERE_DIVERGENCE_NOTE_20260903.md` | 11,068 | `687c2a78311295928f312e7284efb9545e53d89c32b614c2eac5c72a745c9f3b` | R8 source note |
| `symmetry_forces_20260903.py` | 6,493 | `122abc6b72e1f221ecdc9dc226b63c4cd898dcd9b0120aba3a32e8b0c60a4260` | reference script; not production code |

## 2. Packet reconciliation against the graph

- Step 1 is already implemented and receipted at `28935e5d`: EG-1 gains, T2 liveness, and the T4 `bitrev(gray(k))` sequency contract.
- Step 2 is already implemented at `2511ec84` and receipted at `7140a827`: `live` is the default K/V policy; `static` and `midpoint` are retained controls; K=1 live/static identity, S-2 terminal combine, and exact visit schedules are tested.
- D-MC-1's final-plus-one-sampled coda decode and 1.238636× allocation are specified but remain Step-6 runtime work.

Three packet phrases are non-governing defects under its own SEQ-1 rule:

1. `TwoLaneBirkhoffMixer` retires when the real callosum lands at Step 4, not at Step 2 (R0 controls).
2. `DELAY-1` and `MEM-SYN-STATIC` are registered future arms, not merely names, but neither is current build work (R3 controls).
3. The positional shorthand `2K / 2 / 1` is misleading beside `{live, static, midpoint}`. The authoritative mapping is `live → 2K`, `static → 1`, `midpoint → 2` (R6 controls). The implemented receipt uses this mapping.

## 3. R8 implementation

- `SwapLinear.delta_ratio()` computes `||dU dVᵀ||F / ||μ||F` from the two rank Gram matrices without materializing a dense disagreement matrix. Receipt arithmetic is detached FP32 with autocast forcibly disabled.
- Diagnostic model forwards emit an ordered `(fully_qualified_module_name, delta_ratio)` row for every `SwapLinear`.
- `SymmetryCollapseTracker` enforces the exact 1,000-consecutive-eligible-step SYM-COLLAPSE boundary against immutable T7 initialization values. Missing/reordered steps and matrix-set drift fail closed. Its state is JSON-safe and resumable, and a trip exposes the complete triggering receipt.
- `PerBandUnitCircleCombiner.lateralization_index()` emits the literal R8 value `sin(2θ_b)` for every band.
- `CompositionReceipt` carries `lateralization_index` and a backward-compatible nullable `rho_hat_free`. The latter remains null in executable model receipts because Step 6 and its calibration gate do not exist yet.

Production limitation: the repository still has no training runner that can mint T7 initialization baselines, invoke/checkpoint the SYM-COLLAPSE tracker at every eligible optimizer step, or compute `rho_hat_free` from the Step-6 objective. The primitives and schemas are implemented; the production stop is not claimed active.

## 4. Validation

| gate | result |
|---|---|
| focused D-HD-1 / bicameral / accounting suite | **69 passed** |
| all `test_ablation_lm_*` tests | **346 passed**, 18 pre-existing Torch JIT deprecation warnings |
| repository-wide pytest | **1 failed, 4,132 passed, 20 warnings**; sole failure is the unchanged governed Paper-2 evidence-ledger node |
| strict quarantine-aware wrapper | **PASS**; all tests ran, exact one-node match, 4,132 passes and 20 warnings in 199.49 s |
| `git diff --check` | no errors; informational LF→CRLF notices on touched tests only |
| lint | Ruff unavailable; no lint result claimed |

The raw repository remains explicitly red. The due 2026-09-04 quarantine was reviewed rather than carried forward silently. Successor v20 is `training/ablation_lm_engineering_quarantine_20260904_packet_r8.json`, 3,507 bytes, SHA-256 `90c4312fe08b28c7ffa39864c716da1b5871b429766b684780b08ca1f730c058`; it names the same exact node, records the unchanged two missing legacy paths, forbids a repository-wide green claim, and is next due 2026-09-11 or before the next repository-wide receipt. The strict quarantine-aware wrapper passed against that exact successor.

## 5. Fail-closed strategy rulings required

### C-STEP3-1 — zero-angle learned rotor contradicts T2

The handoff makes every rotor plane vector `a_j,b_j` and angle `θ_j` trainable, initializes `θ_j=0`, and requires every active parameter to have a nonzero step-1 gradient. Direct autograd of the ratified formula gives:

```text
||∂L/∂a|| = 0
||∂L/∂b|| = 0
∂L/∂θ     = 1.8602
R(x) == x  = True
```

At `θ=0`, every derivative with respect to the plane vectors is multiplied by `sin θ` or `cos θ−1`; their zero gradients are structural, not a numerical accident.

Choose one literal:

1. **Small nonzero active angle (coding recommendation):** keep structural OFF as a bit-identical bypass and initialize the active arm at a bound nonzero magnitude (for example `10^-3`), retaining learned planes and step-1 liveness.
2. **Fixed planes:** keep `θ=0`, make `a,b` deterministic frozen O-9 buffers, and train only `θ`.
3. **Delayed eligibility:** keep trainable `a,b` and `θ=0`, amend T2/PF-1.5 so plane vectors become eligible only after the first nonzero-angle update, then require them live.

### C-HD1-1 — `sin(2θ)` is not injective

The literal field is implemented, but the prose `0 = consensus read` is not globally valid for unconstrained `θ`: `θ=0` is consensus and `θ=π/2` is pure disagreement, yet both give `sin(2θ)=0`.

Choose one literal:

1. **Keep `θ` unconstrained and amend the interpretation (coding recommendation):** zero means balanced hemisphere magnitude, not uniquely consensus; log the existing `(cos θ, sin θ)` coefficient pair or `θ` beside the index when consensus/disagreement must be distinguished.
2. **Constrain/canonicalize `θ` to a ratified principal interval** on which the intended interpretation is unique, accepting the resulting restriction on the combiner's reachable directions.

### Remaining Step-3 literals

Even after C-STEP3-1, full carrier promotion still needs exact bindings for:

- rank-8 write inputs (corresponding lane or both lanes), normalization, factor shapes/initialization, scalar-gate initialization, and μP/decay classes;
- `bridge_out` lane fusion and whether its residual is scaled by `α_T`;
- fitted carrier-retention gauge projection, fit/intercept, aggregation, O-9 probe identity, and terminal-versus-per-visit decision value;
- μP classes for `a,b` and both low-rank write factors;
- whether the `+ε` plane normalization is accepted as approximate with an operator-norm witness, or replaced by a construction that certifies exact orthogonality.

Until these are bound, Step 3 remains absent rather than partially promoted. Corpus P-A and all authority-complete build work continue independently.
