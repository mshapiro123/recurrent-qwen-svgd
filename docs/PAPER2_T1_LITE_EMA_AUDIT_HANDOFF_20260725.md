# Handoff: T1-lite EMA Failure Localization

**Date:** 2026-07-25  
**Run:** `stage5_paper2_t1_lite_ema_audit_20260725`  
**Mode:** post-hoc, read-only  
**Registered T1-lite verdict:** unchanged `registered_negative`

## Executive result

T1-lite's raw endpoint acquired exact trained-depth control, but the
preregistered final-step EMA endpoint did not. The audit localizes this
divergence to the 178.95M-parameter recurrent block. A raw recurrent block
transplanted into the failed EMA endpoint restored exact depth selection and
answer performance. The reciprocal transplant, an EMA recurrent block in the
healthy raw endpoint, reproduced the failure. Swapping the bridge or control
rows did not reproduce or repair the control collapse.

This is evidence that continuous parameter EMA crossed a functionally sharp
boundary in the recurrent block during the staged curriculum. It is not
evidence that the EMA implementation was mathematically wrong, and it does not
convert the registered negative into a positive result.

## Original T1-lite result

The preregistered primary was the final-step EMA checkpoint. It failed the
joint gate. On 1,024 registered rows it achieved 202 forced-depth answers,
104 self-halted answers, and 128 exact depth selections. Causal overrides were
exact on all 5,632 interventions.

The registered raw-secondary endpoint achieved 967/1,024 for both forced and
self-halted answers, exact selection on 1,024/1,024 rows, and perfect
continue/stop recalls. It missed the chain-preservation threshold of 975 by
eight rows. It also failed to extrapolate its depth selector beyond trained
support.

## Audit design

The audit used the registered raw and EMA endpoint hashes and made no parameter
updates. It evaluated:

1. checkpoint and EMA-recurrence integrity;
2. a fixed 64-row screen, eight rows at each trained depth;
3. raw-to-EMA linear interpolation at seven coefficients;
4. reciprocal single-group swaps for control rows, bridge, and recurrent
   block;
5. 256-row confirmation of the strongest rescue and strongest damage.

The four historical stage checkpoints were absent or unusable in Drive. Their
coverage is 0/4. Archived raw stage receipts are retained, but raw-versus-EMA
stage-lag curves cannot be reconstructed and were not inferred.

## Integrity

- Raw SHA-256: `a83d056cb4fc366a0b3c3e95b10f00d59e2f624b554acbfac02d922119e5826c`
- EMA SHA-256: `1d674a14b7953d72031d72ac8dfd97744120809964b4234ece796efbce849a1e`
- Both checkpoint kinds and step 10,500 matched.
- Key sets and shapes matched; all tensors were finite.
- The scalar EMA recurrence passed with absolute error `1.43e-8`.
- No training or checkpoint mutation occurred.

## Interpolation screen

Alpha is the fraction of EMA endpoint weights in a linear raw-to-EMA blend.

| EMA alpha | Exact depth | Forced answer | Self-halted answer |
|---:|---:|---:|---:|
| 0.00 | 64/64 | 61/64 | 61/64 |
| 0.10 | 64/64 | 62/64 | 62/64 |
| 0.25 | 64/64 | 62/64 | 62/64 |
| 0.50 | 25/64 | 51/64 | 23/64 |
| 0.75 | 8/64 | 19/64 | 8/64 |
| 0.90 | 8/64 | 19/64 | 8/64 |
| 1.00 | 8/64 | 17/64 | 7/64 |

The depth controller remains exact through alpha 0.25 and collapses sharply by
alpha 0.50. This is not a smooth degradation proportional to Euclidean
distance.

## Reciprocal group swaps

Each row below starts from the named endpoint and replaces exactly one group
with the other endpoint's tensors.

| Variant | Exact depth | Forced answer | Self-halted answer |
|---|---:|---:|---:|
| EMA + raw control rows | 8/64 | 15/64 | 7/64 |
| EMA + raw bridge | 8/64 | 8/64 | 7/64 |
| **EMA + raw recurrent block** | **64/64** | **64/64** | **64/64** |
| Raw + EMA control rows | 64/64 | 61/64 | 61/64 |
| Raw + EMA bridge | 64/64 | 64/64 | 64/64 |
| **Raw + EMA recurrent block** | **8/64** | **9/64** | **7/64** |

The 256-row confirmation was equally sharp:

| Confirmed variant | Exact depth | Forced answer | Self-halted answer |
|---|---:|---:|---:|
| EMA + raw recurrent block | 256/256 | 252/256 | 252/256 |
| Raw + EMA recurrent block | 32/256 | 44/256 | 22/256 |

The rescue selected every depth exactly. The damaged model selected only depth
one exactly. The recurrent block is therefore sufficient to transfer the
healthy function into the EMA endpoint and sufficient to transfer the failed
function into the raw endpoint on this audit set.

## Parameter geometry

| Group | Parameters | Raw/EMA cosine | Difference/raw norm |
|---|---:|---:|---:|
| Control rows | 2,688 | 0.999908 | 1.367% |
| Bridge | 1,608,321 | 0.999464 | 3.275% |
| Recurrent block | 178,948,608 | 0.999986 | 0.520% |

The recurrent block has the smallest relative parameter displacement and the
highest cosine, yet it completely determines the functional outcome. Global
parameter proximity is therefore not a useful safety test for this learned
iterative mechanism.

## Interpretation boundaries

Supported:

- The token control pathway was learned by the raw endpoint on trained depths.
- Continuous final-step EMA did not preserve that function.
- The endpoint failure localizes to the recurrent block, not to the bridge or
  the three control-token rows.
- The raw-to-EMA path contains a sharp functional transition between the 0.25
  and 0.50 interpolation points on the screen.

Not supported:

- The registered experiment passed. It did not.
- EMA lag began at a specific curriculum stage. The required EMA stage states
  are unavailable.
- A particular layer, optimizer group, or direction within the recurrent
  block caused the collapse. The current swap resolution is block-level.
- Token control extrapolates beyond trained support. It did not in this run.

## Strategic questions

1. Is the registered negative the final T1-lite verdict, with raw acquisition
   and EMA destruction reported as the mechanistic finding, or is one new
   preregistered replication warranted?
2. If replicated, should raw final weights be the primary endpoint, or should
   the intervention be a stage-reset EMA whose state is reset at each support
   expansion? Either choice must be locked before training.
3. Is block-level localization sufficient for Paper Two, or is a read-only
   layer-group swap valuable before deciding? A layer sweep would be post-hoc
   localization, not a route to changing the verdict.
4. Future staged runs need atomic, hashed stage-checkpoint backups with an
   end-of-run availability manifest. The missing snapshots materially limited
   this audit.

## Recommended next decision

Bank the registered negative and the post-hoc recurrent-block localization.
Do not tune EMA decay on this run. Decide at strategy review whether a single
fresh replication has enough expected information value. If authorized, vary
one declared factor only: either make raw weights primary under the same
training recipe, or test a stage-reset EMA policy against the original
continuous EMA. Preserve both endpoints and every stage boundary with hashes.

The COCONUT horizontal-loop work is independent engineering work and can
continue through its no-training RG-1 to RG-11 preflight while this decision is
reviewed.

## Canonical artifacts

- `outputs/stage5/stage5_paper2_t1_lite_ema_audit_20260725/summary.json`
- `outputs/stage5/stage5_paper2_t1_lite_ema_audit_20260725/receipt.md`
- `outputs/stage5/stage5_paper2_t1_lite_ema_audit_20260725/restore_manifest.json`
- `docs/PAPER2_T1_LITE_EMA_AUDIT_SPEC_20260725.md`
