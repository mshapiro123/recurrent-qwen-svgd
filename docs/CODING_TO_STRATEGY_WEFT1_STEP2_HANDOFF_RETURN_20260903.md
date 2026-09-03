# CODING → STRATEGY — WEFT-1 Step-2 and 2026-09-03 handoff return

**Date:** 2026-09-03
**Status:** BUILD-AXIS CHECKPOINT · implementation complete · promotion held on two exact strategy bindings
**Implementation commit:** `2511ec84ad67724af226dbd6b7abe1c87a9b4892`
**Base commit:** `f5f4bd4447546c2ab5b3dd788558ce72b66ec847`
**Run-axis contact:** none; no tokenizer fit, checkpoint, GPU cell, sealed data, or training compute was consumed

## 0. Outcome

The Step-2 graph now runs end to end through `AblationLM.forward`. It contains
exactly five paired core maps (Q, O, gate, up, down), ordinary shared-weight K/V,
the ratified `live`/`static`/`midpoint` policy switch with `live` as default, a
terminal per-sequency-band unit-circle S-2 combiner, exact execution-derived
schedule receipts, and closed optimizer ownership. The integrated structural-OFF
graph matches an ordinary dense recurrent Transformer at `K = 1, 2, 4, 8`.

This is deliberately a **non-promoted checkpoint**. PF-3.1 does not bind a
stored-tensor class for the low-rank `dV` factors, and reconciliation R-4 does
not identify what exact “engram values” the lane update is to read. Neither was
guessed locally. Steps 3–7 remain absent or deferred exactly as named below.

## 1. Authority verification and precedence

The packet was checked byte-for-byte and SHA-for-SHA. Its complete R0–R7 plus
script register matched the local governed copies and the independently fetched
Drive copies. The immediately operative additions were:

| authority | bytes | SHA-256 |
|---|---:|---|
| `STRATEGY_HANDOFF_PACKET_20260903.md` | 8,679 | `07687b98a37be271318294848338c1588aee435e4ba3f891e1c158c72edd2b1f` |
| D-MC-1 | 2,868 | `9c5822daef5dbb0609bc3e46019cc4b1e332991c30e8a42c1b4432800a747ab1` |
| EG-1 | 4,398 | `36f0255c1cc0e61b2d9019ce86b3b1e7446b0a2c3445a42ce20334213deae780` |
| D-NB-1 / KV-live | 4,329 | `489729907fe672f811015ff961f8731c6a9f775a5119347c819184d368e238b0` |
| Nanbeige A1 | 21,756 | `7afc73725bdb9a60bbf1e8317896a261ac34391ddca360eab413f9f8a3df56ad` |

Two packet summaries were resolved from their source records rather than
silently reconciled:

1. `TwoLaneBirkhoffMixer` is **not** retired in Step 2. R0 R-5 retains it as a
   temporary lane scaffold through Step 3 and retires it when the actual
   callosum lands in Step 4.
2. `DELAY-1` and `MEM-SYN-STATIC` are future registered controls, not current
   build actions. No Nanbeige component or optional `first` K/V policy was built.

## 2. Executed graph and ownership

The actual current graph is:

```text
tied embedding
  → ordinary prelude (optional EG-1 write after block 1)
  → h0; initialize hA=hB=h0 and optional position-aligned lanes
  → repeat K visits with alpha=c/K:
      snapshot/project each block's K/V under kv_policy
      for each unique core block:
        paired attention → paired SwiGLU → lane update
  → terminal S-2 unit-circle combiner
  → optional post-loop LTM-RO arm (OFF by default)
  → ordinary coda once
  → final RMSNorm → tied readout
```

Each core block has shared RMSNorm/QK-normalization/RoPE; paired
`SwapLinear(mu,dU,dV)` for Q/O/gate/up/down; and ordinary shared-weight K/V.
The same unique core blocks are reused across visits and remain registered only
under `core_blocks.*`. Attention dropout remains structurally zero.

The closed Muon allowlist contains all seven ordinary dense maps in each
prelude/coda block and only the two shared K/V maps in each bicameral core block:
`7(P+C)+2B` matrices. All paired `mu`, `dU`, and `dV` tensors stay together in
AdamW; S-2 theta, norms, embeddings, engram tensors, lane tensors, gates, and
other novel parameters also remain outside Muon. The partition is exhaustive
and duplicate-free.

## 3. K/V policy and accounting receipt

Let `B` be the number of unique core blocks and `K` the executed visits.

| policy | projection source | projection rounds | K/V linear calls | serving cache multiplier |
|---|---|---:|---:|---:|
| `live` | each hemisphere's own visit-entry state, same K/V weights | `K` | `4BK` | `2K` |
| `static` | shared `h0`, A/B cache tensors alias | `1` | `2B` | `1` |
| `midpoint` | shared `h0`, then both live states at `floor(K/2)` | `2` for `K≥2` | `6B` | `2` |

The `K=1` multi-block live and static graphs are bit-identical. At the 4/2/4
proxy shape with `B=2`, `K=4`, the executed diagnostics are respectively
`4/32`, `1/4`, and `2/12` for projection rounds / K/V linear calls.
No hemisphere reads the other's K/V.

The recurrence receipt now records `recurrent_steps`, `unique_core_blocks`,
`executed_block_passes`, `recurrence_c`, `residual_scale`, `kv_policy`,
`kv_cache_multiplier_at_serving`, `after_block_modules`, and the exact core
`visit_schedule`. The composition receipt appends the actually executed
`terminal.PerBandUnitCircleCombiner` and rejects an incomplete, reordered, or
caller-preprojected trace that the integrated graph did not execute.

At Step 2 the objective fields truthfully remain
`coda_decodes_per_step = 1` and `lstage_sampled_visit = null`. D-MC-1 runtime is
Step 6; no two-decode claim is made early.

## 4. Owed evidence returned

### T2 and EG-1

The active EG-1 audit now emits `gate_form = "EG-1"`. A direct T2 backward has
finite nonzero step-1 gradients on `W_Q`, both normalization gains
`gamma_q/gamma_k`, every hit table row, `W_V`, and raw residual scale
`gamma_m`. The paired-core T2 asserts a finite nonzero gradient on `mu`, `dU`,
and `dV` for each of the five paired maps, and on both shared K/V matrices.

### T4

The passing ordering is exactly `bitrev(gray(k))`. For widths
`1,2,4,8,16,32,512,1024`, transformed row `k` has exactly `k` sign changes.
The implementation keeps binary inverse scaling `2^-p`; it does not claim a
bit-exact floating-point WHT round trip.

The compact evidence command covering these obligations and the four A7 dense
anchors returned `8 passed in 2.77s`.

### D-MC-1 allocation

The executable Step-6 specification re-derives the K=4 multiplier as
`545/440 = 1.238636...`. Applying it to the prior approximately 234 A100-hour
all-in figure gives approximately **289.84 A100-hours**, +55.84 hours. This
exceeds the referenced allowance and therefore engages the pre-registered
rung-B-first de-scope order, pending the exact Step-3/4 integrated counts before
S2 scheduling. No run was removed locally.

## 5. Exact holds returned to strategy

### C-STEP2-1 — PF-3.1 stored class for `SwapLinear.dV`

At each of `d=128,256,512`, the integrated 4/2/4 inventory has 123 classified
assignments and exactly 10 unresolved tensors: the `dV` factor for five paired
maps in each of two core blocks. Stored `dV` has width-scaling fan-in and
fixed-rank fan-out; PF-3.1 binds no stored parameter class for that shape.
`classify_mup_parameters()` therefore fails closed before PF-3 initialization
or optimizer construction. Strategy must bind its class, initialization,
learning-rate scaling, and decay treatment.

### C-STEP2-2 — R-4 “engram values” lane input

R0 R-4 says the per-block lane update reads `h_A`, `h_B`, and “engram values.”
The current causal engram writes into `h0`, so its contribution reaches both
hemisphere states indirectly; the lane method itself accepts only `h_A/h_B`.
Please rule whether R-4 intended that inherited contribution or an explicit
third tensor. If explicit, bind which tensor (raw retrieved row, normalized row,
gated projected value, or bounded residual), its shape/projection, normalization,
gate, and optimizer ownership. Until then this surface is typed incomplete.

The Step-3/4 executor will also need a typed once-per-visit hook after the block
loop so rotor/write executes before callosum. That is a known implementation
dependency under the already bound order, not a new semantic choice.

## 6. Verification and quarantine

- Targeted handoff evidence: `8 passed in 2.77s`.
- Full ablation/model + PF-C2 suite: `346 passed, 19 warnings in 31.41s`.
- Independent final adversarial subset: `156 passed`; no remaining Step-2
  runtime, semantic, causality, initialization, optimizer, or receipt defect found.
- Repository gate: exact-node gate PASS; the underlying repository remains red
  at `1 failed, 4127 passed, 20 warnings in 199.88s`.
- `compileall` passed; `git diff --check` passed apart from informational
  Windows LF→CRLF conversion notices. Ruff is not installed, so no Ruff claim is made.

The successor exact-node quarantine is v19. It still contains precisely the
single governed Paper-2 evidence-ledger failure and still forbids any
repository-wide green claim. No failure was skipped, xfailed, deselected, or
suppressed. Its next review remains due 2026-09-04.

## 7. Concurrent P-A continuity observation

At `2026-09-03T18:39:34Z`, the durable P-A replay
`pa-v4-7f985158-r4-attempt-05` was alive with both parent and worker present.
It held 159 current-code durable receipts and had covered
18,998,123,536 / 47,632,339,814 source bytes (**39.8849%**): Dolma web 88,
StackEdu 69, Wikipedia/Wikibooks 2. One active StackEdu partial was advancing;
the latest completed receipt was ordinal 68. The log was empty, receipt reads
had zero errors, and the parent final receipt was correctly not yet present.
The most recent 62.4-minute interval advanced 543,427,311 source bytes, which
projects approximately 54.8 hours remaining if that source mix and throughput
hold; this is an operational estimate, not a gate or deadline.

The observation was read-only. The materializer was neither restarted nor
mutated by this build-axis work.

## 8. Boundary

This return closes the executable Step-2 engineering work and supplies the
requested evidence. It does **not** promote the production model, authorize
training, or claim Steps 3–7. Promotion is held on C-STEP2-1 and C-STEP2-2;
run-axis sequencing remains P-A → attribution → P-B freeze → G-TOK.
