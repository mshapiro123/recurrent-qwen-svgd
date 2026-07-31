# Paper Two Phase-2 Pre-Window Build Handoff

Date: 2026-07-31. Status: implemented and locally verified; no experiment in
this package trains a model or opens an exploration window.

## 1. Authority and scope

The implementation is governed by three byte-verified Drive documents:

- `PAPER_TWO_PHASE2_PROGRAM_DECISION_AND_DESIGN_20260731.md`, SHA-256
  `47916cdc88652c3de39c75b634c3f6b2fcc0fd73bf75cef75d3fb8880c97c9de`.
- `THEORETICAL_FOUNDATIONS_AUDIT_20260731.md`, SHA-256
  `a43858623ab88c0710469cdd6b12f3d9ab10c882aea7436291a3ffd4b52e8bdc`.
- `STRATEGY_TO_CODING_AGENT_PHASE2_OPENING_20260731.md`, SHA-256
  `0d9349441df8ad005672da34d507b863b88a9355f5ca3007fdc76162160b6ec7`.

Stage A is recorded as complete, verdict `none`, consequence
`transient_append_retires`, and EVAL-C scoring spent. Mark's program-level
override opens DC2 and D1 only as future separately locked lines. The present
package implements the four authorized no-training jobs and nothing beyond
them.

## 2. Runnable targets

### `paper2_phase2_oracle_overlap`

CPU-only post-processing of the saved Stage A per-batch prediction tensors.
It reports trained and untrained append oracle ceilings plus the exact
trained-append versus in-place depth-2 hurt-set intersection, union, Jaccard,
and directional containment. It does not load a model or rerun EVAL-C scoring.

The governing text calls this post-processing from the immutable cache. The
aggregate immutable JSONL contains per-row counts, not position identities, so
position overlap is reconstructed from the private per-batch prediction tensors
written during the same immutable Stage A pass. Their file-level manifest hashes
are included in the receipt. This is a provenance clarification, not a rescore.

### `paper2_phase2_v1_v2`

L4/A100, read-only DEV-C diagnostic.

- V1 identifies oracle-help positions from registered one-loop predictions and
  the archival trained Stage A append bridge. It records the wrong-versus-teacher
  margin distribution and position-matched upper-stack gain diagnostics for
  `c = 0.01, 0.02, 0.05`, `gamma = 0.05`, and `rho = 0.8`.
- V2 records recurrent-block-only directional gain distributions at iterates
  1 through 4 on the pre-D0 and post-D0 checkpoints.

Centered finite differences implement robust JVP estimates through the existing
Transformers stack. V1 also computes the exact local gradient norm of the
specific wrong-versus-teacher margin. Random JVPs are sampled directional gains,
not certified Lipschitz upper bounds. Consequently the requested overlay is
named `bound_compatible_fraction_using_sampled_max_gain`; the receipt explicitly
forbids treating it as a reachability guarantee. The targeted margin-gradient
overlay is a local first-order diagnostic and is also not a finite-radius proof.

Default deterministic budgets are 128 DEV-C rows, 128 oracle-help probes, 32
V2 rows per checkpoint, and two random directions per position/iterate. Every
probe is independently cached on Drive for exact resume.

### `paper2_phase2_eval_de_freeze`

L4/A100 data-preparation job. It creates EVAL-D and EVAL-E at 200,000 source
tokens each, 50/50 general and code, with separate deterministic seeds. Both
sets are document-disjoint from D0, EVAL-B, DEV-C, EVAL-C, and each other.
The job performs one cached Qwen2.5-7B pass per partition and exposes no scores.

The own-base feature cache stores three boundary states from the frozen post-D0
model: after prelude layer 6, recurrent layer 18, and coda layer 24. Tensors are
private bfloat16 shards with hashes; future full-fp32 losses must upcast them.
This preserves a multi-layer target while leaving exploration loss weights
uncommitted. Public receipts expose hashes and counts only.

## 3. Binding theory amendments carried forward

The future training implementation remains unopened, but the build records all
four mandatory amendments for that work:

- A1: `rho <= 0.9` and realized tube-radius logging.
- A2: margin-improvement loss masked to positions not already correct.
- A3: inter-objective gradient cosines per module.
- A4: scratchpad effective rank, gate-open rates, and tube-radius collapse
  telemetry with thresholds fixed before a window run.

No DC2/D1 training code or confirmatory scorer is introduced here.

## 4. Verification

The local red/green suite covers aggregate oracle arithmetic, exact position
overlap, public/private boundaries, centered finite-difference gain arithmetic,
bound-compatible fractions, boundary-feature extraction through a tiny Qwen2
wrapper, bootstrap targets, and all relevant existing DC1 contracts.

The package is designed for independent execution. Recommended order is the
CPU oracle receipt, V1/V2 on an L4 or A100, and EVAL-D/E preparation in a
separate L4/A100 session. V1/V2 must land and return to strategy review before
either exploration window can open.
