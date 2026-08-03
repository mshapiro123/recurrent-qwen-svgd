# Phase-2 Stage 0A Repair and Experiments 0A/0B Build Receipt

Date: 2026-08-04. Scope: DEV-C only. No frozen evaluation partition is available to these targets.

## Execution order

1. `paper2_phase2_stage0a_repair`: CPU-only recomputation from the durable Stage 0A lattice and audit shards. It performs no model inference and no training.
2. `paper2_phase2_exp0a`: A100-80GB canonicalizer and partial-whitening screening over the cached Qwen2.5-14B states. It performs closed-form fitting only and never trains or mutates the backbone.
3. `paper2_phase2_exp0b`: L4-or-larger interpolation audit plus disposable serial-flow pilots. Its optimizer touches only newly initialized pilot modules.

Every target writes private tensors to its exact Drive run directory, writes an aggregate receipt to Drive, and publishes only the aggregate summary to GitHub.

## Stage 0A repair

The original Stage 0A cache remains immutable. The repair writes separate metric shards and replaces non-finite aggregate behavior with:

- finite-support coarse KL with the clipping floor disclosed;
- explicit student-support-miss mass;
- probability-space scale-coherence cosine;
- finite-only full-logit error summaries plus support-mismatch counts.

The repaired KL remains a sparse coarse diagnostic, not a full-vocabulary KL.

## Experiment 0A

Default linear comparison:

- PCA at matched effective rank;
- predictive reduced-rank regression;
- Tucker-initialized predictive regression using a fitted frozen layer simplex.

Attention pooling and the deterministic autoencoder remain trigger-held under the governing design's linear-underfit rule. Their trigger is recorded but they are not silently added to the default v1 path.

Geometry is fixed at eight slots by 128 coordinates. Four future slots are populated; four trace/span slots remain masked because Stage 0A found no span boundaries. All alpha arms use one frozen PCA orientation. Effective eigenvalues are computed once with `tau=1e-4` and `eps_abs=1e-8`; the forward transform adds no second epsilon. Alpha is screened at `0`, `0.5`, and `1.0` but is not selected.

The source cache contains exactly 200,000 samples, while the design requests both a separate holdout and a fit of at least 200,000. The implementation resolves this transparently: method screening uses a deterministic document-disjoint development split, and candidate artifacts are separately refit on all 200,000 DEV-C samples. The all-sample refit is never reported as held-out evidence.

## Experiment 0B

For every finite linear-method/alpha arm, 0B evaluates affine paths between horizon-one and horizon-four canonical endpoints. It records probe-KL trajectories, monotonicity, second differences, norm contraction, and a small serial-flow trainability pilot. Interpolation targets and persistent states are never renormalized. RMS normalization is confined to pilot-module inputs and innovations. The loop cap is four.

The serial-flow pilot is development scaffolding, not E1 evidence and not the matched built-module alpha-selection pilot. Alpha selection remains deferred until the full student module exists and its matched pilots measure flow convergence, upper-model quality, and verified acceptance.

## Tests

`tests/test_paper2_phase2_stage0ab.py` covers non-finite accounting, support misses, probability-space coherence, document isolation, the single eigenvalue floor, shared alpha orientation, masked unavailable slots, affine interpolation, loop-cap enforcement, a linear fit/probe smoke, and launcher boundaries.

