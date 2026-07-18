# Phase G Multi-Target Curriculum Correction

**Date:** 2026-07-18
**Status:** CPU preparation complete; GPU launch intentionally pending posterior-control threshold locking.

## Purpose

The original Phase G-alpha data exposed one target chain for each prompt. That
cannot identify a target-conditioned posterior: every train prompt had exactly
one target, and each K-trajectory optimizer step repeated that target across
all trajectories. This correction constructs an actual conditional
multimodality task before the one permitted rerun of guided latent width.

## Data Contract

For each generated branching-relation problem, the preparation step enumerates
all distinct exact-depth reachable terminals. It emits one training row per
terminal target. Variants retain an identical prompt, table, start state,
depth, and reachable-set metadata. They differ only in the selected valid
terminal and its exact chain.

Each row records:

- `base_problem_id`: the prompt-level identity shared by variants;
- `target_variant_index` and `target_variant_count`;
- exact `sampled_chain` and `loop_completions` for the selected terminal;
- `posterior_chain_sampling=enumerated_distinct_terminal_target`.

The data validator rejects duplicate targets, prompt differences within a
group, invalid chains, inconsistent loop labels, and missing target support.
The initial corrected run requires every exact reachable terminal, not a
sampled subset.

## Sampling Contract

Corrected Phase G training uses:

```text
uniform base prompt -> uniform target-chain variant
```

This prevents prompts with large reachable sets from receiving proportionally
more gradient updates. The original `row_uniform` policy remains the default,
and it is intentionally omitted from its resume contract to preserve exact
compatibility with existing Phase G-alpha checkpoints. Any non-default policy
is included in the resume contract and trace records.

## Evaluation Contract

The correction has two distinct held-out surfaces:

1. **Posterior-control rows:** repeated prompts with different selected targets.
   The enhanced supervision audit measures K=1 selected-target fidelity and
   group-level switching: a posterior teacher should generate different first
   predictions when only the selected valid chain changes.
2. **Original frozen coverage rows:** retained unchanged for the final
   coverage comparison so the corrected run remains comparable to the prior
   entropy-matched-temperature and iso-compute-depth measurements.

The posterior-control threshold must be locked from a held-out power
calculation before training. The order of gates is fixed:

```text
posterior target control -> K=1 preservation -> coverage versus temperature
-> coverage versus iso-compute depth
```

G-beta, a selector, per-trajectory halting, and SVGD remain closed unless this
corrected prior clears both coverage comparators.

## CPU Entry Point

Prepare the frozen repeated-prompt data with:

```powershell
python colab/run_stage5_phase_g_multitarget_prepare.py
```

It writes training rows, held-out posterior-control rows, manifests, and the
machine-readable pre-registration under:

```text
outputs/stage5/stage5_phase_g_multitarget_prepare_20260718/data/
```

The future GPU runner must assert the keeper checkpoint hash, require
`--sampling_policy base_problem_uniform`, reject a curriculum without multiple
targets per prompt, and record the posterior-control readout before interpreting
coverage.
