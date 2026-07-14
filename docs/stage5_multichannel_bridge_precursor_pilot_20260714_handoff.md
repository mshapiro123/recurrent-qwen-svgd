# Handoff: Multi-Channel Bridge Precursor Pilot

**Run:** `stage5_multichannel_bridge_precursor_pilot_20260714`  
**Status:** finished, eval-only  
**Checkpoint:** `n24_step6000`, SHA-256 `898a259db2ab344ece4545e2910b051840e8408dbe4927f799e7cdb3cdd8c7dc`

## Executive Reading

The pilot is a technically clean partial positive. It confirms that rendered-table retrieval is concentrated in stable attention-head locations beyond a matched random-rotation null. It does **not** show that the carried re-entry state has the stronger head-aligned drift specialization needed to justify replacing the single bridge with a learned multi-channel bridge.

This is a near miss only in the programmatic sense: one useful channel-level observation is positive. It is not a numerical near miss on M1. M1 cleared the random-null check at every eligible late loop, but its effect size remained materially below the locked twofold threshold.

The decision remains `remain_banked`. No bridge intervention, full battery, or queue change is authorized by this result.

## Scope

The tested basis was the final recurrent layer's output-projection query-head input-column blocks, independently orthonormalized. This is meaningful but not privileged, so every claim used 20 matched random orthogonal rotations.

| Measurement | Question | Pilot scope |
|---|---|---|
| M1: subspace drift | Is late-loop drift concentrated in query-head write subspaces beyond random partitions? | One frozen row at each depth 1-14, loops 1-14 |
| M2: retrieval census | Do stable layer-head positions retrieve table tokens beyond a random-rotation null? | Same 14 rows and loops |
| M3: injection sensitivity | Is prompt-injection ablation damage concentrated by head subspace? | Intentionally not run |

## M1: Drift Is Structured but Smeared

M1 was outside the random p95 at **all 9 eligible loops** (6-14). This shows the head basis is not indistinguishable from arbitrary partitions. The locked criterion is stronger: the head basis must carry at least `2.0x` the random top-three drift share at at least 75% of eligible loops.

| Metric | Result |
|---|---:|
| Eligible late loops | 9 |
| Loops outside random p95 | 9 / 9 |
| Loops meeting locked 2.0x bar | 0 / 9 |
| Observed advantage range | `1.370x` to `1.419x` |
| Required qualifying loops | 7 |
| Reading | `smeared` |

Interpretation: late drift has some query-head alignment, but not enough to claim that a small number of head-aligned carry channels are the natural organization of the re-entry state. The relevant question is whether this basis locates materially more drift than an arbitrary matched basis. Under the locked effect-size standard, it does not.

## M2: Retrieval Heads Exist Locally

M2 was positive on the N24 checkpoint.

| Metric | Result |
|---|---:|
| Qualified stable layer-head positions | 37 |
| Per-head requirement | ratio `>= 3.0x` and stability `>= 0.50` |
| Aggregate top-two concentration | `0.7404` |
| Random-rotation p95 | `0.5229` |
| Aggregate random-null result | pass |
| Reading | `retrieval_heads_exist` |

The `3.0x` criterion applies to each individual layer-head's stable retrieval ratio. The `0.7404 > 0.5229` comparison is a separate aggregate random-rotation control. The classifier correctly requires both at least two individually qualifying stable positions and the aggregate null win.

Interpretation: the model uses identifiable, stable head locations to access the rendered function table. This is useful mechanistic evidence for the existing model and supports targeted retrieval diagnostics. It is not yet evidence that the *re-entry carry* needs multiple channels: M2 measures retrieval access, whereas M1 measures the carried-state drift that a multi-channel carry would change.

## What Remains Open

M3 did not run by design: the pilot was intended to validate attention capture and cheaply read M1/M2 before intervention passes. M1 and M2 also have a locked replication rule: both must be evaluated on `n24_step6000` and `backward_recovery`. This pilot has only the first condition. M2 is therefore **positive but pending replication**, not a battery vote.

The staircase prerequisite remains negative:

```text
reading: experiment_stalled_at_matched_dose
reading_one: false
```

Even a full battery pass would remain blocked until the per-position installation-cost condition is established.

## Decision

Activation requires both: (1) at least two of M1/M2/M3 confirmed under their replication rules and (2) a positive staircase reading. Neither condition holds. Building a multi-channel bridge now would turn an interesting retrieval observation into an unearned architectural inference.

The run also repaired a real environment compatibility issue: Transformers 5 Qwen decoder layers discard attention weights at their return boundary. The evaluator now captures them using hooks on each `self_attn` module. This was unit-tested on a real tiny eager Qwen model; the full repository suite passed `1849` tests.

## Recommended Next Actions

Do not change the main inverse/G-alpha queue. This is an eval-only idle-lane precursor.

If an idle L4 slot exists and the strategy group wants to preserve the M2 observation, run one bounded replication on `backward_recovery`: M1/M2 only, one frozen row per depth 1-14, loops 1-14, 20 random rotations, same basis and thresholds, a new run ID, and an explicit checkpoint-SHA receipt. It tests whether M2 is task-relevant recurrent retrieval generally or only an N24-specific pattern. It cannot open the bridge arm alone because the staircase gate remains closed.

Do **not** launch the 64-row four-checkpoint full battery, run M3 merely to search for a second positive vote, weaken M1's `2.0x` bar after observing `1.37x-1.42x`, or claim per-head carry specialization from retrieval-head evidence.

## Questions for Strategy Review

1. Is the bounded `backward_recovery` M1/M2 replication worth an idle L4 slot as a descriptive result, despite being unable to open the bridge arm while the staircase gate is negative?
2. If M2 replicates while M1 remains smeared, should the paper frame retrieval specialization as a characterization of the recurrent block rather than a motivation for changing the bridge?
3. Is there a stronger, independently motivated carried-state basis to test later, such as learned bridge singular directions or an intervention-derived basis? Any new basis must retain a matched random-partition null and be declared before use.

## Source Artifacts

- `outputs/stage5/stage5_multichannel_bridge_precursor_pilot_20260714/summary.json`
- `outputs/stage5/stage5_multichannel_bridge_precursor_pilot_20260714/conditions/n24_step6000/m1_subspace_drift.json`
- `outputs/stage5/stage5_multichannel_bridge_precursor_pilot_20260714/conditions/n24_step6000/m2_retrieval_heads.json`
- `colab/run_stage5_multichannel_bridge_precursor.py`
- `eval/eval_multichannel_bridge_precursor.py`
