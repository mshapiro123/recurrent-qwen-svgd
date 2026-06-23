# Stage 5 CE8 Balanced ARC Depth Curve

## Summary

The CE8 learned-loop-control checkpoint has now been evaluated on balanced
256-example ARC-Easy and ARC-Challenge slices at fixed recurrent depths 1-4
under both cyclic option-permutation scoring and content-question-only scoring.
This is the cleanest depth-allocation result so far.

The result is not a release-grade base-model win. It is a useful mechanism
result:

- shallow recurrence is best for easy/direct behavior;
- deeper recurrence improves the harder ARC-Challenge content readout;
- unconditional deeper recurrence damages ARC-Easy content calibration;
- the learned-loop-control flag did not change the earlier ARC-128 benchmark
  outputs, so the current deployed benchmark gains should be attributed to the
  trained recurrent checkpoint and fixed max-loop depth, not to a working
  learned router.

The next training target is therefore not "make everything deeper." It is
depth-conditional preservation: keep direct/easy rows at depth 1 while routing
harder rows toward depth 2-4.

## Runs

| Depth | Run ID | Elapsed |
|---:|---|---:|
| 1 | `stage5_ce8_balanced_arc256_maxloop1_20260623_075948` | 881.4s |
| 2 | `stage5_ce8_balanced_arc256_maxloop2_20260623_085130` | 1264.6s |
| 3 | `stage5_ce8_balanced_arc256_maxloop3_20260623_085130` | 1691.9s |
| 4 | `stage5_ce8_balanced_arc256_maxloop4_20260623_075948` | 2057.4s |

All four runs used:

- checkpoint:
  `outputs/stage5/stage5_learned_loop_control_ce8_continue_l4_20260623_071135/phase1/phase1_step_400.pt`
- recurrent mode: Phase 1 deterministic recurrent;
- particles/SVGD: off;
- learned loop control: off in these fixed-depth runs;
- examples per benchmark: 256.

## Result Table

| Depth | Benchmark | Scoring | Base | Recurrent | Delta | Wins | Losses | Ties | Sign p |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | ARC-Easy | cyclic | 202/256 | 206/256 | +4 | 4 | 0 | 252 | 0.1250 |
| 1 | ARC-Easy | content | 146/256 | 131/256 | -15 | 3 | 18 | 235 | 0.0015 |
| 1 | ARC-Challenge | cyclic | 154/256 | 151/256 | -3 | 5 | 8 | 243 | 0.5811 |
| 1 | ARC-Challenge | content | 87/256 | 88/256 | +1 | 8 | 7 | 241 | 1.0000 |
| 2 | ARC-Easy | cyclic | 202/256 | 204/256 | +2 | 2 | 0 | 254 | 0.5000 |
| 2 | ARC-Easy | content | 146/256 | 127/256 | -19 | 7 | 26 | 223 | 0.0013 |
| 2 | ARC-Challenge | cyclic | 154/256 | 153/256 | -1 | 8 | 9 | 239 | 1.0000 |
| 2 | ARC-Challenge | content | 87/256 | 90/256 | +3 | 16 | 13 | 227 | 0.7111 |
| 3 | ARC-Easy | cyclic | 202/256 | 204/256 | +2 | 2 | 0 | 254 | 0.5000 |
| 3 | ARC-Easy | content | 146/256 | 127/256 | -19 | 8 | 27 | 221 | 0.0019 |
| 3 | ARC-Challenge | cyclic | 154/256 | 153/256 | -1 | 8 | 9 | 239 | 1.0000 |
| 3 | ARC-Challenge | content | 87/256 | 92/256 | +5 | 18 | 13 | 225 | 0.4731 |
| 4 | ARC-Easy | cyclic | 202/256 | 204/256 | +2 | 2 | 0 | 254 | 0.5000 |
| 4 | ARC-Easy | content | 146/256 | 127/256 | -19 | 8 | 27 | 221 | 0.0019 |
| 4 | ARC-Challenge | cyclic | 154/256 | 153/256 | -1 | 8 | 9 | 239 | 1.0000 |
| 4 | ARC-Challenge | content | 87/256 | 92/256 | +5 | 18 | 13 | 225 | 0.4731 |

## Interpretation

ARC-Challenge content-question-only accuracy improves from depth 1 through
depth 3:

```text
depth 1: +1 over base
depth 2: +3 over base
depth 3: +5 over base
depth 4: +5 over base
```

This is the most direct evidence so far that additional recurrent computation
can help the harder slice. The effect is small and not statistically decisive
at 256 examples, but it is directionally aligned with the depth-ladder thesis.

ARC-Easy content-question-only accuracy is much worse than base at every fixed
depth:

```text
depth 1: -15 versus base
depth 2: -19 versus base
depth 3: -19 versus base
depth 4: -19 versus base
```

This is not just a "too much depth" problem, because even depth 1 is behind
under the content-only readout. The likely issue is answer calibration and
direct-route preservation, not recurrence itself.

Cyclic option-permutation scoring is more favorable to the recurrent checkpoint
on ARC-Easy and less negative on ARC-Challenge:

```text
ARC-Easy cyclic:      +4 at depth 1, +2 at depths 2-4
ARC-Challenge cyclic: -3 at depth 1, -1 at depths 2-4
```

This confirms that bare/content MCQ scoring can overstate label-position
failures, but it does not erase the need for content preservation.

## Consequence For The Next Experiment

The immediate next experiment should train or select depth conditionally:

1. preserve base behavior on direct/easy/base-correct rows with target depth 1;
2. route ambiguous or hard rows toward depth 2-3;
3. keep ARC-Easy content calibration as a hard guardrail;
4. evaluate with both cyclic and content scoring;
5. compare fixed depths, learned router, and a selector baseline.

The target success condition is:

```text
ARC-Easy cyclic remains non-negative,
ARC-Easy content gap shrinks materially,
ARC-Challenge content keeps the depth-3/4 gain,
and learned routing beats any single fixed depth on the paired metric.
```

