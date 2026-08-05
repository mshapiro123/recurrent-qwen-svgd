# Handoff: Phase-2 Staged A1 Completion and A2 Strategy Gate

- Date: 2026-08-05
- Run: `stage5_paper2_phase2_staged_a1_resume_20260805`
- Result commit: `9c5b8ddd`
- Launcher commit: `a4f5e3ab910b1cf1e2b370821af708ed52120a79`
- Resume-amendment lock: `01ae5f28c0b52f4635dc15298bd07269bd853055`
- Original protocol lock: `c0ace7aa02ebea11fc0809298572736d29b24012`
- Status: `complete_with_strategy_gate_required`

## 0. Executive verdict

Stage A1 completed its amended 1,000-step budget in both registered seeds. Both
seeds satisfy the two locked state-construction gates:

1. functional-probe KL improved by at least `0.10` nat; and
2. flow MSE improved.

The observed margins are large. Probe KL improved by `1.3555` and `1.3157`
nats, and flow MSE fell by `19.12%` and `19.16%`. Every matched 51-batch
training audit at steps 200, 400, 600, 800, and 1,000 satisfied the amended
gradient inequalities on every constituent batch. There were no tripwire,
clipping, frozen-lineage, identity, or preservation failures.

The machine verdict is `a1_gate_candidate_pass`, because the locked amendment
requires strategy to bank A1 before A2 can be implemented or launched. The
recommended strategy reading is:

> Bank A1 as a replicated state-construction pass at alpha 0.5, without
> selecting alpha and without claiming useful state use. Authorize A2 from the
> two current step-1,000 checkpoints, but amend A2's loss-share contract before
> launch so static point targets do not become the experiment again.

Do not extend A1 automatically. Flow loss was still improving by about `0.9%`
over the final 100 steps in both seeds, but the amendment explicitly disabled
automatic extension, the registered gates are already exceeded by wide
margins, and seed 1's functional-probe KL slightly worsened over the same final
window. A2 now has greater decision value than another unselected A1 dose.

## 1. Purpose and causal question

The staged repair separates two questions that the failed matched-alpha pilot
had confounded:

- **A1, state construction:** can a frozen flow module learn a teacher-directed
  latent path under a measured, non-dominating objective?
- **A2, state use:** once that flow is frozen, can the bridge, control state,
  and draft head use it to improve speculative acceptance while preserving
  endpoint quality?

A1 trained only `module.flow`. The initializer, bridge, control state, draft
head, student and teacher embeddings, and cached targets were frozen. Executed
bridge and draft gates stayed closed. Therefore A1 cannot improve the deployed
prediction path by construction; it asks only whether the latent state can be
built.

## 2. Why the run resumed at step 200

The first A1 attempt stopped at step 200 because a single 16-row DEV estimate
was compared against weights fitted from 51 training batches of 128 rows. The
matched-estimator audit showed that this was an estimator mismatch rather than
an off-objective training run. On the exact calibration population, both saved
checkpoints satisfied the replacement inequalities on every batch.

Strategy therefore classified the first stop as
`protocol_bug_not_registered_attempt` and authorized continuation from the
preserved step-200 checkpoints under one locked amendment:

- training population owns the hard gradient-share contract;
- 51 batches of 128 rows are measured at steps 200, 400, 600, 800, and 1,000;
- flow share must be at least `0.50`;
- functional-probe share must be at most `0.25`;
- preserve share is descriptive;
- DEV shares are descriptive population-shift telemetry;
- no periodic recalibration;
- no automatic extension;
- stop at step 1,000 for strategy review; and
- A2 remains physically absent.

This is an important methodological result. Replacing an unjustified symmetric
point target with directional inequalities preserved objective identity without
requiring a continuously active shaper.

## 3. Experimental design

| Item | Locked value |
|---|---|
| Alpha | `0.5`, explicitly unselected |
| Seeds | `0`, `1` |
| Batch size | `128` |
| Nominal A1 budget | `1,000` optimizer steps per seed |
| Trainable set | Flow only |
| Optimizer | AdamW |
| Learning rate | `3e-4` |
| Weight decay | `0.01` |
| Warmup | `100` steps |
| Active losses | Flow Huber, functional-probe KL, counterfactual preserve KL |
| Executed gates | Forced closed |
| Hard share estimator | Exact seed-specific training batches 50-100, 51 batches |
| Hard share checks | Steps 200, 400, 600, 800, 1,000 |
| DEV anchors | `8,031` |
| Training anchors | `41,969` |
| Frozen confirmatory partitions | Untouched |

The counterfactual preserve loss read the frozen initialized bridge only for
training. It never changed the executed prediction. Preservation and accepted
length under closed execution are therefore integrity checks, not evidence of
downstream utility.

## 4. Primary results

### 4.1 Locked A1 gates

| Metric | Gate | Seed 0 | Seed 1 | Reading |
|---|---:|---:|---:|---|
| Functional-probe KL improvement | `>= 0.10` nat | `1.3555` | `1.3157` | Pass, both seeds |
| Flow MSE improvement | `> 0` | `0.007936` | `0.007948` | Pass, both seeds |
| Relative flow-MSE reduction | Descriptive | `19.12%` | `19.16%` | Replicated |
| A1 machine verdict | Both gates | candidate pass | candidate pass | Strategy gate required |

The KL improvements are `13.55x` and `13.16x` the registered minimum.

### 4.2 Endpoint trajectories

| Metric | Seed 0, step 0 | Seed 0, step 1,000 | Seed 1, step 0 | Seed 1, step 1,000 |
|---|---:|---:|---:|---:|
| Flow loss | `0.105874` | `0.064350` | `0.105831` | `0.064541` |
| Flow MSE | `0.041519` | `0.033582` | `0.041479` | `0.033531` |
| Functional-probe KL | `5.785360` | `4.429889` | `5.530631` | `4.214959` |
| Functional-probe top-1 | `3.89%` | `5.57%` | `3.75%` | `5.80%` |
| Counterfactual preserve KL | `3.076e-6` | `2.281e-6` | `3.191e-6` | `2.345e-6` |
| Mean endpoint ratio | `0.0458` | `0.2462` | `0.0458` | `0.2965` |
| Maximum endpoint ratio | - | `0.7423` | - | `0.8160` |

Flow loss fell by `39.22%` in seed 0 and `39.01%` in seed 1. The probe-top-1
increase is real but remains small in absolute terms. The positive A1 reading
rests on the registered distributional KL and flow-MSE gates, not on useful
token prediction.

### 4.3 Final-window slope

| Metric, step 900 to 1,000 | Seed 0 | Seed 1 |
|---|---:|---:|
| Relative flow-loss improvement | `0.896%` | `0.909%` |
| Functional-probe KL improvement | `+0.1651` nat | `-0.0277` nat |

Both flow slopes exceed the original `0.5%` extension trigger. They do not
authorize extension under the amendment. The disagreement in probe slope is
also evidence against treating the final flow slope alone as a sufficient
reason to spend another 1,000 A1 steps.

## 5. Gradient-share conformance

The amended contract was satisfied at every matched audit. Each reported
fraction below is the mean over 51 training batches; the batch-level joint-pass
fraction was `1.0` at every checkpoint in both seeds.

### Seed 0

| Step | Flow | Probe | Preserve | Joint contract |
|---:|---:|---:|---:|---|
| 200 | `74.35%` | `13.55%` | `12.10%` | Pass |
| 400 | `77.35%` | `12.88%` | `9.77%` | Pass |
| 600 | `78.17%` | `13.73%` | `8.10%` | Pass |
| 800 | `77.99%` | `14.48%` | `7.53%` | Pass |
| 1,000 | `77.60%` | `15.72%` | `6.68%` | Pass |

### Seed 1

| Step | Flow | Probe | Preserve | Joint contract |
|---:|---:|---:|---:|---|
| 200 | `72.76%` | `17.55%` | `9.69%` | Pass |
| 400 | `76.69%` | `14.57%` | `8.74%` | Pass |
| 600 | `76.90%` | `16.30%` | `6.79%` | Pass |
| 800 | `76.57%` | `17.95%` | `5.48%` | Pass |
| 1,000 | `76.47%` | `18.14%` | `5.39%` | Pass |

The fixed 51-batch DEV estimates also passed the inequalities descriptively at
every measured continuation checkpoint. The close train/DEV tracking weakens
the population-shift concern that motivated the matched audit.

## 6. Health, integrity, and lineage

| Check | Seed 0 | Seed 1 |
|---|---:|---:|
| Completed optimizer steps | `1,000` | `1,000` |
| Non-finite events | `0` | `0` |
| Gradient-clipping activations | `0` | `0` |
| Trust-tripwire exceeding steps | `0/1000` | `0/1000` |
| Frozen hash unchanged | Yes | Yes |
| Source checkpoint preserved | Yes | Yes |
| Zero-loop hidden identity | Bit exact | Bit exact |
| Zero-loop logit identity | Bit exact | Bit exact |
| Quality retention with gates closed | `1.0` | `1.0` |
| Accepted-length delta with gates closed | `0` | `0` |

Seed 0 carries two historical gradient-drift warnings from steps 100 and 200.
The receipt explicitly labels them `superseded_pre_amendment_contract`; they
are not continuation failures. Seed 1 has no warnings.

### Final A1 checkpoints

| Seed | Final checkpoint SHA-256 | Preserved step-200 source SHA-256 |
|---:|---|---|
| 0 | `823c1865878a86079a6423fabf432b6f1d36d431ec4381800846019882afb136` | `9815592e5358fbde535bec27d102717f4f9fe4a0beb9f649f0d0879f88db2c58` |
| 1 | `a9c20510f6cf2561f6208fa8d1915626e2ec6e68a588228d3f0edd9cd0efde89` | `f3538465223c2f09f286bbb276631b3ce9e60a7c3ecd43bf677d4d4c4dfb6e4e` |

The final checkpoint paths are under:

`/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/stage5_paper2_phase2_staged_a1_resume_20260805/private/a1/`

## 7. Interpretation

### Established by this run

1. The frozen flow can learn the registered latent state-construction target at
   alpha 0.5 in two seeds.
2. The functional-probe distribution moves materially toward the teacher while
   flow geometry improves.
3. The amended inequality contract remains satisfied throughout training and
   generalizes descriptively to the fixed DEV estimator.
4. The training is numerically stable and preserves the inactive computation.
5. The prior failure was a protocol/estimator defect, not evidence that the
   state-construction path was untrainable.

### Not established

1. The upper model can use the constructed state.
2. Speculative accepted length or endpoint utility improves.
3. A learned selector can capture oracle headroom.
4. Alpha 0.5 is better than alpha 0 or 1.
5. Another 1,000 A1 steps would or would not help A2.
6. DEV-only construction results will confirm on frozen evaluation material.

### Plain-language summary

The module learned to build a substantially better internal candidate state in
two independent runs. It did so without changing the model's actual answer
path, damaging the frozen model, hitting safety limits, or letting the probe
loss take over training. That closes the state-construction question. It does
not yet show that the rest of the model can read and exploit this state. A2 is
the experiment that answers that next question.

## 8. The A2 contract requires review before launch

The locked A2 outline remains scientifically appropriate:

- freeze each seed's completed A1 flow;
- train bridge, control state, and draft head;
- compare against a matched draft-head-only zero-loop control;
- require at least `+2%` relative hindsight-oracle headroom;
- require the full system to beat the matched control; and
- retain endpoint quality.

However, its present objective contract still uses static point shares:

| A2 loss | Current target |
|---|---:|
| Final CE | `35%` |
| Cumulative KL | `35%` |
| Local CE | `10%` |
| Preserve KL | `20%` |

Those targets use the same calibration formula that required amendment in A1.
The targets may be reasonable, but they are not yet empirically grounded under
the actual A2 graph. Applying them as hard point shapers would risk repeating
the central pilot mistake: a guessed balance could determine what the optimizer
is allowed to learn.

Recommended A2 amendment process, before any training:

1. Run A2's 100-batch, zero-update calibration on both A1 checkpoints.
2. Report raw and weighted per-loss gradient norms, pairwise conflict cosines,
   and batch distributions using the exact training estimator.
3. Classify each A2 mechanism as a tripwire or shaper.
4. Replace unjustified point targets with directional inequalities or
   observe-and-log telemetry, retaining hard stops only for catastrophe,
   frozen-lineage failure, identity failure, and endpoint-quality collapse.
5. Lock the matched draft-head-only control, common rows, seeds, scorer, and
   checkpoint hashes in the amendment.
6. Keep the existing state-use gates and do-not-claim boundaries unchanged
   unless strategy explicitly amends them before launch.

This is not a request for another open-ended pilot. It is a bounded prelaunch
calibration and lock repair using the lesson already established by A1.

## 9. Recommendation and decision tree

### Recommended decision

1. Ratify both `a1_gate_candidate_pass` readings as one replicated A1 pass.
2. Bank the two step-1,000 checkpoints as the A2 sources.
3. Do not extend A1 now.
4. Authorize a zero-update A2 calibration and contract amendment.
5. Launch A2 only after that amendment is locked.

### Why A2 now has higher value than A1 extension

- The registered A1 gates are exceeded by more than an order of magnitude on
  the harder KL criterion.
- Flow MSE improved almost identically in both seeds.
- The late flow slope remains positive, but the probe slope is not consistent
  across seeds.
- A2 directly tests the unresolved causal bottleneck: whether the downstream
  path can use the state.
- If A2 fails for insufficient state quality, the preserved A1 checkpoints and
  slope telemetry support a clean preregistered extension fallback.

## 10. Questions for strategy

1. May the dual-seed candidate result be banked as `a1_pass` under the amended
   protocol?
2. Is the recommendation to proceed from the current step-1,000 checkpoints,
   rather than extending A1, accepted?
3. Should A2 run both seeds symmetrically, including one matched draft-head-only
   control per seed, as the current protocol implies?
4. May the coding lane run zero-update A2 calibration before the amendment is
   locked, with no optimizer and no frozen-slice contact?
5. Which A2 loss relationships are genuinely causal requirements and which are
   only telemetry? In particular, should endpoint preservation be enforced by
   the existing quality tripwire rather than a fixed 20% gradient share?
6. Does the original `+2%` oracle-headroom gate remain the correct adequacy
   threshold after the A1 result?
7. Should low absolute probe top-1 remain descriptive, or does strategy want a
   pre-A2 adequacy floor beyond the already passed KL and MSE gates?

## 11. Do-not-claim boundaries

- Do not call A1 evidence of useful state use.
- Do not say alpha 0.5 was selected.
- Do not call the DEV metrics E1 confirmation evidence.
- Do not claim throughput, speculative speedup, or serving utility.
- Do not claim the final-window slope proves undertraining.
- Do not claim the A2 static point-share targets are validated by A1.
- Do not launch A2 before the required strategy gate and any resulting
  amendment lock.

## 12. Canonical artifacts

- Completion summary:
  `outputs/stage5/stage5_paper2_phase2_staged_a1_resume_20260805/summary.json`
- Matched-estimator audit:
  `outputs/stage5/stage5_paper2_phase2_a1_matched_estimator_audit_20260805/summary.json`
- Original protocol:
  `docs/PAPER2_PHASE2_STAGED_REPILOT_PROTOCOL_DRAFT_20260805.md`
- Resume amendment:
  `docs/PAPER2_PHASE2_STAGED_A1_RESUME_AMENDMENT_20260805.md`
- Machine-readable registration:
  `training/paper2_phase2_staged_repilot_preregistration.json`
- Protocol-stop archaeology:
  `docs/PAPER2_PHASE2_STAGED_A1_PROTOCOL_STOP_HANDOFF_20260805.md`

## 13. Requested strategy response

Please return one explicit branch:

- `BANK_A1_AND_AUTHORIZE_A2_CALIBRATION`, with any required A2 objective
  inequalities and unchanged gates named;
- `EXTEND_A1`, with the added budget and the interpretation of the inconsistent
  terminal probe slope specified before launch; or
- `HOLD`, with the missing receipt or analysis named.

No GPU work is required while this strategy gate is open.
