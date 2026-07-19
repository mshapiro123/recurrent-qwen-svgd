# Arm E Adapter-Parity Closure: Results, Analysis, and Publication Handoff

**Date:** 2026-07-19  
**Program:** Paper One deterministic recurrent Qwen  
**Arm:** E, rank-16 LoRA over recurrent-block projections plus repaired split bridge  
**Locked source checkpoint SHA-256:** `bffa8c4277ce82ae9f662db3243a21a50a08c4c041820c9d7506d8f250e82839`  
**Forward-active trainable parameters:** `6,007,425`  
**Pretrained Qwen parameters:** frozen and hash-checked

## 0. Executive Verdict

The registered Arm E program is complete. No additional Arm E GPU experiment
is required for Paper One.

The five requested readings are now available:

| Item | Question | Result | Status |
|:---|:---|:---|:---|
| E1 | Does matched R16-plus-bridge training reproduce the full-block depth profile? | Pooled parity, registered tail-concentrated deficit | Complete |
| E2 | Does the installed operation persist after intermediate supervision is removed? | Strong persistence | Complete |
| E3a | Does the symbolic mechanism transfer zero-shot to verbal relay and pointer surfaces? | Minimal transfer | Complete |
| E4 | Does adapter-only inverse training relax the acquisition-retention wall? | `wall_holds` | Complete |
| E5 | What is the Arm E depth frontier relative to trained support? | `11.56`, or `1.44x` support | Complete, derived |

E3b, matched natural-surface training, was explicitly excluded from this
battery. It is not needed to interpret E1-E5 and should not delay Paper One.
It remains a separately justified experiment only if the manuscript later
wants to claim adapter-budget natural-surface learnability, or if reviewers
request it.

The combined scientific result is:

> A recurrent operation can be installed with approximately 6.0M trainable
> parameters and can persist after removal of its intermediate-label scaffold.
> Its in-family pooled accuracy nearly matches full recurrent-block training,
> but far-horizon extrapolation, zero-shot surface transfer, and acquisition
> without interference are not budget-independent.

## 1. Registered Design and Lineage

Arm E was initialized from fresh Qwen2.5-0.5B with the corrected recurrent
surgery. It did not descend from a full-block keeper. Its trainable set was:

- rank-16 LoRA over all recurrent-block projections;
- repaired split bridge;
- no pretrained Qwen base parameters;
- no trainable halting parameters;
- no latent, particle, or SVGD parameters.

The trainable budget was `6,007,425`, or `3.327%` of Arm A's `180,556,929`
forward-active trained parameters, a `30.1x` reduction. This is a
trainable-parameter comparison, not a total-model-size or inference-compute
comparison.

The base-weight SHA-256 remained:

`960f8bf265ba2850c9cdd60a388a00f8f366464babe0507521f010cb7f34971f`

The parity battery used the locked Arm E final checkpoint:

`bffa8c4277ce82ae9f662db3243a21a50a08c4c041820c9d7506d8f250e82839`

E2 was an authorization test only. E4 correctly branched from the original
Arm E final checkpoint, not from the post-E2 continuation.

## 2. E1: Matched Depth Profile

### 2.1 Registered pooled result

The frozen Phase A family contained 1,792 paired rows, 128 at each depth 1-14.

| Arm | Correct | Total | Accuracy |
|:---|---:|---:|---:|
| A, full recurrent block | 1,506 | 1,792 | 84.04% |
| E, R16 plus bridge | 1,501 | 1,792 | 83.76% |
| E minus A | -5 | | -0.28 pp |

Paired outcomes were 140 helped, 145 hurt, and 1,507 tied. The exact paired
two-sided McNemar result was `p=0.8128`. Arm E passed the registered pooled
three-point margin.

### 2.2 Registered profile result

The result was not uniform across depth:

| Segment | Arm A | Arm E | E minus A | Paired p |
|:---|---:|---:|---:|---:|
| Trained support, depths 1-8 | 1005/1024 | 1021/1024 | +1.56 pp | 0.000855 |
| Near extrapolation, depths 9-11 | 326/384 | 344/384 | +4.69 pp | 0.0693 |
| Far extrapolation, depths 12-14 | 175/384 | 136/384 | -10.16 pp | 0.00394 |
| All depths | 1506/1792 | 1501/1792 | -0.28 pp | 0.8128 |

The grouped segment tests are post-hoc localization. They do not replace the
registered per-depth parity rule.

Arm E deficits exceeded the allowed 8 rows at:

- depth 12: `75/128` versus `87/128`, deficit 12;
- depth 13: `43/128` versus `57/128`, deficit 14;
- depth 14: `18/128` versus `31/128`, deficit 13.

The registered verdict was therefore `deficit`, with shape
`tail_concentrated`.

### 2.3 Interpretation

Arm E is not a failed small-budget replica. It is nearly perfect on trained
depths, remains competitive through depth 11, and then loses in the far tail.
The clean reading is parameter-efficient mechanism installation with
capacity-sensitive extrapolation.

The data do not isolate whether the far-tail loss comes from LoRA rank, frozen
base geometry, bridge capacity, optimization, or an interaction among them.
No rank sweep was registered.

## 3. E5: Derived Frontier

Arm E crossed the `0.71` threshold between:

- depth 11: `111/128 = 0.8671875`;
- depth 12: `75/128 = 0.5859375`.

Linear interpolation gives:

```text
frontier = 11.559...
frontier / trained support 8 = 1.4449...
```

The banked values are:

- depth frontier: `11.56`;
- frontier-to-support ratio: `1.44x`.

Arm A's ladder-official frontier is `11.61`. The close frontier values do not
erase the profile crossover: Arm E is better at depth 11 and materially worse
at depths 12-14.

## 4. E3a: Zero-Shot Verbal Transfer

### 4.1 Protocol

E3a was evaluation-only:

- frozen relay and pointer rows;
- 128 rows at each depth 1-12;
- forced loops equal row depth;
- same-reader full-symbol scoring;
- no natural-surface training.

The locked reporting bands were:

- strong: at least 70%;
- partial: 40% to under 70%;
- minimal: under 40%.

### 4.2 Results

| Surface | Arm E | Band | Full-block descriptive reference |
|:---|---:|:---|---:|
| Relay | 249/1536 = 16.21% | Minimal | 1321/1536 = 86.00% |
| Pointer | 264/1536 = 17.19% | Minimal | 1213/1536 = 78.97% |

Arm E retained strong depth-1 behavior:

- relay depth 1: `119/128 = 92.97%`;
- pointer depth 1: `125/128 = 97.66%`.

Performance dropped sharply from depth 2 onward. No depth beyond 1 cleared
the `0.71` same-reader threshold.

### 4.3 Interpretation

The adapter-budget recurrent operation did not transfer its multistep symbolic
procedure zero-shot to the verbal surfaces. This is evidence of
surface-specific installation, not evidence that natural-surface adapter
training would fail.

The full-block reference had received natural-surface development; therefore
E3a is descriptive and must not be written as a matched-training parity test.

## 5. E2: Outcome-Only Persistence

### 5.1 Protocol

Starting from the locked Arm E checkpoint:

- 1,000 AdamW steps;
- learning rate `1e-5`;
- bridge-prelude learning-rate multiplier `10`;
- outcome-only supervision from step one;
- intermediate chain-label coefficient fixed at zero;
- R16 LoRA and bridge remained trainable;
- pretrained Qwen weights remained frozen.

The strong gate required:

- active-label diagonal at least `0.93`;
- continuation above the diagonal at least `0.85`.

### 5.2 Results

| Metric | Arm E after outcome-only continuation | Strong floor | Full-block reference |
|:---|---:|---:|---:|
| Active diagonal | 636/640 = 99.38% | 93% | 625/640 = 97.66% |
| Continue above diagonal | 380/384 = 98.96% | 85% | 357/384 = 92.97% |
| Hold above diagonal | 0/384 | descriptive | 1/384 |
| Other | 4/384 = 1.04% | descriptive | not primary |

Registered verdict: `strong`. E4 authorization: `true`.

The post-E2 checkpoint SHA-256 was:

`e69200aa40e17f3fb7bc741a4efad63f77da88c10cf7e30140b5e25ba3253e5a`

### 5.3 Interpretation

The iterative operation survived removal of the intermediate-label scaffold at
the adapter budget. Persistence is therefore not exclusive to full recurrent-
block fine-tuning.

This result does not demonstrate learned halting. The Arm E halting trainable
count was zero, and the readout forced loop positions for diagnosis.

## 6. E4: Adapter-Budget Inverse Retention

### 6.1 Protocol

E4 correctly restarted from the original Arm E final checkpoint. It used:

- explicit inverse-table task capped at depth 3;
- 25% additive forward-synthetic rehearsal;
- intended ceiling of 334 optimizer steps;
- effective batch size 8;
- AdamW at `1e-5`;
- R16 LoRA plus split bridge only;
- inverse acquisition gate `46/64`;
- synthetic retention floor `0.93` in every stratum;
- Arm E own natural baseline with a maximum 3-point drop;
- Tier-1 reference `60/64` with the same 3-point-drop rule.

The Tier-1 accuracy floor is `0.9075`. Therefore `59/64` is green and
`58/64` is red. An earlier prose line saying "below 57/64" was mathematically
inconsistent and has been corrected; the implementation and final result used
the continuous registered floor.

### 6.2 Baselines

Before inverse training:

- natural relay/pointer canary: `60/256 = 23.44%`;
- Tier-1 arithmetic: `59/64 = 92.19%`;
- Tier-1 registered floor: `90.75%`.

The low natural baseline is consistent with E3a's minimal zero-shot transfer.
E4's natural test is therefore a within-arm retention guardrail, not a claim
that Arm E began with strong natural capability.

### 6.3 Step-100 result and hard stop

The first registered checkpoint produced:

| Gate | Result | Requirement | Status |
|:---|---:|---:|:---|
| Inverse acquisition | 2/64 = 3.13% | at least 46/64 | Fail |
| Synthetic minimum | 9.38% | at least 93% | Fail |
| Natural canary | 49/256 = 19.14% | at least 20.44% | Hard-stop fail |
| Natural delta | -4.30 pp | no worse than -3 pp | Hard-stop fail |
| Tier-1 arithmetic | 59/64 = 92.19% | at least 90.75% | Pass |

Training stopped at step 100, before the 334-step ceiling. The pretrained base
hash remained unchanged.

Registered verdict:

```text
wall_holds
joint_pass_any_checkpoint = false
all_retention_checkpoints_green = false
final_joint_pass = false
```

### 6.4 Interpretation

The adapter budget did not provide a clean preservation route for inverse-task
adaptation. Update pressure damaged both the installed synthetic recurrence
and the weak natural baseline before the inverse task was acquired.

This generalizes the boundary at the level justified by the experiment:

- neither full-block nor adapter-budget training produced a registered
  joint-passing inverse-acquisition and retention checkpoint;
- freezing the pretrained base did not prevent interference through the
  trainable recurrent adapters and bridge.

The failure modes are not identical:

- the full-block line acquired the inverse task strongly and then failed one
  or more retention gates along its checkpoint path;
- Arm E was hard-stopped at its first checkpoint with only `2/64` inverse
  accuracy and severe retention loss.

Therefore the paper should not claim that the same quantitative Pareto curve
was reproduced. The defensible statement is that the acquisition-retention
boundary remained unresolved at both trainable budgets, and adapter isolation
did not make it disappear.

## 7. Cross-Experiment Synthesis

The completed battery separates four properties that should not be collapsed
into a single "LoRA works" or "LoRA fails" sentence.

### 7.1 Installation is parameter-efficient

Six million trainable parameters were sufficient for near-perfect behavior
through trained depth 8 and pooled parity with the full-block arm.

### 7.2 Far-horizon extrapolation is budget-sensitive

Arm E's advantage through depth 11 reversed at depths 12-14. This is the
registered reason full-profile parity failed.

### 7.3 The installed operation is persistent

After 1,000 outcome-only steps, the active chain remained intact and continued
iterating above the diagonal. The operation was not merely a transient decoder
artifact sustained by per-loop labels.

### 7.4 Transfer and adaptation remain constrained

The symbolic operation did not transfer zero-shot to verbal multistep tasks,
and inverse-task adaptation damaged prior behavior before acquisition. These
are separate limitations:

- E3a is a surface-transfer negative;
- E4 is an acquisition-retention negative.

Neither result negates E1 or E2. Together they define the scope of the
parameter-efficient positive.

## 8. Manuscript-Ready Findings

### 8.1 Supported compact statement

> A rank-16 adapter-plus-bridge arm trained 6.0M forward-active parameters,
> 3.3% of the full-block arm's trainable budget, and achieved nearly identical
> pooled accuracy on the frozen 1,792-row depth evaluation (83.76% versus
> 84.04%; paired p=0.813). The adapter arm was stronger through trained support
> and near extrapolation but weaker at depths 12-14. Its installed operation
> persisted after 1,000 outcome-only steps, while zero-shot verbal transfer was
> minimal and inverse-task adaptation did not produce a checkpoint that jointly
> passed acquisition and retention.

### 8.2 Abstract-safe statement

> The recurrent operation could be installed with 6.0M trainable parameters
> and persisted after scaffold removal, but far-horizon generalization and
> interference-free adaptation remained sensitive to the trainable budget and
> task surface.

### 8.3 Recommended paper placement

- Depth-profile figure: add Arm E as the fifth curve, annotate 6.0M versus
  180.6M trainable parameters, shade trained depths 1-8, and mark the
  depth-11-to-12 crossover.
- Section 8.4: append E2 persistence results and cross-reference the training
  curriculum.
- Section 9: append E3a as minimal zero-shot verbal transfer, explicitly
  noting unmatched natural-surface training.
- New Section 10.5: report E4 `wall_holds`, its step-100 hard stop, and the
  distinction from the full-block failure path.
- Appendix A.3: replace any unmeasured "frozen-substrate preservation" claim
  with the measured result that freezing base weights did not eliminate
  interference through adapters and bridge.
- Limitations: retain single-seed Arm E training, no rank sweep, no E3b natural
  training, and no learned-halting claim.

## 9. Claims Not Supported

Do not claim:

- full-profile budget independence;
- that 6.0M parameters are the total model size;
- inference-FLOP, latency, or memory equivalence;
- that rank 16 is optimal;
- that LoRA is generally better or worse than full fine-tuning;
- natural-language generalization from E1;
- natural-surface unlearnability from E3a;
- that outcome-only persistence proves learned halting;
- that E4 reproduced the full-block quantitative Pareto path;
- that inverse adaptation is impossible under all adapter curricula;
- seed robustness;
- that every per-depth difference is multiplicity-adjusted.

## 10. Remaining Work

### 10.1 Required before Paper One submission

No additional Arm E GPU work is required. Remaining work is manuscript and
bookkeeping:

1. bank E5 values `11.56` and `1.44x` in the claim ledger;
2. add the Arm E curve and budget annotations to the depth-profile figure;
3. insert the E2, E3a, and E4 paragraphs at the registered paper locations;
4. update Appendix A.3 with the measured E4 result;
5. update the do-not-claim and limitation lists;
6. link all five Arm E receipts from the manuscript artifact table.

### 10.2 Explicitly deferred

E3b remains a separate natural-surface training experiment. It should run only
if one of these conditions is met:

- the manuscript wants a positive or negative claim about natural-surface
  learnability at the adapter budget;
- a reviewer requests matched natural training;
- Paper Two requires the adapter checkpoint as a substrate.

It is not required to close the registered E battery.

## 11. Questions for Strategy Review

1. Should the headline be "parameter-efficient installation with
   capacity-sensitive extrapolation" or the more conservative
   "trainable-budget depth crossover"?
2. Should E2's strong persistence appear in the abstract, or remain a main-text
   result to avoid overloading the headline?
3. Should E4 be described as "the wall holds" in the main text, or should that
   label remain internal while the paper states the measured joint-gate
   failure?
4. Should the post-hoc grouped depth tests appear in the main figure caption or
   in the appendix?
5. Is the single-seed limitation acceptable for submission, with replication
   reserved for reviewer response?
6. Does Paper One need E3b for narrative completeness, or is minimal zero-shot
   transfer plus explicit deferral the cleaner boundary?

## 12. Canonical Artifacts

### E1 and E5

- `outputs/stage5/stage5_adapter_budget_arm_e_20260718/summary.json`
- `outputs/stage5/stage5_adapter_budget_arm_e_20260718/adapter_budget_depth_profile.json`
- `outputs/stage5/stage5_adapter_budget_arm_e_20260718/adapter_budget_posthoc_segments.json`
- `outputs/stage5/stage5_adapter_budget_arm_e_20260718/eval/final_phase_a_1792/summary.json`
- `docs/ARM_E_ADAPTER_BUDGET_PUBLICATION_HANDOFF_20260718.md`

### E3a

- `outputs/stage5/stage5_adapter_parity_e3a_20260719/summary.json`
- `outputs/stage5/stage5_adapter_parity_e3a_20260719/summary.md`
- `outputs/stage5/stage5_adapter_parity_e3a_20260719/eval/relay/summary.json`
- `outputs/stage5/stage5_adapter_parity_e3a_20260719/eval/pointer/summary.json`

### E2

- `outputs/stage5/stage5_adapter_parity_e2_20260719/summary.json`
- `outputs/stage5/stage5_adapter_parity_e2_20260719/summary.md`
- `outputs/stage5/stage5_adapter_parity_e2_20260719/train/outcome_only/train_unfrozen_recurrent_summary.json`

### E4

- `outputs/stage5/stage5_adapter_parity_e4_20260719/summary.json`
- `outputs/stage5/stage5_adapter_parity_e4_20260719/summary.md`
- `outputs/stage5/stage5_adapter_parity_e4_20260719/train/inverse_rehearsal/train_unfrozen_recurrent_summary.json`
- `outputs/stage5/stage5_adapter_parity_e4_20260719/eval/step_100_inverse/summary.json`
- `outputs/stage5/stage5_adapter_parity_e4_20260719/guardrails/step_100_natural/summary.json`
- `outputs/stage5/stage5_adapter_parity_e4_20260719/guardrails/step_100_synthetic/summary.json`
- `outputs/stage5/stage5_adapter_parity_e4_20260719/guardrails/step_100_tier1/summary.json`

### Specification and implementation

- `docs/ARM_E_ADAPTER_PARITY_BATTERY_SPEC_20260719.md`
- `training/adapter_parity_battery.py`
- `colab/run_stage5_adapter_parity_e3a.py`
- `colab/run_stage5_adapter_parity_e2.py`
- `colab/run_stage5_adapter_parity_e4.py`
- `tests/test_adapter_parity_battery.py`

## 13. Final Program Decision

Bank Arm E as complete. Do not run a rank sweep, another Arm E continuation,
or E3b before Paper One submission. Move the active GPU lane away from Arm E.
The next work is manuscript integration and artifact verification, not another
mechanism experiment.
