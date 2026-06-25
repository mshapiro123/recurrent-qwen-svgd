# Latent Recurrent-Particle Adaptation of a Trained Qwen Model

## Abstract

We study whether a pretrained dense language model can be surgically converted
into a recurrent-depth model, then recovered with a small trainable parameter
budget so that reasoning moves partly into latent iterative computation. The
current system wraps `Qwen/Qwen2.5-0.5B-Instruct` into a
Prelude/Recurrent-Block/Coda architecture, adds sequence-level PonderNet-style
halting, and extends the recurrent state into a family of particle trajectories
with stochastic latent injection or SVGD-style repulsion. The strongest
historical non-toy recovery result is a bounded ARC-mixed Phase 1 checkpoint
that beat the unmodified base model on one 256-example ARC-Easy/ARC-Challenge
confirmation slice under content-question scoring while matching or slightly
exceeding base under cyclic option-permutation scoring. Subsequent diagnostics
showed that this was not yet a robust full-split or GPQA claim, and that the
loop-closure path was not trainably reconciled: the bridge was effectively
dead and the loop dynamics looked expansive rather than multistable. The
current scientific gate is therefore Phase 0 re-entry repair, followed by
bounded deterministic depth recovery and a same-curriculum dense-Qwen control.
Only after that control does it become meaningful to return to breadth,
particles, SVGD, or selector-conversion claims.

## 0. Result Maturity

This document is intentionally written like a manuscript draft, but the claim is
still bounded. The project now has an ARC-256 surpass-base result, not a
full-benchmark or release claim. The established results are method and bounded
recovery results:

- exact one-pass identity preservation for the wrapped architecture;
- stable learned recurrence and sequence-level halting;
- substantial recovery of the competence lost by model surgery;
- a bounded ARC-256 recurrent-vs-base win for a targeted deterministic Phase 1
  checkpoint;
- measurable candidate-diversity gains from particle/SVGD mechanisms on
  controlled exact-task suites.

The not-yet-established result is robust benchmark superiority over unmodified
Qwen 0.5B. That now requires a stricter sequence: repair loop closure, recover
deterministic recurrent competence with learned depth, compare against base on
debiased MCQ surfaces, and then compare against a standard dense Qwen LoRA
trained on the same curriculum. The paper should not claim that recurrence,
SVGD, or particles have made the model surpass base until those gates are
logged in `outputs/stage5` and cited from the exact checkpoint summaries.

## 1. Research Question

The central question is not whether a transformer can be looped. The question is
whether a trained dense model can be converted into a recurrent latent-reasoning
model while preserving enough of its original capability that additional
training can make the modified architecture competitive or superior on hard
reasoning.

The fair comparison therefore has three layers:

1. **Base model:** unmodified `Qwen/Qwen2.5-0.5B-Instruct`.
2. **Recovered deterministic recurrent model:** the surgically modified model
   after Phase 1 adapter/controller training.
3. **Recurrent-particle model:** the recurrent model with multiple latent
   trajectories and SVGD-style trajectory separation.

An early recurrent regression versus base is expected, because the architecture
has been changed after pretraining. The research claim becomes interesting only
if minimal additional training recovers the lost capability and if particles
then add hard-tail candidate coverage beyond deterministic recurrence.

## 2. Architectural Intervention

The wrapper divides Qwen into three regions:

- **Prelude:** early transformer layers.
- **Recurrent block:** middle transformer layers repeatedly applied over the
  same hidden sequence.
- **Coda:** final transformer layers, normalization, and language-model head.

For the 0.5B model the working split is `6,18` in the current Colab pipeline.
Earlier handoff notes used a larger-model conceptual split of `8,24`. The
implementation preserves the same attention masks, position ids, rotary/cache
behavior, dtype placement, and gradient-checkpoint compatibility required by the
base model.

The recurrent bridge is initialized as an identity-preserving gated map:

```text
delta = bridge(h) - h
h = h + bridge_gate * delta
```

with identity bridge weights, zero bias, and `bridge_gate=0`. This avoids the
hidden-state doubling bug that would occur from `h = h + bridge(h)` under an
identity-initialized bridge.

## 3. Training Components

Only a small set of parameters is trained in the recurrent adaptation:

- LoRA adapters inside the recurrent block;
- the recurrent bridge and gate;
- a sequence-level halting predictor;
- optional latent policy and latent adapter modules for Phase 2;
- optional SVGD particle update hyperparameters at inference/training time.

The deterministic Phase 1 loss is:

```text
sum_n lambda_n * CE_n + beta * KL(lambda || target_loop_prior)
```

where `lambda_n` is the PonderNet halting probability at loop depth `n`. The
initial stable Colab profile used fp32 trainable adapters/controllers while the
frozen Qwen base remained in bf16/fp16, with fail-fast nonfinite guards.

The Phase 2 objective adds stochastic latent trajectories or SVGD-style particle
updates, plus small latent/diversity regularizers. In practice, the current
stronger evidence for particles comes from inference-time SVGD diagnostics over
the recovered recurrent state rather than from a fully trained Phase 2 model.

## 4. Dataset Program

The immediate reasoning-trace sources are:

- `lordx64/reasoning-distill-opus-4-7-max-sft`, a Qwen-template SFT dataset
  with Opus-generated thinking traces;
- `lordx64/reasoning-distill-claude-opus-4-7-max`, the raw Opus source with
  thinking/response fields and provenance;
- `Jackrong/Claude-opus-4.7-TraceInversion-5000x`, now handled by a dedicated
  `trace_inversion` adapter that preserves `inverted_reasoning` as the
  recurrent latent trace.

Fable is treated differently:

- `Glint-Research/Fable-5-traces` is valuable, but the flat file
  `fable5_cot_merged.jsonl` is long and tool-heavy. It is better suited to
  later agent/tool trajectory diversity experiments than immediate ARC/GPQA
  competence recovery.
- `Glint-Research/Complete-FABLE.5-traces-2M` is a large mining source with
  `row_json` wrappers. It should be audited and filtered in streaming mode
  before any training.

The current policy is to recover deterministic recurrent competence using Opus
and TraceInversion-style reasoning traces first, while holding Fable for a later
coding/tool trajectory track.

The practical reason is credit discipline. Dataset discovery should not consume
A100 time. Opus-style rows are already close to ordinary supervised reasoning
fine-tuning, while Fable rows often encode agent/tool behavior whose value is
likely in trajectory diversity rather than immediate ARC/GPQA competence
recovery. The current dataset policy is therefore:

| Source | Near-term status | Rationale |
|---|---|---|
| `lordx64/reasoning-distill-opus-4-7-max-sft` | trainable now after filtering | SFT-ready Qwen-template reasoning traces; already used in Stage 4/5. |
| `lordx64/reasoning-distill-claude-opus-4-7-max` | audit/filter | Raw source exposes richer fields for trace-length and provenance curriculum. |
| `Jackrong/Claude-opus-4.7-TraceInversion-5000x` | immediate audit candidate | Preserves an explicit inferred reasoning field that maps naturally to latent recurrent supervision. |
| `Glint-Research/Fable-5-traces` | hold for agent/tool filter | Valuable trace corpus, but likely confounds competence recovery if blindly mixed into MCQ/ARC training. |
| `Glint-Research/Complete-FABLE.5-traces-2M` | streaming audit only | Large source for future mining; too broad for direct small-model recovery. |

This is not a judgment that Fable is low value. It is a sequencing decision:
use Opus/TraceInversion to recover deterministic competence first, then use
Fable-like traces when the project is explicitly training multi-path
agent/tool/coding trajectories.

## 5. Evidence So Far

### 5.1 Identity Preservation

The one-pass identity wrapper was verified under strict settings after moving to
float32/eager attention: the wrapped model with `max_loops=1` matched base Qwen
with `max_abs_diff=0.0` and `mean_abs_diff=0.0` in the successful gate. This
establishes that the manual Prelude/Recurrent/Coda path can represent the base
model computation exactly under the identity setting.

### 5.2 Stable Recurrent Halting

The early G4/A100 stabilization work found that fp16 trainable adapters caused
NaNs. Moving LoRA, bridge, halting, and latent modules to fp32 while keeping the
frozen base in low precision stabilized training. A stable Phase 1 continuation
validated with finite held-out loss and non-collapsed loop depth:

```text
expected_ce = 2.742439
halting_kl = 0.448426
loss = 2.778313
mean_expected_loops = 2.903603
mean_halt_entropy = 1.267266
```

This established that deterministic recurrent-depth training can proceed
without halting collapse.

### 5.3 SVGD Candidate-Diversity Signal

On the original five-task exact smoke suite, temperature sampling produced broad
but sparse candidates. SVGD-style recurrent particles with drift/noise and
repulsion improved candidate density:

| Setting | Oracle best-of-K | Candidate hits |
|---|---:|---:|
| Temperature-only, `temp=0.7` | `2.6/5` | `4.0/20` |
| SVGD drift/noise, `repulsion=0` | `1.8/5` | `6.8/20` |
| SVGD drift/noise/repulsion, `repulsion=1` | `2.4/5` | `9.6/20` |

On the broader 14-task exact suite, the same pattern held more weakly:

| Method | Oracle best-of-K | Candidate hits |
|---|---:|---:|
| Temperature-only | `7.333/14` | `18.667/56` |
| SVGD drift/noise, `repulsion=0` | `8.333/14` | `26.667/56` |
| SVGD drift/noise/repulsion, `repulsion=1` | `8.667/14` | `28.667/56` |

The interpretation is that SVGD is not merely adding textual randomness. It can
increase the density of useful candidates. However, this is smoke-suite
evidence, not a benchmark win.

### 5.4 Kernel Geometry Diagnostics

Raw hidden-space repulsion was weak relative to recurrent drift. Random
projection and within-group projection diagnostics showed that the kernel should
be calibrated on within-prompt particle variation rather than global task
variation. A later held-out exact-task diagnostic reported:

```text
Random32 baseline, seeds 0-9:
  best_hits = 77/140
  candidate_hits = 251/560

Within-group PCA dim8 repulsion=2, seeds 0-9:
  best_hits = 89/140
  candidate_hits = 264/560
```

This is a useful mechanism signal: within-group particle geometry improves
oracle/candidate coverage on controlled tasks. It does not yet show that
particles help after the deterministic recurrent model is strong.

### 5.5 Non-Toy ARC Evidence

The Stage 4 modified-Opus fine-tune narrowed the base gap on an ARC-Challenge
128-question slice:

| Model | Accuracy |
|---|---:|
| Base Qwen 0.5B | `72/128` (`56.25%`) |
| Phase 1 deterministic recurrent | `70/128` (`54.69%`) |
| Phase 2/SVGD candidate | `69/128` (`53.91%`) |

This was the first important practical recovery result. It showed that
recurrent adapter training can recover most of the performance lost by surgical
architecture modification. It also showed that the Phase 2/SVGD candidate did
not improve over the stronger Phase 1 recurrent baseline on this slice.

Subsequent Stage 5 recovery work found that a balanced ARC/Opus mix can produce
proxy lift. One selected balanced checkpoint was:

```text
outputs/stage5/stage5_balanced_recovery_autopilot_current_arc_mix/
  arc_mix_nodistill_lr3e6/phase1/phase1_step_150.pt
```

The full balanced MCQ assessment reports:

| Benchmark | Base Qwen | Recurrent Phase 1 | Delta |
|---|---:|---:|---:|
| ARC-Easy | `421/570` (`73.86%`) | `412/570` (`72.28%`) | `-9` |
| ARC-Challenge | `167/299` (`55.85%`) | `169/299` (`56.52%`) | `+2` |
| Combined | `588/869` (`67.66%`) | `581/869` (`66.86%`) | `-7` |

The paired combined sign test was not significant
(`wins=33`, `losses=40`, `ties=796`, `p=0.4828`). The right interpretation is
encouraging but not yet a release claim: the recurrent model can slightly beat
base on the harder ARC-Challenge slice, but it still gives back more points on
ARC-Easy. The full assessment therefore correctly remains
`needs_competence_recovery`.

That checkpoint was superseded by a later proxy-selected full confirmation run
that trailed base on both ARC-Easy and ARC-Challenge under bare-label MCQ
scoring:

```text
run_id = stage5_full_assessment_once_20260622_005522
status = needs_competence_recovery
ARC-Easy:      base 421/570, recurrent 415/570, delta -6
ARC-Challenge: base 167/299, recurrent 164/299, delta -3
Combined:      base 588/869, recurrent 579/869, delta -9
```

The follow-up diagnosis points to answer-calibration drift under the bare
`A/B/C/D` readout. A newer debias diagnostic then showed that this drift is
substantially confounded by MCQ option-label/position bias: on ARC-Easy, loop-4
recurrent accuracy fell from `87/128` base to `82/128` under bare labels, but
rose to `97/128` versus `96/128` base under cyclic option-permutation
aggregation. The current gate is therefore not more training; it is to confirm
the same debiased scoring on ARC-Challenge before deciding whether real content
degradation remains.

See [MCQ_DEBIAS_STATUS.md](MCQ_DEBIAS_STATUS.md).

The follow-up CE8 balanced ARC depth curve gives a cleaner conditional result.
Across fixed depths 1-4 on 256-example balanced slices:

| Depth | Benchmark | Scoring | Base | Recurrent | Delta |
|---:|---|---|---:|---:|---:|
| 1 | ARC-Easy | cyclic | `202/256` | `206/256` | `+4` |
| 1 | ARC-Easy | content | `146/256` | `131/256` | `-15` |
| 2 | ARC-Challenge | content | `87/256` | `90/256` | `+3` |
| 3 | ARC-Challenge | content | `87/256` | `92/256` | `+5` |
| 4 | ARC-Challenge | content | `87/256` | `92/256` | `+5` |

The full result is documented in
[STAGE5_CE8_DEPTH_CURVE_2026_06_23.md](STAGE5_CE8_DEPTH_CURVE_2026_06_23.md)
and summarized at
`outputs/stage5/stage5_ce8_balanced_arc256_depth_curve_summary_20260623/summary.json`.

This changes the current interpretation. The recurrent model is not simply
weaker or stronger than base. It has a conditional profile: shallow recurrence
is best for ARC-Easy cyclic scoring, while depth 3-4 is best for
ARC-Challenge content scoring. The remaining regression is concentrated in
ARC-Easy content calibration, including at depth 1. The next recovery recipe
should therefore train deterministic depth allocation and direct-mode answer
calibration before reintroducing particle width. See
[DEPTH_WIDTH_ROUTING_RECIPE.md](DEPTH_WIDTH_ROUTING_RECIPE.md).

The first targeted ARC-mix follow-up after that diagnosis produced the first
bounded non-toy recurrent-vs-base win. The selected checkpoint is:

```text
outputs/stage5/stage5_content_arcmix_qonly_optiontext_20260623_121707/
  arc_mix_response_w02_lr2e6/phase1/phase1_step_150.pt
```

On the 256-example ARC confirmation slice:

| Benchmark | Scoring | Base | Recurrent | Delta |
|---|---|---:|---:|---:|
| ARC-Easy | cyclic option permutation | `202/256` | `204/256` | `+2` |
| ARC-Easy | content question-only | `146/256` | `155/256` | `+9` |
| ARC-Challenge | cyclic option permutation | `154/256` | `154/256` | `0` |
| ARC-Challenge | content question-only | `87/256` | `97/256` | `+10` |

The run is
`outputs/stage5/stage5_content_arcmix_qonly_optiontext_arc256_check_20260623_123424/summary.json`.
There were no evaluation failures. The paired evidence is positive but still
not decisive on sign-test p-values (`p=0.136` for ARC-Easy content and
`p=0.110` for ARC-Challenge content), so the correct next step is an
independent offset-256 confirmation using the same checkpoint, not immediate
Phase 2 scaling or a public benchmark claim.

## 6. What Has Not Been Shown Yet

The project has not yet shown:

- a recurrent checkpoint that robustly surpasses base Qwen 0.5B across an
  independent offset or full non-toy benchmark split;
- a Phase 2/SVGD checkpoint that beats the strongest recovered Phase 1
  recurrent checkpoint on ARC/GPQA-style tasks;
- a claim-level selector that converts recurrent candidate coverage into
  selected-answer benchmark lift;
- GPQA Diamond results, because access/preparation has not yet been reliable in
  the current account/runtime.

This is not a failure state. It is the correct state for a model-surgery
research program before the competence-recovery and selector gates have passed.

## 7. Training Required To Surpass Base

After the CE8 depth curve, the next training recipe should prioritize
deterministic conditional-depth recovery before further particle training:

1. **Depth-1 preservation.** Keep easy/direct/base-correct rows at depth 1 with
   strong answer-preservation or base-distillation pressure.
2. **Hard-row depth routing.** Route ambiguous and harder rows toward depth 2-3
   using verified difficulty buckets rather than trace length alone.
3. **Dual MCQ scoring.** Optimize and report cyclic option-permutation scoring
   alongside content-question-only scoring. Treat ARC-Easy content loss as a
   hard guardrail.
4. **Selector-first particle gate.** Before training Phase 2 heavily, test
   low-noise K-particle settings against the recovered Phase 1 checkpoint with a
   selector metric. Require helped examples to exceed hurt examples.
5. **Spectrum distillation for particles.** If particle screening passes, train
   particles against multiple correct trace families per problem rather than
   arbitrary hidden-state diversity. Use set/coverage losses so distinct
   particles are rewarded only when they remain correct.
6. **Matched dense control.** Train a standard dense LoRA control through the
   same recipe. Architecture claims require recurrent-vs-dense evidence under
   matched data, steps, base model, and selector.

Only after those gates pass should the project spend heavily on Phase 2/SVGD,
1.5B/3B models, or GPQA Diamond.

## 8. Immediate Credit-Saving Gate

The current A100 policy is deliberately conservative: spend paid GPU only on
the next gate needed for the paper-level claim, and stop whenever the next
action would be exploratory, ambiguous, or repair-oriented. Dataset audits,
notebook editing, GitHub/Drive fixes, documentation, and CPU-sized tests should
not run on A100.

The active paid-GPU gate has moved from MCQ debias confirmation to
deterministic recovery replication. The bounded CE8 depth curve showed where
recurrence helped and where it hurt; the follow-up ARC-mixed content objective
then produced a positive 256-example confirmation slice. The next spend should
replicate that checkpoint on an independent ARC offset before launching more
training.

The prior full balanced ARC assessment reported `needs_competence_recovery`:
the recurrent checkpoint slightly exceeded base on ARC-Challenge but trailed on
ARC-Easy. That result made the next justified paid job a single ARC-mix
recovery proxy, not GPQA, not Phase 2/SVGD, and not another kernel-geometry
sweep. The proxy resumed from the selected full-assessment checkpoint and ran a
single mixed objective with:

- Opus reasoning traces as the general reasoning anchor;
- ARC-Challenge training rows repeated `2x`;
- ARC-Easy training rows repeated `4x`;
- response-level base-logit distillation weight `0.05`;
- learning rate `2e-6`;
- proxy MCQ eval limit `128`.

The full balanced assessment that motivated this recovery proxy was:

```text
run_id = stage5_recovery_full_assessment_current
status = needs_competence_recovery
ARC-Easy:      base 421/570, recurrent 412/570, delta -9
ARC-Challenge: base 167/299, recurrent 169/299, delta +2
Combined:      base 588/869, recurrent 581/869, delta -7
```

The new proxy run was:

```text
run_id = stage5_arc_mix_recovery_once_20260622_003331
status = proxy_lift
base proxy = 68/128
start proxy = 66/128
best recurrent proxy = 68/128
lift vs start = +2
gap vs base = 0
best checkpoint = outputs/stage5/stage5_arc_mix_recovery_once_20260622_003331/arc_mix_response_w005_lr2e6/phase1/phase1_step_50.pt
```

This was the cleanest recovery proxy so far: it improved the recurrent start
and matched base on the 128-row ARC-Challenge proxy. The immediate confirmation
launcher was:

```bash
python colab/run_stage5_full_assessment_once.py
```

It defaulted to
`outputs/stage5/stage5_arc_mix_recovery_once_20260622_003331/summary.json` and
ran the full ARC-Easy/ARC-Challenge balanced assessment for the proxy-selected
checkpoint. That confirmation spend did **not** pass:

```text
run_id = stage5_full_assessment_once_20260622_005522
status = needs_competence_recovery
ARC-Easy:      base 421/570, recurrent 415/570, delta -6
ARC-Challenge: base 167/299, recurrent 164/299, delta -3
Combined:      base 588/869, recurrent 579/869, delta -9
```

The local regression diagnosis at
`outputs/stage5/stage5_full_assessment_once_20260622_005522/mcq_regression_diagnosis.md`
shows that the failure is not just a small count fluctuation. The recurrent
checkpoint shifted answer calibration:

- ARC-Easy: wins/losses/tie-correct/tie-wrong `18/24/397/131`, mean margin
  delta `-1.2373`;
- ARC-Challenge: wins/losses/tie-correct/tie-wrong `11/14/153/121`, mean
  margin delta `-0.3608`;
- answer prior drift: recurrent over-predicts `C` and under-predicts `A` on
  both ARC-Easy and ARC-Challenge.

This gate is intentionally deterministic Phase 1 recovery work. Phase 2/SVGD,
GPQA Diamond, and 1.5B/3B scaling remain deferred until deterministic recurrent
competence is at least base-competitive on the balanced ARC suite.

The proxy gate has therefore been strengthened. Future ARC-mix proxy summaries
include paired margin deltas, correct-answer score deltas, answer-prior shifts,
and a calibration pass/fail flag. A checkpoint that lifts proxy accuracy but
degrades calibration is no longer allowed to trigger a full paid balanced ARC
assessment. This makes the next A100 spend answer the actual recovery question:
whether the recurrent model is improving while preserving the base model's
answer distribution, rather than merely moving errors around.

The newer content-question ARC-mix run supersedes this older failed
confirmation as the active recovery candidate. It shows that additional
ARC-mixed recurrent SFT can cross the base line on a bounded non-toy slice, but
the result still needs independent-offset or full-split confirmation before it
can carry a paper-level benchmark claim. The next hypothesis is no longer
"blindly add traces"; it is that direct content-surface ARC supervision plus
moderate base-response distillation can repair the recurrent answer surface
without sacrificing cyclic/debiased performance.

## 9. A100 Credit Discipline

The current Colab policy should be:

- run dataset audits, planning, summarization, and unit tests on CPU/local
  machines, not on A100;
- use A100 only for bounded training/evaluation jobs with explicit step limits,
  checkpoint backups, and automatic next-action summaries;
- avoid large Phase 2/SVGD training until deterministic Phase 1 is competitive
  with base;
- run small benchmark slices first, then expand only after paired evidence and
  margin/calibration diagnostics are non-negative;
- stop or downgrade runtime after long cells complete.

Recent automation changes support this policy: Stage 4 summaries now identify
their deterministic recurrent checkpoint, and the Stage 5 planner routes a
finished Stage 4 fine-tune directly into a broader benchmark-suite action. This
reduces idle A100 time between training and measurement. The single-runtime
Colab continuation wrapper now defaults to `credit_saver`, which executes one
allowlisted planner action, summarizes, commits safe text artifacts, and stops.
Longer A100 ladders require an explicit profile such as `gate`, `same_recipe`,
or `claim`.

The project should treat UI/auth repair inside Colab as a stop condition, not a
reason to leave an A100 attached. If GitHub, Google Drive, or notebook editor
automation fails, disconnect the runtime and repair the launch path locally.

## 10. Current Conclusion

The project has achieved the hard first step: a pretrained Qwen model can be
converted into a recurrent-depth model with exact identity preservation in the
single-pass gate, stable learned halting, and recoverable benchmark competence
under small-parameter fine-tuning. The latest ARC-mixed Phase 1 checkpoint now
beats the unmodified base model on a bounded 256-example ARC confirmation slice
under the content-question surface, while remaining non-negative under cyclic
option-permutation scoring. SVGD-style particles have shown useful
candidate-density signals on controlled exact tasks, but not yet reliable lift
over the strongest recovered deterministic recurrent model.

The next publishable claim should not be "SVGD beats Qwen" yet. The defensible
claim is narrower and scientifically useful:

> A pretrained dense transformer can be surgically converted into a recurrent
> latent-depth architecture with identity preservation, trained through a small
> adapter/controller budget to recover most lost reasoning competence, and
> instrumented with particle trajectories that produce measurable, selectable
> candidate diversity.

The decisive next result is whether this ARC-mixed recurrent recovery
replicates on an independent offset or full held-out reasoning split. Only
after that should selector-converted particles be treated as the main
benchmark-lift path.
