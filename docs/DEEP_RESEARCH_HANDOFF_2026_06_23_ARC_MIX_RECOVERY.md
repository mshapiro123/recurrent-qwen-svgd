# Deep Research Handoff: ARC-Mix Recovery Crossed Base On A Bounded Slice

## 0. Purpose

This handoff gives the strategy/deep-research agent the newest evidence before
we decide whether the next GPU spend should be:

1. an independent offset confirmation of the current recurrent checkpoint; or
2. immediate depth-routing training from that checkpoint.

The important update is that the project is no longer only in "near-base
recovery" territory. A targeted deterministic Phase 1 checkpoint now beats the
unmodified Qwen 0.5B base model on a bounded ARC-Easy/ARC-Challenge
confirmation slice. The result is promising but not yet robust enough for a
paper-level benchmark claim.

## 1. Current Active Checkpoint

The active deterministic recurrent checkpoint is:

```text
outputs/stage5/stage5_content_arcmix_qonly_optiontext_20260623_121707/
  arc_mix_response_w02_lr2e6/phase1/phase1_step_150.pt
```

The training gate that selected it is:

```text
outputs/stage5/stage5_content_arcmix_qonly_optiontext_20260623_121707/summary.json
```

Training recipe, in compact form:

- model: `Qwen/Qwen2.5-0.5B-Instruct`;
- architecture: deterministic recurrent Phase 1 wrapper, no particles;
- data: ARC train rows only for the final targeted run;
- ARC-Challenge train rows repeated `2x`;
- ARC-Easy train rows repeated `6x`;
- prompt style: `question_only`;
- score target: `option_text`;
- selected arm: `arc_mix_response_w02_lr2e6`;
- response-level base distillation weight: `0.20`;
- learning rate: `2e-6`;
- steps: `150`.

The key design change from earlier failed recovery runs is that the recovery
target attacked the content-answer surface directly instead of optimizing only
bare labels or a generic Opus trace mixture.

## 2. Latest Confirmation Result

The confirmation run is:

```text
outputs/stage5/stage5_content_arcmix_qonly_optiontext_arc256_check_20260623_123424/summary.json
```

Configuration:

- benchmarks: `arc_easy`, `arc_challenge`;
- limit: `256` examples each;
- offset: default `0`;
- recurrent mode: `phase1`;
- trajectories: `1`;
- learned loop control: `False`;
- score targets: `content_question_only`, `cyclic_label_aggregated`;
- failures: none.

Results:

| Benchmark | Scoring | Base | Recurrent | Delta |
|---|---|---:|---:|---:|
| ARC-Easy | cyclic option permutation | `202/256` | `204/256` | `+2` |
| ARC-Easy | content question-only | `146/256` | `155/256` | `+9` |
| ARC-Challenge | cyclic option permutation | `154/256` | `154/256` | `0` |
| ARC-Challenge | content question-only | `87/256` | `97/256` | `+10` |

Paired evidence from the run card:

- ARC-Easy content: recurrent `155/256`, base `146/256`, W/L/T
  `19/10/227`, sign-test p `0.1360`.
- ARC-Challenge content: recurrent `97/256`, base `87/256`, W/L/T
  `21/11/224`, sign-test p `0.1102`.
- ARC-Easy cyclic: recurrent `204/256`, base `202/256`, W/L/T `2/0/254`,
  sign-test p `0.5`.
- ARC-Challenge cyclic: recurrent `154/256`, base `154/256`, W/L/T
  `6/6/244`, sign-test p `1.0`.

Interpretation:

- This is the first bounded non-toy recurrent-vs-base win after the
  architectural surgery.
- It is positive on the content-answer surface, which was previously the main
  failure mode.
- It is non-negative under cyclic/debiased option scoring, so the content win
  does not appear to be purchased by a new obvious label-order collapse.
- It is not yet decisive by paired sign-test and covers only the first
  256-example slice.

## 3. Why This Matters Scientifically

The central project claim is not simply that a looped transformer can be made
to run. The claim is that a trained dense model can be surgically converted into
a recurrent latent-computation model, then recovered or improved with a small
trainable parameter budget.

This result is strong evidence for the recovery part:

- The architecture changed substantially.
- The trainable parameter slice is small relative to the base model.
- Targeted training moved the recurrent model from regression to a bounded
base-model win.

That is already scientifically interesting. If it replicates, the paper can
frame the training result as:

> Surgical recurrent conversion does not doom the model to permanent
> regression; a small adapter/controller recovery recipe can restore and
> sometimes exceed base behavior on non-toy reasoning slices.

What it does not yet prove:

- that recurrence beats base robustly across full ARC;
- that learned depth routing is better than the current fixed/expected-depth
  behavior;
- that SVGD/particles improve over this recovered deterministic checkpoint;
- that the effect transfers to GPQA, GSM8K, ARC-AGI, or larger Qwen scales.

## 4. Strategic Decision Now

There are two plausible next moves.

### Option A: Offset Confirmation First

Run the same checkpoint on a second bounded slice:

```text
ARC-Easy limit=256 offset=256
ARC-Challenge limit=256 offset=256
score_targets=content_question_only,cyclic_label_aggregated
```

Hypothesis:

- If the current checkpoint learned a real recovery, the result should remain
  non-negative or positive on the independent offset.

Success:

- recurrent >= base on both content-question surfaces, or at minimum no large
  content regression;
- cyclic/debiased scores remain non-negative;
- paired wins do not flip strongly against recurrent.

Failure:

- the content lift disappears or reverses on both benchmarks;
- cyclic scores fall materially;
- answer prior or margin drift reappears.

Reason to prefer:

- It is measurement-only.
- It avoids confounding the current positive result with another training run.
- It is the cleanest way to tell whether we have a real checkpoint or a lucky
  slice.

### Option B: Depth-Routing Training Now

Train from the current checkpoint using the depth-gradient idea:

- base-correct/direct/easy rows -> target loop `1`;
- rows missed by 0.5B but solved by 1.5B -> target loop `2`;
- rows missed by 0.5B/1.5B but solved by 3B/7B -> target loop `3` or `4`;
- keep content-question and cyclic/debiased guardrails.

Hypothesis:

- The current checkpoint has repaired the answer surface enough that targeted
  depth supervision can now allocate recurrence rather than merely recover
  base behavior.

Success:

- depth-1 preservation remains non-negative on easy/base-correct examples;
- depth-2/3 rows improve on hard or ambiguous examples;
- selected-depth evaluation beats fixed-depth and base;
- no return of option-label/calibration drift.

Failure:

- depth routing improves hard rows while damaging easy/direct rows;
- target-loop labels overfit the approximate model-scale ladder;
- deeper targets increase entropy/diversity without correctness.

Reason to prefer:

- It advances model capability rather than only confirming it.
- It directly tests the "recurrence substitutes for scale" thesis.
- It uses the strategic insight that easy/deep/wide problems should be trained
  differently.

## 5. Codex Recommendation Before Strategy Review

My current recommendation is a hybrid:

1. Run the offset-256 confirmation now because it is the shortest clean
   measurement that protects us from chasing a slice artifact.
2. In parallel, prepare the depth-routing SFT data and launch cell locally/CPU
   so the next training job is ready.
3. If offset confirmation is non-negative, immediately run a bounded
   depth-routing SFT from the current checkpoint.
4. If offset confirmation fails, do not abandon depth routing; first diagnose
   whether failure is concentrated in ARC-Easy content calibration, cyclic
   scoring, or hard-row routing.

The decision for the strategy agent is whether the value of immediate
depth-routing training outweighs the scientific value of one clean
out-of-slice confirmation. My bias is that one offset confirmation is worth it
because the current result is the first real recovery win and should be
handled carefully.

## 6. Questions For The Deep Research Agent

1. Should one independent offset confirmation be considered mandatory before
   training depth routing, or is the current 256-slice evidence enough to
   proceed?
2. If offset confirmation is positive, should the next training objective be
   explicit target-loop supervision, learned router/halting supervision, or
   answer-preserving distillation plus implicit loop pressure?
3. How should we weight the content-question surface against cyclic/debiased
   scoring during training selection?
4. Are model-scale correctness tiers a good enough proxy for depth labels, or
   should we derive depth from verifier-estimated solution complexity instead?
5. What failure pattern would imply "train more ARC content calibration" versus
   "train depth routing" versus "scale to 1.5B/3B"?
6. If the offset confirmation is positive, what is the minimum next benchmark
   set before packaging a private Hugging Face adapter for broader testing?

## 7. Practical Next Colab Action

The repo now contains a bootstrap target for the offset confirmation:

```text
STAGE5_CURRENT_A100_TARGET=arc_mix_offset_confirm
```

It runs:

```text
STAGE5_DEBIASED_BENCHMARKS=arc_easy,arc_challenge
STAGE5_DEBIASED_ARC_EASY_LIMIT=256
STAGE5_DEBIASED_ARC_CHALLENGE_LIMIT=256
STAGE5_DEBIASED_ARC_EASY_OFFSET=256
STAGE5_DEBIASED_ARC_CHALLENGE_OFFSET=256
STAGE5_DEBIASED_SCORE_TARGETS=content_question_only,cyclic_label_aggregated
```

The action is evaluation-only and should run on L4/T4/A100 depending on
availability. It does not need H100.

