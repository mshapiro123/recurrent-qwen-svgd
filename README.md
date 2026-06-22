# GRAM-Inspired Recurrent-Particle Qwen

This repository studies whether a pretrained dense Qwen model can be surgically
converted into a recurrent latent-reasoning model, recovered with a small
trainable parameter budget, and then extended with multiple particle-like latent
reasoning trajectories. The current base is
`Qwen/Qwen2.5-0.5B-Instruct`; the project deliberately stays small until the
identity, recurrence, recovery, and particle-selection gates are reproducible.

The work is organized like a model-surgery paper rather than a conventional SFT
run. The central question is:

> Can we preserve most of a trained model's competence after converting it into
> a recurrent-depth architecture, then use learned depth and latent particles to
> surpass the original model on hard reasoning?

## Research Contribution Snapshot

This repository is the working artifact for a compact model-surgery study. The
scientific contribution under test is not a larger data run; it is a controlled
conversion of a trained dense transformer into a recurrent latent-computation
system:

1. **Surgery with identity preservation.** Qwen is split into Prelude,
   Recurrent Block, and Coda. The wrapper passes the strict one-pass identity
   gate, so the recurrent architecture can exactly represent the original model
   before training changes behavior.
2. **Learned recurrent depth.** A sequence-level PonderNet-style halting head
   learns a non-collapsed loop distribution while only a small adapter,
   bridge, and controller budget is trained.
3. **Latent particle trajectories.** Stochastic latent injection and SVGD-style
   particle updates create multiple recurrent hidden-state trajectories, giving
   a route to candidate diversity in latent space rather than only in decoded
   text.
4. **Credit-aware empirical gates.** The project treats paid A100 time as a
   scarce experimental reagent. Every GPU run must answer a specific blocker
   and emit planner-readable artifacts before another run is justified.

The current evidence supports the first three method claims. It does **not**
yet support the stronger benchmark claim that the recurrent or SVGD model
surpasses unmodified Qwen 0.5B. The active experimental target is to recover
deterministic recurrent competence to at least base parity on balanced ARC, then
test whether particles and selectors add hard-tail lift.

## Current A100 Answer

We should treat the A100 as a gated measurement instrument, not as the default
workbench. Recent work moved in the right direction: dataset audits, planner
repairs, README/paper writing, and unit tests now run locally or on CPU; A100
jobs are bounded and expected to emit `summary.json` artifacts that decide the
next action. The remaining waste risk is leaving Colab attached while fixing
auth, notebook state, GitHub, or Drive problems. Those are stop conditions.

The current answer to "are we using A100 judiciously?" is: mostly yes now, but
only because the workflow has been narrowed. The project should not use A100 for
open-ended training, notebook debugging, Hugging Face dataset discovery, or
SVGD/kernel exploration while deterministic recurrent recovery is still below
the release bar. Paid GPU is reserved for one bounded proxy or confirmation job
at a time, followed by review.

The latest workflow change makes this stricter: ARC-mix proxy gates now inspect
paired margin movement and answer-prior drift, not just hit count. A proxy that
lifts accuracy but lowers the correct-answer margin or shifts predictions too
far is recorded as a calibration warning and should **not** trigger a full paid
assessment. This is the direct fix for the prior failure where a cheap proxy
matched base but the full balanced ARC run regressed.

As of the current checkpoint, the follow-up ARC-mix recovery proxy matched base
on the 128-row ARC-Challenge proxy, but the full balanced ARC-Easy /
ARC-Challenge confirmation assessment did **not** generalize. The recurrent
checkpoint remains behind base on both ARC-Easy and ARC-Challenge. The decision
tree is intentionally narrow:

| Current evidence | Next GPU action |
|---|---|
| Full balanced ARC assessment trails base | Stop A100 work and revise the data/objective mix locally. |
| A reviewed local diagnosis selects one bounded recovery proxy | Optionally run exactly one stronger-distillation ARC-mix proxy. |
| Auth/Drive/GitHub/notebook failure | Disconnect runtime and repair locally. |

Do not spend A100 credits on GPQA Diamond, Phase 2/SVGD scaling, 1.5B/3B
models, or more kernel geometry until deterministic recurrent recovery is at
least base-competitive on the balanced ARC suite.

## Manuscript Claim Status

The repository is now written as the skeleton of a model-surgery paper. The
defensible current claim is:

> A pretrained dense Qwen model can be surgically converted into a recurrent
> latent-depth architecture with exact one-pass identity preservation, stable
> learned halting, recoverable reasoning competence under a small trainable
> adapter/controller budget, and measurable particle-trajectory diversity.

The stronger claim is not yet established:

> The recurrent or SVGD recurrent-particle model surpasses the unmodified base
> Qwen 0.5B model on held-out reasoning benchmarks.

That stronger claim requires additional training and assessment: deterministic
recurrent recovery must first close the remaining ARC-Easy regression while
preserving the ARC-Challenge gain; then Phase 2/SVGD or selector-converted
particles must beat the recovered deterministic recurrent baseline. Until that
lands, the paper should present the surpass-base result as the next experimental
gate, not as a conclusion.

In paper language, the current manuscript is a methods-and-recovery paper with
a pending benchmark-superiority claim. The additional training required to make
the stronger claim is still being measured: the latest ARC-mix proxy closed the
128-example proxy gap to base, but the subsequent full balanced ARC assessment
remained negative. The next hypothesis is that stronger answer-calibration
preservation is required before recurrence can beat base.

The implementation has three stages:

1. **Identity-preserving surgery.** Split Qwen into Prelude, Recurrent Block,
   and Coda. With one recurrent pass, the wrapper must reproduce the original
   logits exactly under strict float32/eager settings.
2. **Deterministic recurrent recovery.** Train only LoRA adapters in the
   recurrent block, a gated identity bridge, and a sequence-level PonderNet
   halting head. The frozen base remains low precision; trainable controllers
   stay fp32 for stability.
3. **Recurrent particles.** Add stochastic latent trajectories and SVGD-style
   particle updates over recurrent hidden states, then test whether a selector
   can convert candidate diversity into accuracy.

For the current manuscript-style status, evidence, negative results, and next
gates, see [docs/PROJECT_STATUS_PAPER.md](docs/PROJECT_STATUS_PAPER.md). The
program-level strategy is tracked in [docs/PROGRAM_TRACK.md](docs/PROGRAM_TRACK.md).
The no-GPU reasoning-trace data plan is in
[docs/REASONING_TRACE_DATASETS.md](docs/REASONING_TRACE_DATASETS.md).

## Current Result

The project has not yet shown a release-grade recurrent/SVGD win over base
Qwen. It has shown a more useful intermediate result: a trained dense Qwen 0.5B
model can be converted into a recurrent-depth architecture with exact one-pass
identity preservation, stable learned halting, and recoverable benchmark
competence under small-parameter adapter/controller training.

Latest balanced ARC assessment for the selected recurrent checkpoint:

| Benchmark | Base Qwen | Recurrent Phase 1 | Delta |
|---|---:|---:|---:|
| ARC-Easy | `421/570` | `415/570` | `-6` |
| ARC-Challenge | `167/299` | `164/299` | `-3` |
| Combined | `588/869` | `579/869` | `-9` |

The latest proxy-selected checkpoint therefore failed the confirmation gate.
The local regression diagnosis shows answer-calibration drift: the recurrent
checkpoint over-predicts `C`, under-predicts `A`, and lowers the correct-answer
margin on both ARC-Easy and ARC-Challenge. Phase 2/SVGD work resumes only after
deterministic recurrent recovery is competitive with base.

## Credit-Saving Research Rule

Treat A100 time as the scarce experimental reagent. The default answer to
"should we use the A100?" is **no** unless the job satisfies all of these:

1. it answers the next blocker in the paper-level claim;
2. it has a fixed step/eval limit and known checkpoint source;
3. it emits `summary.json`/`summary.md` artifacts that the planner can read;
4. it has an automatic stop/disconnect path or is short enough to supervise;
5. it is not a dataset audit, unit test, notebook repair, README edit, or
   exploratory script-debugging task.

Right now there is exactly one plausible GPU job, and it is **bounded
deterministic repair training**: a direct-mode Phase 1 repair with shallow
halt supervision and base-logit response distillation. The routing diagnostic
from
`outputs/stage5/stage5_routing_diagnostic_20260622_041706/summary.json`
showed that the recurrent checkpoint still harms base-confident direct rows
while over-looping on them, so GPQA, Phase 2/SVGD, wide-particle training, and
scale-up remain premature. Dataset inspection, Hugging Face trace triage,
planner repairs, documentation, and diagnosis should stay local or on a free
CPU runtime.
The maintained next-action wrapper refuses long CPU/data-only dataset actions
on an attached GPU runtime by default, so an A100 session does not sit idle
while a Hugging Face audit runs.

Before attaching or keeping an A100, run the no-GPU spend check:

```bash
python colab/check_stage5_a100_go_no_go.py \
  --source-summary outputs/stage5/stage5_routing_diagnostic_20260622_041706/summary.json
```

Proceed only when it reports `go_routing_repair`, `go_bounded_proxy`,
`go_full_confirmation`, or another explicit `go_*` status that matches the
planned spend. Treat `no_go`, `calibration_warning_no_go`, and inspection
actions as stop signs.
The go/no-go check also verifies that the selected checkpoint is present
locally or visible in the configured Drive artifact backup before it allows a
paid GPU action. A missing checkpoint is a local/Drive repair task, not an
A100 debugging task.

The Colab next-action wrapper enforces the same policy before executing guarded
paid-GPU runners. A copied command can still be run manually, but the maintained
path (`python colab/run_stage5_next_action.py` with execution enabled) records
an `a100_guard` decision, includes checkpoint preflight status, and refuses
unsafe full-assessment or benchmark spends.
It also blocks long CPU/data-only actions such as reasoning dataset audits when
a GPU runtime is attached, unless
`STAGE5_ARC_AGI_NEXT_ACTION_ALLOW_LOCAL_ONLY_ON_GPU=1` is set deliberately.
For the lowest-friction Colab path, use
[`colab/STAGE5_SAFE_CONTINUE_CELL.md`](colab/STAGE5_SAFE_CONTINUE_CELL.md). It
pulls latest GitHub, runs go/no-go, and defaults to dry-run unless
`RUN_A100_ACTION = True` is set deliberately. The same flow is available as
[`colab/08_stage5_safe_continue.ipynb`](colab/08_stage5_safe_continue.ipynb).

## What Has Been Achieved

- **Exact identity gate passed.** The manually wrapped Qwen path can reproduce
  the base model with `max_abs_diff=0.0` under the strict identity setting.
- **Stable learned depth.** Sequence-level halting remains non-collapsed after
  adapter/controller stabilization, with expected loop depth around 2.9 on the
  current recurrent checkpoints.
- **Recoverable competence.** The recurrent model has recovered to near-base
  performance on balanced ARC. An earlier checkpoint slightly exceeded base on
  ARC-Challenge while trailing on ARC-Easy; the latest proxy-selected full
  confirmation trails base on both slices, exposing answer-calibration drift as
  the current blocker.
- **Particle mechanism signal.** SVGD and within-group particle geometry improve
  candidate density on controlled exact-task suites, but have not yet beaten the
  strongest deterministic recurrent checkpoint on non-toy benchmarks.
- **Automation and gates.** Colab runners now emit planner-readable summaries so
  GPU jobs can be bounded, resumed, assessed, and stopped instead of becoming
  open-ended notebook sweeps.

## What Is Still Required To Beat Base

The next scientific result must come from deterministic recurrent recovery, not
more particle geometry. The active recipe is:

1. Continue Phase 1 from the best balanced checkpoint.
2. Mix Opus/TraceInversion reasoning traces with benchmark-style ARC rows.
3. Weight ARC-Easy more heavily to close the easy-regression gap while keeping
   ARC-Challenge non-negative.
4. Preserve answer competence with a small response-level base-distillation
   signal.
5. Re-run the full balanced ARC assessment only after a bounded proxy gate is
   non-negative.

Only after the recurrent model is base-competitive should Phase 2/SVGD, GPQA
Diamond, 1.5B/3B scaling, or Hugging Face release work consume serious A100
time.

## Active Next A100 Action

Credits are tight, so A100 work is deliberately gate-based. The previous full
balanced assessment remained negative overall:

```text
run_id = stage5_recovery_full_assessment_current
status = needs_competence_recovery
ARC-Easy:      base 421/570, recurrent 412/570, delta -9
ARC-Challenge: base 167/299, recurrent 169/299, delta +2
Combined:      base 588/869, recurrent 581/869, delta -7
```

The follow-up low-credit ARC-mix recovery proxy first improved the recurrent
start and matched base:

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

The full balanced confirmation assessment for that proxy checkpoint did not
pass:

```text
run_id = stage5_full_assessment_once_20260622_005522
status = needs_competence_recovery
ARC-Easy:      base 421/570, recurrent 415/570, delta -6
ARC-Challenge: base 167/299, recurrent 164/299, delta -3
Combined:      base 588/869, recurrent 579/869, delta -9
```

The local diagnostic report is:

```bash
outputs/stage5/stage5_full_assessment_once_20260622_005522/mcq_regression_diagnosis.md
```

The subsequent stronger ARC-mix proxy did **not** pass:

```text
run_id = stage5_arc_mix_recovery_once_20260622_030628
status = no_proxy_lift
decision = stop_and_revise_objective
base proxy = 68/128
start proxy = 68/128
best recurrent proxy = 66/128
mean margin delta vs base = -0.308232
```

The bounded routing diagnostic then landed:

```text
run_id = stage5_routing_diagnostic_20260622_041706
status = needs_direct_halting_repair
ARC-Easy direct delta = -2
ARC-Easy mean direct loops = 2.58
ARC-Challenge direct delta = -3
ARC-Challenge mean direct loops = 2.62
ARC-Challenge conceptual delta = +2
```

The next GPU job should therefore be a **bounded direct-mode halting repair**,
not Phase 2/SVGD, GPQA, or scale-up. Use the safe-continue cell from
[`colab/STAGE5_SAFE_CONTINUE_CELL.md`](colab/STAGE5_SAFE_CONTINUE_CELL.md);
with `RUN_A100_ACTION = True`, the planner now selects
`python colab/run_stage5_routing_repair.py`. The selected repair profile
weights ARC-Easy/direct rows heavily, targets loop `1` on ARC-Easy rows, uses
ARC-Challenge only as a light loop-`2` probe, and keeps particles/SVGD off.

The concise run card with the preferred safe-continue path is
[`colab/CURRENT_A100_ACTION.md`](colab/CURRENT_A100_ACTION.md).

The repair runner consumes the diagnostic summary, delegates to the existing
ARC-mix trainer with particles/SVGD off, and records the child run under a
planner-readable summary. A `repair_proxy_lift` or `repair_proxy_matches_base`
result can justify a full balanced ARC confirmation. Calibration warnings or
`repair_no_proxy_lift` are stop signs for objective revision.

The longer-term data plan is captured in
[`docs/CURRICULUM_DATA_PIPELINE.md`](docs/CURRICULUM_DATA_PIPELINE.md): strong
non-Qwen models generate and judge candidate traces, but verified answers and
programmatic checks control labels. Positive SFT loaders may consume only
`positive_*` roles; rationalizations and slips are reserved for verifier or
contrastive training.

When that generated curriculum pipeline completes and its SFT gate reports
`go=true`, the guarded GPU handoff is:

```bash
python colab/run_stage5_curriculum_sft.py
```

This runner refuses unsafe or tiny shards, requires Drive backup by default,
trains only deterministic Phase 1 recurrence from `positive_sft.jsonl`, and
validates on a held-out curriculum split before any particle/SVGD work.

Do not run GPQA, Phase 2/SVGD, or scale-up jobs before this deterministic
recurrent recovery question is resolved.

## A100 Credit Discipline

Use GPU time only for bounded training/evaluation actions that emit summaries,
checkpoints, and planner-readable next steps. The maintained Colab continuation
entrypoint is:

```bash
python colab/run_stage5_colab_continue.py
```

By default it now runs in `credit_saver` mode: one allowlisted planner action,
post-run summaries, safe text-artifact commit, then stop. Set
`STAGE5_ARC_AGI_COLAB_CONTINUE_PROFILE=gate` or `throughput` only when you
intentionally want a three-action loop. Set `same_recipe` or `claim` only after
the preceding evidence gate has landed and the extra A100 spend is deliberate.

## Phase 0 Identity Gate

```bash
python eval/eval_identity.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --split 6,18 \
  --dtype float32 \
  --attn_implementation eager \
  --threshold 1e-3
```

The wrapper path is manual, so a pass means:

```text
embed -> prelude -> recurrent block once -> coda -> norm -> lm_head
```

matches the original model logits with dropout disabled.

## Phase 1 Halting Telemetry

Do not run Phase 1 training until the Phase 0 identity gate passes.

```bash
python eval/eval_halting.py --max_loops 4 --split 6,18
```

For training on JSONL rows with `prompt` + `completion` or `text`:

```bash
python training/train_phase1_ponder.py \
  --config config/qwen_0_5b_phase1.yaml \
  --train_jsonl data/train.jsonl
```

## Hugging Face Reasoning Data

Convert common Opus/Qwen reasoning datasets to this project's JSONL format:

```bash
python training/prepare_hf_reasoning_jsonl.py \
  --dataset_id lordx64/reasoning-distill-opus-4-7-max-sft \
  --tokenizer_name Qwen/Qwen2.5-0.5B-Instruct \
  --output_jsonl data/opus47_train.jsonl \
  --val_jsonl data/opus47_val.jsonl \
  --limit 1000 \
  --max_total_tokens 2048
```

The converter supports datasets with `text`, `messages`, or
`thinking`/`response`-style fields, TraceInversion-style
`inverted_reasoning` rows, explicit Hugging Face JSONL files, and
Complete-FABLE-style `row_json` wrappers. It writes `prompt`, `completion`, and
`cot_tokens`.

For the Jackrong Opus TraceInversion rows, use the dedicated adapter so the
inverted reasoning trace is preserved instead of falling back to answer-only
chat messages:

```bash
python training/prepare_hf_reasoning_jsonl.py \
  --dataset_id Jackrong/Claude-opus-4.7-TraceInversion-5000x \
  --adapter trace_inversion \
  --output_jsonl data/trace_inversion_train.jsonl \
  --val_jsonl data/trace_inversion_val.jsonl \
  --limit 1000 \
  --max_total_tokens 2048
```

For Fable's flat merged trace file, address the file explicitly:

```bash
python training/prepare_hf_reasoning_jsonl.py \
  --dataset_id Glint-Research/Fable-5-traces \
  --hf_file fable5_cot_merged.jsonl \
  --adapter fable_flat \
  --output_jsonl data/fable5_flat_train.jsonl \
  --val_jsonl data/fable5_flat_val.jsonl \
  --limit 1000 \
  --max_total_tokens 4096
```

Audit unfamiliar trace datasets before mixing them into training:

```bash
python training/inspect_hf_reasoning_dataset.py \
  --dataset_id Glint-Research/Fable-5-traces \
  --hf_file fable5_cot_merged.jsonl \
  --adapter fable_flat \
  --limit 1000 \
  --output_json outputs/dataset_audits/fable5_flat.json
```

Known candidate trace sources and their intended roles are tracked in
`config/reasoning_dataset_registry.yaml`. Opus-style reasoning traces are the
current fine-tuning source. Jackrong Opus TraceInversion is an immediate audit
candidate for easy/hard recurrent curriculum work. Fable/Pi-agent traces are
treated as later agent/tool-diversity material unless an audit and filter
explicitly promote them into a training mix.

Current dataset triage:

| Dataset | Immediate role | GPU policy |
|---|---|---|
| [`lordx64/reasoning-distill-opus-4-7-max-sft`](https://huggingface.co/datasets/lordx64/reasoning-distill-opus-4-7-max-sft) | SFT-ready Opus trace source for recurrent competence recovery. | Use filtered subsets in bounded Phase 1 mixes. |
| [`lordx64/reasoning-distill-claude-opus-4-7-max`](https://huggingface.co/datasets/lordx64/reasoning-distill-claude-opus-4-7-max) | Raw Opus source with richer fields for curriculum filtering. | Audit/filter first; do not blindly train full rows. |
| [`Jackrong/Claude-opus-4.7-TraceInversion-5000x`](https://huggingface.co/datasets/Jackrong/Claude-opus-4.7-TraceInversion-5000x) | Immediate audit candidate for alternative reasoning-trace supervision. | CPU audit first, then small mixed pilot only if promoted. |
| [`Glint-Research/Fable-5-traces`](https://huggingface.co/datasets/Glint-Research/Fable-5-traces) | Agent/tool/coding trace diversity source. | Hold for filter design; no blind ARC/GPQA recovery SFT. |
| [`Glint-Research/Complete-FABLE.5-traces-2M`](https://huggingface.co/datasets/Glint-Research/Complete-FABLE.5-traces-2M) | Large trace-mining source. | Streaming CPU audit only until a precise filter exists. |

Additional Opus/Kimi/GLM/Fable/Mythos candidates are registered in
[`config/reasoning_dataset_registry.yaml`](config/reasoning_dataset_registry.yaml)
and summarized in
[`docs/REASONING_TRACE_DATASETS.md`](docs/REASONING_TRACE_DATASETS.md). They
are intentionally marked as audit or later-audit candidates, not automatic
training data.

## Phase 2 Stochastic Trajectories

Do not run Phase 2 training until deterministic halting is non-collapsed.

```bash
python eval/eval_trajectories.py \
  --max_loops 4 \
  --num_trajectories 2 \
  --split 6,18
```

If untrained fp16 trajectories report zero diversity, run a diagnostic-only
amplified check:

```bash
python eval/eval_trajectories.py \
  --max_loops 4 \
  --num_trajectories 2 \
  --split 6,18 \
  --diagnostic_latent_scale 1.0 \
  --diagnostic_adapter_std 0.02
```

```bash
python training/train_phase2_stochastic.py \
  --config config/qwen_0_5b_phase2.yaml \
  --train_jsonl data/train.jsonl
```

## Phase 2 SVGD Particles

SVGD mode keeps the frozen Qwen base path deterministic but treats each
trajectory as a particle after the recurrent block. The update combines the
ordinary recurrent transition with an RBF-kernel repulsion term across the
trajectory axis. This is the preferred next experiment when testing whether
particle diversity can create useful answer candidates without relying on large
latent noise.

Start with the smoke config:

```bash
python training/train_phase2_stochastic.py \
  --config config/qwen_0_5b_phase2_svgd.yaml \
  --train_jsonl data/opus47_train.jsonl \
  --device cuda
```

Validate with the SVGD switches enabled:

```bash
python eval/eval_jsonl.py \
  --data_jsonl data/opus47_val.jsonl \
  --checkpoint outputs/qwen_0_5b_phase2_svgd_smoke25/phase2_step_25.pt \
  --max_loops 4 \
  --num_trajectories 4 \
  --particle_update_mode svgd \
  --particle_init_noise 0.02 \
  --svgd_repulsion_scale 0.5 \
  --svgd_repulsion_max_norm 1.0 \
  --max_length 512 \
  --beta 0.08 \
  --rho 1e-3 \
  --dtype bfloat16 \
  --adapter_dtype float32 \
  --device cuda
```

Compare the Phase 1 checkpoint against a Phase 2 checkpoint on small exact
generation tasks:

```bash
python eval/eval_best_of_k_jsonl.py \
  --phase1_checkpoint outputs/qwen_0_5b_phase1_a100_beta008_continue_150/phase1_step_150.pt \
  --phase2_checkpoint outputs/qwen_0_5b_phase2_svgd_smoke25/phase2_step_25.pt \
  --phase2_num_trajectories 4 \
  --phase2_particle_update_mode svgd \
  --particle_init_noise 0.02 \
  --svgd_repulsion_scale 0.5 \
  --svgd_repulsion_max_norm 1.0 \
  --max_new_tokens 64 \
  --dtype bfloat16 \
  --adapter_dtype float32 \
  --device cuda
```

To initialize Phase 2 from a Phase 1 trainable checkpoint, set `resume_from` in
the Phase 2 config, for example:

```yaml
resume_from: outputs/qwen_0_5b_phase1/phase1_step_100.pt
```

Validate a trainable checkpoint on held-out JSONL:

```bash
python eval/eval_jsonl.py \
  --data_jsonl data/opus47_val.jsonl \
  --checkpoint outputs/qwen_0_5b_phase1_opus47_200/phase1_step_200.pt \
  --max_loops 4 \
  --max_length 1024
```

For constrained Colab GPUs such as G4, use the 50-step stability profile first:

```bash
python training/train_phase1_ponder.py \
  --config config/qwen_0_5b_phase1_g4_stability.yaml \
  --train_jsonl data/opus47_train.jsonl \
  --device cuda
```

Then validate the 50-step checkpoint before any longer run:

```bash
python eval/eval_jsonl.py \
  --data_jsonl data/opus47_val.jsonl \
  --checkpoint outputs/qwen_0_5b_phase1_g4_stability_50/phase1_step_50.pt \
  --max_loops 4 \
  --max_length 512 \
  --dtype float16 \
  --adapter_dtype float32
```

This keeps trainable adapters/controllers in fp32 and aborts on nonfinite loss,
metrics, gradients, or trainable parameters. Do not resume from the discarded
NaN checkpoint at
`outputs/qwen_0_5b_phase1_opus47_beta005_lr2e5_200/phase1_step_200.pt`.

## Slow Recurrent Inference

```bash
python infer_recurrent.py \
  --prompt "Find one valid 4-queens placement." \
  --max_loops 4 \
  --num_trajectories 2 \
  --sample_latents
```

Generation is intentionally no-cache and slow because recurrent passes reuse the
same sequence. KV cache is only valid on the single-pass identity-shaped path.

## GPQA-Style Multiple Choice

```bash
python eval/eval_gpqa.py \
  --data_jsonl data/gpqa_lite.jsonl \
  --max_loops 4 \
  --split 6,18
```

Rows should contain `question`, `choices` or `options`, and `answer`.

## Colab

Open [colab/GRAM_Recurrent_Qwen_Colab.ipynb](colab/GRAM_Recurrent_Qwen_Colab.ipynb)
in Google Colab, set the runtime to GPU, upload this project as a zip, and run
the notebook from top to bottom through the Phase 0 identity gate first.

For Colab Pro+, use [colab/COLAB_PRO_PLUS_RUNBOOK.md](colab/COLAB_PRO_PLUS_RUNBOOK.md):
select H100 GPU if available, turn High-RAM on, and keep `num_trajectories=2`
until the small-model gates pass.

## RunPod

For a more stable GPU workbench, use [runpod/RUNPOD_HANDOFF.md](runpod/RUNPOD_HANDOFF.md).
The short path is:

```bash
bash scripts/runpod_setup.sh
bash scripts/run_smoke_gates.sh
```
