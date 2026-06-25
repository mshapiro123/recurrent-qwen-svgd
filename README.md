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

The latest workflow change makes this stricter: MCQ benchmark claims must
separate content competence from option-label/position bias, and training spend
must now respect the master sequence: loop-closure re-entry first,
deterministic depth recovery second, same-curriculum dense control third, and
particles/SVGD only after correct-bearing breadth is measurable. The ARC-Easy
debias diagnostic at
`outputs/stage5/stage5_mcq_debias_direct_20260622_194346/summary.json` reported
`selection_bias_likely`: the loop-4 recurrent checkpoint looked worse under
bare `A/B/C/D` scoring, but matched or slightly exceeded base after cyclic
option-permutation aggregation. See
[docs/MCQ_DEBIAS_STATUS.md](docs/MCQ_DEBIAS_STATUS.md).

The current front-of-queue action is the Phase 0 re-entry repair chain. Stage 1
found a dead bridge and Stage 2 found eval-only `entry_rms` re-entry
normalization safe enough for a tiny trainable smoke. The intended immediate
sequence is:

| Stage | Runtime | Purpose |
|---|---|---|
| `reentry_repair_smoke` | L4/T4 | Make the bridge and re-entry adapter gradient-live while preserving loop-1 behavior. |
| `reentry_recovery_training` | L4/T4, G4 only if needed | Run bounded deterministic recovery SFT from the repaired checkpoint with learned loop control and target-loop supervision. |
| `debiased_benchmark_suite` | L4/T4 | Compare base Qwen 0.5B versus the repaired recurrent checkpoint on ARC-Easy, ARC-Challenge, and GPQA-lite surfaces. |
| `dense_mcq_trace_sft_control` | L4/T4 | Train/evaluate standard dense Qwen LoRA on the same curriculum so architecture lift is separated from data-recipe lift. |

Do not spend A100 credits on GPQA Diamond, Phase 2/SVGD scaling, 1.5B/3B
models, or more kernel geometry until the repaired deterministic recurrent path
produces a sane Stage 4 checkpoint and a paired base/recurrent/dense-control
benchmark diagnostic.
Auth, Drive, GitHub, notebook-state, source-pointer, and dataset-prep failures
are CPU/local repair tasks, not GPU tasks.

The detailed hypothesis-driven experiment queue is maintained in
[docs/SEQUENCED_EXPERIMENT_PLAN.md](docs/SEQUENCED_EXPERIMENT_PLAN.md). Use it
as the handoff for strategy/deep-research review and as the default ordering
for deciding whether the next Colab job should be CPU, L4/T4, A100, or deferred.

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

That stronger claim requires additional training and assessment: first,
bare-label MCQ results must be regenerated under cyclic/permutation scoring;
then deterministic recurrent recovery must be at least base-competitive under
that debiased metric; finally Phase 2/SVGD or selector-converted particles must
beat the recovered deterministic recurrent baseline. Until that lands, the
paper should present the surpass-base result as the next experimental gate, not
as a conclusion.

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
program-level dependency sequence is tracked in
[docs/PROGRAM_TRACK_MASTER_SEQUENCE.md](docs/PROGRAM_TRACK_MASTER_SEQUENCE.md);
the older strategy note remains at [docs/PROGRAM_TRACK.md](docs/PROGRAM_TRACK.md).
The MCQ scoring-confound note is in
[docs/MCQ_DEBIAS_STATUS.md](docs/MCQ_DEBIAS_STATUS.md). The current
deep-research handoff, including the direct-route preservation questions that
remain relevant if debiased scoring still shows a gap, is
[docs/DEEP_RESEARCH_HANDOFF_2026_06_22.md](docs/DEEP_RESEARCH_HANDOFF_2026_06_22.md).
The current master-sequence strategy packet is
[docs/DEEP_RESEARCH_HANDOFF_2026_06_25_MASTER_SEQUENCE.md](docs/DEEP_RESEARCH_HANDOFF_2026_06_25_MASTER_SEQUENCE.md).
The no-GPU reasoning-trace data plan is in
[docs/REASONING_TRACE_DATASETS.md](docs/REASONING_TRACE_DATASETS.md). The
wide/deep curriculum data contract is in
[docs/CURRICULUM_DATA_PIPELINE.md](docs/CURRICULUM_DATA_PIPELINE.md).

## Current Result

The project has not yet shown a release-grade recurrent/SVGD win over base
Qwen. It has shown a more useful intermediate result: a trained dense Qwen 0.5B
model can be converted into a recurrent-depth architecture with exact one-pass
identity preservation, stable learned halting, and recoverable benchmark
competence under small-parameter adapter/controller training.

The latest CE8 balanced ARC depth curve is now the best current readout. It
uses balanced 256-example ARC-Easy and ARC-Challenge slices and compares fixed
recurrent depths 1-4 under both cyclic option-permutation scoring and
content-question-only scoring. The key result is conditional:

| Slice | Best fixed depth | Base Qwen | Recurrent | Delta |
|---|---:|---:|---:|---:|
| ARC-Easy, cyclic | `1` | `202/256` | `206/256` | `+4` |
| ARC-Easy, content-only | `1` | `146/256` | `131/256` | `-15` |
| ARC-Challenge, cyclic | `2-4` | `154/256` | `153/256` | `-1` |
| ARC-Challenge, content-only | `3-4` | `87/256` | `92/256` | `+5` |

This is not a clean benchmark win, but it is the sharpest mechanism result so
far. Shallow recurrence preserves easy/cyclic behavior best, while deeper
recurrence improves the harder ARC-Challenge content readout. At the same time,
ARC-Easy content-only scoring remains badly behind base at every fixed depth,
so answer calibration and direct-route preservation are now the main blockers.

Detailed artifact:
[docs/STAGE5_CE8_DEPTH_CURVE_2026_06_23.md](docs/STAGE5_CE8_DEPTH_CURVE_2026_06_23.md).

## Credit-Saving Research Rule

Treat A100 time as the scarce experimental reagent. The default answer to
"should we use the A100?" is **no** unless the job satisfies all of these:

1. it answers the next blocker in the paper-level claim;
2. it has a fixed step/eval limit and known checkpoint source;
3. it emits `summary.json`/`summary.md` artifacts that the planner can read;
4. it has an automatic stop/disconnect path or is short enough to supervise;
5. it is not a dataset audit, unit test, notebook repair, README edit, or
   exploratory script-debugging task.

Right now the plausible GPU job is **depth-conditional preservation/routing
training**, not Phase 2/SVGD or GPQA. The depth curve shows that harder
ARC-Challenge content can benefit from depth 3-4, but easy/content behavior
needs a stronger depth-1 preservation objective. GPQA, Phase 2/SVGD,
wide-particle training, and scale-up remain premature until this deterministic
depth spine is cleaner.
Dataset inspection, Hugging Face trace triage, planner repairs, documentation,
and diagnosis should stay local or on a free CPU runtime.
The maintained next-action wrapper refuses long CPU/data-only dataset actions
on an attached GPU runtime by default, so an A100 session does not sit idle
while a Hugging Face audit runs.

Before attaching or keeping an A100, run the no-GPU spend check:

```bash
python colab/check_stage5_a100_go_no_go.py \
  --source-summary outputs/stage5/stage5_routing_diagnostic_20260622_041706/summary.json
```

Proceed only when it reports an explicit `go_*` status that matches the planned
spend, such as the bounded MCQ debias diagnostic or a later proven recovery
action. Treat `no_go`, `calibration_warning_no_go`, and inspection actions as
stop signs.
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
  performance on balanced ARC. The latest CE8 depth curve shows a positive
  hard-slice content signal at depth 3-4 (`+5/256` on ARC-Challenge
  content-only) while exposing a serious ARC-Easy content regression. The
  blocker is now conditional depth allocation plus direct-route calibration.
- **Particle mechanism signal.** SVGD and within-group particle geometry improve
  candidate density on controlled exact-task suites, but have not yet beaten the
  strongest deterministic recurrent checkpoint on non-toy benchmarks.
- **Automation and gates.** Colab runners now emit planner-readable summaries so
  GPU jobs can be bounded, resumed, assessed, and stopped instead of becoming
  open-ended notebook sweeps.

## What Is Still Required To Beat Base

The next scientific result must come from deterministic recurrent recovery, not
more particle geometry. The active recipe is:

1. Preserve depth-1 behavior on easy/direct/base-correct rows.
2. Train or select depth 2-3 behavior for ambiguous and harder rows.
3. Keep cyclic option-permutation scoring and content-only scoring side by
   side.
4. Treat ARC-Easy content regression as a hard guardrail.
5. Re-run particles/SVGD only after selected deterministic depth beats the best
   fixed-depth recurrent baseline.

Only after the recurrent model is base-competitive should Phase 2/SVGD, GPQA
Diamond, 1.5B/3B scaling, or Hugging Face release work consume serious A100
time.

## Active Next A100 Action

Credits are tight, so GPU work remains gate-based. The current front-of-queue
job is the re-entry architecture repair gate, not more ARC-mix depth training
or particle/SVGD geometry. Stage 1 showed the current recovered checkpoint has
a dead bridge: `bridge_gate=0.0`, zero bridge delta, and zero bridge projection
gradients. Until that loop-closure path is repaired, further particle diversity
experiments are likely to amplify noise rather than produce useful reasoning
paths.

Current reviewer state:

```text
latest stage: stage2_norm
latest status: entry_rms_safe_for_smoke
current source summary: outputs/stage5/stage5_reentry_norm_20260625_013527/summary.json
next target: reentry_repair_smoke
```

Stage 2 has already cleared the eval-only re-entry normalization gate. Prefer
the GitHub-resolved launcher below from a restarted or blank runtime; the
shorter local `exec(open(...))` form is only safe after the repo has already
been freshly cloned or reset to `main`.

Fresh-runtime Stage 3 launch:

```python
import base64, json, os, time, urllib.request
from google.colab import userdata

REPO = "mshapiro123/recurrent-qwen-svgd"
TARGET = "reentry_repair_smoke"

gh = userdata.get("GH_TOKEN") or userdata.get("GITHUB_TOKEN")
assert gh, "Missing GH_TOKEN or GITHUB_TOKEN in Colab secrets."
hf = userdata.get("HF_TOKEN") or userdata.get("HUGGINGFACE_HUB_TOKEN")
if hf:
    os.environ["HF_TOKEN"] = hf
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf

os.environ["STAGE5_CURRENT_A100_TARGET"] = TARGET

headers = {
    "Authorization": f"Bearer {gh}",
    "Accept": "application/vnd.github+json",
}
ref_req = urllib.request.Request(
    f"https://api.github.com/repos/{REPO}/git/refs/heads/main?cache_bust={time.time_ns()}",
    headers=headers,
)
with urllib.request.urlopen(ref_req, timeout=30) as response:
    resolved_ref = json.load(response)["object"]["sha"]

file_req = urllib.request.Request(
    f"https://api.github.com/repos/{REPO}/contents/colab/CURRENT_A100_BOOTSTRAP_CELL.py"
    f"?ref={resolved_ref}&cache_bust={time.time_ns()}",
    headers=headers,
)
with urllib.request.urlopen(file_req, timeout=30) as response:
    payload = json.load(response)

code = base64.b64decode(payload["content"]).decode("utf-8")
required = [
    "sha_resolved_nested_fetch_v3",
    TARGET,
    "STAGE5_REENTRY_REPAIR_SMOKE_CELL_VERSION",
    "stage5_reentry_repair_smoke_v1_trainable",
    "stage2_norm_assessment",
    "Loop-1 Preservation",
    "colab/assess_stage5_reentry.py",
]
missing = [marker for marker in required if marker not in code]
assert not missing, f"Fetched stale or incomplete bootstrap: {missing}"
print("Fetched bootstrap sha:", payload.get("sha"), "commit:", resolved_ref[:12], "target:", TARGET)
exec(compile(code, "colab/CURRENT_A100_BOOTSTRAP_CELL.py", "exec"))
```

After Stage 3 publishes, run the CPU-only reviewer. In a fresh runtime, change
`TARGET` in the paste-anywhere launcher above to `master_sequence_status`;
only use the shorter repo-local form if `/content/recurrent-qwen-svgd` has
already been freshly cloned or reset to `main` in the current runtime.
Continue only if the reviewer recommends
`run_bounded_recovery_training_with_reentry_repair`:

```python
import os
os.environ["STAGE5_CURRENT_A100_TARGET"] = "master_sequence_status"
exec(open("colab/CURRENT_A100_BOOTSTRAP_CELL.py").read())
```

Only if Stage 3 assessment recommends
`run_bounded_recovery_training_with_reentry_repair`, launch bounded recovery
SFT. Again, in a fresh runtime prefer the paste-anywhere launcher above with
`TARGET = "reentry_recovery_training"`; the snippet below is only for a
runtime where the repo already exists locally:

```python
import os
os.environ["STAGE5_CURRENT_A100_TARGET"] = "reentry_recovery_training"
exec(open("colab/CURRENT_A100_BOOTSTRAP_CELL.py").read())
```

Minimum run contract:

- keep Phase 2/SVGD and inference-time particle noise off;
- do not mutate checkpoints during Stage 2;
- require loop-1 preservation during Stage 3;
- require finite validation and target-loop/depth-gradient metrics during Stage
  4;
- pause for review after Stage 2 and Stage 3 before spending additional GPU.

The concise run card is maintained here:
[`colab/CURRENT_A100_ACTION.md`](colab/CURRENT_A100_ACTION.md).
Use the CPU-only reviewer `colab/review_stage5_reentry.py` after each run.

The longer-term data plan is captured in
[`docs/CURRICULUM_DATA_PIPELINE.md`](docs/CURRICULUM_DATA_PIPELINE.md): strong
non-Qwen models generate and judge candidate traces, but verified answers and
programmatic checks control labels. Positive SFT loaders may consume only
`positive_*` roles; rationalizations and slips are reserved for verifier or
contrastive training.

When that generated curriculum pipeline completes and its SFT gate reports
`go=true`, the guarded GPU handoff is:

```bash
STAGE5_CURRICULUM_WORK_DIR=data/curriculum/programmatic_direct_deep_001 \
STAGE5_CURRICULUM_MIN_POSITIVE_ROWS=2000 \
STAGE5_CURRICULUM_MIN_MODE_ROWS=direct=1000,deep_narrow=1000 \
python colab/run_stage5_curriculum_sft.py
```

This runner refuses unsafe or tiny shards, requires Drive backup by default,
trains only deterministic Phase 1 recurrence from `positive_sft.jsonl`, and
validates on a held-out curriculum split before any particle/SVGD work.
The go/no-go guard also requires this explicit mode-coverage gate before paid
GPU SFT. Keep the direct/deep default for the current 2000-row programmatic
calibration phase; change it deliberately, for example to `wide=64`, only for a
later width/particle curriculum.

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

The audit now writes a `curriculum_signal` block. Treat it as a routing
recommendation, not permission to train blindly:

- `direct_recovery_candidate`: short reasoning traces likely useful for
  restoring depth-1 competence.
- `deep_narrow_candidate`: longer reasoning traces that can supervise learned
  recurrence/depth.
- `hold_for_wide_or_agentic_filter`: Fable/tool/agent traces that may help
  trajectory diversity later, but should not be mixed into ARC/GPQA recovery
  without a domain filter.
- `fit_rates_total_tokens`: quick context-budget sanity check for 512/1024/2048
  token runs.

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
training data. The strong-model API workflow for converting verified problems
into width/depth/mode-labeled curriculum records is documented in
[`docs/CURRICULUM_DATA_PIPELINE.md`](docs/CURRICULUM_DATA_PIPELINE.md).

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
