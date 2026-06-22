# Single A100 Colab Runbook

The preferred workflow is **one Colab notebook attached to one A100 runtime**.
Do not open each stage as a separate notebook while an expensive runtime is
active. The staged notebooks remain in `colab/` for review, but the execution
path below stays inside the current runtime.

## Current Low-Credit Stage 5 Action

For the shortest current instruction, use
[`CURRENT_A100_ACTION.md`](CURRENT_A100_ACTION.md). It includes the direct
GitHub-Colab URL and the post-run stop/continue decisions.

The active next action is **not** more training. The latest ARC-mix recovery
proxy landed as `stage5_arc_mix_recovery_once_20260622_030628` and reported
`no_proxy_lift` with `decision = stop_and_revise_objective`:

- proxy base: `68/128`;
- proxy start: `68/128`;
- best recurrent proxy: `66/128`;
- mean margin delta versus base: `-0.308232`.

The default Colab entrypoint is
[`STAGE5_SAFE_CONTINUE_CELL.md`](STAGE5_SAFE_CONTINUE_CELL.md), which pulls the
latest repo, runs the no-GPU go/no-go check, and dry-runs the guarded
next-action path unless `RUN_A100_ACTION = True` is set in the cell. It now
disconnects the runtime by default after printing the dry-run or guarded-action
result; set `DISCONNECT_RUNTIME_WHEN_DONE = False` only when you intentionally
want to keep the session attached. In dry-run or blocked states it also skips
`pip install -r requirements.txt`, so a status check is mostly Git plus stdlib
planner work.

The guarded action selected from that source is one bounded depth/width routing
diagnostic via [`run_stage5_routing_diagnostic.py`](run_stage5_routing_diagnostic.py).
It benchmarks ARC-Easy and ARC-Challenge with loop diagnostics enabled, writes
`routing_assessment.json` / `routing_assessment.md`, and stops. Prefer L4/T4
for this diagnostic if available; use A100 only when that is the available
runtime and the short run is intentional.

Do not use the ARC-mix recovery cell again from this state. The next training
run should be chosen only after the routing assessment identifies direct-mode
halting repair versus deep-narrow recovery.
When that assessment does identify a repair, the safe-continue planner selects
[`run_stage5_routing_repair.py`](run_stage5_routing_repair.py), which launches
one bounded deterministic Phase 1 repair profile and keeps particles/SVGD off.

If the ARC-mix review explicitly says a full confirmation is justified,
[`STAGE5_FULL_ASSESSMENT_CELL.md`](STAGE5_FULL_ASSESSMENT_CELL.md) and
[`07_stage5_full_arc_assessment.ipynb`](07_stage5_full_arc_assessment.ipynb)
now apply the same policy: mount Drive, run go/no-go, install dependencies only
after approval, run exactly one full balanced assessment, and disconnect on
completion or setup failure.

## Legacy Stage 1 Cell

The following older cell is kept for reference when restarting the full staged
program. Do not run it for the current diagnosis / ARC-mix recovery gate.

```python
import os, subprocess, sys
from pathlib import Path
from google.colab import userdata

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")

def secret(name):
    try:
        return userdata.get(name)
    except Exception:
        return None

GH_TOKEN = (
    os.environ.get("GH_TOKEN")
    or os.environ.get("GITHUB_TOKEN")
    or secret("GH_TOKEN")
    or secret("GITHUB_TOKEN")
)
HF_TOKEN = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    or secret("HF_TOKEN")
    or secret("HUGGINGFACE_HUB_TOKEN")
)
assert GH_TOKEN, "Missing GH_TOKEN or GITHUB_TOKEN in Colab secrets."
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

def run(cmd, cwd=None, check=True):
    printable = " ".join(map(str, cmd)).replace(GH_TOKEN, "****")
    print("$", printable)
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(proc.stdout)
    if check and proc.returncode:
        raise RuntimeError(f"failed: {printable}")
    return proc

if ROOT.exists():
    run(["git", "remote", "set-url", "origin", f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"], cwd=ROOT)
    run(["git", "fetch", "origin", "main"], cwd=ROOT)
    run(["git", "checkout", "main"], cwd=ROOT)
    run(["git", "pull", "--ff-only", "origin", "main"], cwd=ROOT)
else:
    run(["git", "clone", f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git", str(ROOT)])

run(["git", "config", "user.email", "colab-runner@local"], cwd=ROOT)
run(["git", "config", "user.name", "Colab Runner"], cwd=ROOT)
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=ROOT)

if HF_TOKEN:
    from huggingface_hub import HfApi, login
    login(token=HF_TOKEN, add_to_git_credential=False)
    who = HfApi(token=HF_TOKEN).whoami()
    print("HF auth OK:", who.get("name") or who.get("email") or "authenticated user")

run([sys.executable, "colab/run_stage1_single_runtime.py"], cwd=ROOT)
```

## Optional Notebook Reference

Open `colab/08_stage5_safe_continue.ipynb` when you want the safest current
single-notebook path. It defaults to a no-GPU dry run and requires
`RUN_A100_ACTION = True` before executing the guarded next action. Once it is
open, keep all stage execution inside that one notebook.

Open `colab/00_single_a100_runbook.ipynb` only when you want the older broad
launcher layout.

The older split notebooks are kept only as references:

1. `01_stage1_svgd_seed_replication.ipynb`
2. `02_stage2_benchmark_harness.ipynb`
3. `03_stage3_hf_packaging.ipynb`
4. `04_stage4_modified_opus_finetune.ipynb`
5. `05_stage5_benchmarks.ipynb`
6. `06_stage6_writeup_and_release.ipynb`

## Stage Map

1. Finish heldout seed `5-9` replication for random32 vs recreated
   within-group PCA.
2. Build/validate the MCQ benchmark harness.
3. Export and push adapter/controller package to Hugging Face.
4. Audit reasoning-trace sources before expanding the modified-Opus mix. Run
   `colab/run_stage5_reasoning_dataset_pipeline.py` when you want one cell to
   audit and then execute the planner-selected next action. For audit-only
   mode, run `colab/run_stage5_reasoning_dataset_audit.py`. Both read
   `config/reasoning_dataset_registry.yaml`, audit Opus/Fable-style schemas,
   write `outputs/stage5/<run_id>/summary.{json,md}`, and classify each dataset
   as immediate trace-SFT material, audit-only material, or later
   agent/tool-diversity material. Use this to keep Opus competence recovery
   separate from Fable tool/agent trajectory experiments.
5. Fine-tune on modified Opus traces, or on a generated width/depth curriculum
   shard after it passes `training/check_curriculum_sft_gate.py`. The guarded
   generated-curriculum handoff is `colab/run_stage5_curriculum_sft.py`: it
   requires a green SFT gate, enough `positive_*` rows, explicit
   `STAGE5_CURRICULUM_MIN_MODE_ROWS`, and Drive backup before launching Phase 1
   recurrent training. The current default is
   `direct=64,deep_narrow=64`; switch it deliberately for later width-only
   shards. The copy/paste Colab helper is `colab/STAGE5_CURRICULUM_SFT_CELL.py`
   with notes in
   `colab/STAGE5_CURRICULUM_SFT_CELL.md`. Keep Phase 2/SVGD off until this
   deterministic recurrent checkpoint validates.
6. Run base vs recurrent benchmarks, then run
   `colab/run_stage5_arc_agi_candidate_gate.py` to separate model-only,
   symbolic-only, and hybrid candidate value before more particle tuning.
   If symbolic candidates help, run `colab/run_stage5_arc_agi_trace_sft_gate.py`
   to compare grid-only ARC SFT against symbolic-trace ARC SFT on trace-covered
   examples. Then run `colab/run_stage5_arc_agi_distill_sft_gate.py` to test
   whether frozen-base logit distillation preserves competence during ARC SFT.
   For a controlled recurrent-recovery curriculum, run
   `colab/run_stage5_arc_agi_sft.py` with
   `STAGE5_ARC_AGI_SYNTHETIC_TASKS=200`,
   `STAGE5_ARC_AGI_TRACE_MODE=symbolic_program`, and
   `STAGE5_ARC_AGI_TRACE_FILTER=covered`. That tests whether targeted
   symbolic ARC program traces improve the deterministic recurrent model
   before attributing value to particles/SVGD.
   For a more recurrent-specific curriculum arm, set
   `STAGE5_ARC_AGI_TRACE_MODE=symbolic_state_trace`; this includes compact
   intermediate grid states after each symbolic operation, and should be
   compared against `symbolic_program` before scaling.
   Every ARC SFT run also writes `training_signal.json` and
   `training_signal.md`, which profile how many rows came from public ARC,
   synthetic families, candidate distillation, symbolic traces, and
   program-style traces. Use that audit before scaling a run that appears to
   improve or regress unexpectedly.
   The most direct combined gate is
   `colab/run_stage5_arc_agi_recovery_particle_gate.py`: it runs that synthetic
   recurrent-recovery SFT first, then tests low-noise K-particle/SVGD variants
   against the tuned recurrent checkpoint. It defaults to `symbolic_program`
   traces and `STAGE5_ARC_AGI_PROGRAM_PARSE_MODE=prefer`, so executable
   transformations are measured before final-grid formatting. Use its two
   decisions to keep "training helped" separate from "particles helped." It
   also evaluates a disjoint synthetic holdout, configured by
   `STAGE5_ARC_AGI_SYNTHETIC_EVAL_TASKS`, so we can see whether program-trace
   training generalizes inside the taught operation family.
   For unattended A100 time under the current low-credit policy, use
   `colab/CURRENT_A100_BOOTSTRAP_CELL.py` with
   `STAGE5_CURRENT_A100_TARGET=safe_continue_execute`, or run
   `colab/run_stage5_colab_continue.py` inside the already-attached runtime.
   These maintained paths run the current go/no-go guard, execute one
   allowlisted planner action by default, write summaries, and stop. The older
   `colab/run_stage5_arc_agi_autopilot.py` remains available only when the
   planner or run card explicitly selects that ARC-AGI branch.
   To keep a live A100 session moving after any Stage 5 result lands, run
   `colab/run_stage5_colab_continue.py`. It prints GPU state, runs focused
   Stage 5 tests including the Gate 1 assessor and next-action executor, runs
   `colab/run_stage5_next_action.py`, writes the progress ledger, and commits
   changed `outputs/stage5` artifacts. It now defaults to a credit-saving
   single-action loop: one allowlisted planner action, one summary, then stop.
   Set `STAGE5_ARC_AGI_COLAB_CONTINUE_PROFILE=gate` or
   `STAGE5_ARC_AGI_COLAB_CONTINUE_PROFILE=throughput` for the older bounded
   three-action loop when you intentionally want candidate, trace-SFT, and
   distill/dense-control decisions to chain inside one A100 session. Set
   `STAGE5_ARC_AGI_COLAB_CONTINUE_MAX_ACTIONS=N` to override the action count
   directly. Set `STAGE5_ARC_AGI_COLAB_CONTINUE_PROFILE=same_recipe` for
   a six-action ladder intended to continue through candidate gate, trace gate,
   distill/dense-control selection, dense control, matched recurrent SFT, and
   the same-recipe architecture assessment when each prior step completes. Set
   `STAGE5_ARC_AGI_COLAB_CONTINUE_PROFILE=claim` for a ten-action bounded
   release/SOTA-readiness ladder after same-recipe evidence exists; it can
   continue through release gate, broader benchmark suite, claim packet,
   same-size ARC-AGI comparison, and baseline-registry validation.
   `colab/run_stage5_next_action.py` remains available directly. By default it
   writes a dry-run summary of the planner's top action. Set
   `STAGE5_ARC_AGI_NEXT_ACTION_EXECUTE=1` to execute the selected allowlisted
   Stage 5 runner without copying commands by hand. Set
   `STAGE5_ARC_AGI_NEXT_ACTION_MAX_ACTIONS=2` or higher only when you want a
   bounded planner-runner loop; repeated commands stop the loop unless
   `STAGE5_ARC_AGI_NEXT_ACTION_ALLOW_REPEAT=1` is set.
   To summarize all saved Stage 5 evidence without using the GPU, run
   `colab/summarize_stage5_progress.py`. It scans `outputs/stage5`, writes a
   compact progress ledger, and reports the best base/recurrent/recovered arms,
   recovered-vs-base gaps, and the latest planner-compatible source summary.
   The continuation wrapper only commits safe text/report artifacts from
   `outputs/stage5` and `outputs/hf_exports` (`.json`, `.jsonl`, `.md`, `.log`,
   `.txt`, `.yaml`, `.yml`, `.csv`, `.html`) and skips checkpoints such as
   `.pt`/`.safetensors`; keep large trainable adapters in Drive or Hugging Face.
6. Write the report, model card, and release notes.
