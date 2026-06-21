# Single A100 Colab Runbook

The preferred workflow is **one Colab notebook attached to one A100 runtime**.
Do not open each stage as a separate notebook while an expensive runtime is
active. The staged notebooks remain in `colab/` for review, but the execution
path below stays inside the current runtime.

## Start or Resume Stage 1 In-Place

Paste this single cell into the already-attached A100 notebook. It clones or
updates the private GitHub repo, installs dependencies, validates optional HF
auth, then runs Stage 1 directly in the current runtime.

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

Open `colab/00_single_a100_runbook.ipynb` only when you want a clean notebook
layout. Once it is open, keep all stage execution inside that one notebook.

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
4. Fine-tune on modified Opus traces.
5. Run base vs recurrent benchmarks, then run
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
   For unattended A100 time, run `colab/run_stage5_arc_agi_autopilot.py`
   instead; it branches through those gates with explicit thresholds and writes
   one decision report.
   To keep a live A100 session moving after any Stage 5 result lands, run
   `colab/run_stage5_colab_continue.py`. It prints GPU state, runs focused
   Stage 5 tests including the Gate 1 assessor and next-action executor, runs
   `colab/run_stage5_next_action.py`, writes the progress ledger, and commits
   changed `outputs/stage5` artifacts. It defaults to a bounded two-action loop
   so a cheap Gate 1 assessment can run and the planner can immediately react
   to the assessment. Set `STAGE5_ARC_AGI_COLAB_CONTINUE_MAX_ACTIONS=1` for a
   strictly single-action continuation.
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
6. Write the report, model card, and release notes.
