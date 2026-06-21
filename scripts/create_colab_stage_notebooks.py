"""Generate staged Colab notebooks for the recurrent Qwen SVGD project."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COLAB = ROOT / "colab"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip() + "\n",
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "A100"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


BOOTSTRAP = r"""
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

GH_TOKEN = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or secret("GH_TOKEN") or secret("GITHUB_TOKEN")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or secret("HF_TOKEN") or secret("HUGGINGFACE_HUB_TOKEN")
assert GH_TOKEN, "Missing GH_TOKEN or GITHUB_TOKEN in Colab secrets."
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

def run(cmd, cwd=None, check=True):
    printable = " ".join(map(str, cmd))
    if GH_TOKEN:
        printable = printable.replace(GH_TOKEN, "****")
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

run(["git", "log", "--oneline", "-5"], cwd=ROOT)
"""


STAGE1_RUN = r"""
import json, os, subprocess, sys
from pathlib import Path

ROOT = Path("/content/recurrent-qwen-svgd")
RUN_ID = "stage1_seed5_9"
PHASE1 = "outputs/qwen_0_5b_phase1_recreated_beta008_150/phase1_step_150.pt"
PHASE2 = "outputs/qwen_0_5b_phase2_svgd_recreated_smoke25/phase2_step_25.pt"
PROJ = "outputs/calibration/recreated_within_group_pca_projection.pt"
SEEDS = "5,6,7,8,9"

def run(cmd, check=True):
    print("$", " ".join(map(str, cmd)))
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(proc.stdout)
    if check and proc.returncode:
        raise RuntimeError("command failed")
    return proc

for path in [PHASE1, PHASE2, PROJ, "outputs/heldout_task_splits/fold0_heldout.jsonl", "outputs/heldout_task_splits/fold1_heldout.jsonl"]:
    assert (ROOT / path).exists(), f"missing required artifact: {path}"

common = [
    sys.executable, "eval/eval_best_of_k_jsonl.py",
    "--skip_phase1",
    "--compact",
    "--seeds", SEEDS,
    "--phase1_checkpoint", PHASE1,
    "--phase2_checkpoint", PHASE2,
    "--phase2_num_trajectories", "4",
    "--phase2_particle_update_mode", "svgd",
    "--particle_init_noise", "0.05",
    "--particle_noise_every_step",
    "--particle_noise_steps", "16",
    "--svgd_repulsion_max_norm", "none",
    "--temperature", "0.0",
    "--max_new_tokens", "140",
    "--dtype", "bfloat16",
    "--adapter_dtype", "float32",
    "--device", "cuda",
]

jobs = [
    (
        "extended_fold0_random32_rep05_seeds5_9",
        "outputs/heldout_task_splits/fold0_heldout.jsonl",
        "outputs/diagnostics/extended_fold0_random32_rep05_seeds5_9.jsonl",
        ["--svgd_kernel_geometry", "euclidean", "--svgd_kernel_projection_dim", "32", "--svgd_projection_seed", "123", "--svgd_repulsion_scale", "0.5"],
    ),
    (
        "extended_fold0_wg_dim8_rep2_seeds5_9",
        "outputs/heldout_task_splits/fold0_heldout.jsonl",
        "outputs/diagnostics/extended_fold0_wg_dim8_rep2_seeds5_9.jsonl",
        ["--svgd_kernel_geometry", "euclidean", "--svgd_kernel_projection_path", PROJ, "--svgd_kernel_projection_dim", "8", "--svgd_repulsion_scale", "2"],
    ),
    (
        "extended_fold1_random32_rep05_seeds5_9",
        "outputs/heldout_task_splits/fold1_heldout.jsonl",
        "outputs/diagnostics/extended_fold1_random32_rep05_seeds5_9.jsonl",
        ["--svgd_kernel_geometry", "euclidean", "--svgd_kernel_projection_dim", "32", "--svgd_projection_seed", "123", "--svgd_repulsion_scale", "0.5"],
    ),
    (
        "extended_fold1_wg_dim8_rep2_seeds5_9",
        "outputs/heldout_task_splits/fold1_heldout.jsonl",
        "outputs/diagnostics/extended_fold1_wg_dim8_rep2_seeds5_9.jsonl",
        ["--svgd_kernel_geometry", "euclidean", "--svgd_kernel_projection_path", PROJ, "--svgd_kernel_projection_dim", "8", "--svgd_repulsion_scale", "2"],
    ),
]

for label, tasks, out, extra in jobs:
    print("\n\n====", label, "====")
    cmd = common + ["--tasks_jsonl", tasks, "--output_jsonl", out] + extra
    log_path = ROOT / "outputs" / "diagnostics" / f"{label}.log"
    proc = run(cmd, check=False)
    log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode:
        raise RuntimeError(f"{label} failed; see {log_path}")

def read_rows(path):
    rows = []
    for line in (ROOT / path).read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows

def metrics(paths):
    rows = []
    for path in paths:
        rows.extend(read_rows(path))
    grouped = {}
    for row in rows:
        grouped.setdefault((row.get("seed"), row.get("task")), []).append(row)
    return {
        "best_hits": sum(any(item.get("hit") for item in items) for items in grouped.values()),
        "total_tasks": len(grouped),
        "candidate_hits": sum(1 for row in rows if row.get("hit")),
        "total_candidates": len(rows),
    }

summary_sets = {
    "fold0_seeds0_4_random32": ["outputs/diagnostics/recreated_fold0_random32_rep05.jsonl"],
    "fold0_seeds0_4_wg_dim8": ["outputs/diagnostics/recreated_fold0_wg_dim8_rep2.jsonl"],
    "fold1_seeds0_4_random32": ["outputs/diagnostics/recreated_fold1_random32_rep05.jsonl"],
    "fold1_seeds0_4_wg_dim8": ["outputs/diagnostics/recreated_fold1_wg_dim8_rep2.jsonl"],
    "fold0_seeds5_9_random32": ["outputs/diagnostics/extended_fold0_random32_rep05_seeds5_9.jsonl"],
    "fold0_seeds5_9_wg_dim8": ["outputs/diagnostics/extended_fold0_wg_dim8_rep2_seeds5_9.jsonl"],
    "fold1_seeds5_9_random32": ["outputs/diagnostics/extended_fold1_random32_rep05_seeds5_9.jsonl"],
    "fold1_seeds5_9_wg_dim8": ["outputs/diagnostics/extended_fold1_wg_dim8_rep2_seeds5_9.jsonl"],
    "heldout_seeds0_9_random32": [
        "outputs/diagnostics/recreated_fold0_random32_rep05.jsonl",
        "outputs/diagnostics/recreated_fold1_random32_rep05.jsonl",
        "outputs/diagnostics/extended_fold0_random32_rep05_seeds5_9.jsonl",
        "outputs/diagnostics/extended_fold1_random32_rep05_seeds5_9.jsonl",
    ],
    "heldout_seeds0_9_wg_dim8": [
        "outputs/diagnostics/recreated_fold0_wg_dim8_rep2.jsonl",
        "outputs/diagnostics/recreated_fold1_wg_dim8_rep2.jsonl",
        "outputs/diagnostics/extended_fold0_wg_dim8_rep2_seeds5_9.jsonl",
        "outputs/diagnostics/extended_fold1_wg_dim8_rep2_seeds5_9.jsonl",
    ],
}

lines = []
for name, paths in summary_sets.items():
    line = f"{name}: {metrics(paths)}"
    print(line)
    lines.append(line)

rand = metrics(summary_sets["heldout_seeds0_9_random32"])
wg = metrics(summary_sets["heldout_seeds0_9_wg_dim8"])
delta = {
    "best_hits": wg["best_hits"] - rand["best_hits"],
    "candidate_hits": wg["candidate_hits"] - rand["candidate_hits"],
}
lines.append(f"heldout_delta_wg_minus_random32: {delta}")
print(lines[-1])

summary_path = ROOT / "outputs" / "diagnostics" / "extended_heldout_seeds0_9_summary.txt"
summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("saved", summary_path)

run(["git", "status", "-sb"])
run(["git", "add", "-f", "outputs/diagnostics/extended_*", "outputs/diagnostics/recreated_*", "outputs/heldout_task_splits/*.jsonl"])
status = run(["git", "diff", "--cached", "--quiet"], check=False)
if status.returncode == 0:
    print("No staged changes to commit.")
else:
    run(["git", "commit", "-m", "Extend heldout SVGD diagnostics seeds 5-9"])
    run(["git", "push", "origin", "main"])
"""


SINGLE_RUNTIME_STAGE1_CELL = BOOTSTRAP + "\n\n" + STAGE1_RUN


STAGE5_CONTINUE_RUN = r"""
import os, subprocess, sys, time
from pathlib import Path

ROOT = Path("/content/recurrent-qwen-svgd")
RUN_ID = os.environ.get("STAGE5_ARC_AGI_NEXT_ACTION_RUN_ID") or time.strftime("stage5_arc_agi_colab_continue_%Y%m%d_%H%M%S")

# Conservative defaults: execute one allowlisted next action. Increase MAX_ACTIONS
# only when you want the planner to chain safe follow-up actions in the same A100 session.
os.environ.setdefault("STAGE5_ARC_AGI_NEXT_ACTION_RUN_ID", RUN_ID)
os.environ.setdefault("STAGE5_ARC_AGI_NEXT_ACTION_EXECUTE", "1")
os.environ.setdefault("STAGE5_ARC_AGI_NEXT_ACTION_MAX_ACTIONS", "1")
os.environ.setdefault("STAGE5_ARC_AGI_NEXT_ACTION_ALLOW_REPEAT", "0")
os.environ.setdefault("STAGE5_ARC_AGI_AUTOPILOT_TRACE_SFT_GATE_ARMS", "grid_only,symbolic_program_trace_covered,symbolic_state_trace_covered")
os.environ.setdefault("STAGE5_ARC_AGI_NEXT_PLAN_TRACE_SFT_GATE_ARMS", "grid_only,symbolic_program_trace_covered,symbolic_state_trace_covered")
os.environ.setdefault("DRIVE_BACKUP_DIR", "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts")

def run(cmd, check=True):
    print("$", " ".join(map(str, cmd)))
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(proc.stdout)
    if check and proc.returncode:
        raise RuntimeError(f"failed: {' '.join(map(str, cmd))}")
    return proc

try:
    from google.colab import drive

    drive.mount("/content/drive")
except Exception as exc:
    print("Drive mount skipped/failed:", exc)

run(["nvidia-smi"], check=False)
run([sys.executable, "-m", "pytest", "-q", "tests/test_stage5_autopilot.py", "tests/test_stage5_next_plan.py", "tests/test_stage5_sft_gates.py"])

print("RUN_ID", RUN_ID)
run([sys.executable, "colab/run_stage5_next_action.py"])
run([sys.executable, "colab/summarize_stage5_progress.py"], check=False)

run(["git", "status", "-sb"])
run(["git", "add", "-f", "outputs/stage5"])
status = run(["git", "diff", "--cached", "--quiet"], check=False)
if status.returncode == 0:
    print("No Stage 5 outputs to commit.")
else:
    run(["git", "commit", "-m", f"Record Stage 5 continuation {RUN_ID}"])
    run(["git", "push", "origin", "main"], check=False)
"""


SINGLE_RUNTIME_STAGE5_CELL = BOOTSTRAP + "\n\n" + STAGE5_CONTINUE_RUN


def write_notebook(name: str, cells: list[dict]) -> None:
    COLAB.mkdir(parents=True, exist_ok=True)
    path = COLAB / name
    path.write_text(json.dumps(notebook(cells), indent=2) + "\n", encoding="utf-8")
    print(path)


def main() -> int:
    write_notebook(
        "00_single_a100_runbook.ipynb",
        [
            md(
                """
                # Single A100 Runbook - Recurrent Qwen SVGD

                This is the preferred Colab workflow. Keep one A100 runtime attached
                to this notebook and run the stage cells in order. Do not hop between
                notebooks unless you intentionally want a separate session.

                The first executable cell is the current Stage 5 continuation path:
                clone/pull latest GitHub, verify auth, run focused tests, execute one
                allowlisted next action, summarize progress, and push run summaries.
                """
            ),
            code(BOOTSTRAP),
            md(
                """
                ## Current Stage 5 Continuation

                Runs one planner-selected, allowlisted next action. Change
                `STAGE5_ARC_AGI_NEXT_ACTION_MAX_ACTIONS` in the cell only when you
                intentionally want a bounded multi-action loop in the same runtime.
                """
            ),
            code(STAGE5_CONTINUE_RUN),
            md(
                """
                ## Historical Stage 1 - SVGD Heldout Seed Replication

                Kept for reproducing the earlier SVGD heldout seed diagnostics.
                """
            ),
            code(STAGE1_RUN),
            md(
                """
                ## Benchmark Harness Gate

                Run after a recovered recurrent checkpoint is selected by Stage 5.
                """
            ),
            code(
                r"""
%cd /content/recurrent-qwen-svgd

print('''
Benchmark order:
1. ARC-AGI recovered recurrent vs base at increasing limits
2. TTA/selector sweeps only after recurrent recovery is non-negative
3. MCQ harness smoke
4. GPQA-lite/sample
5. GPQA Diamond only after packaging/reload is deterministic
''')
"""
            ),
            md(
                """
                ## Stage 4 - Modified Opus Fine-Tune Gate

                Resume modified Opus training only after Stage 1 replication and the
                benchmark harness are stable.
                """
            ),
            code(
                r"""
%cd /content/recurrent-qwen-svgd

from google.colab import drive
drive.mount('/content/drive')

!python colab/run_stage4_opus_finetune.py
"""
            ),
            md(
                """
                ## Stage 5 - Benchmark Runs Gate

                Run serious benchmarks after packaging/reload is deterministic.
                """
            ),
            code(
                r"""
%cd /content/recurrent-qwen-svgd

print('''
Benchmark order:
1. exact smoke v2/v3
2. tiny handcrafted MCQ sanity set
3. GPQA-lite/sample
4. ARC-Challenge or science MCQ subset
5. GPQA Diamond full

Report separately:
- deterministic option-likelihood accuracy
- Phase 2 seed/K settings
- best-of-K oracle where applicable
- selector/verifier-selected score only if a selector exists
''')
"""
            ),
            md(
                """
                ## Stage 6 - Write-Up and Release Gate

                Turn the run into a reproducible result bundle.
                """
            ),
            code(
                r"""
%cd /content/recurrent-qwen-svgd

print('''
Stage 6 deliverables:
1. docs/U5_WITHIN_GROUP_SVGD_REPORT.md
2. docs/EXPERIMENT_LOG.md update
3. HF model card
4. benchmark result JSONLs
5. exact reproduction commands
6. limitations section: small model, no verifier, slow no-cache recurrent generation
''')
"""
            ),
        ],
    )

    write_notebook(
        "00_stage_launcher.ipynb",
        [
            md(
                """
                # Recurrent Qwen SVGD Single-Runtime Launcher

                Run the single code cell below in the already-attached Colab A100
                runtime. It clones/updates the private GitHub repo and runs the
                Stage 5 continuation directly in this notebook. It does not open or route
                you to another notebook.
                """
            ),
            code(SINGLE_RUNTIME_STAGE5_CELL),
        ],
    )

    write_notebook(
        "01_stage1_svgd_seed_replication.ipynb",
        [
            md(
                """
                # Stage 1 - SVGD Seed Replication

                Goal: finish the heldout seed `5-9` replication for the current best diagnostic:

                - baseline: random projection dim 32, repulsion 0.5
                - candidate: recreated within-group PCA dim 8, repulsion 2

                This notebook uses the recreated artifacts already committed to GitHub.
                """
            ),
            code(BOOTSTRAP),
            code(STAGE1_RUN),
        ],
    )

    write_notebook(
        "02_stage2_benchmark_harness.ipynb",
        [
            md(
                """
                # Stage 2 - Benchmark Harness

                Goal: make MCQ/science benchmark evaluation fair before GPQA Diamond.

                Deliverables:
                - base Qwen 0.5B scorer
                - recurrent Phase 1 scorer
                - recurrent Phase 2/SVGD scorer
                - JSONL result output
                - per-question prediction records
                - seed-aware Phase 2 evaluation
                """
            ),
            code(BOOTSTRAP),
            code(
                r"""
%cd /content/recurrent-qwen-svgd
!python colab/run_stage2_mcq_smoke.py
"""
            ),
        ],
    )

    write_notebook(
        "03_stage3_hf_packaging.ipynb",
        [
            md(
                """
                # Stage 3 - Hugging Face Packaging

                Goal: publish a loadable adapter/controller artifact first, then optionally
                a `trust_remote_code` full recurrent wrapper repo.
                """
            ),
            code(BOOTSTRAP),
            code(
                r"""
%cd /content/recurrent-qwen-svgd

MODEL_REPO = "mshapiro123/recurrent-qwen-svgd-0.5b-adapter"
print("Target HF repo:", MODEL_REPO)
print('''
Stage 3 implementation checklist:
1. Export trainable checkpoint to safetensors.
2. Save adapter config: base_model, split, max_loops, SVGD defaults, projection path metadata.
3. Write README/model card with limitations and exact commands.
4. Push adapter package to HF private repo first.
5. Add a reload smoke test from HF.
''')
"""
            ),
        ],
    )

    write_notebook(
        "04_stage4_modified_opus_finetune.ipynb",
        [
            md(
                """
                # Stage 4 - Modified Opus Reasoning Fine-Tune

                Goal: resume training on modified Opus reasoning traces only after Stage 1
                and Stage 2 are stable.
                """
            ),
            code(BOOTSTRAP),
            code(
                r"""
%cd /content/recurrent-qwen-svgd

from google.colab import drive
drive.mount('/content/drive')

!python colab/run_stage4_opus_finetune.py
"""
            ),
        ],
    )

    write_notebook(
        "05_stage5_benchmarks.ipynb",
        [
            md(
                """
                # Stage 5 - Benchmarks

                Goal: compare unmodified Qwen 0.5B, Phase 1, and Phase 2/SVGD on
                increasingly serious benchmarks before claiming GPQA Diamond progress.
                """
            ),
            code(BOOTSTRAP),
            code(
                r"""
%cd /content/recurrent-qwen-svgd

print('''
Benchmark order:
1. exact smoke v2/v3
2. tiny handcrafted MCQ sanity set
3. GPQA-lite/sample
4. ARC-Challenge or science MCQ subset
5. GPQA Diamond full

Report separately:
- deterministic option-likelihood accuracy
- Phase 2 seed/K settings
- best-of-K oracle where applicable
- selector/verifier-selected score only if a selector exists
''')
"""
            ),
        ],
    )

    write_notebook(
        "06_stage6_writeup_and_release.ipynb",
        [
            md(
                """
                # Stage 6 - Write-Up and Release

                Goal: turn the experiment into a reproducible report and release bundle.
                """
            ),
            code(BOOTSTRAP),
            code(
                r"""
%cd /content/recurrent-qwen-svgd

print('''
Stage 6 deliverables:
1. docs/U5_WITHIN_GROUP_SVGD_REPORT.md
2. docs/EXPERIMENT_LOG.md update
3. HF model card
4. benchmark result JSONLs
5. exact reproduction commands
6. limitations section: small model, no verifier, slow no-cache recurrent generation
''')
"""
            ),
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
