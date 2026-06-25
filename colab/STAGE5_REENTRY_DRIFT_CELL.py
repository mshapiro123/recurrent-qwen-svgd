"""Colab launcher: Stage 5 recurrent re-entry drift diagnostic.

This is a bounded read-only diagnostic. It measures the recurrent loop-closure
path before any bridge/re-entry repair or additional training.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from google.colab import drive, runtime, userdata

STAGE5_REENTRY_DRIFT_CELL_VERSION = "stage5_reentry_drift_v1_readonly"
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DRIVE_ARTIFACT_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts")

DEFAULT_CHECKPOINT = (
    "outputs/stage5/stage5_content_arcmix_qonly_optiontext_20260623_121707/"
    "arc_mix_response_w02_lr2e6/phase1/phase1_step_150.pt"
)


def secret(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
        try:
            value = userdata.get(name)
        except Exception:
            value = None
        if value:
            return value
    return None


GH_TOKEN = secret("GH_TOKEN", "GITHUB_TOKEN")
HF_TOKEN = secret("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
assert GH_TOKEN, "Missing GH_TOKEN or GITHUB_TOKEN in Colab secrets."
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
    print("HF token loaded", flush=True)
else:
    print("HF token not found; downloads will be anonymous.", flush=True)


def run(
    cmd: list[str | os.PathLike[str]],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    printable = " ".join(map(str, cmd)).replace(GH_TOKEN, "****")
    print(f"$ {printable}", flush=True)
    proc = subprocess.run(
        list(map(str, cmd)),
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(proc.stdout, flush=True)
    if check:
        proc.check_returncode()
    return proc


def ensure_repo() -> None:
    clone_url = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
    if ROOT.exists():
        run(["git", "remote", "set-url", "origin", clone_url])
        run(["git", "fetch", "origin", "main"])
        run(["git", "checkout", "main"])
        run(["git", "reset", "--hard", "origin/main"])
    else:
        run(["git", "clone", clone_url, ROOT], cwd=Path("/content"))
    run(["git", "config", "user.email", "colab-runner@local"])
    run(["git", "config", "user.name", "Colab Runner"])
    run(["git", "log", "--oneline", "-5"])


def restore_checkpoint(rel_path: str) -> Path:
    target = ROOT / rel_path
    if target.exists():
        print(f"checkpoint already local: {target}", flush=True)
        return target

    print("Mounting Drive to restore checkpoint.", flush=True)
    drive.mount("/content/drive", force_remount=False)

    candidates = [
        DRIVE_ARTIFACT_ROOT / rel_path,
        Path("/content/drive/MyDrive/recurrent-qwen-svgd") / rel_path,
    ]
    rel = Path(rel_path)
    run_id = rel.parts[2] if rel_path.startswith("outputs/stage5/") and len(rel.parts) > 2 else ""
    for root in [DRIVE_ARTIFACT_ROOT, Path("/content/drive/MyDrive/recurrent-qwen-svgd")]:
        if root.exists() and run_id:
            candidates.extend(p for p in root.rglob(target.name) if run_id in str(p))

    for candidate in candidates:
        if candidate.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, target)
            print(f"restored_checkpoint={candidate} -> {target}", flush=True)
            return target

    raise FileNotFoundError(f"Could not restore checkpoint from Drive: {rel_path}")


def write_markdown(summary: dict[str, object], path: Path) -> None:
    aggregate = summary.get("aggregate") if isinstance(summary.get("aggregate"), dict) else {}
    bridge = summary.get("bridge") if isinstance(summary.get("bridge"), dict) else {}
    liveness = summary.get("bridge_gradient_liveness") if isinstance(summary.get("bridge_gradient_liveness"), dict) else {}
    subspace = aggregate.get("entry_exit_subspace") if isinstance(aggregate.get("entry_exit_subspace"), dict) else {}
    loops = aggregate.get("loop_summary") if isinstance(aggregate.get("loop_summary"), dict) else {}
    lines = [
        f"# Stage 5 Re-entry Drift - {summary.get('run_id', '')}",
        "",
        f"- Checkpoint: `{summary.get('checkpoint', '')}`",
        f"- Prompts: `{summary.get('prompts', '')}`",
        f"- Max loops: `{summary.get('max_loops', '')}`",
        f"- Cell version: `{summary.get('cell_version', '')}`",
        "",
        "## Aggregate",
        f"- Mean entry RMS: `{aggregate.get('mean_entry_rms')}`",
        f"- Mean exit RMS: `{aggregate.get('mean_exit_rms')}`",
        f"- Mean exit/entry RMS: `{aggregate.get('mean_exit_over_entry_rms')}`",
        f"- Mean pooled entry/exit cosine: `{aggregate.get('mean_pooled_entry_exit_cosine')}`",
        f"- Subspace overlap: `{subspace.get('overlap')}`",
        f"- Aligned dims >= 0.8: `{subspace.get('aligned_dims_cos_ge_0p8')}`",
        f"- Aligned dims >= 0.9: `{subspace.get('aligned_dims_cos_ge_0p9')}`",
        "",
        "## Bridge",
        f"- Bridge gate: `{bridge.get('bridge_gate')}`",
        f"- Projection identity max abs diff: `{bridge.get('proj_identity_max_abs_diff')}`",
        f"- Bridge delta RMS: `{bridge.get('sample_bridge_delta_rms')}`",
        f"- Gate grad abs: `{liveness.get('gate_grad_abs')}`",
        f"- Weight grad RMS: `{liveness.get('weight_grad_rms')}`",
        f"- Bias grad RMS: `{liveness.get('bias_grad_rms')}`",
        "",
        "## Loop Drift",
        "| loop | input/entry RMS | output/entry RMS | output/input RMS | bridge delta RMS | input-output cosine |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for loop, row in sorted(loops.items(), key=lambda item: int(item[0])):
        if not isinstance(row, dict):
            continue
        lines.append(
            "| {loop} | {input_entry} | {output_entry} | {output_input} | {bridge_delta} | {cosine} |".format(
                loop=loop,
                input_entry=row.get("input_over_entry_rms"),
                output_entry=row.get("output_over_entry_rms"),
                output_input=row.get("output_over_input_rms"),
                bridge_delta=row.get("bridge_delta_rms"),
                cosine=row.get("pooled_input_output_cosine"),
            )
        )
    lines.extend(
        [
            "",
            "## Readout Pause",
            "This run intentionally stops after Stage 1. Review these numbers before running re-entry normalization or bridge repair.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def publish_outputs(out_dir: Path, run_id: str) -> None:
    run(["git", "status", "-sb"])
    run(["git", "add", "-f", str(out_dir.relative_to(ROOT))])
    diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=str(ROOT))
    if diff.returncode == 0:
        print("No staged changes to commit.", flush=True)
        return
    run(["git", "commit", "-m", f"Record Stage 5 re-entry drift {run_id} [skip ci]"])
    if os.environ.get("STAGE5_REENTRY_DRIFT_PUSH", "1").strip().lower() in {"1", "true", "yes", "y"}:
        pushed = run(["git", "push", "origin", "main"], check=False)
        if pushed.returncode != 0:
            print("Initial push failed; attempting one fast rebase and retry.", flush=True)
            run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
            run(["git", "push", "origin", "main"])


def main() -> None:
    print(f"cell_version={STAGE5_REENTRY_DRIFT_CELL_VERSION}", flush=True)
    run_id = os.environ.get("STAGE5_REENTRY_DRIFT_RUN_ID") or time.strftime("stage5_reentry_drift_%Y%m%d_%H%M%S")
    checkpoint = os.environ.get("STAGE5_REENTRY_DRIFT_CHECKPOINT", DEFAULT_CHECKPOINT)
    prompts = os.environ.get("STAGE5_REENTRY_DRIFT_PROMPTS", "eval/smoke_exact_tasks_v2.jsonl")
    max_loops = os.environ.get("STAGE5_REENTRY_DRIFT_MAX_LOOPS", "8")
    limit = os.environ.get("STAGE5_REENTRY_DRIFT_LIMIT", "8")
    dtype = os.environ.get("STAGE5_REENTRY_DRIFT_DTYPE", "bfloat16")
    disconnect = os.environ.get("STAGE5_REENTRY_DRIFT_DISCONNECT", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }

    ensure_repo()
    run(["nvidia-smi"], cwd=Path("/content"))
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run([sys.executable, "-m", "pytest", "-q", "tests/test_eval_reentry_drift.py", "tests/test_bridge.py"])
    restore_checkpoint(checkpoint)

    out_dir = ROOT / "outputs" / "stage5" / run_id
    diag_dir = out_dir / "reentry_drift"
    diag_dir.mkdir(parents=True, exist_ok=True)
    out_json = diag_dir / "reentry_drift.json"
    out_jsonl = diag_dir / "reentry_drift.jsonl"
    run(
        [
            sys.executable,
            "eval/eval_reentry_drift.py",
            "--checkpoint",
            checkpoint,
            "--prompts_jsonl",
            prompts,
            "--limit",
            limit,
            "--max_loops",
            max_loops,
            "--max_length",
            os.environ.get("STAGE5_REENTRY_DRIFT_MAX_LENGTH", "256"),
            "--subspace_rank",
            os.environ.get("STAGE5_REENTRY_DRIFT_SUBSPACE_RANK", "8"),
            "--dtype",
            dtype,
            "--adapter_dtype",
            os.environ.get("STAGE5_REENTRY_DRIFT_ADAPTER_DTYPE", "float32"),
            "--device",
            os.environ.get("STAGE5_REENTRY_DRIFT_DEVICE", "cuda"),
            "--output_json",
            str(out_json.relative_to(ROOT)),
            "--output_jsonl",
            str(out_jsonl.relative_to(ROOT)),
        ]
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    payload.update(
        {
            "run_id": run_id,
            "cell_version": STAGE5_REENTRY_DRIFT_CELL_VERSION,
            "checkpoint": checkpoint,
            "prompts": prompts,
        }
    )
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(payload, out_dir / "summary.md")
    print((out_dir / "summary.md").read_text(encoding="utf-8"), flush=True)

    if Path("/content/drive/MyDrive").exists():
        backup_dir = DRIVE_ARTIFACT_ROOT / "outputs" / "stage5" / run_id
        backup_dir.parent.mkdir(parents=True, exist_ok=True)
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(out_dir, backup_dir)
        print(f"drive_backup={backup_dir}", flush=True)

    publish_outputs(out_dir, run_id)

    if disconnect:
        print("Disconnecting Colab runtime to conserve credits after Stage 1 readout.", flush=True)
        runtime.unassign()


main()
