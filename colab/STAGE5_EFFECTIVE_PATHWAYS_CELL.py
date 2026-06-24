"""Colab launcher: effective-pathway dynamics diagnostic.

This is a bounded Stage 5 diagnostic, not a training run. It probes whether the
deterministic recurrent map preserves multiple latent pathways for a fixed
prompt when SVGD and latent sampling are disabled.
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

STAGE5_EFFECTIVE_PATHWAYS_CELL_VERSION = "stage5_effective_pathways_v1"
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


def run(cmd: list[str | os.PathLike[str]], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
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
    proc.check_returncode()
    return proc


def ensure_repo() -> None:
    clone_url = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
    if ROOT.exists():
        run(["git", "remote", "set-url", "origin", clone_url])
        run(["git", "fetch", "origin", "main"])
        run(["git", "checkout", "main"])
        run(["git", "pull", "--ff-only", "origin", "main"])
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
    run_id = Path(rel_path).parts[2] if rel_path.startswith("outputs/stage5/") and len(Path(rel_path).parts) > 2 else ""
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


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def float_label(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def main() -> None:
    print(f"cell_version={STAGE5_EFFECTIVE_PATHWAYS_CELL_VERSION}", flush=True)
    run_id = os.environ.get("STAGE5_EFFECTIVE_PATHWAYS_RUN_ID") or time.strftime("stage5_effective_pathways_%Y%m%d_%H%M%S")
    checkpoint = os.environ.get("STAGE5_EFFECTIVE_PATHWAYS_CHECKPOINT", DEFAULT_CHECKPOINT)
    prompts = os.environ.get("STAGE5_EFFECTIVE_PATHWAYS_PROMPTS", "eval/smoke_exact_tasks_v2.jsonl")
    loop_sweep = parse_ints(os.environ.get("STAGE5_EFFECTIVE_PATHWAYS_LOOP_SWEEP", "4,8"))
    num_particles = os.environ.get("STAGE5_EFFECTIVE_PATHWAYS_NUM_PARTICLES", "16")
    noise_sweep = parse_floats(
        os.environ.get(
            "STAGE5_EFFECTIVE_PATHWAYS_NOISE_SWEEP",
            os.environ.get("STAGE5_EFFECTIVE_PATHWAYS_NOISE", "0.05"),
        )
    )
    print(f"noise_sweep={noise_sweep}", flush=True)
    limit = os.environ.get("STAGE5_EFFECTIVE_PATHWAYS_LIMIT", "8")
    dtype = os.environ.get("STAGE5_EFFECTIVE_PATHWAYS_DTYPE", "bfloat16")
    disconnect = os.environ.get("STAGE5_EFFECTIVE_PATHWAYS_DISCONNECT", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }

    ensure_repo()
    run(["nvidia-smi"], cwd=Path("/content"))
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run([sys.executable, "-m", "pytest", "-q", "tests/test_pathway_diversity.py", "tests/test_eval_effective_pathways.py"])
    restore_checkpoint(checkpoint)

    out_dir = ROOT / "outputs" / "stage5" / run_id
    diag_dir = out_dir / "effective_pathways"
    diag_dir.mkdir(parents=True, exist_ok=True)

    result_paths: list[str] = []
    for particle_noise in noise_sweep:
        noise_tag = float_label(particle_noise)
        for loops in loop_sweep:
            out_json = diag_dir / f"effective_pathways_noise{noise_tag}_loops{loops}.json"
            out_jsonl = diag_dir / f"effective_pathways_noise{noise_tag}_loops{loops}.jsonl"
            run(
                [
                    sys.executable,
                    "eval/eval_effective_pathways.py",
                    "--checkpoint",
                    checkpoint,
                    "--prompts_jsonl",
                    prompts,
                    "--limit",
                    limit,
                    "--max_loops",
                    str(loops),
                    "--num_particles",
                    num_particles,
                    "--particle_init_noise",
                    str(particle_noise),
                    "--max_length",
                    os.environ.get("STAGE5_EFFECTIVE_PATHWAYS_MAX_LENGTH", "256"),
                    "--dtype",
                    dtype,
                    "--adapter_dtype",
                    os.environ.get("STAGE5_EFFECTIVE_PATHWAYS_ADAPTER_DTYPE", "float32"),
                    "--device",
                    os.environ.get("STAGE5_EFFECTIVE_PATHWAYS_DEVICE", "cuda"),
                    "--output_json",
                    str(out_json.relative_to(ROOT)),
                    "--output_jsonl",
                    str(out_jsonl.relative_to(ROOT)),
                ]
            )
            result_paths.append(str(out_json.relative_to(ROOT)))

    summary = {
        "kind": "stage5_effective_pathways_run",
        "run_id": run_id,
        "cell_version": STAGE5_EFFECTIVE_PATHWAYS_CELL_VERSION,
        "checkpoint": checkpoint,
        "prompts": prompts,
        "loop_sweep": loop_sweep,
        "num_particles": int(num_particles),
        "noise_sweep": noise_sweep,
        "limit": int(limit),
        "result_paths": result_paths,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(
        "\n".join(
            [
                f"# Stage 5 Effective Pathways - {run_id}",
                "",
                f"- Checkpoint: `{checkpoint}`",
                f"- Prompts: `{prompts}`",
                f"- Loop sweep: `{loop_sweep}`",
                f"- Num particles: `{num_particles}`",
                f"- Noise sweep: `{noise_sweep}`",
                "",
                "## Result Files",
                *[f"- `{path}`" for path in result_paths],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(summary_path.read_text(encoding="utf-8"), flush=True)

    if Path("/content/drive/MyDrive").exists():
        backup_dir = DRIVE_ARTIFACT_ROOT / "outputs" / "stage5" / run_id
        backup_dir.parent.mkdir(parents=True, exist_ok=True)
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(out_dir, backup_dir)
        print(f"drive_backup={backup_dir}", flush=True)

    run(["git", "status", "-sb"])
    # `outputs/` is ignored by default, but selected Stage 5 summaries are
    # intentionally versioned as evidence artifacts.
    run(["git", "add", "-f", str(out_dir.relative_to(ROOT))])
    diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=str(ROOT))
    if diff.returncode != 0:
        run(["git", "commit", "-m", f"Record Stage 5 effective pathways {run_id} [skip ci]"])
        if os.environ.get("STAGE5_EFFECTIVE_PATHWAYS_PUSH", "1").strip().lower() in {"1", "true", "yes", "y"}:
            run(["git", "push", "origin", "main"])
    else:
        print("No staged changes to commit.", flush=True)

    if disconnect:
        print("Disconnecting Colab runtime to conserve credits.", flush=True)
        runtime.unassign()


main()
