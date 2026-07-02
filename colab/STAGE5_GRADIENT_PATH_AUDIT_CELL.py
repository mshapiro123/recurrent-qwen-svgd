"""Stage 5 gradient-path audit launcher.

This target is read-only: it restores the completed chain-supervision
checkpoint, audits one chain-labeled batch, publishes the diagnostic artifacts,
and leaves the runtime connected by default for review.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from google.colab import drive, runtime, userdata


STAGE5_GRADIENT_PATH_AUDIT_CELL_VERSION = "gradient_path_audit_v1"
# Safety marker: read-only gradient matrix plus finite_difference_bridge_prelude.
# Safety marker: CoherenceAccumulator and multiplier_consumption_check are required in eval/eval_gradient_path_audit.py.
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")


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
    print("HF token missing; model downloads will use anonymous Hub access.", flush=True)


def redact(text: str) -> str:
    out = str(text)
    for token in (GH_TOKEN, HF_TOKEN):
        if token:
            out = out.replace(token, "****")
    return out


def run(
    cmd: list[str | os.PathLike[str]],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    print("$", redact(" ".join(map(str, cmd))), flush=True)
    process = subprocess.Popen(
        list(map(str, cmd)),
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    chunks: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        safe = redact(line)
        print(safe, end="", flush=True)
        chunks.append(safe)
    stdout = "".join(chunks)
    proc = subprocess.CompletedProcess(list(map(str, cmd)), process.wait(), stdout, None)
    if check and proc.returncode:
        print("FAILED_COMMAND_TAIL_START", flush=True)
        print("\n".join(stdout.splitlines()[-180:]), flush=True)
        print("FAILED_COMMAND_TAIL_END", flush=True)
        raise subprocess.CalledProcessError(proc.returncode, proc.args, output=stdout)
    return proc


def env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def path_for_cli(path: str | Path) -> str:
    candidate = Path(path)
    try:
        return candidate.relative_to(ROOT).as_posix()
    except ValueError:
        return str(candidate).replace("\\", "/")


def sync_repo() -> None:
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
    run(["git", "log", "--oneline", "-5"], check=False)
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def require_gpu_runtime() -> None:
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("Attach a GPU runtime first. L4/T4 is sufficient for the gradient-path audit.")
    run(["nvidia-smi"], cwd=Path("/content"), check=False)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(resolve_repo_path(path).read_text(encoding="utf-8"))


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def latest_chain_stage(source_summary: Path, *, stage_name: str | None = None) -> dict[str, Any]:
    payload = read_json(source_summary)
    stages = payload.get("chain_stages") or []
    if not stages:
        raise RuntimeError(f"No chain_stages found in {source_summary}")
    if stage_name:
        for stage in stages:
            if str(stage.get("stage_name")) == stage_name:
                return dict(stage)
        raise RuntimeError(f"No chain stage named {stage_name!r} found in {source_summary}")
    return dict(stages[-1])


def restore_checkpoint(stage: dict[str, Any]) -> Path:
    local = ROOT / str(stage["checkpoint"])
    if local.exists():
        print(f"checkpoint already local: {path_for_cli(local)}", flush=True)
        return local
    backup_raw = stage.get("checkpoint_drive_backup")
    if not backup_raw:
        raise FileNotFoundError(f"No checkpoint_drive_backup in chain stage: {stage}")
    drive.mount("/content/drive", force_remount=False)
    backup = Path(str(backup_raw))
    if not backup.exists():
        raise FileNotFoundError(f"Missing checkpoint backup: {backup}")
    local.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(backup, local)
    print(f"restored checkpoint: {path_for_cli(local)}", flush=True)
    return local


def publish(paths: list[Path], *, message: str) -> None:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"], check=False)
    for path in paths:
        if path.exists():
            run(["git", "add", "-f", path_for_cli(path)], check=False)
    diff = run(["git", "diff", "--cached", "--quiet"], check=False)
    if diff.returncode == 0:
        print("No gradient-path audit changes to publish.", flush=True)
        return
    run(["git", "commit", "-m", message])
    push = run(["git", "push", "origin", "main"], check=False)
    if push.returncode == 0:
        return
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "push", "origin", "main"])


def disconnect(reason: str) -> None:
    if env_flag("STAGE5_GRADIENT_PATH_AUDIT_DISCONNECT", "0"):
        print(f"Disconnecting Colab runtime to conserve credits: {reason}", flush=True)
        runtime.unassign()
    else:
        print(f"Leaving Colab runtime connected: {reason}", flush=True)


try:
    require_gpu_runtime()
    sync_repo()
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_eval_gradient_path_audit.py",
            "tests/test_stage5_notebooks.py::test_gradient_path_audit_target_is_wired_and_guarded",
        ]
    )

    source_summary = Path(
        os.environ.get(
            "STAGE5_GRADIENT_PATH_AUDIT_SOURCE_SUMMARY",
            "outputs/stage5/stage5_synthetic_depth_chain_supervision_20260701_201715/summary.json",
        )
    )
    stage = latest_chain_stage(
        source_summary,
        stage_name=os.environ.get("STAGE5_GRADIENT_PATH_AUDIT_STAGE_NAME", "").strip() or None,
    )
    checkpoint = restore_checkpoint(stage)
    train_jsonl = ROOT / str(stage["train_jsonl"])
    train_config = ROOT / str(stage["train_config"])
    if not train_jsonl.exists():
        raise FileNotFoundError(f"Missing chain audit train_jsonl: {train_jsonl}")
    if not train_config.exists():
        raise FileNotFoundError(f"Missing chain audit train_config: {train_config}")

    run_id = os.environ.get("STAGE5_GRADIENT_PATH_AUDIT_RUN_ID") or time.strftime(
        "stage5_gradient_path_audit_%Y%m%d_%H%M%S"
    )
    out_dir = ROOT / "outputs" / "stage5" / run_id
    max_loops_raw = os.environ.get("STAGE5_GRADIENT_PATH_AUDIT_MAX_LOOPS", "auto").strip().lower()
    max_loops = str(stage.get("max_loops", 2)) if max_loops_raw in {"", "auto"} else max_loops_raw
    min_active_raw = os.environ.get("STAGE5_GRADIENT_PATH_AUDIT_MIN_ACTIVE_LOOP_LABELS", "auto").strip().lower()
    min_active = "auto" if min_active_raw in {"", "auto"} else min_active_raw
    num_rows = os.environ.get("STAGE5_GRADIENT_PATH_AUDIT_NUM_ROWS", "48")
    depths = os.environ.get("STAGE5_GRADIENT_PATH_AUDIT_DEPTHS", "1,2,3,4")
    cross_loop_fd = os.environ.get("STAGE5_GRADIENT_PATH_AUDIT_CROSS_LOOP_FD", "2:4")
    match_train_precision = os.environ.get("STAGE5_GRADIENT_PATH_AUDIT_MATCH_TRAIN_PRECISION", "1").strip().lower()
    print("gradient_path_audit_source_summary:", source_summary, flush=True)
    print("gradient_path_audit_checkpoint:", path_for_cli(checkpoint), flush=True)
    print("gradient_path_audit_train_jsonl:", path_for_cli(train_jsonl), flush=True)
    print("gradient_path_audit_run_id:", run_id, flush=True)

    run(
        [
            sys.executable,
            "eval/eval_gradient_path_audit.py",
            "--train_jsonl",
            path_for_cli(train_jsonl),
            "--checkpoint",
            path_for_cli(checkpoint),
            "--config",
            path_for_cli(train_config),
            "--output_dir",
            path_for_cli(out_dir),
            "--max_loops",
            max_loops,
            "--min_active_loop_labels",
            min_active,
            "--num_rows",
            num_rows,
            "--depths",
            depths,
            "--fd_rows",
            os.environ.get("STAGE5_GRADIENT_PATH_AUDIT_FD_ROWS", "8"),
            "--cross_loop_fd",
            cross_loop_fd,
            "--cross_loop_fd_rows",
            os.environ.get("STAGE5_GRADIENT_PATH_AUDIT_CROSS_LOOP_FD_ROWS", "8"),
            "--manual_loss_scale",
            os.environ.get("STAGE5_GRADIENT_PATH_AUDIT_MANUAL_LOSS_SCALE", "1.0"),
            "--max_length",
            os.environ.get("STAGE5_GRADIENT_PATH_AUDIT_MAX_LENGTH", "512"),
            "--fd_epsilon",
            os.environ.get("STAGE5_GRADIENT_PATH_AUDIT_FD_EPSILON", "0.01"),
            "--dtype",
            os.environ.get("STAGE5_GRADIENT_PATH_AUDIT_DTYPE", "bfloat16"),
            "--adapter_dtype",
            os.environ.get("STAGE5_GRADIENT_PATH_AUDIT_ADAPTER_DTYPE", "float32"),
            "--device",
            os.environ.get("STAGE5_GRADIENT_PATH_AUDIT_DEVICE", "cuda"),
        ]
        + (["--match_train_precision"] if match_train_precision in {"1", "true", "yes", "y", "on"} else [])
    )
    publish(
        [out_dir],
        message=f"Record Stage 5 gradient-path audit {run_id} [skip ci]",
    )
    disconnect("gradient-path audit finished")
except Exception:
    print("Gradient-path audit errored; leaving runtime connected.", flush=True)
    raise
