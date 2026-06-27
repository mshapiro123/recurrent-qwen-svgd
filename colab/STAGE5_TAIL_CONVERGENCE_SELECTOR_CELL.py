"""Colab launcher: tail-convergence selector diagnostic.

This target is GPU-bounded and read-only. It restores the checkpoint used by
the existing forced-depth selector-transfer sweeps, computes cross-loop tail
movement features, and tests whether those features transfer as rescue/harm
selectors.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from google.colab import runtime, userdata


STAGE5_TAIL_CONVERGENCE_SELECTOR_CELL_VERSION = "tail_convergence_selector_v1"
# Bootstrap safety markers: tail_deceleration_12_minus_23, stage5_current_tail_convergence_selector_summary.
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


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


def mask(text: str, token: str | None) -> str:
    return text.replace(token, "****") if token else text


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, check: bool = True):
    print("$", mask(" ".join(map(str, cmd)), GH_TOKEN), flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = mask(proc.stdout or "", GH_TOKEN)
    if output:
        print(output, flush=True)
    if check and proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout)
    return proc


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def resolve_path(value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pointer(path: Path) -> Path | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return resolve_path(value) if value else None


def latest_selector_summary() -> Path:
    override = os.environ.get("STAGE5_TAIL_CONVERGENCE_SELECTOR_SOURCE_SUMMARY", "").strip()
    if override:
        return resolve_path(override)
    ptr = pointer(ROOT / "config" / "stage5_current_rescue_selector_transfer_summary.txt")
    if ptr and ptr.exists():
        return ptr
    candidates = []
    for path in (ROOT / "outputs" / "stage5").glob("stage5_rescue_selector_transfer_*/summary.json"):
        try:
            if read_json(path).get("kind") == "stage5_rescue_selector_transfer":
                candidates.append(path)
        except Exception:
            continue
    if not candidates:
        raise FileNotFoundError("No rescue-selector transfer summary available.")
    return sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)[0]


def first_recurrent_jsonl(sweep_summary: Path, benchmark: str, score_target: str) -> Path:
    sweep = read_json(sweep_summary)
    run_id = str(sweep["loop_run_ids"][0])
    path = ROOT / "outputs" / "stage5" / run_id / f"{benchmark}_recurrent_{score_target}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def infer_checkpoint(selector_summary: Path) -> Path:
    payload = read_json(selector_summary)
    discovery = resolve_path(payload["discovery_sweep_summary"])
    benchmark = str(payload.get("discovery_benchmark") or "arc_challenge")
    score_target = str(payload.get("score_target") or "content_question_only")
    for row in read_jsonl(first_recurrent_jsonl(discovery, benchmark, score_target)):
        value = row.get("checkpoint")
        if value:
            return resolve_path(str(value))
    raise ValueError(f"Could not infer checkpoint from {selector_summary}")


def restore_checkpoint(checkpoint: Path) -> None:
    if checkpoint.exists():
        print("checkpoint_present:", path_for_cli(checkpoint), flush=True)
        return
    from colab.run_stage5_benchmark_suite import drive_diagnostics, restore_checkpoint_from_drive

    restored = restore_checkpoint_from_drive(checkpoint)
    if restored and restored.exists():
        return
    raise FileNotFoundError(f"Missing checkpoint {path_for_cli(checkpoint)}\n{drive_diagnostics()}")


def publish(out_dir: Path, pointer_path: Path) -> None:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"], check=False)
    run(["git", "add", "-f", path_for_cli(out_dir)])
    run(["git", "add", "-f", path_for_cli(pointer_path)])
    diff = run(["git", "diff", "--cached", "--quiet"], check=False)
    if diff.returncode == 0:
        print("No tail-convergence selector outputs changed.", flush=True)
        return
    run(["git", "commit", "-m", f"Record Stage 5 tail convergence selector {out_dir.name} [skip ci]"])
    pushed = run(["git", "push", "origin", "main"], check=False)
    if pushed.returncode == 0:
        return
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "push", "origin", "main"])


GH_TOKEN = secret("GH_TOKEN", "GITHUB_TOKEN")
assert GH_TOKEN, "Missing GH_TOKEN/GITHUB_TOKEN in Colab secrets."

HF_TOKEN = secret("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

try:
    print(
        f"STAGE5_TAIL_CONVERGENCE_SELECTOR_CELL_VERSION={STAGE5_TAIL_CONVERGENCE_SELECTOR_CELL_VERSION}",
        flush=True,
    )
    if shutil.which("nvidia-smi"):
        run(["nvidia-smi"], cwd=Path("/content"), check=False)
    else:
        print("No GPU detected; this hidden-state probe should run on an L4/T4 or better.", flush=True)

    authed = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
    if ROOT.exists():
        run(["git", "remote", "set-url", "origin", authed])
        run(["git", "fetch", "origin", "main"])
        run(["git", "checkout", "main"])
        run(["git", "reset", "--hard", "origin/main"])
    else:
        run(["git", "clone", authed, str(ROOT)], cwd=Path("/content"))
        run(["git", "remote", "set-url", "origin", authed])

    run(["git", "config", "user.email", "colab-runner@local"])
    run(["git", "config", "user.name", "Colab Runner"])
    run(["git", "log", "--oneline", "-5"])
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_tail_convergence_selector.py",
            "tests/test_stage5_notebooks.py::test_current_bootstrap_exposes_tail_convergence_selector_target",
        ]
    )

    selector_summary = latest_selector_summary()
    selector_payload = read_json(selector_summary)
    checkpoint = infer_checkpoint(selector_summary)
    restore_checkpoint(checkpoint)

    run_id = os.environ.get(
        "STAGE5_TAIL_CONVERGENCE_SELECTOR_RUN_ID",
        "stage5_tail_convergence_selector_" + time.strftime("%Y%m%d_%H%M%S"),
    )
    out_dir = ROOT / "outputs" / "stage5" / run_id
    discovery = resolve_path(selector_payload["discovery_sweep_summary"])
    heldout = resolve_path(selector_payload["heldout_sweep_summary"])
    print("tail_convergence_selector_source:", path_for_cli(selector_summary), flush=True)
    print("tail_convergence_discovery_sweep:", path_for_cli(discovery), flush=True)
    print("tail_convergence_heldout_sweep:", path_for_cli(heldout), flush=True)
    print("tail_convergence_checkpoint:", path_for_cli(checkpoint), flush=True)

    run(
        [
            sys.executable,
            "eval/evaluate_tail_convergence_selector.py",
            "--discovery_sweep_summary",
            path_for_cli(discovery),
            "--heldout_sweep_summary",
            path_for_cli(heldout),
            "--checkpoint",
            path_for_cli(checkpoint),
            "--output_dir",
            path_for_cli(out_dir),
            "--run_id",
            run_id,
            "--score_target",
            os.environ.get(
                "STAGE5_TAIL_CONVERGENCE_SELECTOR_SCORE_TARGET",
                str(selector_payload.get("score_target") or "content_question_only"),
            ),
            "--aggregate",
            os.environ.get(
                "STAGE5_TAIL_CONVERGENCE_SELECTOR_AGGREGATE",
                str(selector_payload.get("aggregate") or "mean"),
            ),
            "--n_tail",
            os.environ.get("STAGE5_TAIL_CONVERGENCE_N_TAIL", "7"),
            "--drop_top",
            os.environ.get("STAGE5_TAIL_CONVERGENCE_DROP_TOP", "1"),
            "--max_examples_per_benchmark",
            os.environ.get("STAGE5_TAIL_CONVERGENCE_MAX_EXAMPLES_PER_BENCHMARK", "0"),
            "--dtype",
            os.environ.get("DTYPE", "bfloat16"),
            "--adapter_dtype",
            os.environ.get("ADAPTER_DTYPE", "float32"),
            "--device",
            os.environ.get("DEVICE", "cuda"),
        ]
    )
    pointer_path = ROOT / "config" / "stage5_current_tail_convergence_selector_summary.txt"
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path.write_text(path_for_cli(out_dir / "summary.json") + "\n", encoding="utf-8")
    publish(out_dir, pointer_path)
finally:
    if env_bool("STAGE5_TAIL_CONVERGENCE_SELECTOR_DISCONNECT", True):
        print("Disconnecting Colab runtime after tail-convergence selector diagnostic.", flush=True)
        try:
            runtime.unassign()
        except Exception as exc:
            print("runtime.unassign failed:", repr(exc), flush=True)
    else:
        print("Leaving Colab runtime attached because STAGE5_TAIL_CONVERGENCE_SELECTOR_DISCONNECT=0.", flush=True)
