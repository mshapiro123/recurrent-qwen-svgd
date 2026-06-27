"""Colab cell: transfer discovery depth-selector gates to held-out sweeps.

This target is CPU-only. It takes the forced-depth discovery sweep used for the
rescue-predictability precursor, applies those threshold gates to the held-out
router validation sweep, and records the rescue/harm trade curve.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from google.colab import runtime, userdata


STAGE5_RESCUE_SELECTOR_TRANSFER_CELL_VERSION = "rescue_selector_transfer_v1"
# Bootstrap safety markers: transferred_curve_summary, stage5_current_rescue_selector_transfer_summary.
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y"}


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
        return str(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def summary_pointer(path: Path) -> Path | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return resolve_path(value) if value else None


def latest_summary(pattern: str, *, kind: str) -> Path:
    candidates: list[Path] = []
    for path in (ROOT / "outputs" / "stage5").glob(pattern):
        try:
            payload = read_json(path)
        except Exception:
            continue
        if payload.get("kind") == kind:
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No {pattern} artifact with kind={kind!r} found.")
    return sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)[0]


def discovery_sweep_summary() -> Path:
    override = os.environ.get("STAGE5_RESCUE_SELECTOR_DISCOVERY_SWEEP_SUMMARY", "").strip()
    if override:
        return resolve_path(override)
    precursor = summary_pointer(ROOT / "config" / "stage5_current_rescue_predictability_summary.txt")
    if precursor and precursor.exists():
        source = read_json(precursor).get("source_sweep_summary")
        if source:
            return resolve_path(str(source))
    return latest_summary("stage5_forced_depth*/summary.json", kind="stage5_forced_depth_diagnostic")


def heldout_sweep_summary() -> Path:
    override = os.environ.get("STAGE5_RESCUE_SELECTOR_HELDOUT_SWEEP_SUMMARY", "").strip()
    if override:
        return resolve_path(override)
    return latest_summary("stage5_heldout_router_validation*/summary.json", kind="stage5_heldout_router_validation_sweep")


def publish(out_dir: Path, pointer: Path) -> None:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"], check=False)
    run(["git", "add", "-f", path_for_cli(out_dir)])
    run(["git", "add", "-f", path_for_cli(pointer)])
    diff = run(["git", "diff", "--cached", "--quiet"], check=False)
    if diff.returncode == 0:
        print("No rescue-selector transfer outputs changed.", flush=True)
        return
    run(["git", "commit", "-m", f"Record Stage 5 rescue selector transfer {out_dir.name} [skip ci]"])
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
        f"STAGE5_RESCUE_SELECTOR_TRANSFER_CELL_VERSION={STAGE5_RESCUE_SELECTOR_TRANSFER_CELL_VERSION}",
        flush=True,
    )
    if shutil.which("nvidia-smi"):
        run(["nvidia-smi"], cwd=Path("/content"), check=False)
    else:
        print("No GPU attached; rescue-selector transfer is CPU-only and will continue.", flush=True)

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
            "tests/test_evaluate_rescue_selector_transfer.py",
            "tests/test_analyze_rescue_predictability.py",
            "tests/test_stage5_notebooks.py::test_current_bootstrap_exposes_rescue_selector_transfer_target",
        ]
    )

    discovery = discovery_sweep_summary()
    heldout = heldout_sweep_summary()
    run_id = os.environ.get(
        "STAGE5_RESCUE_SELECTOR_TRANSFER_RUN_ID",
        "stage5_rescue_selector_transfer_" + time.strftime("%Y%m%d_%H%M%S"),
    )
    out_dir = ROOT / "outputs" / "stage5" / run_id
    score_target = os.environ.get("STAGE5_RESCUE_SELECTOR_SCORE_TARGET", "content_question_only")
    aggregate = os.environ.get("STAGE5_RESCUE_SELECTOR_AGGREGATE", "mean")
    manual_thresholds = os.environ.get(
        "STAGE5_RESCUE_SELECTOR_MANUAL_BASE_MARGIN_THRESHOLDS",
        "0.19658637046813965,0.3647139072418213,0.4698638916015625",
    )
    print("rescue_selector_discovery_sweep:", path_for_cli(discovery), flush=True)
    print("rescue_selector_heldout_sweep:", path_for_cli(heldout), flush=True)
    run(
        [
            sys.executable,
            "eval/evaluate_rescue_selector_transfer.py",
            "--discovery_sweep_summary",
            path_for_cli(discovery),
            "--heldout_sweep_summary",
            path_for_cli(heldout),
            "--score_target",
            score_target,
            "--aggregate",
            aggregate,
            "--manual_base_margin_thresholds",
            manual_thresholds,
            "--run_id",
            run_id,
            "--output_dir",
            path_for_cli(out_dir),
        ]
    )
    pointer = ROOT / "config" / "stage5_current_rescue_selector_transfer_summary.txt"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(path_for_cli(out_dir / "summary.json") + "\n", encoding="utf-8")
    publish(out_dir, pointer)
finally:
    if env_bool("STAGE5_RESCUE_SELECTOR_TRANSFER_DISCONNECT", True):
        print("Disconnecting Colab runtime after CPU-only rescue-selector transfer analysis.", flush=True)
        try:
            runtime.unassign()
        except Exception as exc:
            print("runtime.unassign failed:", repr(exc), flush=True)
    else:
        print("Leaving Colab runtime attached because STAGE5_RESCUE_SELECTOR_TRANSFER_DISCONNECT=0.", flush=True)
