"""Colab cell: run a bounded debiased Stage 5 benchmark suite.

This is a measurement-only GPU action. It compares base Qwen against the
current recurrent checkpoint with bare-label, content-question-only, and
cyclic-label-aggregated MCQ scoring. The default suite includes ARC-Easy for
easy-item preservation, ARC-Challenge for hard-tail depth, and GPQA-lite as a
small out-of-domain STEM check.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from google.colab import drive, runtime, userdata


STAGE5_DEBIASED_BENCHMARK_SUITE_CELL_VERSION = "debiased_benchmark_suite_v3_spectral_health_gate"
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


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    shown = mask(" ".join(map(str, cmd)), GH_TOKEN)
    print("$", shown, flush=True)
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
    if proc.returncode:
        print("FAILED_COMMAND_TAIL_START", flush=True)
        print("\n".join(output.splitlines()[-160:]), flush=True)
        print("FAILED_COMMAND_TAIL_END", flush=True)
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout)


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_path(value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def read_json(path: str | Path) -> dict:
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def current_source_summary() -> Path:
    override = os.environ.get("STAGE5_DEBIASED_BENCHMARK_SOURCE_SUMMARY", "").strip()
    if override:
        return resolve_path(override)
    pointer = ROOT / "config" / "stage5_current_source_summary.txt"
    value = pointer.read_text(encoding="utf-8").strip()
    assert value, "config/stage5_current_source_summary.txt is empty."
    return resolve_path(value)


def payload_has_checkpoint(payload: dict) -> bool:
    for key in ("checkpoint", "phase1_checkpoint", "final_checkpoint", "tuned_checkpoint", "resume_checkpoint"):
        if payload.get(key):
            return True
    for key_path in (
        ("metadata", "checkpoint"),
        ("metadata", "recovered_checkpoint"),
        ("compact", "final_checkpoint"),
        ("autopilot_compact", "final_checkpoint"),
        ("selected_checkpoint", "checkpoint"),
    ):
        cursor = payload
        for key in key_path:
            if not isinstance(cursor, dict):
                cursor = {}
                break
            cursor = cursor.get(key)
        if cursor:
            return True
    return False


def benchmark_source_summary(start: Path) -> Path:
    """Follow scoring-policy/debias wrappers to the checkpoint-bearing summary."""

    seen: set[Path] = set()
    path = start
    for _depth in range(8):
        path = resolve_path(path)
        if path in seen:
            raise RuntimeError(f"Cycle while resolving benchmark source summary: {path}")
        seen.add(path)
        payload = read_json(path)
        kind = payload.get("kind")
        if payload_has_checkpoint(payload):
            return path
        if kind == "stage5_mcq_scoring_policy" and payload.get("source_summary"):
            path = resolve_path(payload["source_summary"])
            continue
        if kind == "stage5_mcq_debias_pair_assessment":
            source_summaries = payload.get("source_summaries") or {}
            next_summary = source_summaries.get("arc_challenge") or source_summaries.get("arc_easy")
            if next_summary:
                path = resolve_path(next_summary)
                continue
        if kind == "stage5_mcq_debias_diagnostic":
            next_summary = payload.get("nested_source_summary") or payload.get("source_summary")
            if next_summary:
                path = resolve_path(next_summary)
                continue
        return path
    raise RuntimeError(f"Could not resolve checkpoint-bearing source summary from {start}")


def benchmark_source_override_requested() -> bool:
    return bool(os.environ.get("STAGE5_DEBIASED_BENCHMARK_SOURCE_SUMMARY", "").strip())


def positive_float(value: object) -> bool:
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def spectral_source_health_override(source_summary: Path, payload: dict, health: dict) -> bool:
    """Allow benchmarking when a stale affine-only health check mislabels spectral liveness."""

    issues = set(health.get("issues") or [])
    if issues != {"reentry_adapter_gradient_not_live_after_recovery"}:
        return False
    drift_info = payload.get("post_reentry_drift") if isinstance(payload.get("post_reentry_drift"), dict) else {}
    drift_summary = str(drift_info.get("summary_path") or "").strip()
    if not drift_summary:
        return False
    drift_path = resolve_path(drift_summary)
    if not drift_path.exists():
        sibling = source_summary.parent / Path(drift_summary).name
        drift_path = sibling if sibling.exists() else drift_path
    if not drift_path.exists():
        return False
    drift = read_json(drift_path)
    adapter = drift.get("reentry_adapter") if isinstance(drift.get("reentry_adapter"), dict) else {}
    liveness = (
        drift.get("reentry_adapter_gradient_liveness")
        if isinstance(drift.get("reentry_adapter_gradient_liveness"), dict)
        else {}
    )
    if str(adapter.get("mode") or liveness.get("mode") or "") != "spectral":
        return False
    if not (
        positive_float(liveness.get("spectral_u_grad_rms"))
        and positive_float(liveness.get("spectral_v_grad_rms"))
        and positive_float(liveness.get("spectral_theta_grad_abs"))
    ):
        return False
    print(
        "stage4_benchmark_source_gate=spectral_health_override "
        f"drift={path_for_cli(drift_path)} issues={sorted(issues)}",
        flush=True,
    )
    return True


def validate_stage4_benchmark_source(source_summary: Path, payload: dict, *, allow_override: bool) -> None:
    if allow_override:
        print(
            "stage4_benchmark_source_gate=override "
            f"source={path_for_cli(source_summary)} kind={payload.get('kind')}",
            flush=True,
        )
        return
    kind = payload.get("kind")
    if kind != "stage5_reentry_recovery_training":
        raise RuntimeError(
            "debiased_benchmark_suite follows the master sequence and requires "
            "config/stage5_current_source_summary.txt to resolve to a Stage 4 "
            f"re-entry recovery summary, got kind={kind!r} at {path_for_cli(source_summary)}. "
            "Run reentry_recovery_training first, or set "
            "STAGE5_DEBIASED_BENCHMARK_SOURCE_SUMMARY for an intentional older-artifact test."
        )
    health = payload.get("post_reentry_health_checks") if isinstance(payload.get("post_reentry_health_checks"), dict) else {}
    status = str(health.get("status") or "")
    if status != "reentry_health_sane":
        if spectral_source_health_override(source_summary, payload, health):
            return
        raise RuntimeError(
            "Stage 4 recovery source is not benchmark-ready: "
            f"post_reentry_health_checks.status={status!r}, issues={health.get('issues', [])!r}. "
            "Review or rerun reentry_recovery_training before benchmarking."
        )
    print("stage4_benchmark_source_gate=passed", flush=True)


def fixed_tail_damper_env(payload: dict) -> dict[str, str]:
    fixed = payload.get("fixed_tail_damper")
    if not isinstance(fixed, dict):
        return {}
    damper_path = str(fixed.get("damper_path") or "").strip()
    strength = str(fixed.get("strength") or "").strip()
    if not damper_path or not strength:
        return {}
    return {
        "STAGE5_BENCHMARK_REENTRY_TAIL_DAMPER_PATH": damper_path,
        "STAGE5_BENCHMARK_REENTRY_TAIL_DAMPER_STRENGTH": strength,
    }


GH_TOKEN = secret("GH_TOKEN", "GITHUB_TOKEN")
assert GH_TOKEN, "Missing GH_TOKEN/GITHUB_TOKEN in Colab secrets."

HF_TOKEN = secret("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

try:
    gpu_check = shutil.which("nvidia-smi")
    assert gpu_check, "Attach an A100/H100/L4/T4 GPU runtime before running this benchmark action."
    run(["nvidia-smi"], cwd=Path("/content"))

    mount_drive_first = os.environ.get("STAGE5_DEBIASED_MOUNT_DRIVE_FIRST", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    if mount_drive_first:
        drive.mount("/content/drive", force_remount=False)
    else:
        print("Skipping upfront Drive mount; benchmark runner will request Drive only if checkpoint restore is needed.", flush=True)
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
            "tests/test_stage5_benchmark_suite.py",
            "tests/test_stage5_benchmark_assessment.py",
            "tests/test_mcq_debias.py",
            "tests/test_stage5_notebooks.py::test_debiased_benchmark_suite_threads_fixed_tail_damper",
        ]
    )

    current_summary = current_source_summary()
    source_summary = benchmark_source_summary(current_summary)
    source_payload = read_json(source_summary)
    validate_stage4_benchmark_source(
        source_summary,
        source_payload,
        allow_override=benchmark_source_override_requested(),
    )
    adjacent_adapter = source_summary.parent / "recurrent_adapter_checkpoint.pt"
    assert payload_has_checkpoint(source_payload) or adjacent_adapter.exists(), (
        "Resolved benchmark source summary does not expose a checkpoint path "
        f"and has no adjacent adapter checkpoint: {path_for_cli(source_summary)}"
    )
    print("current_summary:", path_for_cli(current_summary), flush=True)
    print("benchmark_source_summary:", path_for_cli(source_summary), flush=True)

    env = os.environ.copy()
    env.setdefault(
        "STAGE5_BENCHMARK_SUITE_RUN_ID",
        "stage5_debiased_benchmark_suite_" + time.strftime("%Y%m%d_%H%M%S"),
    )
    env["STAGE5_BENCHMARK_SOURCE_SUMMARY"] = path_for_cli(source_summary)
    env["STAGE5_BENCHMARK_SUITE_PROFILE"] = os.environ.get("STAGE5_DEBIASED_BENCHMARK_SUITE_PROFILE", "default")
    env["STAGE5_BENCHMARKS"] = os.environ.get(
        "STAGE5_DEBIASED_BENCHMARKS",
        "arc_easy,arc_challenge,gpqa_lite",
    )
    env["STAGE5_BENCHMARK_ARC_CHALLENGE_LIMIT"] = os.environ.get("STAGE5_DEBIASED_ARC_CHALLENGE_LIMIT", "128")
    env["STAGE5_BENCHMARK_ARC_EASY_LIMIT"] = os.environ.get("STAGE5_DEBIASED_ARC_EASY_LIMIT", "128")
    env["STAGE5_BENCHMARK_OPEN_HARD_ARC_CHALLENGE_LIMIT"] = os.environ.get(
        "STAGE5_DEBIASED_OPEN_HARD_ARC_CHALLENGE_LIMIT",
        "256",
    )
    env["STAGE5_BENCHMARK_ARC_CHALLENGE_OFFSET"] = os.environ.get("STAGE5_DEBIASED_ARC_CHALLENGE_OFFSET", "0")
    env["STAGE5_BENCHMARK_ARC_EASY_OFFSET"] = os.environ.get("STAGE5_DEBIASED_ARC_EASY_OFFSET", "0")
    env["STAGE5_BENCHMARK_OPEN_HARD_ARC_CHALLENGE_OFFSET"] = os.environ.get(
        "STAGE5_DEBIASED_OPEN_HARD_ARC_CHALLENGE_OFFSET",
        "0",
    )
    env["STAGE5_BENCHMARK_GPQA_LIMIT"] = os.environ.get("STAGE5_DEBIASED_GPQA_LIMIT", "16")
    env["STAGE5_BENCHMARK_SCORE_TARGETS"] = os.environ.get(
        "STAGE5_DEBIASED_SCORE_TARGETS",
        "label,content_question_only,cyclic_label_aggregated",
    )
    env["STAGE5_BENCHMARK_AGGREGATES"] = "mean"
    env["STAGE5_BENCHMARK_USE_LEARNED_LOOP_CONTROL"] = os.environ.get(
        "STAGE5_DEBIASED_USE_LEARNED_LOOP_CONTROL",
        "1",
    )
    if env_bool("STAGE5_DEBIASED_USE_FIXED_TAIL_DAMPER", True):
        for key, value in fixed_tail_damper_env(source_payload).items():
            if not env.get(key):
                env[key] = os.environ.get(key) or value
        if env.get("STAGE5_BENCHMARK_REENTRY_TAIL_DAMPER_PATH"):
            print(
                "benchmark_fixed_tail_damper="
                f"{env['STAGE5_BENCHMARK_REENTRY_TAIL_DAMPER_PATH']} "
                f"strength={env.get('STAGE5_BENCHMARK_REENTRY_TAIL_DAMPER_STRENGTH', '')}",
                flush=True,
            )
    env["STAGE5_BENCHMARK_PUSH"] = "1"
    env["MODEL_NAME"] = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
    env["DTYPE"] = os.environ.get("DTYPE", "bfloat16")
    env["ADAPTER_DTYPE"] = os.environ.get("ADAPTER_DTYPE", "float32")
    env["DEVICE"] = os.environ.get("DEVICE", "cuda")
    run([sys.executable, "colab/run_stage5_benchmark_suite.py"], env=env)

    benchmark_summary = (ROOT / "config" / "stage5_current_source_summary.txt").read_text(encoding="utf-8").strip()
    print("benchmark_summary:", benchmark_summary, flush=True)

    assess_env = os.environ.copy()
    assess_env.setdefault(
        "STAGE5_BENCHMARK_ASSESS_RUN_ID",
        "stage5_debiased_benchmark_assessment_" + time.strftime("%Y%m%d_%H%M%S"),
    )
    assess_env["STAGE5_BENCHMARK_ASSESS_SCORE_TARGET"] = "cyclic_label_aggregated"
    assess_env["STAGE5_BENCHMARK_ASSESS_AGGREGATE"] = "permutation_mean"
    assess_env["STAGE5_BENCHMARK_ASSESS_MIN_ARC_EXAMPLES"] = env["STAGE5_BENCHMARK_ARC_CHALLENGE_LIMIT"]
    assess_env["STAGE5_BENCHMARK_ASSESS_MIN_GPQA_EXAMPLES"] = env["STAGE5_BENCHMARK_GPQA_LIMIT"]
    assess_env["STAGE5_BENCHMARK_ASSESS_REQUIRED_BENCHMARKS"] = os.environ.get(
        "STAGE5_DEBIASED_ASSESS_REQUIRED_BENCHMARKS",
        "arc_challenge,gpqa_lite",
    )
    assess_env["STAGE5_BENCHMARK_ASSESS_PUSH"] = "1"
    run([sys.executable, "colab/assess_stage5_benchmark_suite.py", "--summary_json", benchmark_summary], env=assess_env)

    assessment_summary = (ROOT / "config" / "stage5_current_source_summary.txt").read_text(encoding="utf-8").strip()
    print("assessment_summary:", assessment_summary, flush=True)
    assessment_md = ROOT / assessment_summary.replace("summary.json", "summary.md")
    if assessment_md.exists():
        print(assessment_md.read_text(encoding="utf-8"), flush=True)

finally:
    if env_bool("STAGE5_DEBIASED_BENCHMARK_DISCONNECT", True):
        print("Disconnecting Colab runtime to conserve credits.", flush=True)
        try:
            runtime.unassign()
        except Exception as exc:
            print("runtime.unassign failed:", repr(exc), flush=True)
    else:
        print("Leaving Colab runtime attached because STAGE5_DEBIASED_BENCHMARK_DISCONNECT=0.", flush=True)
