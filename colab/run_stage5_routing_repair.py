"""Run the bounded Phase 1 repair selected by a routing diagnostic.

The routing diagnostic decides whether the current deterministic recurrent
checkpoint is mostly failing direct/base-confident items or deep/numeric items.
This wrapper translates that diagnosis into one bounded Phase 1 ARC-mix repair
profile and delegates to the existing, tested ARC-mix trainer.

It intentionally keeps particles/SVGD off. This is Phase A from the depth/width
curriculum: repair direct calibration and deterministic depth before spending on
width.
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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RUN_ID = os.environ.get("STAGE5_ROUTING_REPAIR_RUN_ID") or time.strftime(
    "stage5_routing_repair_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_SUMMARY = Path(os.environ.get("STAGE5_ROUTING_REPAIR_SOURCE_SUMMARY", ""))
if str(SOURCE_SUMMARY):
    SOURCE_SUMMARY = SOURCE_SUMMARY if SOURCE_SUMMARY.is_absolute() else ROOT / SOURCE_SUMMARY

PUSH_RESULTS = os.environ.get("STAGE5_ROUTING_REPAIR_PUSH", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}

SAFE_OUTPUT_SUFFIXES = {".csv", ".html", ".json", ".jsonl", ".log", ".md", ".txt", ".yaml", ".yml"}
MAX_COMMIT_ARTIFACT_BYTES = int(os.environ.get("STAGE5_ROUTING_REPAIR_COMMIT_MAX_ARTIFACT_BYTES", "25000000"))
CHILD_STDOUT_TAIL_CHARS = int(os.environ.get("STAGE5_ROUTING_REPAIR_CHILD_STDOUT_TAIL_CHARS", "12000"))


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def tail_text(value: str, *, limit: int = CHILD_STDOUT_TAIL_CHARS) -> str:
    if len(value) <= limit:
        return value
    return f"...[truncated {len(value) - limit} chars]\n{value[-limit:]}"


def resolve_source_summary() -> Path:
    if str(SOURCE_SUMMARY).strip() and SOURCE_SUMMARY.is_file():
        return SOURCE_SUMMARY
    candidates = sorted(
        (ROOT / "outputs" / "stage5").glob("stage5*_routing_diagnostic*/summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        "Missing routing diagnostic summary. Set STAGE5_ROUTING_REPAIR_SOURCE_SUMMARY="
        "outputs/stage5/<routing_run>/summary.json"
    )


def run(cmd: list[str], *, env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(map(str, cmd)), flush=True)
    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    chunks: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        chunks.append(line)
    proc = subprocess.CompletedProcess(cmd, process.wait(), "".join(chunks), None)
    if check and proc.returncode:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}")
    return proc


def benchmark_summary_from_assessment(payload: dict[str, Any], source_summary: Path) -> Path:
    value = payload.get("benchmark_summary")
    if value:
        path = Path(str(value))
        path = path if path.is_absolute() else ROOT / path
        if path.exists():
            return path
    fallback = source_summary.parent / "benchmark_run" / "summary.json"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No benchmark summary found for routing assessment {source_summary}")


def repair_profile(status: str) -> dict[str, str]:
    if status == "needs_direct_halting_repair":
        return {
            "repair_mode": "direct_halting",
            "STAGE5_ARC_MIX_ARMS": "arc_mix_response_w02_lr2e6",
            "STAGE5_ARC_MIX_ARC_EASY_REPEAT": "8",
            "STAGE5_ARC_MIX_ARC_CHALLENGE_REPEAT": "1",
            "STAGE5_ARC_MIX_ARC_EASY_TARGET_LOOP": "1",
            "STAGE5_ARC_MIX_ARC_CHALLENGE_TARGET_LOOP": "2",
            "STAGE5_ARC_MIX_ARC_EASY_ROUTING_TYPE": "direct",
            "STAGE5_ARC_MIX_ARC_CHALLENGE_ROUTING_TYPE": "deep_narrow_probe",
            "STAGE5_ARC_MIX_EVAL_CONFIG": "ARC-Easy",
            "STAGE5_ARC_MIX_OPUS_LIMIT": "1500",
            "STAGE5_ARC_MIX_ARC_EVAL_LIMIT": "128",
            "STAGE5_ARC_MIX_MIN_MARGIN_DELTA": "0.0",
            "STAGE5_ARC_MIX_MAX_PREDICTION_SHIFT": "8",
        }
    if status == "needs_deep_narrow_recovery":
        return {
            "repair_mode": "deep_narrow",
            "STAGE5_ARC_MIX_ARMS": "arc_mix_response_w005_lr2e6,arc_mix_response_w01_lr2e6",
            "STAGE5_ARC_MIX_ARC_EASY_REPEAT": "2",
            "STAGE5_ARC_MIX_ARC_CHALLENGE_REPEAT": "5",
            "STAGE5_ARC_MIX_ARC_EASY_TARGET_LOOP": "1",
            "STAGE5_ARC_MIX_ARC_CHALLENGE_TARGET_LOOP": "3",
            "STAGE5_ARC_MIX_ARC_EASY_ROUTING_TYPE": "direct_anchor",
            "STAGE5_ARC_MIX_ARC_CHALLENGE_ROUTING_TYPE": "deep_narrow",
            "STAGE5_ARC_MIX_EVAL_CONFIG": "ARC-Challenge",
            "STAGE5_ARC_MIX_OPUS_LIMIT": "3000",
            "STAGE5_ARC_MIX_ARC_EVAL_LIMIT": "128",
            "STAGE5_ARC_MIX_MIN_MARGIN_DELTA": "-0.05",
            "STAGE5_ARC_MIX_MAX_PREDICTION_SHIFT": "16",
        }
    if status == "routing_diagnostic_pass":
        return {"repair_mode": "none"}
    raise ValueError(f"Unsupported routing diagnostic status: {status}")


def profile_arm_names(profile: dict[str, str]) -> list[str]:
    return [
        item.strip()
        for item in profile.get("STAGE5_ARC_MIX_ARMS", "").split(",")
        if item.strip()
    ]


def profile_objective_audit(profile: dict[str, str]) -> dict[str, Any]:
    """Make the delegated ARC-mix objective explicit in this wrapper's report."""
    repair_mode = profile["repair_mode"]
    if repair_mode == "none":
        return {
            "repair_mode": repair_mode,
            "requires_distillation": False,
            "distillation_ok": True,
            "arms": [],
        }

    from colab.run_stage5_balanced_arc_mix_gate import arm_config

    arms = []
    for name in profile_arm_names(profile):
        arm = arm_config(name)
        distill_enabled = arm.distill_enabled == "1"
        distill_weight = float(arm.distill_weight)
        arms.append(
            {
                "name": arm.name,
                "learning_rate": float(arm.learning_rate),
                "beta": float(arm.beta),
                "steps": int(arm.steps),
                "distillation": {
                    "enabled": distill_enabled,
                    "weight": distill_weight,
                    "temperature": float(arm.distill_temperature),
                    "on": arm.distill_on,
                },
            }
        )

    requires_distillation = repair_mode in {"direct_halting", "deep_narrow"}
    distillation_ok = (not requires_distillation) or (
        bool(arms)
        and all(
            arm["distillation"]["enabled"] and arm["distillation"]["weight"] > 0.0
            for arm in arms
        )
    )
    if requires_distillation and not distillation_ok:
        raise ValueError(
            f"{repair_mode} profile must use only response-distillation ARC-mix arms; "
            f"got {profile.get('STAGE5_ARC_MIX_ARMS')!r}"
        )
    return {
        "repair_mode": repair_mode,
        "requires_distillation": requires_distillation,
        "distillation_ok": distillation_ok,
        "arms": arms,
        "rationale": (
            "The routing repair is a deterministic Phase 1 recovery run. "
            "It uses ARC label supervision plus frozen-base response KL to repair "
            "direct/deep allocation without launching particles."
        ),
    }


def build_child_env(*, source_summary: Path, profile: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    env.update({key: value for key, value in profile.items() if key.startswith("STAGE5_")})
    env["STAGE5_ARC_MIX_RUN_ID"] = os.environ.get(
        "STAGE5_ROUTING_REPAIR_ARC_MIX_RUN_ID",
        f"{RUN_ID}_{profile['repair_mode']}_arc_mix",
    )
    env["STAGE5_ARC_MIX_SOURCE_SUMMARY"] = path_for_cli(source_summary)
    env["STAGE5_ARC_MIX_INCLUDE_LOOP_DIAGNOSTICS"] = "1"
    env["STAGE5_ARC_MIX_PUSH"] = "0"
    return env


def child_summary_path(env: dict[str, str]) -> Path:
    return ROOT / "outputs" / "stage5" / env["STAGE5_ARC_MIX_RUN_ID"] / "summary.json"


def child_best_checkpoint(child_payload: dict[str, Any]) -> dict[str, Any] | None:
    best_arm = child_payload.get("best_arm")
    if isinstance(best_arm, dict):
        best_checkpoint = best_arm.get("best_checkpoint")
        if isinstance(best_checkpoint, dict):
            return best_checkpoint
    best_checkpoint = child_payload.get("best_checkpoint")
    return best_checkpoint if isinstance(best_checkpoint, dict) else None


def child_passed(child_payload: dict[str, Any]) -> bool:
    status = str(child_payload.get("status", ""))
    return bool(child_payload.get("passed")) or status in {"proxy_lift", "proxy_matches_base"}


def child_proxy_alignment(profile: dict[str, str], child_payload: dict[str, Any]) -> dict[str, Any]:
    """Check that the delegated proxy eval matched this repair's objective."""

    expected = str(profile.get("STAGE5_ARC_MIX_EVAL_CONFIG") or "").strip()
    actual = str(child_payload.get("arc_eval_config") or (child_payload.get("data") or {}).get("arc_eval_config") or "").strip()
    if not expected:
        return {
            "checked": False,
            "ok": True,
            "expected_arc_eval_config": None,
            "actual_arc_eval_config": actual or None,
            "reason": "Repair profile did not request a specific proxy eval config.",
        }
    return {
        "checked": True,
        "ok": actual == expected,
        "expected_arc_eval_config": expected,
        "actual_arc_eval_config": actual or None,
        "reason": (
            "Child ARC-mix proxy matched the repair objective."
            if actual == expected
            else "Child ARC-mix proxy did not match the repair objective; do not advance this checkpoint."
        ),
    }


def repair_child_outcome(profile: dict[str, str], child_payload: dict[str, Any]) -> dict[str, Any]:
    proxy_alignment = child_proxy_alignment(profile, child_payload)
    child_status = str(child_payload.get("status", "unknown"))
    passed = child_passed(child_payload) and bool(proxy_alignment.get("ok", True))
    status = f"repair_{child_status}" if proxy_alignment.get("ok", True) else "repair_proxy_misaligned"
    next_step = child_payload.get("next_step") or "Review ARC-mix child summary."
    if not proxy_alignment.get("ok", True):
        next_step = (
            "Do not use this repair result for full assessment; rerun the routing repair after confirming "
            "STAGE5_ARC_MIX_EVAL_CONFIG matches the selected repair mode."
        )
    return {
        "status": status,
        "passed": passed,
        "next_step": next_step,
        "proxy_alignment": proxy_alignment,
    }


def copy_child_run(env: dict[str, str]) -> Path:
    child_dir = ROOT / "outputs" / "stage5" / env["STAGE5_ARC_MIX_RUN_ID"]
    target = RUN_DIR / "repair_run"
    if child_dir.exists():
        shutil.copytree(child_dir, target, dirs_exist_ok=True)
    return target


def is_safe_output_artifact(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() not in SAFE_OUTPUT_SUFFIXES:
        return False
    try:
        return path.stat().st_size <= MAX_COMMIT_ARTIFACT_BYTES
    except OSError:
        return False


def current_source_summary_file() -> Path:
    return ROOT / "config" / "stage5_current_source_summary.txt"


def update_current_source_summary(summary_path: Path) -> Path:
    """Advance the no-argument planner/go-no-go pointer to this repair result."""

    pointer = current_source_summary_file()
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(path_for_cli(summary_path) + "\n", encoding="utf-8")
    return pointer


def commit_results() -> None:
    if not PUSH_RESULTS:
        return
    files = sorted(path.relative_to(ROOT).as_posix() for path in RUN_DIR.rglob("*") if is_safe_output_artifact(path))
    pointer = current_source_summary_file()
    if pointer.exists():
        files.append(pointer.relative_to(ROOT).as_posix())
    files = sorted(set(files))
    if not files:
        print("No safe routing repair artifacts to commit.")
        return
    for index in range(0, len(files), 100):
        run(["git", "add", "-f", *files[index : index + 100]], check=False)
    if run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("No routing repair outputs changed.")
        return
    run(["git", "commit", "-m", f"Record Stage 5 routing repair {RUN_ID}"])
    run(["git", "push", "origin", "main"], check=False)


def write_report(payload: dict[str, Any]) -> None:
    summary_path = RUN_DIR / "summary.json"
    write_json(summary_path, payload)
    update_current_source_summary(summary_path)
    lines = [
        f"# Stage 5 Routing Repair - {RUN_ID}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Repair mode: `{payload['repair_mode']}`",
        f"- Source routing summary: `{payload['source_summary']}`",
        f"- Benchmark source summary: `{payload.get('benchmark_summary')}`",
        f"- ARC-mix child summary: `{payload.get('arc_mix_summary')}`",
        f"- Objective audit: distillation_ok=`{payload['profile_objective']['distillation_ok']}`",
        f"- Proxy alignment: `{(payload.get('proxy_alignment') or {}).get('ok', 'n/a')}`",
        f"- Next step: {payload['next_step']}",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))


def main() -> int:
    source_summary = resolve_source_summary()
    source_payload = read_json(source_summary)
    status = str(source_payload.get("status", ""))
    profile = repair_profile(status)
    repair_mode = profile["repair_mode"]
    profile_objective = profile_objective_audit(profile)
    benchmark_summary = benchmark_summary_from_assessment(source_payload, source_summary)

    if repair_mode == "none":
        payload = {
            "run_id": RUN_ID,
            "kind": "stage5_routing_repair",
            "status": "no_repair_needed",
            "repair_mode": repair_mode,
            "source_summary": path_for_cli(source_summary),
            "benchmark_summary": path_for_cli(benchmark_summary),
            "profile_objective": profile_objective,
            "next_step": "Run a larger confirmation or proceed to the bounded direct/deep recovery ladder.",
        }
        write_report(payload)
        commit_results()
        return 0

    child_env = build_child_env(source_summary=benchmark_summary, profile=profile)
    child_proc = run([sys.executable, "colab/run_stage5_balanced_arc_mix_gate.py"], env=child_env, check=False)
    child_summary = child_summary_path(child_env)
    if child_proc.returncode != 0:
        copied_child_dir = copy_child_run(child_env)
        payload = {
            "run_id": RUN_ID,
            "kind": "stage5_routing_repair",
            "status": "repair_child_failed",
            "passed": False,
            "repair_mode": repair_mode,
            "source_summary": path_for_cli(source_summary),
            "benchmark_summary": path_for_cli(benchmark_summary),
            "profile": profile,
            "profile_objective": profile_objective,
            "arc_mix_summary": path_for_cli(child_summary) if child_summary.exists() else None,
            "arc_mix_returncode": child_proc.returncode,
            "arc_mix_stdout_tail": tail_text(child_proc.stdout or ""),
            "repair_run_dir": path_for_cli(copied_child_dir),
            "next_step": (
                "The delegated ARC-mix repair command failed before producing a usable routing-repair result. "
                "Inspect arc_mix_stdout_tail, the copied repair_run directory, and any child logs before retrying."
            ),
        }
        if child_summary.exists():
            payload["arc_mix"] = read_json(child_summary)
        write_report(payload)
        commit_results()
        return child_proc.returncode or 1
    if not child_summary.exists():
        copied_child_dir = copy_child_run(child_env)
        payload = {
            "run_id": RUN_ID,
            "kind": "stage5_routing_repair",
            "status": "repair_child_missing_summary",
            "passed": False,
            "repair_mode": repair_mode,
            "source_summary": path_for_cli(source_summary),
            "benchmark_summary": path_for_cli(benchmark_summary),
            "profile": profile,
            "profile_objective": profile_objective,
            "arc_mix_summary": path_for_cli(child_summary),
            "repair_run_dir": path_for_cli(copied_child_dir),
            "next_step": (
                "The delegated ARC-mix command exited successfully but did not write its summary.json. "
                "Treat this as a failed repair and inspect the child run directory before retrying."
            ),
        }
        write_report(payload)
        commit_results()
        return 1
    child_payload = read_json(child_summary)
    copy_child_run(child_env)
    best_checkpoint = child_best_checkpoint(child_payload)
    child_outcome = repair_child_outcome(profile, child_payload)
    payload = {
        "run_id": RUN_ID,
        "kind": "stage5_routing_repair",
        "status": child_outcome["status"],
        "passed": child_outcome["passed"],
        "repair_mode": repair_mode,
        "source_summary": path_for_cli(source_summary),
        "benchmark_summary": path_for_cli(benchmark_summary),
        "profile": profile,
        "profile_objective": profile_objective,
        "proxy_alignment": child_outcome["proxy_alignment"],
        "arc_mix_summary": path_for_cli(child_summary),
        "arc_mix": child_payload,
        "best_checkpoint": best_checkpoint,
        "next_step": child_outcome["next_step"],
    }
    write_report(payload)
    commit_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
