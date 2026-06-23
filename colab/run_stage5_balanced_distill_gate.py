"""Run a competence-preserving Phase 1 distillation gate from the balanced checkpoint.

The balanced MCQ assessment currently selects the deterministic recurrent
checkpoint that best trades ARC-Easy preservation against ARC-Challenge lift.
This runner resumes from that checkpoint and runs one or more short Phase 1
continuations intended to reduce the remaining base-model competence tax before
returning to particle/SVGD work.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.run_stage5_recovered_phase1_arc_gate import (  # noqa: E402
    path_for_cli,
    restore_checkpoint_if_needed,
)


RUN_ID = os.environ.get("STAGE5_BALANCED_DISTILL_RUN_ID") or time.strftime(
    "stage5_balanced_distill_gate_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID

SOURCE_SUMMARY = Path(
    os.environ.get(
        "STAGE5_BALANCED_DISTILL_SOURCE_SUMMARY",
        "outputs/stage5/stage5_balanced_mcq_current/summary.json",
    )
)
if not SOURCE_SUMMARY.is_absolute():
    SOURCE_SUMMARY = ROOT / SOURCE_SUMMARY

ARM_NAMES = [
    item.strip()
    for item in os.environ.get("STAGE5_BALANCED_DISTILL_ARMS", "response_w005_lr3e6").split(",")
    if item.strip()
]
OPUS_LIMIT = os.environ.get("STAGE5_BALANCED_DISTILL_OPUS_LIMIT", "6000")
ARC_LIMIT = os.environ.get("STAGE5_BALANCED_DISTILL_ARC_LIMIT", "128")
PUSH_RESULTS = os.environ.get("STAGE5_BALANCED_DISTILL_PUSH", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}


@dataclass(frozen=True)
class ArmConfig:
    name: str
    learning_rate: str
    beta: str
    steps: str
    save_every: str
    distill_enabled: str
    distill_weight: str
    distill_temperature: str
    distill_on: str


ARM_PRESETS: dict[str, ArmConfig] = {
    "response_w005_lr3e6": ArmConfig(
        name="response_w005_lr3e6",
        learning_rate="3e-6",
        beta="0.12",
        steps="100",
        save_every="50",
        distill_enabled="1",
        distill_weight="0.05",
        distill_temperature="2.0",
        distill_on="response",
    ),
    "response_w010_lr3e6": ArmConfig(
        name="response_w010_lr3e6",
        learning_rate="3e-6",
        beta="0.12",
        steps="100",
        save_every="50",
        distill_enabled="1",
        distill_weight="0.10",
        distill_temperature="2.0",
        distill_on="response",
    ),
    "nodistill_lr3e6": ArmConfig(
        name="nodistill_lr3e6",
        learning_rate="3e-6",
        beta="0.12",
        steps="100",
        save_every="50",
        distill_enabled="0",
        distill_weight="0.0",
        distill_temperature="2.0",
        distill_on="response",
    ),
}


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checkpoint_run_id(checkpoint: str | Path) -> str:
    path = Path(checkpoint)
    parts = list(path.parts)
    for idx, part in enumerate(parts):
        if part == "stage5" and idx + 1 < len(parts):
            return parts[idx + 1]
    raise ValueError(f"Could not infer Stage 5 run id from checkpoint path: {checkpoint}")


def selected_checkpoint(source_payload: dict[str, Any]) -> Path:
    best = source_payload.get("best_checkpoint") or {}
    checkpoint = best.get("checkpoint")
    if not checkpoint:
        raise ValueError("Balanced MCQ summary does not contain best_checkpoint.checkpoint")
    return resolve_path(str(checkpoint))


def arm_config(name: str) -> ArmConfig:
    try:
        return ARM_PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown balanced distill arm {name!r}; choose one of {sorted(ARM_PRESETS)}") from exc


def run(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    check: bool = True,
    log_name: str | None = None,
) -> subprocess.CompletedProcess[str]:
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
    stdout = "".join(chunks)
    proc = subprocess.CompletedProcess(cmd, process.wait(), stdout, None)
    if log_name:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        (RUN_DIR / log_name).write_text(stdout, encoding="utf-8")
    if check and proc.returncode:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}")
    return proc


def child_env(config: ArmConfig, *, child_run_id: str, resume_from: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "STAGE5_RUN_ID": child_run_id,
            "STAGE5_RESUME_FROM": path_for_cli(resume_from),
            "STAGE5_PHASE1_EXTRA_STEPS": config.steps,
            "STAGE5_PHASE1_SAVE_EVERY": config.save_every,
            "STAGE5_PHASE1_LR": config.learning_rate,
            "STAGE5_PHASE1_BETA": config.beta,
            "STAGE5_PHASE1_DISTILL": config.distill_enabled,
            "STAGE5_PHASE1_DISTILL_WEIGHT": config.distill_weight,
            "STAGE5_PHASE1_DISTILL_TEMPERATURE": config.distill_temperature,
            "STAGE5_PHASE1_DISTILL_ON": config.distill_on,
            "STAGE5_OPUS_LIMIT": OPUS_LIMIT,
            "STAGE5_ARC_LIMIT": ARC_LIMIT,
            "STAGE5_RUN_FULL_ARC_FINAL": "0",
            "STAGE5_PUSH": "0",
        }
    )
    return env


def mean_arc(summary: dict[str, Any]) -> dict[str, Any]:
    mean = (summary.get("mean") or {}) if isinstance(summary, dict) else {}
    return {
        "correct": int(mean.get("correct", 0) or 0),
        "total": int(mean.get("total", 0) or 0),
        "accuracy": float(mean.get("accuracy", 0.0) or 0.0),
    }


def compact_child_summary(child_summary: Path, *, arm: ArmConfig) -> dict[str, Any]:
    payload = read_json(child_summary)
    start_arc = mean_arc((payload.get("phase1_start") or {}).get("arc") or {})
    base_arc = mean_arc(payload.get("base_arc") or {})
    best = payload.get("best_checkpoint") or {}
    best_arc = mean_arc(best.get("arc") or {})
    return {
        "arm": arm.name,
        "child_run_id": child_summary.parent.name,
        "summary": path_for_cli(child_summary),
        "checkpoint": best.get("checkpoint"),
        "start_arc": start_arc,
        "base_arc": base_arc,
        "best_arc": best_arc,
        "lift_vs_start": best_arc["correct"] - start_arc["correct"],
        "gap_vs_base": best_arc["correct"] - base_arc["correct"],
        "distillation": {
            "enabled": arm.distill_enabled == "1",
            "weight": float(arm.distill_weight),
            "temperature": float(arm.distill_temperature),
            "on": arm.distill_on,
        },
        "learning_rate": float(arm.learning_rate),
        "beta": float(arm.beta),
        "steps": int(arm.steps),
    }


def build_gate_summary(
    *,
    source_summary: Path,
    source_payload: dict[str, Any],
    resume_checkpoint: Path,
    arm_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    ranked = sorted(
        arm_summaries,
        key=lambda row: (
            int(row["lift_vs_start"]),
            int(row["gap_vs_base"]),
            float(row["best_arc"]["accuracy"]),
            str(row["arm"]),
        ),
        reverse=True,
    )
    best = ranked[0] if ranked else None
    if best is None:
        status = "no_arms"
        next_step = "Run at least one balanced distillation arm."
    elif int(best["lift_vs_start"]) > 0:
        status = "proxy_lift"
        next_step = "Run full ARC-Easy and ARC-Challenge balanced benchmark on the best child checkpoint."
    elif int(best["gap_vs_base"]) >= 0:
        status = "proxy_matches_base"
        next_step = "Run full balanced benchmark on the best child checkpoint; proxy no longer trails base."
    else:
        status = "no_proxy_lift"
        next_step = (
            "Do not extend this distillation setting. Try a mixed ARC-train supervision gate or revisit "
            "training data before particle/SVGD work."
        )

    return {
        "run_id": RUN_ID,
        "kind": "stage5_balanced_distill_gate",
        "source_summary": path_for_cli(source_summary),
        "source_status": source_payload.get("status"),
        "resume_checkpoint": path_for_cli(resume_checkpoint),
        "resume_run_id": checkpoint_run_id(resume_checkpoint),
        "arc_limit": ARC_LIMIT,
        "opus_limit": OPUS_LIMIT,
        "status": status,
        "passed": status in {"proxy_lift", "proxy_matches_base"},
        "next_step": next_step,
        "best_arm": best,
        "arms": ranked,
    }


def write_report(payload: dict[str, Any]) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"# Stage 5 Balanced Distillation Gate - {RUN_ID}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Passed: `{payload['passed']}`",
        f"- Source summary: `{payload['source_summary']}`",
        f"- Resume checkpoint: `{payload['resume_checkpoint']}`",
        f"- ARC proxy limit: `{payload['arc_limit']}`",
        f"- Next step: {payload['next_step']}",
        "",
        "## Arms",
        "",
        "| arm | best proxy | start proxy | base proxy | lift vs start | gap vs base | checkpoint |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["arms"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['arm']}`",
                    f"{row['best_arc']['correct']}/{row['best_arc']['total']}",
                    f"{row['start_arc']['correct']}/{row['start_arc']['total']}",
                    f"{row['base_arc']['correct']}/{row['base_arc']['total']}",
                    str(row["lift_vs_start"]),
                    str(row["gap_vs_base"]),
                    f"`{row['checkpoint']}`",
                ]
            )
            + " |"
        )
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))


def commit_results(child_run_ids: list[str]) -> None:
    if not PUSH_RESULTS:
        return
    run(["git", "add", "-f", path_for_cli(RUN_DIR)], check=False)
    for run_id in child_run_ids:
        run(["git", "add", "-f", f"outputs/stage5/{run_id}"], check=False)
    if run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("No balanced distillation gate outputs changed.")
        return
    run(["git", "commit", "-m", f"Record balanced Stage 5 distillation gate {RUN_ID} [skip ci]"])
    run(["git", "push", "origin", "main"], check=False)


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    source_payload = read_json(SOURCE_SUMMARY)
    resume_checkpoint = selected_checkpoint(source_payload)
    restore_checkpoint_if_needed(resume_checkpoint, run_id=checkpoint_run_id(resume_checkpoint))

    arm_summaries: list[dict[str, Any]] = []
    child_run_ids: list[str] = []
    for name in ARM_NAMES:
        config = arm_config(name)
        child_run_id = f"{RUN_ID}_{config.name}"
        child_run_ids.append(child_run_id)
        env = child_env(config, child_run_id=child_run_id, resume_from=resume_checkpoint)
        run(
            [sys.executable, "colab/run_stage5_phase1_recovery_ladder.py"],
            env=env,
            log_name=f"{config.name}.log",
        )
        arm_summaries.append(
            compact_child_summary(
                ROOT / "outputs" / "stage5" / child_run_id / "summary.json",
                arm=config,
            )
        )

    payload = build_gate_summary(
        source_summary=SOURCE_SUMMARY,
        source_payload=source_payload,
        resume_checkpoint=resume_checkpoint,
        arm_summaries=arm_summaries,
    )
    write_report(payload)
    commit_results(child_run_ids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
