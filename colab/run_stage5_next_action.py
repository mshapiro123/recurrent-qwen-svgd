"""Run the Stage 5 planner and optionally execute its top safe action.

This is the "keep the A100 moving" wrapper. It deliberately does not execute
arbitrary shell text. Planner commands are parsed into environment assignments
plus an allowlisted ``python colab/...py`` runner, or a read-only ``cat`` action.
Set ``STAGE5_ARC_AGI_NEXT_ACTION_EXECUTE=1`` to run the selected action.
Without that flag, the script writes a dry-run summary.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get("STAGE5_ARC_AGI_NEXT_ACTION_RUN_ID") or time.strftime(
    "stage5_arc_agi_next_action_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_SUMMARY = os.environ.get("STAGE5_ARC_AGI_NEXT_ACTION_SOURCE_SUMMARY") or os.environ.get(
    "STAGE5_ARC_AGI_NEXT_PLAN_SOURCE_SUMMARY",
    "",
)
ACTION_INDEX = int(os.environ.get("STAGE5_ARC_AGI_NEXT_ACTION_INDEX", "0"))
MAX_ACTIONS = int(os.environ.get("STAGE5_ARC_AGI_NEXT_ACTION_MAX_ACTIONS", "1"))
ALLOW_REPEAT = os.environ.get("STAGE5_ARC_AGI_NEXT_ACTION_ALLOW_REPEAT", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
EXECUTE = os.environ.get("STAGE5_ARC_AGI_NEXT_ACTION_EXECUTE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}

ENV_ASSIGNMENT_RE = re.compile(r"^[A-Z][A-Z0-9_]*=.*$")
ALLOWED_PYTHON_SCRIPTS = {
    "colab/assess_stage5_gate1.py",
    "colab/assess_stage5_gate2.py",
    "colab/assess_stage5_recipe_control.py",
    "colab/assess_stage5_recipe_selector_conversion.py",
    "colab/assess_stage5_benchmark_suite.py",
    "colab/assess_stage5_release_gate.py",
    "colab/assess_stage5_selector_replication.py",
    "colab/build_stage5_arc_agi_reproduced_baseline_registry.py",
    "colab/build_stage5_arc_agi_sota_comparison.py",
    "colab/build_stage5_claim_packet.py",
    "colab/plan_stage5_next_run.py",
    "colab/validate_arc_agi_baseline_registry.py",
    "colab/run_stage5_arc_agi_autopilot_followup.py",
    "colab/run_stage5_arc_agi_candidate_gate.py",
    "colab/run_stage5_arc_agi_candidate_distill_gate.py",
    "colab/run_stage5_arc_agi_curriculum_particle_autopilot.py",
    "colab/run_stage5_arc_agi_recovered_benchmark.py",
    "colab/run_stage5_arc_agi_recovery_particle_gate.py",
    "colab/run_stage5_arc_agi_rescore_selectors.py",
    "colab/run_stage5_arc_agi_sft.py",
    "colab/run_stage5_arc_agi_dense_sft.py",
    "colab/run_stage5_arc_agi_trace_sft_gate.py",
    "colab/run_stage5_arc_agi_tta_sweep.py",
    "colab/run_stage5_benchmark_suite.py",
    "colab/run_stage5_phase1_recovery_ladder.py",
    "colab/run_stage5_publish_hf_adapter.py",
    "colab/run_stage5_recovered_phase1_arc_gate.py",
    "colab/run_stage5_recovered_phase1_particle_arc_gate.py",
    "colab/run_stage5_recovered_phase2_smoke.py",
}


@dataclass(frozen=True)
class ParsedCommand:
    env: dict[str, str]
    argv: list[str]
    kind: str


def path_for_cli(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    check: bool = True,
    log_name: str | None = None,
) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(map(str, cmd)))
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
    proc = subprocess.CompletedProcess(cmd, process.wait(), stdout=stdout, stderr=None)
    if log_name:
        (RUN_DIR / log_name).write_text(stdout, encoding="utf-8")
    if check and proc.returncode:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}")
    return proc


def parse_action_command(command: str) -> ParsedCommand:
    tokens = shlex.split(command, posix=True)
    if not tokens:
        raise ValueError("Planner action command is empty.")

    env: dict[str, str] = {}
    while tokens and ENV_ASSIGNMENT_RE.match(tokens[0]):
        key, value = tokens.pop(0).split("=", 1)
        env[key] = value
    if not tokens:
        raise ValueError("Planner action command has environment assignments but no executable.")

    executable = tokens[0]
    if executable == "cat":
        return ParsedCommand(env=env, argv=tokens, kind="cat")

    if executable != "python":
        raise ValueError(f"Unsupported planner executable: {executable!r}")
    if len(tokens) < 2:
        raise ValueError("Planner python command is missing a script path.")

    script = tokens[1].replace("\\", "/")
    if script not in ALLOWED_PYTHON_SCRIPTS:
        raise ValueError(f"Planner script is not allowlisted: {script!r}")

    return ParsedCommand(env=env, argv=[sys.executable, script, *tokens[2:]], kind="python")


def run_cat(argv: list[str]) -> subprocess.CompletedProcess[str]:
    chunks: list[str] = []
    for value in argv[1:]:
        path = Path(value)
        path = path if path.is_absolute() else ROOT / path
        if not path.exists():
            raise FileNotFoundError(path)
        text = path.read_text(encoding="utf-8")
        print(text, end="" if text.endswith("\n") else "\n")
        chunks.append(text)
    return subprocess.CompletedProcess(argv, 0, "".join(chunks), None)


def execute_parsed_command(parsed: ParsedCommand, *, log_name: str = "selected_action.log") -> subprocess.CompletedProcess[str]:
    if parsed.kind == "cat":
        return run_cat(parsed.argv)
    env = os.environ.copy()
    env.update(parsed.env)
    return run(parsed.argv, env=env, log_name=log_name)


def bootstrap_plan(plan_run_id: str, *, reason: str) -> dict[str, Any]:
    """First Stage 5 action when no prior result summary exists."""

    return {
        "run_id": plan_run_id,
        "source_kind": "bootstrap",
        "source_summary": None,
        "reason": reason,
        "actions": [
            {
                "name": "Run Stage 5 ARC-AGI candidate gate",
                "reason": (
                    "No Stage 5 summary exists yet; start with the candidate-source gate to measure "
                    "base, recurrent, symbolic, and hybrid ARC-AGI exact-grid behavior before SFT or particles."
                ),
                "command": (
                    f"STAGE5_ARC_AGI_GATE_RUN_ID={RUN_ID}_candidate_gate "
                    "python colab/run_stage5_arc_agi_candidate_gate.py"
                ),
                "priority": 10,
            }
        ],
    }


def run_planner(*, step: int | None = None) -> tuple[Path, dict[str, Any], subprocess.CompletedProcess[str]]:
    plan_run_id = f"{RUN_ID}_plan" if step is None else f"{RUN_ID}_plan{step:02d}"
    env = os.environ.copy()
    env["STAGE5_ARC_AGI_NEXT_PLAN_RUN_ID"] = plan_run_id
    if SOURCE_SUMMARY:
        env["STAGE5_ARC_AGI_NEXT_PLAN_SOURCE_SUMMARY"] = SOURCE_SUMMARY
    proc = run([sys.executable, "colab/plan_stage5_next_run.py"], env=env, check=False, log_name="planner.log")
    plan_summary = ROOT / "outputs" / "stage5" / plan_run_id / "summary.json"
    if proc.returncode:
        if not SOURCE_SUMMARY and "No Stage 5 result summary found" in proc.stdout:
            plan = bootstrap_plan(plan_run_id, reason="planner_found_no_stage5_summary")
            write_json(plan_summary, plan)
            return plan_summary, plan, proc
        raise RuntimeError(f"command failed: {sys.executable} colab/plan_stage5_next_run.py")
    if not plan_summary.exists():
        raise FileNotFoundError(plan_summary)
    return plan_summary, read_json(plan_summary), proc


def selected_action(plan: dict[str, Any], *, action_index: int = ACTION_INDEX) -> dict[str, Any]:
    actions = plan.get("actions") or []
    if not actions:
        raise ValueError("Planner returned no actions.")
    if action_index < 0 or action_index >= len(actions):
        raise IndexError(f"STAGE5_ARC_AGI_NEXT_ACTION_INDEX={action_index} out of range for {len(actions)} actions.")
    return actions[action_index]


def action_fingerprint(action: dict[str, Any]) -> str:
    return str(action.get("command", "")).strip()


def is_repeat_action(action: dict[str, Any], seen_commands: set[str], *, allow_repeat: bool = ALLOW_REPEAT) -> bool:
    return not allow_repeat and action_fingerprint(action) in seen_commands


def execute_action_loop(*, execute: bool = EXECUTE, max_actions: int = MAX_ACTIONS) -> list[dict[str, Any]]:
    if max_actions < 1:
        raise ValueError("STAGE5_ARC_AGI_NEXT_ACTION_MAX_ACTIONS must be >= 1.")
    steps: list[dict[str, Any]] = []
    seen_commands: set[str] = set()
    total_steps = max_actions if execute else 1

    for step_index in range(total_steps):
        plan_summary_path, plan, _planner_proc = run_planner(step=step_index if total_steps > 1 else None)
        action = selected_action(plan)
        parsed = parse_action_command(str(action.get("command", "")))
        repeated = is_repeat_action(action, seen_commands)
        command_text = action_fingerprint(action)
        step_payload: dict[str, Any] = {
            "step": step_index,
            "planner_summary": path_for_cli(plan_summary_path),
            "planner": plan,
            "selected_action": action,
            "parsed_command": {"kind": parsed.kind, "env": parsed.env, "argv": parsed.argv},
            "repeat_detected": repeated,
        }
        if repeated:
            step_payload["execution"] = {
                "executed": False,
                "stopped": True,
                "reason": "Selected action command repeated; set STAGE5_ARC_AGI_NEXT_ACTION_ALLOW_REPEAT=1 to override.",
            }
            steps.append(step_payload)
            break

        seen_commands.add(command_text)
        if execute:
            proc = execute_parsed_command(parsed, log_name=f"selected_action_{step_index:02d}.log")
            step_payload["execution"] = {"executed": True, "returncode": proc.returncode}
            steps.append(step_payload)
            if proc.returncode:
                break
        else:
            step_payload["execution"] = {
                "executed": False,
                "dry_run": True,
                "reason": "Set STAGE5_ARC_AGI_NEXT_ACTION_EXECUTE=1 to run the selected action.",
            }
            steps.append(step_payload)
            break
    return steps


def backup_to_drive() -> None:
    if not Path("/content/drive/MyDrive").exists():
        try:
            from google.colab import drive  # type: ignore

            drive.mount("/content/drive")
        except Exception as exc:  # pragma: no cover - Colab only
            print(f"Drive mount skipped/failed: {exc}")
            return
    drive_root = Path(os.environ.get("DRIVE_BACKUP_DIR", "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts"))
    if not drive_root.exists():
        print(f"Drive backup skipped; missing {drive_root}")
        return
    backup = drive_root / RUN_ID
    backup.mkdir(parents=True, exist_ok=True)
    shutil.copytree(RUN_DIR, backup / "run_dir", dirs_exist_ok=True)
    print(f"backed_up_to={backup}")


def write_report(payload: dict[str, Any]) -> None:
    write_json(RUN_DIR / "summary.json", payload)
    steps = payload.get("steps") or []
    first_step = steps[0] if steps else {}
    action = first_step.get("selected_action") or {}
    parsed = first_step.get("parsed_command") or {}
    execution = first_step.get("execution") or {}
    lines = [
        f"# Stage 5 Next Action - {RUN_ID}",
        "",
        f"- Planner summary: `{first_step.get('planner_summary')}`",
        f"- Execute requested: `{payload['execute_requested']}`",
        f"- Action index: `{payload['action_index']}`",
        f"- Max actions: `{payload['max_actions']}`",
        f"- Completed steps: `{len(steps)}`",
        f"- Selected action: `{action.get('name')}`",
        f"- Priority: `{action.get('priority')}`",
        f"- Reason: {action.get('reason')}",
        f"- Command: `{action.get('command')}`",
        f"- Parsed kind: `{parsed.get('kind')}`",
        f"- Parsed argv: `{parsed.get('argv')}`",
        f"- Execution: `{execution}`",
    ]
    if len(steps) > 1:
        lines.extend(["", "## Action Loop", ""])
        for step in steps:
            step_action = step.get("selected_action") or {}
            lines.append(
                f"- Step `{step.get('step')}`: `{step_action.get('name')}` "
                f"repeat `{step.get('repeat_detected')}` execution `{step.get('execution')}`"
            )
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print(
            "Run Stage 5 planner and optionally execute the selected allowlisted action. "
            "Set STAGE5_ARC_AGI_NEXT_ACTION_EXECUTE=1 to execute. "
            "Set STAGE5_ARC_AGI_NEXT_ACTION_MAX_ACTIONS=N for a bounded loop."
        )
        return 0

    steps = execute_action_loop()
    last_execution = (steps[-1].get("execution") if steps else {}) or {}
    payload = {
        "run_id": RUN_ID,
        "action_index": ACTION_INDEX,
        "max_actions": MAX_ACTIONS,
        "allow_repeat": ALLOW_REPEAT,
        "execute_requested": EXECUTE,
        "steps": steps,
    }
    write_report(payload)
    backup_to_drive()
    return int(last_execution.get("returncode", 0))


if __name__ == "__main__":
    raise SystemExit(main())
