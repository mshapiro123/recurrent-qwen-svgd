"""Run deterministic ARC curriculum, then replicated particle/SVGD gate.

This is the one-shot Colab launcher for the current Stage 5 question:

1. Can staged deterministic recurrent SFT recover ARC-style skills?
2. Once recovered, do particles/SVGD add replicated value over that selected
   recurrent checkpoint?

The child scripts write detailed reports. This wrapper keeps one A100 session
moving and writes a compact parent decision summary.
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
RUN_ID = os.environ.get("STAGE5_ARC_AGI_CURRICULUM_PARTICLE_AUTOPILOT_RUN_ID") or time.strftime(
    "stage5_arc_curriculum_particle_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

PUSH_RESULTS = os.environ.get("STAGE5_ARC_AGI_CURRICULUM_PARTICLE_AUTOPILOT_PUSH", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
RUN_CANDIDATE_DISTILL_GATE = os.environ.get(
    "STAGE5_ARC_AGI_CURRICULUM_PARTICLE_AUTOPILOT_RUN_CANDIDATE_DISTILL_GATE",
    "1",
).strip().lower() in {"1", "true", "yes", "y"}
REQUIRE_CANDIDATE_DISTILL_PASS = os.environ.get(
    "STAGE5_ARC_AGI_CURRICULUM_PARTICLE_AUTOPILOT_REQUIRE_CANDIDATE_DISTILL_PASS",
    "1",
).strip().lower() in {"1", "true", "yes", "y"}
MIN_CANDIDATE_DISTILL_ROWS = int(
    os.environ.get("STAGE5_ARC_AGI_CURRICULUM_PARTICLE_AUTOPILOT_MIN_CANDIDATE_DISTILL_ROWS", "1")
)
MIN_CANDIDATE_DISTILL_SELECTED_DELTA = int(
    os.environ.get("STAGE5_ARC_AGI_CURRICULUM_PARTICLE_AUTOPILOT_MIN_CANDIDATE_DISTILL_SELECTED_DELTA", "0")
)
MIN_CANDIDATE_DISTILL_BEST_DELTA = int(
    os.environ.get("STAGE5_ARC_AGI_CURRICULUM_PARTICLE_AUTOPILOT_MIN_CANDIDATE_DISTILL_BEST_DELTA", "0")
)
MIN_CANDIDATE_DISTILL_VALID_RATE_DELTA = float(
    os.environ.get("STAGE5_ARC_AGI_CURRICULUM_PARTICLE_AUTOPILOT_MIN_CANDIDATE_DISTILL_VALID_RATE_DELTA", "-0.05")
)


def child_run_id(label: str) -> str:
    return f"{RUN_ID}_{label}"


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


def path_for_cli(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def child_summary_path(label: str) -> Path:
    return ROOT / "outputs" / "stage5" / child_run_id(label) / "summary.json"


def run_child(label: str, script: str, env_updates: dict[str, str], *, log_name: str) -> dict[str, Any]:
    summary_path = child_summary_path(label)
    if summary_path.exists() and os.environ.get("STAGE5_ARC_AGI_CURRICULUM_PARTICLE_AUTOPILOT_RESUME", "1") in {
        "1",
        "true",
        "yes",
    }:
        print(f"reusing_child_summary={summary_path}")
        return read_json(summary_path)
    env = os.environ.copy()
    env.update(env_updates)
    run([sys.executable, script], env=env, log_name=log_name)
    return read_json(summary_path)


def candidate_distill_rows(candidate_gate: dict[str, Any] | None) -> int:
    if not candidate_gate:
        return 0
    comparison = candidate_gate.get("comparison", {})
    distill = comparison.get("candidate_distill", {})
    return int(distill.get("candidate_distill_rows", 0))


def decide_candidate_distill_gate(candidate_gate: dict[str, Any] | None) -> tuple[bool, dict[str, Any]]:
    if not candidate_gate:
        return False, {"reason": "candidate distillation gate did not run"}
    delta = candidate_gate.get("delta_candidate_distill_vs_baseline", {})
    rows = candidate_distill_rows(candidate_gate)
    selected_delta = int(delta.get("best_selected_delta", 0))
    best_delta = int(delta.get("best_best_delta", 0))
    valid_rate_delta = float(delta.get("best_valid_rate_delta", 0.0))
    passed = (
        rows >= MIN_CANDIDATE_DISTILL_ROWS
        and selected_delta >= MIN_CANDIDATE_DISTILL_SELECTED_DELTA
        and best_delta >= MIN_CANDIDATE_DISTILL_BEST_DELTA
        and valid_rate_delta >= MIN_CANDIDATE_DISTILL_VALID_RATE_DELTA
    )
    return passed, {
        "candidate_distill_rows": rows,
        "min_candidate_distill_rows": MIN_CANDIDATE_DISTILL_ROWS,
        "best_selected_delta": selected_delta,
        "min_best_selected_delta": MIN_CANDIDATE_DISTILL_SELECTED_DELTA,
        "best_best_delta": best_delta,
        "min_best_best_delta": MIN_CANDIDATE_DISTILL_BEST_DELTA,
        "best_valid_rate_delta": valid_rate_delta,
        "min_best_valid_rate_delta": MIN_CANDIDATE_DISTILL_VALID_RATE_DELTA,
    }


def candidate_distill_curriculum_env(candidate_gate: dict[str, Any]) -> dict[str, str]:
    candidate_jsonl = candidate_gate.get("candidate_source", {}).get("candidate_jsonl")
    if not candidate_jsonl:
        raise ValueError("candidate distillation gate summary is missing candidate_source.candidate_jsonl")
    metadata = candidate_gate.get("metadata", {})
    return {
        "STAGE5_ARC_AGI_CANDIDATE_DISTILL_JSONLS": str(candidate_jsonl),
        "STAGE5_ARC_AGI_CANDIDATE_DISTILL_CHOICE": str(metadata.get("distill_choice", "best_exact")),
        "STAGE5_ARC_AGI_CANDIDATE_DISTILL_COMPLETION_SOURCE": str(
            metadata.get("distill_completion_source", "trace_then_canonical_grid")
        ),
    }


def summarize_autopilot(
    curriculum: dict[str, Any] | None,
    particle: dict[str, Any] | None,
    candidate_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidate_passed, candidate_evidence = decide_candidate_distill_gate(candidate_gate)
    particle_decision = (particle or {}).get("particle_decision", {})
    evidence = particle_decision.get("evidence", {})
    return {
        "candidate_distillation_passed": candidate_passed,
        "candidate_distillation_evidence": candidate_evidence,
        "candidate_distillation_jsonl": (candidate_gate or {}).get("candidate_source", {}).get("candidate_jsonl"),
        "final_checkpoint": (curriculum or {}).get("final_checkpoint"),
        "curriculum_stages": [
            {
                "name": row.get("stage", {}).get("name"),
                "modes": row.get("stage", {}).get("synthetic_modes"),
                "selected_checkpoint": row.get("selected_checkpoint", {}).get("checkpoint"),
                "selected_summary": row.get("selected_checkpoint", {}).get("summary", {}),
                "candidate_distill_rows": row.get("candidate_distill_rows", 0),
            }
            for row in (curriculum or {}).get("stages", [])
        ],
        "particle_passed": particle_decision.get("passed", False),
        "best_replicated_particle_variant": evidence.get("best_replicated_variant"),
        "particle_variant_mean_deltas": {
            name: row.get("mean_delta_vs_tuned", {})
            for name, row in (evidence.get("variants") or {}).items()
        },
    }


def recommended_next_actions(compact: dict[str, Any]) -> list[dict[str, str]]:
    """Return concrete follow-up actions for the next Colab cell.

    The parent autopilot is intentionally conservative: it gates candidate
    distillation, then curriculum recovery, then particle value. These
    recommendations make the outcome operational without requiring someone to
    inspect nested child summaries by hand after a long run.
    """
    actions: list[dict[str, str]] = []
    curriculum_summary = f"outputs/stage5/{child_run_id('curriculum')}/summary.json"

    if not compact["candidate_distillation_passed"]:
        actions.append(
            {
                "name": "Inspect candidate distillation gate",
                "reason": "Candidate distillation did not clear the promotion gate; inspect whether it failed from too few exact candidates, lower selected accuracy, or lower valid rate.",
                "command": f"cat outputs/stage5/{child_run_id('candidate_distill_gate')}/summary.md",
            }
        )
        actions.append(
            {
                "name": "Run baseline curriculum without candidate distillation",
                "reason": "This preserves A100 momentum if candidate distillation is not yet a useful training signal.",
                "command": (
                    "STAGE5_ARC_AGI_CURRICULUM_PARTICLE_AUTOPILOT_RUN_CANDIDATE_DISTILL_GATE=0 "
                    f"STAGE5_ARC_AGI_CURRICULUM_PARTICLE_AUTOPILOT_RUN_ID={RUN_ID}_baseline_no_candidate_distill "
                    "python colab/run_stage5_arc_agi_curriculum_particle_autopilot.py"
                ),
            }
        )
        return actions

    final_checkpoint = compact.get("final_checkpoint")
    if final_checkpoint:
        actions.append(
            {
                "name": "Run recovered-vs-base ARC benchmark",
                "reason": "The core question is whether curriculum tuning closes the gap from recurrent start to base Qwen and how much remains.",
                "command": (
                    f"STAGE5_ARC_AGI_CURRICULUM_SUMMARY={curriculum_summary} "
                    "STAGE5_ARC_AGI_LIMIT=50 "
                    f"STAGE5_ARC_AGI_RECOVERED_BENCHMARK_RUN_ID={RUN_ID}_recovered_benchmark_limit50 "
                    "python colab/run_stage5_arc_agi_recovered_benchmark.py"
                ),
            }
        )
        actions.append(
            {
                "name": "Run recovered TTA and selector sweep",
                "reason": "If recurrent recovery is real, TTA/selector sweeps identify whether inference-time structure adds value before more training.",
                "command": (
                    f"STAGE5_ARC_AGI_CURRICULUM_SUMMARY={curriculum_summary} "
                    "STAGE5_ARC_AGI_LIMIT=50 "
                    f"STAGE5_ARC_AGI_TTA_SWEEP_RUN_ID={RUN_ID}_tta_sweep_limit50 "
                    "python colab/run_stage5_arc_agi_tta_sweep.py"
                ),
            }
        )
    else:
        actions.append(
            {
                "name": "Inspect curriculum logs",
                "reason": "Candidate distillation passed, but no final curriculum checkpoint was recorded.",
                "command": f"cat outputs/stage5/{child_run_id('curriculum')}/summary.md",
            }
        )

    if compact["particle_passed"]:
        actions.append(
            {
                "name": "Replicate winning particle variant at larger limit",
                "reason": "The particle gate passed; larger limits are needed before treating the replicated lift as reliable.",
                "command": f"cat outputs/stage5/{child_run_id('post_curriculum_particle')}/summary.md",
            }
        )
    else:
        actions.append(
            {
                "name": "Defer particle/SVGD training",
                "reason": "Particle inference did not clear the replicated gate; prioritize deterministic recurrent recovery and benchmark comparison.",
                "command": f"cat outputs/stage5/{child_run_id('post_curriculum_particle')}/summary.md",
            }
        )
    return actions


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
    for label in ("candidate_distill_gate", "curriculum", "post_curriculum_particle"):
        child_dir = ROOT / "outputs" / "stage5" / child_run_id(label)
        if child_dir.exists():
            shutil.copytree(child_dir, backup / label, dirs_exist_ok=True)
    print(f"backed_up_to={backup}")


def git_commit_results() -> None:
    for pattern in ["*.json", "*.md", "*.log"]:
        run(["git", "add", "-f", str((RUN_DIR / pattern).relative_to(ROOT))], check=False)
    if run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("No curriculum-particle autopilot outputs changed.")
        return
    run(["git", "commit", "-m", "Record Stage 5 curriculum particle autopilot"])
    run(["git", "push", "origin", "main"], check=False)


def write_report(payload: dict[str, Any]) -> None:
    (RUN_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    compact = payload["compact"]
    lines = [
        f"# Stage 5 ARC Curriculum + Particle Autopilot - {RUN_ID}",
        "",
        "## Decision Summary",
        "",
        f"- Candidate distillation gate passed: `{compact['candidate_distillation_passed']}`",
        f"- Candidate distillation JSONL: `{compact['candidate_distillation_jsonl']}`",
        f"- Final curriculum checkpoint: `{compact['final_checkpoint']}`",
        f"- Particle gate passed: `{compact['particle_passed']}`",
        f"- Best replicated particle variant: `{compact['best_replicated_particle_variant']}`",
        "",
        "## Curriculum Stages",
        "",
    ]
    for row in compact["curriculum_stages"]:
        lines.append(
            f"- `{row['name']}` modes `{row['modes']}` selected `{row['selected_checkpoint']}` "
            f"candidate_rows `{row['candidate_distill_rows']}` summary `{row['selected_summary']}`"
        )
    lines.extend(["", "## Candidate Distillation Evidence", "", f"`{compact['candidate_distillation_evidence']}`"])
    lines.extend(["", "## Particle Mean Deltas", ""])
    for name, delta in compact["particle_variant_mean_deltas"].items():
        lines.append(f"- `{name}`: `{delta}`")
    lines.extend(["", "## Recommended Next Actions", ""])
    for index, action in enumerate(compact.get("recommended_next_actions", []), start=1):
        lines.extend(
            [
                f"{index}. **{action['name']}**",
                f"   - Reason: {action['reason']}",
                f"   - Command: `{action['command']}`",
            ]
        )
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print("Run from Colab to execute curriculum recovery followed by post-curriculum particle gate.")
        return 0

    candidate_gate = None
    candidate_passed = True
    candidate_env: dict[str, str] = {}
    if RUN_CANDIDATE_DISTILL_GATE:
        candidate_gate = run_child(
            "candidate_distill_gate",
            "colab/run_stage5_arc_agi_candidate_distill_gate.py",
            {
                "STAGE5_ARC_AGI_CANDIDATE_DISTILL_GATE_RUN_ID": child_run_id("candidate_distill_gate"),
                "STAGE5_ARC_AGI_CANDIDATE_DISTILL_GATE_PUSH": "0",
            },
            log_name="candidate_distill_gate.log",
        )
        candidate_passed, candidate_evidence = decide_candidate_distill_gate(candidate_gate)
        print(f"candidate_distill_gate_passed={candidate_passed} evidence={candidate_evidence}")
        if candidate_passed:
            candidate_env = candidate_distill_curriculum_env(candidate_gate)
        elif REQUIRE_CANDIDATE_DISTILL_PASS:
            payload = {
                "run_id": RUN_ID,
                "child_run_ids": {"candidate_distill_gate": child_run_id("candidate_distill_gate")},
                "candidate_distill_gate": candidate_gate,
                "curriculum": None,
                "post_curriculum_particle": None,
                "compact": summarize_autopilot(None, None, candidate_gate),
                "stopped_after_candidate_distill_gate": True,
            }
            payload["compact"]["recommended_next_actions"] = recommended_next_actions(payload["compact"])
            write_report(payload)
            backup_to_drive()
            if PUSH_RESULTS:
                git_commit_results()
            return 0

    curriculum = run_child(
        "curriculum",
        "colab/run_stage5_arc_agi_curriculum.py",
        {
            "STAGE5_ARC_AGI_CURRICULUM_RUN_ID": child_run_id("curriculum"),
            "STAGE5_ARC_AGI_CURRICULUM_PUSH": "0",
            **candidate_env,
        },
        log_name="curriculum.log",
    )
    curriculum_summary = child_summary_path("curriculum")
    particle = run_child(
        "post_curriculum_particle",
        "colab/run_stage5_arc_agi_post_curriculum_particle_gate.py",
        {
            "STAGE5_ARC_AGI_POST_CURRICULUM_PARTICLE_RUN_ID": child_run_id("post_curriculum_particle"),
            "STAGE5_ARC_AGI_POST_CURRICULUM_PARTICLE_PUSH": "0",
            "STAGE5_ARC_AGI_CURRICULUM_SUMMARY": path_for_cli(curriculum_summary),
        },
        log_name="post_curriculum_particle.log",
    )
    payload = {
        "run_id": RUN_ID,
        "child_run_ids": {
            "candidate_distill_gate": child_run_id("candidate_distill_gate") if RUN_CANDIDATE_DISTILL_GATE else None,
            "curriculum": child_run_id("curriculum"),
            "post_curriculum_particle": child_run_id("post_curriculum_particle"),
        },
        "candidate_distill_gate": candidate_gate,
        "curriculum": curriculum,
        "post_curriculum_particle": particle,
        "compact": summarize_autopilot(curriculum, particle, candidate_gate),
    }
    payload["compact"]["recommended_next_actions"] = recommended_next_actions(payload["compact"])
    write_report(payload)
    backup_to_drive()
    if PUSH_RESULTS:
        git_commit_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
