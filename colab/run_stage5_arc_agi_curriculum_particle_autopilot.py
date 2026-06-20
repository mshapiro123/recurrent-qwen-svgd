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


def summarize_autopilot(curriculum: dict[str, Any], particle: dict[str, Any]) -> dict[str, Any]:
    particle_decision = particle.get("particle_decision", {})
    evidence = particle_decision.get("evidence", {})
    return {
        "final_checkpoint": curriculum.get("final_checkpoint"),
        "curriculum_stages": [
            {
                "name": row.get("stage", {}).get("name"),
                "modes": row.get("stage", {}).get("synthetic_modes"),
                "selected_checkpoint": row.get("selected_checkpoint", {}).get("checkpoint"),
                "selected_summary": row.get("selected_checkpoint", {}).get("summary", {}),
            }
            for row in curriculum.get("stages", [])
        ],
        "particle_passed": particle_decision.get("passed", False),
        "best_replicated_particle_variant": evidence.get("best_replicated_variant"),
        "particle_variant_mean_deltas": {
            name: row.get("mean_delta_vs_tuned", {})
            for name, row in (evidence.get("variants") or {}).items()
        },
    }


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
    for label in ("curriculum", "post_curriculum_particle"):
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
        f"- Final curriculum checkpoint: `{compact['final_checkpoint']}`",
        f"- Particle gate passed: `{compact['particle_passed']}`",
        f"- Best replicated particle variant: `{compact['best_replicated_particle_variant']}`",
        "",
        "## Curriculum Stages",
        "",
    ]
    for row in compact["curriculum_stages"]:
        lines.append(f"- `{row['name']}` modes `{row['modes']}` selected `{row['selected_checkpoint']}` summary `{row['selected_summary']}`")
    lines.extend(["", "## Particle Mean Deltas", ""])
    for name, delta in compact["particle_variant_mean_deltas"].items():
        lines.append(f"- `{name}`: `{delta}`")
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print("Run from Colab to execute curriculum recovery followed by post-curriculum particle gate.")
        return 0

    curriculum = run_child(
        "curriculum",
        "colab/run_stage5_arc_agi_curriculum.py",
        {
            "STAGE5_ARC_AGI_CURRICULUM_RUN_ID": child_run_id("curriculum"),
            "STAGE5_ARC_AGI_CURRICULUM_PUSH": "0",
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
            "curriculum": child_run_id("curriculum"),
            "post_curriculum_particle": child_run_id("post_curriculum_particle"),
        },
        "curriculum": curriculum,
        "post_curriculum_particle": particle,
        "compact": summarize_autopilot(curriculum, particle),
    }
    write_report(payload)
    backup_to_drive()
    if PUSH_RESULTS:
        git_commit_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
