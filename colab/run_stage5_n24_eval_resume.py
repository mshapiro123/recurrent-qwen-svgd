"""Resume the eval-only tail of the Stage 5 N-24 support-12 rung.

This runner is intentionally narrow: it does not regenerate data and it does
not retrain. It restores the backed-up N-24 checkpoints after a Colab restart,
runs the frozen depth-22 active-label eval, and publishes partial/final
summaries. The mid checkpoint at step 4000 is optional because the original
training runner only backed up the ramp and final checkpoints to Drive.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import colab.run_stage5_n24_support12_rung as n24_runner
from colab.run_stage5_n24_support12_rung import parse_checkpoints, write_markdown
from colab.stage5_chain_consolidation_utils import (
    DRIVE_CHECKPOINT_ROOT,
    ROOT,
    path_for_cli,
    publish_run,
    read_json,
    root_path,
    write_json,
)
from colab.stage5_n24_rung import N24_CHECKPOINTS


DEFAULT_RUN_ID = "stage5_n24_support12_rung_20260707_140139"


def local_checkpoint_path(run_dir: Path, step: int) -> Path:
    return run_dir / "train" / "chain_continuation" / f"unfrozen_recurrent_step_{int(step)}.pt"


def maybe_path(raw: str | Path | None) -> Path | None:
    if not raw:
        return None
    candidate = Path(raw)
    candidates = [candidate] if candidate.is_absolute() else [ROOT / candidate, candidate]
    for path in candidates:
        if path.exists():
            return path
    return None


def drive_candidates(run_id: str, step: int, summary: dict[str, Any]) -> list[Path]:
    name = f"unfrozen_recurrent_step_{int(step)}.pt"
    explicit: list[str | None] = []
    if step == 2000:
        explicit.extend([summary.get("ramp_checkpoint_drive_backup"), summary.get("ramp_checkpoint")])
    if step == 6000:
        explicit.extend([summary.get("final_checkpoint_drive_backup"), summary.get("final_checkpoint")])
    candidates = [Path(item) for item in explicit if item]
    candidates.extend(
        [
            DRIVE_CHECKPOINT_ROOT / run_id / "anneal_to_outcome_ramp" / name,
            DRIVE_CHECKPOINT_ROOT / run_id / "anneal_to_outcome_final" / name,
            DRIVE_CHECKPOINT_ROOT / run_id / "anneal_to_outcome_interval" / name,
            DRIVE_CHECKPOINT_ROOT / run_id / "chain_continuation" / name,
            DRIVE_CHECKPOINT_ROOT / run_id / name,
        ]
    )
    return candidates


def restore_checkpoint_for_step(run_dir: Path, step: int, summary: dict[str, Any]) -> Path | None:
    dest = local_checkpoint_path(run_dir, step)
    if dest.exists():
        print(f"checkpoint_step_{step}_already_local={dest}", flush=True)
        return dest
    print(f"checkpoint_step_{step}_restore_search_start", flush=True)
    for raw in drive_candidates(str(summary["run_id"]), step, summary):
        print(f"checkpoint_step_{step}_try_source={raw}", flush=True)
        source = maybe_path(raw)
        if source is None:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        print(f"restored_checkpoint_step_{step}={dest} from {source}", flush=True)
        return dest
    return None


def run(cmd: list[str | os.PathLike[str]], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(map(str, cmd)), flush=True)
    proc = subprocess.Popen(
        list(map(str, cmd)),
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line)
        print(line, end="", flush=True)
    returncode = proc.wait()
    output = "".join(lines)
    if check and returncode:
        raise subprocess.CalledProcessError(returncode, list(map(str, cmd)), output=output)
    return subprocess.CompletedProcess(list(map(str, cmd)), returncode, stdout=output, stderr=None)


def write_resume_status(run_dir: Path, status: dict[str, Any]) -> None:
    payload = {
        "kind": "stage5_n24_eval_resume_status",
        "updated_at_unix": time.time(),
        **status,
    }
    write_json(run_dir / "eval" / "resume_status.json", payload)
    print("resume_status:", payload, flush=True)


def main() -> int:
    run_id = os.environ.get("STAGE5_N24_RESUME_RUN_ID", DEFAULT_RUN_ID)
    run_dir = ROOT / "outputs" / "stage5" / run_id
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    summary = read_json(summary_path)
    if summary.get("run_id") != run_id:
        raise RuntimeError(f"Run-id mismatch in {summary_path}: {summary.get('run_id')!r}")

    eval_steps = parse_checkpoints(
        os.environ.get("STAGE5_N24_EVAL_CHECKPOINTS", ",".join(map(str, N24_CHECKPOINTS)))
    )
    strict = os.environ.get("STAGE5_N24_RESUME_STRICT_CHECKPOINTS", "0") == "1"
    dtype = os.environ.get("STAGE5_N24_DTYPE", "bfloat16")
    value_prefix = os.environ.get("STAGE5_N24_VALUE_PREFIX", "letter:")
    threshold = float(os.environ.get("STAGE5_N24_THRESHOLD", str(summary.get("threshold", 0.71))))

    frozen = summary.get("frozen_eval_set") or read_json(
        ROOT / "outputs" / "stage5" / os.environ.get("STAGE5_N24_FROZEN_EVAL_ID", "stage5_synthetic_depth_frozen_eval_v3_depth22_n24") / "summary.json"
    )
    frozen_chain = root_path(frozen["test_chain_mcq"])
    if not frozen_chain.exists():
        raise FileNotFoundError(frozen_chain)

    n24_runner.run = run
    print(
        {
            "event": "resume_start",
            "run_id": run_id,
            "summary": path_for_cli(summary_path),
            "eval_steps": eval_steps,
            "dtype": dtype,
            "frozen_chain": path_for_cli(frozen_chain),
        },
        flush=True,
    )
    write_resume_status(
        run_dir,
        {
            "status": "started",
            "run_id": run_id,
            "eval_steps": eval_steps,
            "completed_steps": [],
            "skipped_steps": [],
        },
    )

    existing_evals = list(summary.get("checkpoint_evals") or [])
    existing_steps = {int(item.get("step")) for item in existing_evals if item.get("step") is not None}
    checkpoint_evals = existing_evals
    skipped: list[dict[str, Any]] = list(summary.get("skipped_checkpoints") or [])

    for step in eval_steps:
        if step in existing_steps:
            print(f"checkpoint_step_{step}_already_evaluated", flush=True)
            continue
        write_resume_status(
            run_dir,
            {
                "status": "restoring_checkpoint",
                "run_id": run_id,
                "current_step": int(step),
                "completed_steps": [int(item.get("step")) for item in checkpoint_evals],
                "skipped_steps": [int(item.get("step")) for item in skipped],
            },
        )
        checkpoint = restore_checkpoint_for_step(run_dir, step, summary)
        if checkpoint is None:
            message = f"checkpoint step {step} is unavailable locally and in known Drive backup paths"
            if strict:
                raise FileNotFoundError(message)
            print(f"skipping_checkpoint_step_{step}: {message}", flush=True)
            skipped.append({"step": int(step), "reason": message})
            write_resume_status(
                run_dir,
                {
                    "status": "checkpoint_skipped",
                    "run_id": run_id,
                    "current_step": int(step),
                    "completed_steps": [int(item.get("step")) for item in checkpoint_evals],
                    "skipped_steps": [int(item.get("step")) for item in skipped],
                    "reason": message,
                },
            )
            continue
        write_resume_status(
            run_dir,
            {
                "status": "eval_start",
                "run_id": run_id,
                "current_step": int(step),
                "checkpoint": path_for_cli(checkpoint),
                "completed_steps": [int(item.get("step")) for item in checkpoint_evals],
                "skipped_steps": [int(item.get("step")) for item in skipped],
            },
        )
        checkpoint_evals.append(
            n24_runner.eval_checkpoint(
                run_dir=run_dir,
                frozen_chain=path_for_cli(frozen_chain),
                checkpoint=checkpoint,
                step=step,
                dtype=dtype,
                value_prefix=value_prefix,
                threshold=threshold,
            )
        )
        latest = checkpoint_evals[-1]
        print(
            {
                "event": "eval_done",
                "step": int(step),
                "verdict": latest["score"]["verdict"],
                "selected_correct": latest["score"]["selected_correct"],
                "overall_pass": latest["score"]["overall_pass"],
            },
            flush=True,
        )
        summary.update(
            {
                "kind": "stage5_n24_support12_rung",
                "status": "partial_eval_finished",
                "resume_eval_only": True,
                "checkpoint_evals": checkpoint_evals,
                "skipped_checkpoints": skipped,
            }
        )
        write_json(summary_path, summary)
        write_markdown(run_dir, summary)
        write_resume_status(
            run_dir,
            {
                "status": "publishing_step",
                "run_id": run_id,
                "current_step": int(step),
                "completed_steps": [int(item.get("step")) for item in checkpoint_evals],
                "skipped_steps": [int(item.get("step")) for item in skipped],
            },
        )
        publish_run(run_dir, message=f"Record Stage 5 N-24 resumed eval {run_id} step {step} [skip ci]")
        write_resume_status(
            run_dir,
            {
                "status": "published_step",
                "run_id": run_id,
                "current_step": int(step),
                "completed_steps": [int(item.get("step")) for item in checkpoint_evals],
                "skipped_steps": [int(item.get("step")) for item in skipped],
            },
        )

    if not checkpoint_evals:
        raise RuntimeError("No checkpoint evaluations were available or completed.")

    final = checkpoint_evals[-1]
    summary.update(
        {
            "kind": "stage5_n24_support12_rung",
            "status": "finished_with_frozen_eval",
            "resume_eval_only": True,
            "checkpoint_evals": checkpoint_evals,
            "skipped_checkpoints": skipped,
            "decision_read": {
                "question": "Does the final synthetic rung confirm a four-point frontier law or characterize a ceiling?",
                "final_step": final["step"],
                "final_verdict": final["score"]["verdict"],
                "synthetic_line_closed": True,
                "resume_note": "Eval-only resume after the original Colab run completed training before checkpoint evals landed.",
            },
        }
    )
    write_json(summary_path, summary)
    write_markdown(run_dir, summary)
    write_resume_status(
        run_dir,
        {
            "status": "publishing_final",
            "run_id": run_id,
            "completed_steps": [int(item.get("step")) for item in checkpoint_evals],
            "skipped_steps": [int(item.get("step")) for item in skipped],
            "final_verdict": summary["decision_read"]["final_verdict"],
        },
    )
    publish_run(run_dir, message=f"Record Stage 5 N-24 support-12 resumed final rung {run_id} [skip ci]")
    write_resume_status(
        run_dir,
        {
            "status": "finished",
            "run_id": run_id,
            "completed_steps": [int(item.get("step")) for item in checkpoint_evals],
            "skipped_steps": [int(item.get("step")) for item in skipped],
            "final_verdict": summary["decision_read"]["final_verdict"],
        },
    )
    print(
        {
            "run_id": run_id,
            "status": summary["status"],
            "evaluated_steps": [item["step"] for item in checkpoint_evals],
            "skipped_steps": [item["step"] for item in skipped],
            "summary": path_for_cli(summary_path),
        },
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
