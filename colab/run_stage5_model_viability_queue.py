"""Run bounded no-training viability probes across Qwen model scales.

This is the scale-up orchestration layer around
``colab/run_stage5_model_viability_probe.py``.  It keeps each individual probe
small and restartable, while letting a single Colab session test the next model
sizes that are worth considering for recurrent training.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RUN_ID = os.environ.get("STAGE5_MODEL_QUEUE_RUN_ID") or time.strftime(
    "stage5_model_viability_queue_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
DEFAULT_QUEUE = (
    "qwen_3b=Qwen/Qwen2.5-3B-Instruct|auto|1,2|22|32|32|float32;"
    "qwen_7b=Qwen/Qwen2.5-7B-Instruct|auto|1,2|39|24|24|float32"
)
QUEUE_SPECS = os.environ.get("STAGE5_MODEL_QUEUE_MODELS", DEFAULT_QUEUE)
ALLOW_INSUFFICIENT_VRAM = os.environ.get(
    "STAGE5_MODEL_QUEUE_ALLOW_INSUFFICIENT_VRAM", "0"
).strip().lower() in {"1", "true", "yes", "y"}
CONTINUE_ON_FAILURE = os.environ.get("STAGE5_MODEL_QUEUE_CONTINUE_ON_FAILURE", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
PUSH_RESULTS = os.environ.get("STAGE5_MODEL_QUEUE_PUSH", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}


@dataclass(frozen=True)
class ModelProbeSpec:
    label: str
    model_name: str
    layer_split: str = "auto"
    loops: str = "1,2"
    min_vram_gb: float | None = None
    arc_easy_limit: str = "32"
    arc_challenge_limit: str = "32"
    identity_dtype: str = "float32"


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def parse_model_queue(value: str) -> list[ModelProbeSpec]:
    specs: list[ModelProbeSpec] = []
    for item in [part.strip() for part in value.split(";") if part.strip()]:
        if "=" not in item:
            raise ValueError(f"Invalid model queue entry {item!r}; expected label=model|...")
        label, rest = item.split("=", 1)
        fields = [field.strip() for field in rest.split("|")]
        if not label.strip() or not fields[0]:
            raise ValueError(f"Invalid model queue entry {item!r}; label and model are required")
        min_vram = None
        if len(fields) > 3 and fields[3]:
            min_vram = float(fields[3])
        specs.append(
            ModelProbeSpec(
                label=label.strip(),
                model_name=fields[0],
                layer_split=fields[1] if len(fields) > 1 and fields[1] else "auto",
                loops=fields[2] if len(fields) > 2 and fields[2] else "1,2",
                min_vram_gb=min_vram,
                arc_easy_limit=fields[4] if len(fields) > 4 and fields[4] else "32",
                arc_challenge_limit=fields[5] if len(fields) > 5 and fields[5] else "32",
                identity_dtype=fields[6] if len(fields) > 6 and fields[6] else "float32",
            )
        )
    labels = [spec.label for spec in specs]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Duplicate model labels in STAGE5_MODEL_QUEUE_MODELS: {labels}")
    if not specs:
        raise ValueError("STAGE5_MODEL_QUEUE_MODELS must contain at least one model")
    return specs


def attached_gpu_memory_gb() -> float | None:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    values: list[float] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(line) / 1024.0)
        except ValueError:
            continue
    return max(values) if values else None


def should_skip_for_vram(
    spec: ModelProbeSpec,
    *,
    available_vram_gb: float | None,
    allow_insufficient_vram: bool = False,
) -> str | None:
    if spec.min_vram_gb is None or allow_insufficient_vram:
        return None
    if available_vram_gb is None:
        return f"no GPU VRAM reading; requires >= {spec.min_vram_gb:g} GB"
    if available_vram_gb + 0.25 < spec.min_vram_gb:
        return (
            f"available VRAM {available_vram_gb:.1f} GB is below "
            f"required {spec.min_vram_gb:g} GB"
        )
    return None


def run(cmd: list[str], *, log_name: str, env: dict[str, str] | None = None, check: bool = True):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
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
    proc = subprocess.CompletedProcess(cmd, process.wait(), stdout=stdout, stderr=None)
    (RUN_DIR / log_name).write_text(stdout, encoding="utf-8")
    if check and proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout)
    return proc


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def mean_loop_delta(summary: dict[str, Any], loop: str) -> int | None:
    deltas: list[int] = []
    for score_targets in (summary.get("comparisons") or {}).values():
        for loops in score_targets.values():
            aggregate = (loops.get(loop) or {}).get("mean")
            if isinstance(aggregate, dict) and isinstance(aggregate.get("delta"), int):
                deltas.append(int(aggregate["delta"]))
    return min(deltas) if deltas else None


def assess_child(summary: dict[str, Any] | None, returncode: int) -> dict[str, Any]:
    if not summary:
        return {
            "status": "failed_without_summary",
            "returncode": returncode,
            "promote_to_training_probe": False,
        }
    identity_passed = bool((summary.get("identity") or {}).get("passed"))
    loop1_min_delta = mean_loop_delta(summary, "1")
    loop2_min_delta = mean_loop_delta(summary, "2")
    promote = bool(identity_passed and loop1_min_delta is not None and loop1_min_delta >= -1)
    if not identity_passed:
        status = "identity_failed"
    elif loop1_min_delta is None:
        status = "missing_loop1_comparison"
    elif loop1_min_delta < -1:
        status = "loop1_regression_too_large"
    else:
        status = "viable_for_training_probe"
    return {
        "status": status,
        "returncode": returncode,
        "identity_passed": identity_passed,
        "loop1_min_delta": loop1_min_delta,
        "loop2_min_delta": loop2_min_delta,
        "promote_to_training_probe": promote,
    }


def child_env(spec: ModelProbeSpec, run_id: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "STAGE5_MODEL_PROBE_RUN_ID": run_id,
            "STAGE5_MODEL_PROBE_MODEL_NAME": spec.model_name,
            "STAGE5_MODEL_PROBE_MODEL_LABEL": spec.label,
            "STAGE5_MODEL_PROBE_LAYER_SPLIT": spec.layer_split,
            "STAGE5_MODEL_PROBE_LOOPS": spec.loops,
            "STAGE5_MODEL_PROBE_ARC_EASY_LIMIT": spec.arc_easy_limit,
            "STAGE5_MODEL_PROBE_ARC_CHALLENGE_LIMIT": spec.arc_challenge_limit,
            "STAGE5_MODEL_PROBE_SCORE_TARGETS": os.environ.get(
                "STAGE5_MODEL_QUEUE_SCORE_TARGETS", "label,content_question_only"
            ),
            "STAGE5_MODEL_PROBE_IDENTITY_DTYPE": spec.identity_dtype,
            "STAGE5_MODEL_PROBE_IDENTITY_ATTN": os.environ.get("STAGE5_MODEL_QUEUE_IDENTITY_ATTN", "eager"),
            "STAGE5_MODEL_PROBE_EVAL_DTYPE": os.environ.get("STAGE5_MODEL_QUEUE_EVAL_DTYPE", "bfloat16"),
            "STAGE5_MODEL_PROBE_ADAPTER_DTYPE": os.environ.get("STAGE5_MODEL_QUEUE_ADAPTER_DTYPE", "float32"),
            "STAGE5_MODEL_PROBE_DEVICE": os.environ.get("STAGE5_MODEL_QUEUE_DEVICE", "cuda"),
            "STAGE5_MODEL_PROBE_PUSH": os.environ.get("STAGE5_MODEL_QUEUE_CHILD_PUSH", "1"),
        }
    )
    return env


def write_summary(payload: dict[str, Any]) -> Path:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    summary = RUN_DIR / "summary.json"
    summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"# Stage 5 Model Viability Queue - {RUN_ID}",
        "",
        f"- Available VRAM GB: `{payload['available_vram_gb']}`",
        f"- Allow insufficient VRAM: `{payload['allow_insufficient_vram']}`",
        "",
        "## Results",
        "",
    ]
    for result in payload["results"]:
        spec = result["spec"]
        assessment = result.get("assessment") or {}
        lines.append(
            f"- `{spec['label']}` `{spec['model_name']}`: `{result['status']}`; "
            f"assessment `{assessment.get('status')}`; promote "
            f"`{assessment.get('promote_to_training_probe')}`; "
            f"loop1_min_delta `{assessment.get('loop1_min_delta')}`"
        )
    (RUN_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"), flush=True)
    return summary


def update_pointer(summary_path: Path) -> None:
    pointer = ROOT / "config" / "stage5_current_source_summary.txt"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(path_for_cli(summary_path) + "\n", encoding="utf-8")


def commit_results(summary_path: Path) -> None:
    if not PUSH_RESULTS:
        return
    update_pointer(summary_path)
    run(["git", "status", "-sb"], log_name="git_status_before_queue_commit.log", check=False)
    run(["git", "add", "-f", path_for_cli(RUN_DIR)], log_name="git_add_queue.log", check=False)
    run(["git", "add", "-f", "config/stage5_current_source_summary.txt"], log_name="git_add_pointer.log", check=False)
    status = run(["git", "diff", "--cached", "--quiet"], log_name="git_diff_cached_queue.log", check=False)
    if status.returncode == 0:
        return
    run(["git", "commit", "-m", f"Record Stage 5 model viability queue {RUN_ID}"], log_name="git_commit_queue.log")
    push = run(["git", "push", "origin", "main"], log_name="git_push_queue.log", check=False)
    if push.returncode == 0:
        return
    run(["git", "pull", "--rebase", "origin", "main"], log_name="git_pull_rebase_queue.log")
    run(["git", "push", "origin", "main"], log_name="git_push_retry_queue.log")


def main() -> int:
    started = time.time()
    specs = parse_model_queue(QUEUE_SPECS)
    available_vram_gb = attached_gpu_memory_gb()
    results: list[dict[str, Any]] = []
    overall_returncode = 0
    for spec in specs:
        skip_reason = should_skip_for_vram(
            spec,
            available_vram_gb=available_vram_gb,
            allow_insufficient_vram=ALLOW_INSUFFICIENT_VRAM,
        )
        spec_payload = {
            "label": spec.label,
            "model_name": spec.model_name,
            "layer_split": spec.layer_split,
            "loops": spec.loops,
            "min_vram_gb": spec.min_vram_gb,
            "arc_easy_limit": spec.arc_easy_limit,
            "arc_challenge_limit": spec.arc_challenge_limit,
            "identity_dtype": spec.identity_dtype,
        }
        if skip_reason:
            results.append({"spec": spec_payload, "status": "skipped", "skip_reason": skip_reason})
            continue
        child_run_id = f"stage5_model_viability_{safe_id(spec.label)}_{time.strftime('%Y%m%d_%H%M%S')}"
        proc = run(
            [sys.executable, "colab/run_stage5_model_viability_probe.py"],
            env=child_env(spec, child_run_id),
            log_name=f"{child_run_id}.log",
            check=False,
        )
        child_summary_path = ROOT / "outputs" / "stage5" / child_run_id / "summary.json"
        child_summary = load_json(child_summary_path)
        assessment = assess_child(child_summary, proc.returncode)
        result = {
            "spec": spec_payload,
            "status": "completed" if proc.returncode == 0 else "failed",
            "returncode": proc.returncode,
            "summary_path": path_for_cli(child_summary_path) if child_summary_path.exists() else None,
            "assessment": assessment,
        }
        results.append(result)
        if proc.returncode != 0:
            overall_returncode = proc.returncode
            if not CONTINUE_ON_FAILURE:
                break

    payload = {
        "kind": "stage5_model_viability_queue",
        "run_id": RUN_ID,
        "queue_specs": QUEUE_SPECS,
        "available_vram_gb": available_vram_gb,
        "allow_insufficient_vram": ALLOW_INSUFFICIENT_VRAM,
        "continue_on_failure": CONTINUE_ON_FAILURE,
        "results": results,
        "elapsed_seconds": time.time() - started,
    }
    summary_path = write_summary(payload)
    commit_results(summary_path)
    return overall_returncode


if __name__ == "__main__":
    raise SystemExit(main())
