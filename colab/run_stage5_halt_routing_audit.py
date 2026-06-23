"""Audit halt-depth routing over the full positive curriculum shard.

Curriculum SFT validation splits can be small. This diagnostic evaluates a
finished recurrent checkpoint over every positive SFT row from its source
curriculum and reports the direct/deep loop separation without doing any
training.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RUN_ID = os.environ.get("STAGE5_HALT_ROUTING_AUDIT_RUN_ID") or time.strftime(
    "stage5_halt_routing_audit_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
SOURCE_SUMMARY = os.environ.get("STAGE5_HALT_ROUTING_AUDIT_SOURCE_SUMMARY", "").strip()
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
DTYPE = os.environ.get("DTYPE", "bfloat16")
ADAPTER_DTYPE = os.environ.get("ADAPTER_DTYPE", "float32")
DEVICE = os.environ.get("DEVICE", "cuda")
PUSH_RESULTS = os.environ.get("STAGE5_HALT_ROUTING_AUDIT_PUSH", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_path(value: str | Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run(cmd: list[str], *, check: bool = True, log_name: str | None = None) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(map(str, cmd)), flush=True)
    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
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
        (RUN_DIR / log_name).write_text(stdout, encoding="utf-8")
    if check and proc.returncode:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}")
    return proc


def current_source_summary() -> Path:
    pointer = ROOT / "config" / "stage5_current_source_summary.txt"
    if SOURCE_SUMMARY:
        return resolve_path(SOURCE_SUMMARY)
    if pointer.exists():
        for line in pointer.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return resolve_path(stripped)
    raise FileNotFoundError("Set STAGE5_HALT_ROUTING_AUDIT_SOURCE_SUMMARY or config/stage5_current_source_summary.txt")


def update_current_source_summary(summary_path: Path) -> None:
    pointer = ROOT / "config" / "stage5_current_source_summary.txt"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(path_for_cli(summary_path) + "\n", encoding="utf-8")


def metric_lines(stdout: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        key, raw = line.split("=", 1)
        try:
            metrics[key.strip()] = float(raw.strip())
        except ValueError:
            continue
    return metrics


def grouped(metrics: dict[str, float], *, group_field: str = "curriculum_mode") -> dict[str, dict[str, float]]:
    prefix = f"group/{group_field}/"
    result: dict[str, dict[str, float]] = {}
    for key, value in metrics.items():
        if not key.startswith(prefix):
            continue
        tail = key[len(prefix):]
        group, sep, metric = tail.partition("/")
        if sep:
            result.setdefault(group, {})[metric] = value
    return result


def source_positive_sft(payload: dict[str, Any]) -> Path:
    dataset = payload.get("dataset") or {}
    if dataset.get("source_positive_sft"):
        return resolve_path(str(dataset["source_positive_sft"]))
    config = payload.get("config") or {}
    if config.get("work_dir"):
        return resolve_path(str(config["work_dir"])) / "positive_sft.jsonl"
    raise KeyError("source summary lacks dataset.source_positive_sft and config.work_dir")


def commit_results() -> None:
    if not PUSH_RESULTS:
        return
    run(["git", "config", "user.email", "colab-runner@local"], check=False)
    run(["git", "config", "user.name", "Colab Runner"], check=False)
    run(["git", "add", "-f", path_for_cli(RUN_DIR)], check=False)
    pointer = ROOT / "config" / "stage5_current_source_summary.txt"
    if pointer.exists():
        run(["git", "add", "-f", path_for_cli(pointer)], check=False)
    if run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("No halt routing audit outputs changed.", flush=True)
        return
    run(["git", "commit", "-m", f"Record Stage 5 halt routing audit {RUN_ID}"])
    run(["git", "pull", "--rebase", "origin", "main"], check=False)
    run(["git", "push", "origin", "main"], check=False)


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    source_summary = current_source_summary()
    payload = read_json(source_summary)
    checkpoint = resolve_path(str(payload["phase1_checkpoint"]))
    positive_sft = source_positive_sft(payload)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if not positive_sft.exists():
        raise FileNotFoundError(positive_sft)

    proc = run(
        [
            sys.executable,
            "eval/eval_jsonl.py",
            "--model_name",
            MODEL_NAME,
            "--data_jsonl",
            path_for_cli(positive_sft),
            "--checkpoint",
            path_for_cli(checkpoint),
            "--split",
            "6,18",
            "--max_loops",
            str((payload.get("config") or {}).get("max_loops", 4)),
            "--max_length",
            str((payload.get("config") or {}).get("max_length", 512)),
            "--beta",
            str((payload.get("config") or {}).get("beta", 0.08)),
            "--dtype",
            DTYPE,
            "--adapter_dtype",
            ADAPTER_DTYPE,
            "--device",
            DEVICE,
            "--group_by_field",
            "curriculum_mode",
        ],
        log_name="full_positive_sft_eval.log",
    )
    metrics = metric_lines(proc.stdout)
    by_mode = grouped(metrics)
    direct = by_mode.get("direct", {}).get("mean_expected_loops")
    deep = by_mode.get("deep_narrow", {}).get("mean_expected_loops")
    margin = None if direct is None or deep is None else float(deep) - float(direct)
    summary = {
        "run_id": RUN_ID,
        "kind": "stage5_halt_routing_audit",
        "source_summary": path_for_cli(source_summary),
        "checkpoint": path_for_cli(checkpoint),
        "positive_sft": path_for_cli(positive_sft),
        "metrics": metrics,
        "by_mode": by_mode,
        "depth_gradient": {
            "direct_mean_expected_loops": direct,
            "deep_narrow_mean_expected_loops": deep,
            "margin": margin,
            "observed": None if margin is None else margin >= 0.25,
        },
    }
    write_json(RUN_DIR / "summary.json", summary)
    update_current_source_summary(RUN_DIR / "summary.json")
    lines = [
        f"# Stage 5 Halt Routing Audit - {RUN_ID}",
        "",
        f"- Source summary: `{summary['source_summary']}`",
        f"- Checkpoint: `{summary['checkpoint']}`",
        f"- Positive SFT: `{summary['positive_sft']}`",
        f"- Mean loops: `{metrics.get('mean_expected_loops')}`",
        f"- Direct loops: `{direct}`",
        f"- Deep-narrow loops: `{deep}`",
        f"- Margin: `{margin}`",
        f"- Depth gradient observed: `{summary['depth_gradient']['observed']}`",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"), flush=True)
    commit_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
