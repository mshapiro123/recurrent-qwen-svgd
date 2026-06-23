"""Build a small capability-ladder probe from ARC MCQ model-scale scores.

This is a bounded GPU scoring stage, not a training run. It asks whether a
simple capability ladder exists on an ARC slice:

* Qwen 0.5B correct -> depth 1 / direct preservation row;
* Qwen 0.5B misses and Qwen 1.5B correct -> depth 2 row;
* Qwen 0.5B and 1.5B miss and Qwen 3B correct -> depth 3 row.

The generated rows use MCQ predictions as minimal answer-only traces, so the
artifact is a depth-label probe and a candidate SFT shard, not a final rich
reasoning-trace curriculum.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.mcq_debias import aggregate_permutation_scores, cyclic_permutation_rows, read_jsonl, write_jsonl  # noqa: E402
from training.build_capability_ladder_curriculum import parse_model_ladder  # noqa: E402


RUN_ID = os.environ.get("STAGE5_CAPABILITY_LADDER_RUN_ID") or time.strftime(
    "stage5_capability_ladder_mcq_probe_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
PRIVATE_DATA_DIR = ROOT / "data" / "stage5_capability_ladder" / RUN_ID
WORK_DIR = ROOT / "data" / "curriculum" / RUN_ID

ARC_CONFIG = os.environ.get("STAGE5_CAPABILITY_LADDER_ARC_CONFIG", "ARC-Challenge")
ARC_SPLIT = os.environ.get("STAGE5_CAPABILITY_LADDER_ARC_SPLIT", "train")
ARC_LIMIT = int(os.environ.get("STAGE5_CAPABILITY_LADDER_ARC_LIMIT", "48"))
ARC_SEED = int(os.environ.get("STAGE5_CAPABILITY_LADDER_ARC_SEED", "0"))
SCORE_MODE = os.environ.get("STAGE5_CAPABILITY_LADDER_SCORE_MODE", "content_question_only")
MODEL_SPECS = os.environ.get(
    "STAGE5_CAPABILITY_LADDER_MODELS",
    ",".join(
        [
            "qwen_0_5b=Qwen/Qwen2.5-0.5B-Instruct",
            "qwen_1_5b=Qwen/Qwen2.5-1.5B-Instruct",
            "qwen_3b=Qwen/Qwen2.5-3B-Instruct",
        ]
    ),
)
BASE_KEY = os.environ.get("STAGE5_CAPABILITY_LADDER_BASE_KEY", "qwen_0_5b")
MID_KEY = os.environ.get("STAGE5_CAPABILITY_LADDER_MID_KEY", "qwen_1_5b")
HIGH_KEYS = os.environ.get("STAGE5_CAPABILITY_LADDER_HIGH_KEYS", "qwen_3b")
HIGH_TARGET_LOOP = int(os.environ.get("STAGE5_CAPABILITY_LADDER_HIGH_TARGET_LOOP", "3"))
MODEL_LADDER = os.environ.get("STAGE5_CAPABILITY_LADDER_MODEL_LADDER", "").strip()
MIN_POSITIVE_ROWS = int(os.environ.get("STAGE5_CAPABILITY_LADDER_MIN_POSITIVE_ROWS", "1"))
MIN_DIRECT_ROWS = int(os.environ.get("STAGE5_CAPABILITY_LADDER_MIN_DIRECT_ROWS", "1"))
MIN_DEEP_ROWS = int(os.environ.get("STAGE5_CAPABILITY_LADDER_MIN_DEEP_ROWS", "1"))
DTYPE = os.environ.get("DTYPE", "bfloat16")
DEVICE = os.environ.get("DEVICE", "cuda")
PUSH_RESULTS = os.environ.get("STAGE5_CAPABILITY_LADDER_PUSH", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
BACKUP_DRIVE = os.environ.get("STAGE5_CAPABILITY_LADDER_BACKUP_DRIVE", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_name: str


@dataclass(frozen=True)
class ScoreConfig:
    public_name: str
    prompt_style: str
    score_target: str
    cyclic: bool = False


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def current_source_summary_file() -> Path:
    return ROOT / "config" / "stage5_current_source_summary.txt"


def update_current_source_summary(summary_path: Path) -> Path:
    pointer = current_source_summary_file()
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(path_for_cli(summary_path) + "\n", encoding="utf-8")
    return pointer


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_model_specs(value: str) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for item in parse_csv(value):
        if "=" not in item:
            raise ValueError(f"Invalid model spec {item!r}; expected key=model_name.")
        key, model_name = item.split("=", 1)
        key = key.strip()
        model_name = model_name.strip()
        if not key or not model_name:
            raise ValueError(f"Invalid model spec {item!r}; expected nonempty key and model_name.")
        specs.append(ModelSpec(key=key, model_name=model_name))
    keys = [spec.key for spec in specs]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate model keys in STAGE5_CAPABILITY_LADDER_MODELS: {keys}")
    return specs


def active_model_ladder() -> list[dict[str, Any]] | None:
    if not MODEL_LADDER:
        return None
    return parse_model_ladder(MODEL_LADDER)


def required_model_keys() -> list[str]:
    ladder = active_model_ladder()
    if ladder:
        return [str(entry["key"]) for entry in ladder]
    return sorted({BASE_KEY, MID_KEY, *parse_csv(HIGH_KEYS)})


def max_target_loop() -> int:
    ladder = active_model_ladder()
    if ladder:
        return max(int(entry["target_loop_count"]) for entry in ladder)
    return max(HIGH_TARGET_LOOP, 3)


def score_config(mode: str) -> ScoreConfig:
    if mode == "label":
        return ScoreConfig(public_name="label", prompt_style="with_options", score_target="label")
    if mode == "content_question_only":
        return ScoreConfig(
            public_name="content_question_only",
            prompt_style="question_only",
            score_target="option_text",
        )
    if mode == "cyclic_label_aggregated":
        return ScoreConfig(
            public_name="cyclic_label_aggregated",
            prompt_style="with_options",
            score_target="label",
            cyclic=True,
        )
    raise ValueError(
        "STAGE5_CAPABILITY_LADDER_SCORE_MODE must be one of "
        "label, content_question_only, cyclic_label_aggregated."
    )


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object.")
    return payload


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


def summarize_score_rows(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    total = len(rows)
    correct = sum(1 for row in rows if row.get("hit") is True or row.get("correct") is True)
    return {
        "path": path_for_cli(path),
        "rows": total,
        "correct": correct,
        "accuracy": correct / max(total, 1),
        "prediction_counts": count_field(rows, "prediction"),
        "answer_counts": count_field(rows, "answer"),
    }


def count_field(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(field))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def prepare_arc_tasks() -> Path:
    path = PRIVATE_DATA_DIR / "arc_mcq.jsonl"
    run(
        [
            sys.executable,
            "eval/prepare_arc_mcq.py",
            "--config",
            ARC_CONFIG,
            "--split",
            ARC_SPLIT,
            "--limit",
            str(ARC_LIMIT),
            "--seed",
            str(ARC_SEED),
            "--output_jsonl",
            path_for_cli(path),
        ],
        log_name="prepare_arc_mcq.log",
    )
    return path


def cyclic_task_path(tasks_jsonl: Path) -> Path:
    path = PRIVATE_DATA_DIR / "arc_mcq_cyclic_permuted.jsonl"
    write_jsonl(path, cyclic_permutation_rows(read_jsonl(tasks_jsonl)))
    return path


def score_model(spec: ModelSpec, tasks_jsonl: Path, config: ScoreConfig, *, seed: int) -> Path:
    public_output = RUN_DIR / f"{spec.key}_{config.public_name}.jsonl"
    public_output.unlink(missing_ok=True)
    eval_input = tasks_jsonl
    eval_output = public_output
    if config.cyclic:
        eval_input = cyclic_task_path(tasks_jsonl)
        eval_output = RUN_DIR / f"{spec.key}_{config.public_name}_raw.jsonl"
        eval_output.unlink(missing_ok=True)
    run(
        [
            sys.executable,
            "eval/eval_mcq.py",
            "--model_name",
            spec.model_name,
            "--data_jsonl",
            path_for_cli(eval_input),
            "--mode",
            "base",
            "--output_jsonl",
            path_for_cli(eval_output),
            "--prompt_style",
            config.prompt_style,
            "--score_target",
            config.score_target,
            "--aggregates",
            "mean",
            "--dtype",
            DTYPE,
            "--device",
            DEVICE,
            "--seed",
            str(seed),
        ],
        log_name=f"score_{spec.key}_{config.public_name}.log",
    )
    if not config.cyclic:
        return public_output

    permutation_rows = read_jsonl(eval_input)
    scored_rows = read_jsonl(eval_output)
    output_rows = aggregate_permutation_scores(scored_rows, permutation_rows)
    for row in output_rows:
        row["mode"] = "base"
        row["model_name"] = spec.model_name
        row["prompt_style"] = "with_options"
        row["score_target"] = "cyclic_label_aggregated"
        row["aggregate"] = "permutation_mean"
    write_jsonl(public_output, output_rows)
    return public_output


def merge_score_rows(tasks_jsonl: Path, result_paths: dict[str, Path]) -> Path:
    output = PRIVATE_DATA_DIR / "scored_capability_rows.jsonl"
    cmd = [
        sys.executable,
        "training/merge_capability_score_rows.py",
        "--tasks_jsonl",
        path_for_cli(tasks_jsonl),
        "--output_jsonl",
        path_for_cli(output),
        "--verified_by",
        "benchmark_ground_truth",
        "--assume_decontaminated",
        "--prediction_as_solution",
    ]
    for key, path in result_paths.items():
        cmd.extend(["--result", f"{key}={path_for_cli(path)}"])
    run(cmd, log_name="merge_capability_score_rows.log")
    return output


def build_capability_ladder(scored_jsonl: Path) -> Path:
    cmd = [
        sys.executable,
        "training/build_capability_ladder_curriculum.py",
        "--input_jsonl",
        path_for_cli(scored_jsonl),
        "--work_dir",
        path_for_cli(WORK_DIR),
        "--allow_answer_only",
        "--assume_decontaminated",
    ]
    if MODEL_LADDER:
        cmd.extend(["--model_ladder", MODEL_LADDER])
    else:
        cmd.extend(
            [
                "--base_key",
                BASE_KEY,
                "--mid_key",
                MID_KEY,
                "--high_keys",
                HIGH_KEYS,
                "--high_target_loop",
                str(HIGH_TARGET_LOOP),
            ]
        )
    run(cmd, log_name="build_capability_ladder_curriculum.log")
    return WORK_DIR / "summary.json"


def gate_capability_ladder(summary_json: Path) -> Path:
    output = RUN_DIR / "curriculum_sft_gate.json"
    output_md = RUN_DIR / "curriculum_sft_gate.md"
    run(
        [
            sys.executable,
            "training/check_curriculum_sft_gate.py",
            "--summary_json",
            path_for_cli(summary_json),
            "--output_json",
            path_for_cli(output),
            "--output_md",
            path_for_cli(output_md),
            "--min_positive_rows",
            str(MIN_POSITIVE_ROWS),
            "--min_mode_rows",
            f"direct={MIN_DIRECT_ROWS},deep_narrow={MIN_DEEP_ROWS}",
            "--max_loop_target",
            str(max_target_loop()),
            "--allow_cross_model_only_answers",
        ],
        check=False,
        log_name="check_curriculum_sft_gate.log",
    )
    return output


def probe_status(curriculum_summary: dict[str, Any], gate_payload: dict[str, Any] | None) -> str:
    counts = curriculum_summary.get("counts") if isinstance(curriculum_summary.get("counts"), dict) else {}
    typed = int(counts.get("typed_records") or 0)
    mode_counts = counts.get("mode_counts") if isinstance(counts.get("mode_counts"), dict) else {}
    direct = int(mode_counts.get("direct") or 0)
    deep = int(mode_counts.get("deep_narrow") or 0)
    if typed == 0:
        return "capability_ladder_probe_no_signal"
    if gate_payload and gate_payload.get("go") is True:
        return "capability_ladder_probe_gate_ready"
    if direct > 0 and deep > 0:
        return "capability_ladder_probe_needs_review"
    return "capability_ladder_probe_sparse"


def backup_to_drive(paths: list[Path]) -> dict[str, Any]:
    if not BACKUP_DRIVE:
        return {"enabled": False}
    drive_root = Path("/content/drive/MyDrive")
    if not drive_root.exists():
        return {"enabled": True, "available": False}
    dest_root = drive_root / "recurrent-qwen-svgd" / "stage5_capability_ladder" / RUN_ID
    copied: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        dest = dest_root / path.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        if path.is_dir():
            shutil.copytree(path, dest)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        copied.append(str(dest))
    return {"enabled": True, "available": True, "dest_root": str(dest_root), "copied": copied}


def safe_commit(summary_path: Path) -> None:
    if not PUSH_RESULTS:
        return
    pointer = update_current_source_summary(summary_path)
    run(["git", "add", "-f", path_for_cli(RUN_DIR), path_for_cli(pointer)], check=False, log_name="git_add.log")
    diff = run(["git", "diff", "--cached", "--quiet"], check=False)
    if diff.returncode == 0:
        print("No safe result changes to commit.")
        return
    run(["git", "commit", "-m", "Record capability-ladder MCQ probe"], check=True, log_name="git_commit.log")
    push = run(["git", "push", "origin", "main"], check=False, log_name="git_push.log")
    if push.returncode == 0:
        return
    run(["git", "pull", "--rebase", "origin", "main"], check=True, log_name="git_pull_rebase.log")
    run(["git", "push", "origin", "main"], check=True, log_name="git_push_retry.log")


def write_probe_summary(
    *,
    tasks_jsonl: Path,
    score_paths: dict[str, Path],
    scored_jsonl: Path,
    curriculum_summary_path: Path,
    gate_path: Path,
    drive_backup: dict[str, Any],
) -> Path:
    curriculum_summary = read_json(curriculum_summary_path)
    gate_payload = read_json(gate_path) if gate_path.exists() else None
    score_summaries = {key: summarize_score_rows(path) for key, path in score_paths.items()}
    payload = {
        "run_id": RUN_ID,
        "kind": "stage5_capability_ladder_mcq_probe",
        "status": probe_status(curriculum_summary, gate_payload),
        "score_mode": SCORE_MODE,
        "arc": {
            "config": ARC_CONFIG,
            "split": ARC_SPLIT,
            "limit": ARC_LIMIT,
            "seed": ARC_SEED,
            "tasks_jsonl": path_for_cli(tasks_jsonl),
        },
        "models": {spec.key: spec.model_name for spec in parse_model_specs(MODEL_SPECS)},
        "ladder_keys": {
            "base_key": BASE_KEY,
            "mid_key": MID_KEY,
            "high_keys": parse_csv(HIGH_KEYS),
            "high_target_loop": HIGH_TARGET_LOOP,
            "model_ladder": active_model_ladder(),
        },
        "score_summaries": score_summaries,
        "curriculum": {
            "summary_json": path_for_cli(curriculum_summary_path),
            "work_dir": path_for_cli(WORK_DIR),
            "counts": curriculum_summary.get("counts", {}),
        },
        "gate": gate_payload,
        "artifacts": {
            "run_dir": path_for_cli(RUN_DIR),
            "scored_capability_rows": path_for_cli(scored_jsonl),
            "curriculum_summary": path_for_cli(curriculum_summary_path),
            "curriculum_gate": path_for_cli(gate_path),
        },
        "drive_backup": drive_backup,
        "caveat": (
            "This probe uses answer-only MCQ predictions as minimal traces. Use it to "
            "select/size a depth curriculum, then replace or enrich rows with verified "
            "reasoning traces before claiming reasoning SFT quality."
        ),
    }
    payload["next_action"] = (
        "If status is gate_ready or needs_review, inspect tier counts and use richer "
        "trace generation for the deep rows before recurrent SFT."
    )
    summary = RUN_DIR / "summary.json"
    write_json(summary, payload)
    (RUN_DIR / "summary.md").write_text(markdown_summary(payload), encoding="utf-8")
    return summary


def markdown_summary(payload: dict[str, Any]) -> str:
    counts = payload.get("curriculum", {}).get("counts", {})
    lines = [
        f"# Capability-Ladder MCQ Probe: {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Score mode: `{payload['score_mode']}`",
        f"- ARC: `{payload['arc']['config']}` `{payload['arc']['split']}` limit `{payload['arc']['limit']}`",
        f"- Typed records: `{counts.get('typed_records', 0)}`",
        f"- Positive SFT rows: `{counts.get('positive_sft_rows', 0)}`",
        f"- Mode counts: `{counts.get('mode_counts', {})}`",
        f"- Target loop counts: `{counts.get('target_loop_counts', {})}`",
        "",
        "## Model Scores",
    ]
    for key, summary in sorted(payload.get("score_summaries", {}).items()):
        lines.append(
            f"- `{key}`: `{summary['correct']}/{summary['rows']}` "
            f"accuracy `{summary['accuracy']:.4f}`"
        )
    lines.extend(["", "## Caveat", "", payload["caveat"], ""])
    return "\n".join(lines)


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    PRIVATE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config = score_config(SCORE_MODE)
    specs = parse_model_specs(MODEL_SPECS)
    available_keys = {spec.key for spec in specs}
    required_keys = set(required_model_keys())
    missing = sorted(required_keys - available_keys)
    if missing:
        raise SystemExit(f"Missing model specs for ladder keys: {missing}")

    print(f"run_id={RUN_ID}")
    print(f"score_mode={SCORE_MODE}")
    print(f"arc_config={ARC_CONFIG} split={ARC_SPLIT} limit={ARC_LIMIT}")
    print(f"models={','.join(f'{spec.key}={spec.model_name}' for spec in specs)}")

    tasks_jsonl = prepare_arc_tasks()
    score_paths: dict[str, Path] = {}
    for index, spec in enumerate(specs):
        score_paths[spec.key] = score_model(spec, tasks_jsonl, config, seed=ARC_SEED + index)
    scored_jsonl = merge_score_rows(tasks_jsonl, score_paths)
    curriculum_summary_path = build_capability_ladder(scored_jsonl)
    gate_path = gate_capability_ladder(curriculum_summary_path)
    drive_backup = backup_to_drive([RUN_DIR, PRIVATE_DATA_DIR, WORK_DIR])
    summary_path = write_probe_summary(
        tasks_jsonl=tasks_jsonl,
        score_paths=score_paths,
        scored_jsonl=scored_jsonl,
        curriculum_summary_path=curriculum_summary_path,
        gate_path=gate_path,
        drive_backup=drive_backup,
    )
    print(f"summary_json={path_for_cli(summary_path)}")
    print(f"status={read_json(summary_path)['status']}")
    safe_commit(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
