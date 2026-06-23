"""Run a bounded MCQ content/cyclic surface-alignment repair.

This is the follow-up to the surface-mismatch diagnosis.  It does not replace
the broader competence-preserving pipeline; it tests a more specific repair:
teach the direct/content MCQ surface to agree with the cyclic surface on rows
where cyclic aggregation already recovers the correct answer.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.colab_auth import ensure_hf_token_from_colab  # noqa: E402
from colab.run_stage5_benchmark_suite import resolve_checkpoint  # noqa: E402


RUN_ID = os.environ.get("STAGE5_SURFACE_ALIGN_RUN_ID") or time.strftime(
    "stage5_surface_alignment_repair_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
PRIVATE_DATA_DIR = ROOT / "data" / "stage5_surface_alignment" / RUN_ID
SOURCE_SUMMARY = Path(
    os.environ.get(
        "STAGE5_SURFACE_ALIGN_SOURCE_SUMMARY",
        "outputs/stage5/stage5_traced_sft_direct_preservation_20260623_scale64_confirm_assessment/summary.json",
    )
)
if not SOURCE_SUMMARY.is_absolute():
    SOURCE_SUMMARY = ROOT / SOURCE_SUMMARY

ARC_EASY_LIMIT = int(os.environ.get("STAGE5_SURFACE_ALIGN_ARC_EASY_LIMIT", "256"))
MAX_STEPS = int(os.environ.get("STAGE5_SURFACE_ALIGN_MAX_STEPS", "50"))
SAVE_EVERY = int(os.environ.get("STAGE5_SURFACE_ALIGN_SAVE_EVERY", "25"))
LEARNING_RATE = os.environ.get("STAGE5_SURFACE_ALIGN_LR", "5e-7")
DISTILL_WEIGHT = os.environ.get("STAGE5_SURFACE_ALIGN_DISTILL_WEIGHT", "0.05")
CONTENT_REPEAT = int(os.environ.get("STAGE5_SURFACE_ALIGN_CONTENT_REPEAT", "2"))
CYCLIC_REPEAT = int(os.environ.get("STAGE5_SURFACE_ALIGN_CYCLIC_REPEAT", "1"))
CYCLIC_ROWS_PER_ITEM_RAW = os.environ.get("STAGE5_SURFACE_ALIGN_CYCLIC_ROWS_PER_ITEM", "4").strip()
CYCLIC_ROWS_PER_ITEM = None if CYCLIC_ROWS_PER_ITEM_RAW.lower() in {"", "none", "all"} else int(CYCLIC_ROWS_PER_ITEM_RAW)
INCLUDE_UNRESCUED = os.environ.get("STAGE5_SURFACE_ALIGN_INCLUDE_UNRESCUED", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
PUSH_RESULTS = os.environ.get("STAGE5_SURFACE_ALIGN_PUSH", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def resolve_path(value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def current_source_summary_file() -> Path:
    return ROOT / "config" / "stage5_current_source_summary.txt"


def update_current_source_summary(summary_path: Path) -> Path:
    pointer = current_source_summary_file()
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(path_for_cli(summary_path) + "\n", encoding="utf-8")
    return pointer


def source_benchmark_summary(source_payload: dict[str, Any], source_summary: Path) -> Path:
    if source_payload.get("kind") == "stage5_benchmark_suite":
        return source_summary
    raw = str(source_payload.get("source_summary") or "").strip()
    if raw:
        candidate = resolve_path(raw)
        if candidate.exists() and read_json(candidate).get("kind") == "stage5_benchmark_suite":
            return candidate
    raise ValueError(f"Could not resolve benchmark suite summary from {path_for_cli(source_summary)}")


def arc_easy_eval_file(benchmark_dir: Path, stem: str) -> Path:
    path = benchmark_dir / stem
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def ensure_surface_diagnosis(benchmark_summary: Path) -> Path:
    benchmark_dir = benchmark_summary.parent
    diagnosis = benchmark_dir / "arc_easy_surface_mismatch_diagnosis.json"
    if diagnosis.exists():
        return diagnosis
    run(
        [
            sys.executable,
            "eval/analyze_mcq_surface_mismatch.py",
            "--benchmark",
            f"arc_easy_{benchmark_dir.name}",
            "--base_content",
            path_for_cli(arc_easy_eval_file(benchmark_dir, "arc_easy_base_content_question_only.jsonl")),
            "--candidate_content",
            path_for_cli(arc_easy_eval_file(benchmark_dir, "arc_easy_recurrent_content_question_only.jsonl")),
            "--candidate_cyclic",
            path_for_cli(arc_easy_eval_file(benchmark_dir, "arc_easy_recurrent_cyclic_label_aggregated.jsonl")),
            "--base_cyclic",
            path_for_cli(arc_easy_eval_file(benchmark_dir, "arc_easy_base_cyclic_label_aggregated.jsonl")),
            "--output_json",
            path_for_cli(diagnosis),
            "--output_md",
            path_for_cli(benchmark_dir / "arc_easy_surface_mismatch_diagnosis.md"),
        ],
        log_name="surface_diagnosis.log",
    )
    return diagnosis


def train_config(*, checkpoint: Path, train_jsonl: Path) -> dict[str, Any]:
    return {
        "model_name": os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
        "dtype": os.environ.get("TRAIN_DTYPE", "bfloat16"),
        "adapter_dtype": os.environ.get("ADAPTER_DTYPE", "float32"),
        "layer_split": os.environ.get("LAYER_SPLIT", "6,18"),
        "max_length": 512,
        "max_loops": 4,
        "initial_halt_prob": 0.15,
        "beta": 0.08,
        "batch_size": 1,
        "learning_rate": float(LEARNING_RATE),
        "weight_decay": 0.0,
        "max_grad_norm": 0.3,
        "max_steps": MAX_STEPS,
        "save_every": SAVE_EVERY,
        "log_every": 5,
        "train_on_prompt": False,
        "use_target_loop_control": True,
        "halt_target_nll_weight": 0.05,
        "output_dir": path_for_cli(RUN_DIR / "phase1_surface_align"),
        "resume_from": path_for_cli(checkpoint),
        "lora": {
            "enabled": True,
            "rank": 8,
            "alpha": 16,
            "dropout": 0.0,
        },
        "distillation": {
            "enabled": float(DISTILL_WEIGHT) > 0.0,
            "teacher_model_name": os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
            "dtype": os.environ.get("TRAIN_DTYPE", "bfloat16"),
            "weight": float(DISTILL_WEIGHT),
            "temperature": 2.0,
            "on": "response",
        },
    }


def final_checkpoint() -> Path:
    return RUN_DIR / "phase1_surface_align" / f"phase1_step_{MAX_STEPS}.pt"


def benchmark_and_assess(checkpoint: Path) -> tuple[Path, Path]:
    bench_run_id = f"{RUN_ID}_benchmark"
    assess_run_id = f"{RUN_ID}_assessment"
    bench_env = os.environ.copy()
    bench_env.update(
        {
            "STAGE5_BENCHMARK_SUITE_RUN_ID": bench_run_id,
            "STAGE5_BENCHMARK_SOURCE_SUMMARY": path_for_cli(SOURCE_SUMMARY),
            "STAGE5_BENCHMARK_CHECKPOINT": path_for_cli(checkpoint),
            "STAGE5_BENCHMARKS": "arc_easy,arc_challenge",
            "STAGE5_BENCHMARK_ARC_EASY_LIMIT": str(ARC_EASY_LIMIT),
            "STAGE5_BENCHMARK_ARC_CHALLENGE_LIMIT": str(ARC_EASY_LIMIT),
            "STAGE5_BENCHMARK_SCORE_TARGETS": "content_question_only,cyclic_label_aggregated",
            "STAGE5_BENCHMARK_AGGREGATES": "mean",
            "STAGE5_BENCHMARK_MAX_LOOPS": "4",
            "STAGE5_BENCHMARK_NUM_TRAJECTORIES": "1",
            "STAGE5_BENCHMARK_USE_LEARNED_LOOP_CONTROL": "0",
            "STAGE5_BENCHMARK_PUSH": "1" if PUSH_RESULTS else "0",
        }
    )
    run([sys.executable, "colab/run_stage5_benchmark_suite.py"], env=bench_env, log_name="benchmark.log")
    bench_summary = ROOT / "outputs" / "stage5" / bench_run_id / "summary.json"

    assess_env = os.environ.copy()
    assess_env.update(
        {
            "STAGE5_BENCHMARK_ASSESS_RUN_ID": assess_run_id,
            "STAGE5_BENCHMARK_ASSESS_SCORE_TARGET": "content_question_only",
            "STAGE5_BENCHMARK_ASSESS_AGGREGATE": "mean",
            "STAGE5_BENCHMARK_ASSESS_MIN_ARC_EXAMPLES": str(min(128, ARC_EASY_LIMIT)),
            "STAGE5_BENCHMARK_ASSESS_ALLOWED_NEGATIVE_DELTA": "0",
            "STAGE5_BENCHMARK_ASSESS_PUSH": "1" if PUSH_RESULTS else "0",
        }
    )
    run(
        [
            sys.executable,
            "colab/assess_stage5_benchmark_suite.py",
            "--summary_json",
            path_for_cli(bench_summary),
        ],
        env=assess_env,
        log_name="assessment.log",
    )
    assess_summary = ROOT / "outputs" / "stage5" / assess_run_id / "summary.json"
    return bench_summary, assess_summary


def assess_surface_repair(source_benchmark: Path, repaired_benchmark: Path) -> dict[str, Any]:
    output_json = RUN_DIR / "surface_repair_assessment.json"
    output_md = RUN_DIR / "surface_repair_assessment.md"
    run(
        [
            sys.executable,
            "colab/assess_stage5_surface_repair.py",
            "--source_benchmark_summary",
            path_for_cli(source_benchmark),
            "--repaired_benchmark_summary",
            path_for_cli(repaired_benchmark),
            "--output_json",
            path_for_cli(output_json),
            "--output_md",
            path_for_cli(output_md),
            "--allowed_challenge_regression",
            os.environ.get("STAGE5_SURFACE_ALIGN_ALLOWED_CHALLENGE_REGRESSION", "0"),
        ],
        log_name="surface_repair_assessment.log",
    )
    return read_json(output_json)


def write_summary(payload: dict[str, Any]) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    summary = RUN_DIR / "summary.json"
    summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    update_current_source_summary(summary)
    lines = [
        f"# Stage 5 Surface-Alignment Repair - {RUN_ID}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Source summary: `{payload['source_summary']}`",
        f"- Benchmark source: `{payload['benchmark_source_summary']}`",
        f"- Diagnosis: `{payload['surface_diagnosis']}`",
        f"- Train rows: `{payload['surface_alignment_rows']}`",
        f"- Checkpoint: `{payload.get('checkpoint') or 'not_trained'}`",
        f"- Assessment: `{payload.get('assessment_summary') or 'not_run'}`",
        f"- Surface repair assessment: `{payload.get('surface_repair_assessment_summary') or 'not_run'}`",
        f"- Next step: {payload['next_step']}",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))


def commit_results(extra_paths: list[Path]) -> None:
    if not PUSH_RESULTS:
        return
    paths = [RUN_DIR, current_source_summary_file(), *extra_paths]
    for path in paths:
        if path.exists():
            run(["git", "add", "-f", path_for_cli(path)], check=False)
    if run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("No surface-alignment outputs changed.")
        return
    run(["git", "commit", "-m", f"Record Stage 5 surface-alignment repair {RUN_ID} [skip ci]"])
    run(["git", "pull", "--rebase", "origin", "main"], check=False)
    run(["git", "push", "origin", "main"], check=False)


def main() -> int:
    ensure_hf_token_from_colab()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    PRIVATE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not SOURCE_SUMMARY.exists():
        raise FileNotFoundError(SOURCE_SUMMARY)
    source_payload = read_json(SOURCE_SUMMARY)
    benchmark_summary = source_benchmark_summary(source_payload, SOURCE_SUMMARY)
    benchmark_payload = read_json(benchmark_summary)
    checkpoint = resolve_checkpoint(benchmark_summary, benchmark_payload)
    diagnosis = ensure_surface_diagnosis(benchmark_summary)

    mcq_jsonl = PRIVATE_DATA_DIR / f"arc_easy_validation_{ARC_EASY_LIMIT}.jsonl"
    run(
        [
            sys.executable,
            "eval/prepare_arc_mcq.py",
            "--config",
            "ARC-Easy",
            "--split",
            "validation",
            "--seed",
            "0",
            "--limit",
            str(ARC_EASY_LIMIT),
            "--output_jsonl",
            path_for_cli(mcq_jsonl),
        ],
        log_name="prepare_arc_easy.log",
    )
    train_jsonl = PRIVATE_DATA_DIR / "surface_alignment_train.jsonl"
    surface_summary_json = PRIVATE_DATA_DIR / "surface_alignment_train_summary.json"
    prepare_cmd = [
        sys.executable,
        "training/prepare_mcq_surface_alignment_jsonl.py",
        "--mcq_jsonl",
        path_for_cli(mcq_jsonl),
        "--diagnosis_json",
        path_for_cli(diagnosis),
        "--output_jsonl",
        path_for_cli(train_jsonl),
        "--summary_json",
        path_for_cli(surface_summary_json),
        "--target_loop_count",
        "1",
        "--content_repeat",
        str(CONTENT_REPEAT),
        "--cyclic_repeat",
        str(CYCLIC_REPEAT),
    ]
    if CYCLIC_ROWS_PER_ITEM is not None:
        prepare_cmd.extend(["--cyclic_rows_per_item", str(CYCLIC_ROWS_PER_ITEM)])
    if INCLUDE_UNRESCUED:
        prepare_cmd.append("--include_unrescued")
    run(prepare_cmd, log_name="prepare_surface_alignment.log")
    surface_data_summary = read_json(surface_summary_json)

    if int(surface_data_summary.get("output_rows", 0)) <= 0:
        payload = {
            "kind": "stage5_surface_alignment_repair",
            "run_id": RUN_ID,
            "status": "no_surface_alignment_rows",
            "passed": False,
            "source_summary": path_for_cli(SOURCE_SUMMARY),
            "benchmark_source_summary": path_for_cli(benchmark_summary),
            "surface_diagnosis": path_for_cli(diagnosis),
            "surface_alignment_rows": 0,
            "next_step": "Inspect the surface diagnosis; no trainable surface-alignment rows were selected.",
        }
        write_summary(payload)
        commit_results([diagnosis, diagnosis.with_suffix(".md")])
        return 0

    config_path = RUN_DIR / "surface_alignment_phase1.yaml"
    config_path.write_text(yaml.safe_dump(train_config(checkpoint=checkpoint, train_jsonl=train_jsonl), sort_keys=False), encoding="utf-8")
    run(
        [
            sys.executable,
            "training/train_phase1_ponder.py",
            "--config",
            path_for_cli(config_path),
            "--train_jsonl",
            path_for_cli(train_jsonl),
            "--device",
            os.environ.get("DEVICE", "cuda"),
        ],
        log_name="train_surface_alignment.log",
    )
    checkpoint_path = final_checkpoint()
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    bench_summary, assessment_summary = benchmark_and_assess(checkpoint_path)
    assessment_payload = read_json(assessment_summary)
    surface_repair_payload = assess_surface_repair(benchmark_summary, bench_summary)
    passed = bool(assessment_payload.get("passed"))
    surface_repair_passed = bool(surface_repair_payload.get("passed"))
    payload = {
        "kind": "stage5_surface_alignment_repair",
        "run_id": RUN_ID,
        "status": (
            "surface_alignment_passed"
            if passed and surface_repair_passed
            else "surface_alignment_partial"
            if surface_repair_payload.get("status") == "surface_repair_partial"
            else "surface_alignment_tradeoff"
            if surface_repair_payload.get("status") == "surface_repair_tradeoff"
            else "surface_alignment_not_passed"
        ),
        "passed": passed and surface_repair_passed,
        "source_summary": path_for_cli(SOURCE_SUMMARY),
        "source_status": source_payload.get("status"),
        "benchmark_source_summary": path_for_cli(benchmark_summary),
        "resume_checkpoint": path_for_cli(checkpoint),
        "surface_diagnosis": path_for_cli(diagnosis),
        "surface_alignment_train_jsonl": path_for_cli(train_jsonl),
        "surface_alignment_train_summary": surface_data_summary,
        "surface_alignment_rows": surface_data_summary.get("output_rows"),
        "config": {
            "arc_easy_limit": ARC_EASY_LIMIT,
            "max_steps": MAX_STEPS,
            "learning_rate": LEARNING_RATE,
            "distill_weight": DISTILL_WEIGHT,
            "content_repeat": CONTENT_REPEAT,
            "cyclic_repeat": CYCLIC_REPEAT,
            "cyclic_rows_per_item": CYCLIC_ROWS_PER_ITEM,
            "include_unrescued": INCLUDE_UNRESCUED,
        },
        "checkpoint": path_for_cli(checkpoint_path),
        "benchmark_summary": path_for_cli(bench_summary),
        "assessment_summary": path_for_cli(assessment_summary),
        "surface_repair_assessment_summary": path_for_cli(RUN_DIR / "surface_repair_assessment.json"),
        "surface_repair_assessment_status": surface_repair_payload.get("status"),
        "surface_repair_assessment": surface_repair_payload,
        "assessment_status": assessment_payload.get("status"),
        "assessment": assessment_payload,
        "next_step": (
            "Run same-recipe dense control or larger held-out confirmation."
            if passed and surface_repair_passed
            else "Inspect content/cyclic deltas; if cyclic remains strong but content lags, add explicit score-level alignment."
        ),
    }
    write_summary(payload)
    commit_results([diagnosis, diagnosis.with_suffix(".md")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
