"""Train a dense Qwen LoRA on the same traced MCQ curriculum as recurrent SFT.

This is the standard-transformer control arm for the recurrent trace SFT work.
It answers a narrow but critical question:

    If ordinary Qwen 0.5B receives the same Qwen-7B trace curriculum, does it
    recover the same ARC-Easy/Challenge lift as the recurrent checkpoint?

The runner trains a dense LoRA checkpoint from the traced ``positive_sft``
rows, then compares base-vs-dense on ARC-Easy and ARC-Challenge using the same
content and cyclic MCQ surfaces used by the recurrent benchmark suite.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.stage5_publish_utils import publishable_artifact_paths  # noqa: E402
from colab.colab_auth import ensure_hf_token_from_colab  # noqa: E402
from colab.run_stage5_curriculum_sft import depth_hint_for_row, insert_depth_hint, split_train_val  # noqa: E402
from eval.mcq_debias import aggregate_permutation_scores, cyclic_permutation_rows, read_jsonl, write_jsonl  # noqa: E402


RUN_ID = os.environ.get("STAGE5_DENSE_MCQ_RUN_ID") or time.strftime("stage5_dense_mcq_trace_sft_%Y%m%d_%H%M%S")
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
PRIVATE_DATA_DIR = ROOT / "data" / "stage5_dense_mcq_trace_sft" / RUN_ID

SOURCE_SUMMARY = os.environ.get("STAGE5_DENSE_MCQ_SOURCE_SUMMARY", "").strip()
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
DTYPE = os.environ.get("DTYPE", "bfloat16")
ADAPTER_DTYPE = os.environ.get("ADAPTER_DTYPE", "float32")
DEVICE = os.environ.get("DEVICE", "cuda")
MAX_LENGTH = int(os.environ.get("STAGE5_DENSE_MCQ_MAX_LENGTH", "512"))
DENSE_LORA_LAYER_RANGE = os.environ.get("STAGE5_DENSE_MCQ_LORA_LAYER_RANGE", "6,18")
TRAIN_STEPS_ENV = os.environ.get("STAGE5_DENSE_MCQ_STEPS", "").strip()
SAVE_EVERY_ENV = os.environ.get("STAGE5_DENSE_MCQ_SAVE_EVERY", "").strip()
LEARNING_RATE_ENV = os.environ.get("STAGE5_DENSE_MCQ_LR", "").strip()
EXTRA_TRAIN_JSONL_ENV = os.environ.get("STAGE5_DENSE_MCQ_EXTRA_TRAIN_JSONL", "").strip()
VAL_FRACTION = float(os.environ.get("STAGE5_DENSE_MCQ_VAL_FRACTION", "0.10"))
VAL_MIN_ROWS = int(os.environ.get("STAGE5_DENSE_MCQ_VAL_MIN_ROWS", "1"))
SPLIT_SEED = int(os.environ.get("STAGE5_DENSE_MCQ_SPLIT_SEED", "17"))

BENCHMARKS = os.environ.get("STAGE5_DENSE_MCQ_BENCHMARKS", "arc_easy,arc_challenge")
ARC_EASY_LIMIT = int(os.environ.get("STAGE5_DENSE_MCQ_ARC_EASY_LIMIT", "256"))
ARC_CHALLENGE_LIMIT = int(os.environ.get("STAGE5_DENSE_MCQ_ARC_CHALLENGE_LIMIT", "256"))
SCORE_TARGETS = os.environ.get("STAGE5_DENSE_MCQ_SCORE_TARGETS", "content_question_only,cyclic_label_aggregated")
AGGREGATES = os.environ.get("STAGE5_DENSE_MCQ_AGGREGATES", "mean")
PUSH_RESULTS = os.environ.get("STAGE5_DENSE_MCQ_PUSH", "1").strip().lower() in {"1", "true", "yes", "y"}
COMMIT_CHECKPOINT = os.environ.get("STAGE5_DENSE_MCQ_COMMIT_CHECKPOINT", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
BACKUP_TO_DRIVE = os.environ.get("STAGE5_DENSE_MCQ_BACKUP_DRIVE", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
RECURRENT_BENCHMARK_SUMMARY = os.environ.get(
    "STAGE5_DENSE_MCQ_RECURRENT_BENCHMARK_SUMMARY",
    "",
).strip()
SUMMARY_CHAIN_KEYS = (
    "source_summary",
    "nested_source_summary",
    "benchmark_source_summary",
    "child_summary",
    "trace_summary",
)


def resolve_path(value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


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
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        (RUN_DIR / log_name).write_text(stdout, encoding="utf-8")
    if check and proc.returncode:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}")
    return proc


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def current_source_summary_file() -> Path:
    return ROOT / "config" / "stage5_current_source_summary.txt"


def source_summary_path() -> Path:
    raw = SOURCE_SUMMARY
    if not raw or raw == "config/stage5_current_source_summary.txt":
        pointer = current_source_summary_file()
        if not pointer.exists():
            raise FileNotFoundError(pointer)
        raw = pointer.read_text(encoding="utf-8").strip()
        if not raw:
            raise FileNotFoundError("config/stage5_current_source_summary.txt is empty")
    path = resolve_path(raw)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def direct_positive_sft_path(payload: dict[str, Any]) -> Path | None:
    dataset = payload.get("dataset") if isinstance(payload.get("dataset"), dict) else {}
    if dataset.get("source_positive_sft"):
        return resolve_path(str(dataset["source_positive_sft"]))
    gate = payload.get("gate") if isinstance(payload.get("gate"), dict) else {}
    artifacts = gate.get("artifacts") if isinstance(gate.get("artifacts"), dict) else {}
    if artifacts.get("positive_sft"):
        return resolve_path(str(artifacts["positive_sft"]))
    curriculum = payload.get("curriculum") if isinstance(payload.get("curriculum"), dict) else {}
    if curriculum.get("work_dir"):
        return resolve_path(str(curriculum["work_dir"])) / "positive_sft.jsonl"
    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    if config.get("work_dir"):
        return resolve_path(str(config["work_dir"])) / "positive_sft.jsonl"
    return None


def source_positive_sft_path(payload: dict[str, Any], *, _seen: set[Path] | None = None) -> Path:
    direct = direct_positive_sft_path(payload)
    if direct is not None:
        return direct
    seen = _seen if _seen is not None else set()
    for key in SUMMARY_CHAIN_KEYS:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = resolve_path(value)
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        try:
            return source_positive_sft_path(read_json(candidate), _seen=seen)
        except KeyError:
            continue
    raise KeyError("Source summary does not expose a positive_sft path or curriculum work_dir")


def resolve_curriculum_source(
    payload: dict[str, Any],
    *,
    source_path: Path | None = None,
    _seen: set[Path] | None = None,
) -> tuple[Path | None, dict[str, Any]]:
    """Return the summary payload that owns the SFT rows and training defaults.

    Benchmark and assessment summaries are useful comparison wrappers, but the
    dense control must inherit data/depth/training defaults from the underlying
    curriculum or SFT summary. Otherwise a benchmark wrapper can supply the
    right rows through ``source_positive_sft_path`` while silently falling back
    to stale dense-control defaults for depth hints, steps, or LR.
    """

    if direct_positive_sft_path(payload) is not None:
        return source_path, payload

    seen = _seen if _seen is not None else set()
    if source_path is not None:
        seen.add(source_path.resolve())
    for key in SUMMARY_CHAIN_KEYS:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = resolve_path(value)
        resolved = candidate.resolve()
        if resolved in seen or not candidate.exists():
            continue
        seen.add(resolved)
        try:
            return resolve_curriculum_source(read_json(candidate), source_path=candidate, _seen=seen)
        except KeyError:
            continue
    raise KeyError("Source summary chain does not expose a curriculum/SFT source")


def resolve_benchmark_suite_summary(
    source_path: Path,
    *,
    _seen: set[Path] | None = None,
) -> Path | None:
    """Find the benchmark-suite summary represented by a front-of-queue summary.

    The queue pointer often advances from a benchmark suite to an assessment of
    that suite. Dense-control assessment still needs the benchmark suite itself
    because it contains the recurrent artifact paths. This resolver follows the
    local summary chain while avoiding stale hard-coded run IDs.
    """

    seen = _seen if _seen is not None else set()
    resolved = source_path.resolve()
    if resolved in seen or not source_path.exists():
        return None
    seen.add(resolved)
    payload = read_json(source_path)
    if payload.get("kind") == "stage5_benchmark_suite":
        return source_path
    if payload.get("gate") == "stage5_broader_benchmark_suite" and isinstance(payload.get("source_summary"), str):
        candidate = resolve_path(str(payload["source_summary"]))
        if candidate.exists():
            candidate_payload = read_json(candidate)
            if candidate_payload.get("kind") == "stage5_benchmark_suite":
                return candidate
    for key in ("benchmark_summary", "recurrent_benchmark_summary", "source_summary", "benchmark_source_summary"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = resolve_path(value)
        found = resolve_benchmark_suite_summary(candidate, _seen=seen)
        if found is not None:
            return found
    return None


def recurrent_benchmark_summary_path() -> Path | None:
    if RECURRENT_BENCHMARK_SUMMARY:
        explicit = resolve_path(RECURRENT_BENCHMARK_SUMMARY)
        if not explicit.exists():
            raise FileNotFoundError(explicit)
        return resolve_benchmark_suite_summary(explicit) or explicit
    return resolve_benchmark_suite_summary(source_summary_path())


def mount_drive_if_possible() -> None:
    if Path("/content/drive/MyDrive").exists():
        return
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")
    except Exception as exc:  # pragma: no cover - Colab only
        print(f"Drive mount skipped/failed: {exc}", flush=True)


def restore_positive_sft_if_needed(path: Path, source_payload: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        return {"restored": False, "positive_sft": path_for_cli(path), "source": "local"}
    config = source_payload.get("config") if isinstance(source_payload.get("config"), dict) else {}
    curriculum = source_payload.get("curriculum") if isinstance(source_payload.get("curriculum"), dict) else {}
    work_dir = resolve_path(str(config.get("work_dir") or curriculum.get("work_dir") or path.parent))
    drive_root = Path(
        str(config.get("curriculum_input_backup_dir") or "/content/drive/MyDrive/recurrent-qwen-svgd/curriculum_runs")
    )
    mount_drive_if_possible()
    candidates = [
        drive_root / work_dir.name,
        drive_root / path.parent.name,
    ]
    for candidate in candidates:
        source = candidate / "positive_sft.jsonl"
        if source.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(candidate, path.parent, dirs_exist_ok=True)
            return {"restored": True, "positive_sft": path_for_cli(path), "source": str(candidate)}
    raise FileNotFoundError(f"Missing positive_sft rows at {path}. Drive candidates: {[str(c) for c in candidates]}")


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")


def maybe_apply_depth_hints(rows: list[dict[str, Any]], source_payload: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    config = source_payload.get("config") if isinstance(source_payload.get("config"), dict) else {}
    style = os.environ.get("STAGE5_DENSE_MCQ_DEPTH_HINT_STYLE", str(config.get("depth_hint_style") or "none")).strip().lower()
    if style in {"", "none", "off", "0"}:
        return rows, "none"
    if style not in {"natural", "natural_language"}:
        raise ValueError("STAGE5_DENSE_MCQ_DEPTH_HINT_STYLE must be one of: none, natural")
    hinted: list[dict[str, Any]] = []
    for row in rows:
        updated = dict(row)
        updated["prompt"] = insert_depth_hint(str(row.get("prompt") or ""), depth_hint_for_row(row))
        updated["depth_hint_style"] = style
        hinted.append(updated)
    return hinted, style


def mode_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        mode = str(row.get("curriculum_mode") or row.get("routing_type") or "ungrouped")
        counts[mode] = counts.get(mode, 0) + 1
    return counts


def target_loop_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        target_loop = row.get("target_loop_count")
        if isinstance(target_loop, bool):
            continue
        if isinstance(target_loop, int):
            key = str(target_loop)
        elif isinstance(target_loop, str) and target_loop.strip().isdigit():
            key = str(int(target_loop.strip()))
        else:
            continue
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def extra_train_jsonl_paths() -> list[Path]:
    return [resolve_path(item) for item in parse_csv(EXTRA_TRAIN_JSONL_ENV)]


def read_extra_train_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for path in extra_train_jsonl_paths():
        if not path.exists():
            raise FileNotFoundError(path)
        path_rows = read_jsonl_rows(path)
        rows.extend(path_rows)
        metadata.append(
            {
                "path": path_for_cli(path),
                "rows": len(path_rows),
                "mode_counts": mode_counts(path_rows),
                "target_loop_counts": target_loop_counts(path_rows),
            }
        )
    return rows, metadata


def prepare_train_val(source_payload: dict[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    positive_sft = source_positive_sft_path(source_payload)
    restore = restore_positive_sft_if_needed(positive_sft, source_payload)
    source_rows, depth_hint_style = maybe_apply_depth_hints(read_jsonl_rows(positive_sft), source_payload)
    train_rows, val_rows = split_train_val(source_rows, val_fraction=VAL_FRACTION, val_min_rows=VAL_MIN_ROWS, seed=SPLIT_SEED)
    extra_rows, extra_metadata = read_extra_train_rows()
    if extra_rows:
        train_rows = [*train_rows, *extra_rows]
    train_jsonl = PRIVATE_DATA_DIR / "dense_trace_train.jsonl"
    val_jsonl = PRIVATE_DATA_DIR / "dense_trace_val.jsonl"
    write_jsonl_rows(train_jsonl, train_rows)
    write_jsonl_rows(val_jsonl, val_rows)
    return train_jsonl, val_jsonl, {
        "restore": restore,
        "source_positive_sft": path_for_cli(positive_sft),
        "source_rows": len(source_rows),
        "rows": len(source_rows) + len(extra_rows),
        "extra_train_jsonls": extra_metadata,
        "extra_train_rows": len(extra_rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "val_fraction": VAL_FRACTION,
        "val_min_rows": VAL_MIN_ROWS,
        "split_seed": SPLIT_SEED,
        "depth_hint_style": depth_hint_style,
        "source_mode_counts": mode_counts(source_rows),
        "source_target_loop_counts": target_loop_counts(source_rows),
        "extra_train_mode_counts": mode_counts(extra_rows),
        "extra_train_target_loop_counts": target_loop_counts(extra_rows),
        "train_mode_counts": mode_counts(train_rows),
        "val_mode_counts": mode_counts(val_rows),
        "train_target_loop_counts": target_loop_counts(train_rows),
        "val_target_loop_counts": target_loop_counts(val_rows),
    }


def source_default(source_payload: dict[str, Any], key: str, default: Any) -> Any:
    config = source_payload.get("config") if isinstance(source_payload.get("config"), dict) else {}
    return config.get(key, default)


def train_dense_lora(train_jsonl: Path, source_payload: dict[str, Any]) -> Path:
    steps = int(TRAIN_STEPS_ENV or source_default(source_payload, "max_steps", 200))
    save_every = int(SAVE_EVERY_ENV or max(steps, 1))
    lr = float(LEARNING_RATE_ENV or source_default(source_payload, "learning_rate", 2e-4))
    output_dir = RUN_DIR / "dense_lora"
    cfg = {
        "model_name": MODEL_NAME,
        "dtype": DTYPE,
        "adapter_dtype": ADAPTER_DTYPE,
        "layer_split": DENSE_LORA_LAYER_RANGE,
        "max_length": MAX_LENGTH,
        "batch_size": 1,
        "learning_rate": lr,
        "weight_decay": 0.0,
        "max_grad_norm": 0.3,
        "max_steps": steps,
        "save_every": save_every,
        "log_every": 25,
        "train_on_prompt": False,
        "output_dir": path_for_cli(output_dir),
        "lora": {"enabled": True, "rank": 8, "alpha": 16, "dropout": 0.0, "layer_range": DENSE_LORA_LAYER_RANGE},
        "distillation": {"enabled": False},
    }
    cfg_path = RUN_DIR / "dense_lora_trace_sft.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    run(
        [
            sys.executable,
            "training/train_dense_lora.py",
            "--config",
            path_for_cli(cfg_path),
            "--train_jsonl",
            path_for_cli(train_jsonl),
            "--device",
            DEVICE,
        ],
        log_name="dense_lora_train.log",
    )
    checkpoint = output_dir / f"dense_lora_step_{steps}.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def benchmark_data_specs() -> dict[str, Path]:
    specs: dict[str, Path] = {}
    for name in parse_csv(BENCHMARKS):
        if name == "arc_easy":
            config, limit = "ARC-Easy", ARC_EASY_LIMIT
        elif name == "arc_challenge":
            config, limit = "ARC-Challenge", ARC_CHALLENGE_LIMIT
        else:
            raise ValueError(f"Unsupported dense MCQ benchmark: {name}")
        output = PRIVATE_DATA_DIR / f"{name}_validation_{limit}.jsonl"
        run(
            [
                sys.executable,
                "eval/prepare_arc_mcq.py",
                "--config",
                config,
                "--split",
                "validation",
                "--seed",
                "0",
                "--limit",
                str(limit),
                "--output_jsonl",
                path_for_cli(output),
            ],
            log_name=f"prepare_{name}.log",
        )
        specs[name] = output
    return specs


def eval_score_target(score_target: str) -> str:
    if score_target == "content_question_only":
        return "option_text"
    if score_target == "cyclic_label_aggregated":
        return "label"
    return score_target


def prompt_style(score_target: str) -> str:
    return "question_only" if score_target == "content_question_only" else "with_options"


def eval_arm(
    *,
    benchmark: str,
    data_jsonl: Path,
    arm: str,
    score_target: str,
    dense_checkpoint: Path | None,
) -> Path:
    eval_data = data_jsonl
    raw_output = RUN_DIR / f"{benchmark}_{arm}_{score_target}.jsonl"
    output = raw_output
    permutation_jsonl: Path | None = None
    if score_target == "cyclic_label_aggregated":
        permutation_jsonl = PRIVATE_DATA_DIR / f"{data_jsonl.stem}_cyclic_permuted.jsonl"
        if not permutation_jsonl.exists():
            write_jsonl(permutation_jsonl, cyclic_permutation_rows(read_jsonl(data_jsonl)))
        eval_data = permutation_jsonl
        raw_output = RUN_DIR / f"{benchmark}_{arm}_cyclic_label_raw.jsonl"
        output = RUN_DIR / f"{benchmark}_{arm}_cyclic_label_aggregated.jsonl"
    for path in (raw_output, output):
        if path.exists():
            path.unlink()
    cmd = [
        sys.executable,
        "eval/eval_mcq.py",
        "--data_jsonl",
        path_for_cli(eval_data),
        "--mode",
        "base",
        "--prompt_style",
        prompt_style(score_target),
        "--score_target",
        eval_score_target(score_target),
        "--aggregates",
        AGGREGATES,
        "--dtype",
        DTYPE,
        "--adapter_dtype",
        ADAPTER_DTYPE,
        "--device",
        DEVICE,
        "--quiet_rows",
        "--output_jsonl",
        path_for_cli(raw_output),
    ]
    if arm == "dense":
        assert dense_checkpoint is not None
        cmd.extend(["--checkpoint", path_for_cli(dense_checkpoint), "--base_lora_layer_range", DENSE_LORA_LAYER_RANGE])
    run(cmd, log_name=f"{raw_output.stem}.log")
    if permutation_jsonl is not None:
        scored_rows = read_jsonl(raw_output)
        permutation_rows = read_jsonl(permutation_jsonl)
        output_rows: list[dict[str, Any]] = []
        for aggregate in sorted({str(row.get("aggregate") or "mean") for row in scored_rows}):
            aggregate_rows = [row for row in scored_rows if str(row.get("aggregate") or "mean") == aggregate]
            for row in aggregate_permutation_scores(aggregate_rows, permutation_rows):
                row["aggregate"] = f"permutation_{aggregate}"
                output_rows.append(row)
        write_jsonl(output, output_rows)
    return output


def summarize_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    by_aggregate: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_aggregate.setdefault(str(row.get("aggregate") or "mean"), []).append(row)
    return {
        aggregate: {
            "correct": sum(1 for row in aggregate_rows if row.get("hit")),
            "total": len(aggregate_rows),
            "accuracy": sum(1 for row in aggregate_rows if row.get("hit")) / max(len(aggregate_rows), 1),
        }
        for aggregate, aggregate_rows in sorted(by_aggregate.items())
    }


def two_sided_sign_p_value(wins: int, losses: int) -> float | None:
    trials = wins + losses
    if trials == 0:
        return None
    observed = min(wins, losses)
    return min(1.0, 2.0 * sum(math.comb(trials, k) for k in range(observed + 1)) / (2**trials))


def rows_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in read_jsonl(path) if row.get("id") is not None}


def paired_dense_vs_base(base_path: Path, dense_path: Path) -> dict[str, dict[str, Any]]:
    base_rows = read_jsonl(base_path)
    dense_rows = read_jsonl(dense_path)
    aggregates = sorted({str(row.get("aggregate") or "mean") for row in base_rows + dense_rows})
    result: dict[str, dict[str, Any]] = {}
    for aggregate in aggregates:
        base_by_id = {
            str(row["id"]): row
            for row in base_rows
            if row.get("id") is not None and str(row.get("aggregate") or "mean") == aggregate
        }
        dense_by_id = {
            str(row["id"]): row
            for row in dense_rows
            if row.get("id") is not None and str(row.get("aggregate") or "mean") == aggregate
        }
        common = sorted(set(base_by_id) & set(dense_by_id))
        paired = [(bool(base_by_id[item].get("hit")), bool(dense_by_id[item].get("hit"))) for item in common]
        base_correct = sum(1 for base_hit, _dense_hit in paired if base_hit)
        dense_correct = sum(1 for _base_hit, dense_hit in paired if dense_hit)
        wins = sum(1 for base_hit, dense_hit in paired if dense_hit and not base_hit)
        losses = sum(1 for base_hit, dense_hit in paired if base_hit and not dense_hit)
        ties = len(paired) - wins - losses
        result[aggregate] = {
            "paired_examples": len(paired),
            "base_correct": base_correct,
            "dense_correct": dense_correct,
            "correct_delta_dense_vs_base": dense_correct - base_correct,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "sign_test_p_value": two_sided_sign_p_value(wins, losses),
        }
    return result


def run_benchmarks(dense_checkpoint: Path) -> dict[str, Any]:
    specs = benchmark_data_specs()
    results: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    for benchmark, data_jsonl in specs.items():
        results[benchmark] = {}
        paired[benchmark] = {}
        artifacts[benchmark] = {"data_jsonl": path_for_cli(data_jsonl)}
        for score_target in parse_csv(SCORE_TARGETS):
            base_path = eval_arm(
                benchmark=benchmark,
                data_jsonl=data_jsonl,
                arm="base",
                score_target=score_target,
                dense_checkpoint=None,
            )
            dense_path = eval_arm(
                benchmark=benchmark,
                data_jsonl=data_jsonl,
                arm="dense",
                score_target=score_target,
                dense_checkpoint=dense_checkpoint,
            )
            results[benchmark][score_target] = {
                "base": summarize_rows(base_path),
                "dense": summarize_rows(dense_path),
            }
            paired[benchmark][score_target] = paired_dense_vs_base(base_path, dense_path)
            artifacts[benchmark][score_target] = {
                "base": path_for_cli(base_path),
                "dense": path_for_cli(dense_path),
            }
    return {"results": results, "paired_comparisons": paired, "artifacts": artifacts}


def drive_backup() -> dict[str, Any]:
    if not BACKUP_TO_DRIVE:
        return {"backed_up": False, "reason": "disabled"}
    mount_drive_if_possible()
    root = Path(os.environ.get("DRIVE_BACKUP_DIR", "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts"))
    if not root.exists():
        return {"backed_up": False, "drive_root": str(root)}
    target = root / RUN_ID
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(RUN_DIR, target / "run_dir", dirs_exist_ok=True)
    for source in PRIVATE_DATA_DIR.glob("*.jsonl"):
        out = target / "data" / source.name
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, out)
    return {"backed_up": True, "path": str(target)}


def update_current_source_summary(summary_path: Path) -> None:
    pointer = current_source_summary_file()
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(path_for_cli(summary_path) + "\n", encoding="utf-8")


def front_of_queue_summary_path(payload: dict[str, Any], dense_summary_json: Path) -> Path:
    assessment = payload.get("recipe_control_assessment")
    if isinstance(assessment, dict) and assessment.get("ran") is True and assessment.get("summary_json"):
        assessment_path = resolve_path(str(assessment["summary_json"]))
        if assessment_path.exists():
            return assessment_path
    return dense_summary_json


def write_summary(payload: dict[str, Any]) -> Path:
    summary_json = RUN_DIR / "summary.json"
    write_json(summary_json, payload)
    update_current_source_summary(front_of_queue_summary_path(payload, summary_json))
    lines = [
        f"# Stage 5 Dense MCQ Trace-SFT Control - {RUN_ID}",
        "",
        "## Question",
        "Does standard dense Qwen get the same ARC MCQ lift from the traced curriculum as recurrent Qwen?",
        "",
        f"- Source summary: `{payload['source_summary']}`",
        f"- Curriculum source summary: `{payload.get('curriculum_source_summary')}`",
        f"- Total SFT rows: `{payload['dataset']['rows']}`",
        f"- Source rows: `{payload['dataset'].get('source_rows')}`",
        f"- Extra train rows: `{payload['dataset'].get('extra_train_rows')}`",
        f"- Train rows: `{payload['dataset'].get('train_rows')}`",
        f"- Validation rows: `{payload['dataset'].get('val_rows')}`",
        f"- Train mode counts: `{payload['dataset'].get('train_mode_counts')}`",
        f"- Validation mode counts: `{payload['dataset'].get('val_mode_counts')}`",
        f"- Train target-loop counts: `{payload['dataset'].get('train_target_loop_counts')}`",
        f"- Validation target-loop counts: `{payload['dataset'].get('val_target_loop_counts')}`",
        f"- Dense checkpoint: `{payload['dense_checkpoint']}`",
        f"- Dense LoRA layer range: `{payload['config']['dense_lora_layer_range']}`",
        "",
        "## Dense vs Base",
        "",
    ]
    for benchmark, score_targets in payload["paired_comparisons"].items():
        lines.append(f"### {benchmark}")
        for score_target, aggregates in score_targets.items():
            lines.append(f"- `{score_target}`")
            for aggregate, stats in aggregates.items():
                lines.append(
                    f"  - `{aggregate}`: dense `{stats['dense_correct']}` / `{stats['paired_examples']}`, "
                    f"base `{stats['base_correct']}` / `{stats['paired_examples']}`, "
                    f"delta `{stats['correct_delta_dense_vs_base']}`, W/L/T "
                    f"`{stats['wins']}/{stats['losses']}/{stats['ties']}`, p `{stats['sign_test_p_value']}`"
                )
    assessment = payload.get("recipe_control_assessment")
    if isinstance(assessment, dict):
        lines.extend(
            [
                "",
                "## Dense vs Recurrent Same-Recipe Assessment",
                "",
                f"- Ran: `{assessment.get('ran')}`",
                f"- Status: `{assessment.get('status')}`",
                f"- Passed: `{assessment.get('passed')}`",
                f"- Reason: {assessment.get('reason')}",
                f"- Next step: {assessment.get('next_step')}",
            ]
        )
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"), flush=True)
    return summary_json


def run_recipe_control_assessment(summary_json: Path, recurrent_summary: Path | None) -> dict[str, Any]:
    if recurrent_summary is None:
        return {
            "ran": False,
            "reason": (
                "No recurrent benchmark summary was resolved from "
                "STAGE5_DENSE_MCQ_RECURRENT_BENCHMARK_SUMMARY or the current source chain."
            ),
        }
    if not recurrent_summary.exists():
        return {"ran": False, "reason": f"missing recurrent benchmark summary: {path_for_cli(recurrent_summary)}"}
    output_json = RUN_DIR / "mcq_recipe_control_assessment.json"
    output_md = RUN_DIR / "mcq_recipe_control_assessment.md"
    run(
        [
            sys.executable,
            "colab/assess_stage5_mcq_recipe_control.py",
            "--dense_summary_json",
            path_for_cli(summary_json),
            "--recurrent_summary_json",
            path_for_cli(recurrent_summary),
            "--output_json",
            path_for_cli(output_json),
            "--output_md",
            path_for_cli(output_md),
        ],
        log_name="mcq_recipe_control_assessment.log",
    )
    assessment = read_json(output_json)
    return {
        "ran": True,
        "summary_json": path_for_cli(output_json),
        "summary_md": path_for_cli(output_md),
        "status": assessment.get("status"),
        "passed": assessment.get("passed"),
        "reason": assessment.get("reason"),
        "next_step": assessment.get("next_step"),
    }


def commit_results(dense_checkpoint: Path) -> None:
    if not PUSH_RESULTS:
        return
    paths = publishable_artifact_paths(RUN_DIR)
    paths.append(current_source_summary_file())
    if COMMIT_CHECKPOINT:
        paths.append(dense_checkpoint)
    for path in paths:
        if path.exists():
            run(["git", "add", "-f", path_for_cli(path)], check=False)
    if run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("No dense MCQ control outputs changed.", flush=True)
        return
    run(["git", "commit", "-m", f"Record Stage 5 dense MCQ trace-SFT control {RUN_ID} [skip ci]"])
    pushed = run(["git", "push", "origin", "main"], check=False)
    if pushed.returncode == 0:
        return
    print("Initial dense MCQ control push failed; attempting one autostash rebase and retry.", flush=True)
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "push", "origin", "main"])


def main() -> int:
    ensure_hf_token_from_colab()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    PRIVATE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    source_path = source_summary_path()
    source_payload = read_json(source_path)
    curriculum_source_path, curriculum_source_payload = resolve_curriculum_source(source_payload, source_path=source_path)
    recurrent_benchmark = recurrent_benchmark_summary_path()
    train_jsonl, val_jsonl, dataset = prepare_train_val(curriculum_source_payload)
    dense_checkpoint = train_dense_lora(train_jsonl, curriculum_source_payload)
    benchmark_payload = run_benchmarks(dense_checkpoint)
    backup = drive_backup()
    payload = {
        "run_id": RUN_ID,
        "kind": "stage5_dense_mcq_trace_sft_control",
        "source_summary": path_for_cli(source_path),
        "source_kind": source_payload.get("kind"),
        "curriculum_source_summary": path_for_cli(curriculum_source_path)
        if curriculum_source_path is not None
        else path_for_cli(source_path),
        "curriculum_source_kind": curriculum_source_payload.get("kind"),
        "dataset": dataset,
        "config": {
            "model_name": MODEL_NAME,
            "dtype": DTYPE,
            "adapter_dtype": ADAPTER_DTYPE,
            "max_length": MAX_LENGTH,
            "dense_lora_layer_range": DENSE_LORA_LAYER_RANGE,
            "benchmarks": parse_csv(BENCHMARKS),
            "score_targets": parse_csv(SCORE_TARGETS),
            "aggregates": parse_csv(AGGREGATES),
            "extra_train_jsonl": parse_csv(EXTRA_TRAIN_JSONL_ENV),
            "commit_checkpoint": COMMIT_CHECKPOINT,
        },
        "recurrent_benchmark_summary": path_for_cli(recurrent_benchmark) if recurrent_benchmark is not None else None,
        "train_jsonl": path_for_cli(train_jsonl),
        "val_jsonl": path_for_cli(val_jsonl),
        "dense_checkpoint": path_for_cli(dense_checkpoint),
        "backup": backup,
        **benchmark_payload,
    }
    summary_json = write_summary(payload)
    payload["recipe_control_assessment"] = run_recipe_control_assessment(summary_json, recurrent_benchmark)
    write_summary(payload)
    commit_results(dense_checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
