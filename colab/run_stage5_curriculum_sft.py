"""Train Phase 1 recurrent Qwen from a gated generated-curriculum shard.

This runner starts after the strong-model curriculum pipeline has produced a
completed work directory with ``positive_sft.jsonl``. It deliberately refuses
to spend GPU on ambiguous data:

1. rerun the no-GPU curriculum SFT gate;
2. require enough positive rows for a real fine-tune;
3. require a visible Drive backup directory unless explicitly overridden;
4. train a bounded deterministic Phase 1 recurrent checkpoint;
5. validate on a held-out split and back up the data/checkpoints to Drive.

It is not a particle/SVGD runner. Particle value should be tested only after a
deterministic recurrent checkpoint trained on the generated curriculum is sane.
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

from training.check_curriculum_sft_gate import (  # noqa: E402
    build_gate_payload,
    parse_args as parse_gate_args,
    write_outputs as write_gate_outputs,
)


RUN_ID = os.environ.get("STAGE5_CURRICULUM_SFT_RUN_ID") or time.strftime(
    "stage5_curriculum_sft_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
DATA_DIR = RUN_DIR / "data"

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
WORK_DIR = Path(
    os.environ.get(
        "STAGE5_CURRICULUM_WORK_DIR",
        "data/curriculum/programmatic_direct_deep_001",
    )
)
SUMMARY_JSON = os.environ.get("STAGE5_CURRICULUM_SUMMARY_JSON", "")
MIN_POSITIVE_ROWS = int(os.environ.get("STAGE5_CURRICULUM_MIN_POSITIVE_ROWS", "2000"))
MIN_MODE_ROWS = os.environ.get(
    "STAGE5_CURRICULUM_MIN_MODE_ROWS",
    "direct=1000,deep_narrow=1000",
).strip()
MIN_TARGET_LOOP_ROWS = os.environ.get("STAGE5_CURRICULUM_MIN_TARGET_LOOP_ROWS", "").strip()
VAL_FRACTION = float(os.environ.get("STAGE5_CURRICULUM_VAL_FRACTION", "0.10"))
VAL_MIN_ROWS = int(os.environ.get("STAGE5_CURRICULUM_VAL_MIN_ROWS", "1"))
SPLIT_SEED = int(os.environ.get("STAGE5_CURRICULUM_SPLIT_SEED", "17"))

DTYPE = os.environ.get("DTYPE", "bfloat16")
ADAPTER_DTYPE = os.environ.get("ADAPTER_DTYPE", "float32")
DEVICE = os.environ.get("DEVICE", "cuda")
MAX_LENGTH = int(os.environ.get("STAGE5_CURRICULUM_MAX_LENGTH", "512"))
MAX_LOOPS = int(os.environ.get("STAGE5_CURRICULUM_MAX_LOOPS", "4"))
MAX_STEPS = int(os.environ.get("STAGE5_CURRICULUM_PHASE1_STEPS", "150"))
SAVE_EVERY = int(os.environ.get("STAGE5_CURRICULUM_PHASE1_SAVE_EVERY", "0"))
LEARNING_RATE = float(os.environ.get("STAGE5_CURRICULUM_PHASE1_LR", "1e-5"))
BETA = float(os.environ.get("STAGE5_CURRICULUM_PHASE1_BETA", "0.08"))
HALT_TARGET_NLL_WEIGHT = float(os.environ.get("STAGE5_CURRICULUM_HALT_TARGET_NLL_WEIGHT", "0.0"))
OPTIMIZER_MODULES = os.environ.get("STAGE5_CURRICULUM_OPTIMIZER_MODULES", "all").strip() or "all"
DEPTH_HINT_STYLE = os.environ.get("STAGE5_CURRICULUM_DEPTH_HINT_STYLE", "none").strip().lower()
USE_TARGET_LOOP_CONTROL = os.environ.get(
    "STAGE5_CURRICULUM_USE_TARGET_LOOP_CONTROL",
    "0",
).strip().lower() in {"1", "true", "yes", "y"}
USE_LEARNED_LOOP_CONTROL = os.environ.get(
    "STAGE5_CURRICULUM_USE_LEARNED_LOOP_CONTROL",
    "0",
).strip().lower() in {"1", "true", "yes", "y"}
LOOP_CONTROL_CE_WEIGHT = float(os.environ.get("STAGE5_CURRICULUM_LOOP_CONTROL_CE_WEIGHT", "0.0"))
REENTRY_RESCALE_MODE = os.environ.get("STAGE5_CURRICULUM_REENTRY_RESCALE_MODE", "none").strip().lower()
if REENTRY_RESCALE_MODE not in {"none", "entry_rms"}:
    raise ValueError("STAGE5_CURRICULUM_REENTRY_RESCALE_MODE must be one of: none, entry_rms")
USE_REENTRY_ADAPTER = os.environ.get(
    "STAGE5_CURRICULUM_USE_REENTRY_ADAPTER",
    "0",
).strip().lower() in {"1", "true", "yes", "y"}
if USE_TARGET_LOOP_CONTROL and USE_LEARNED_LOOP_CONTROL:
    raise ValueError("Use either target-loop oracle control or learned loop control, not both.")
MAX_GRAD_NORM = float(os.environ.get("STAGE5_CURRICULUM_PHASE1_MAX_GRAD_NORM", "0.3"))
MIN_MEAN_EXPECTED_LOOPS = float(os.environ.get("STAGE5_CURRICULUM_SFT_MIN_MEAN_EXPECTED_LOOPS", "1.05"))
DEPTH_GRADIENT_MARGIN = float(os.environ.get("STAGE5_CURRICULUM_SFT_DEPTH_GRADIENT_MARGIN", "0.25"))
TARGET_LOOP_GRADIENT_MARGIN = float(
    os.environ.get("STAGE5_CURRICULUM_SFT_TARGET_LOOP_GRADIENT_MARGIN", "0.10")
)
REQUIRE_DEPTH_GRADIENT = os.environ.get("STAGE5_CURRICULUM_SFT_REQUIRE_DEPTH_GRADIENT", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
REQUIRE_TARGET_LOOP_GRADIENT = os.environ.get(
    "STAGE5_CURRICULUM_SFT_REQUIRE_TARGET_LOOP_GRADIENT",
    "0",
).strip().lower() in {"1", "true", "yes", "y"}
ALLOW_ANSWER_LINE_VERIFICATION = os.environ.get(
    "STAGE5_CURRICULUM_ALLOW_ANSWER_LINE_VERIFICATION",
    "0",
).strip().lower() in {"1", "true", "yes", "y"}
ALLOW_CROSS_MODEL_ONLY_ANSWERS = os.environ.get(
    "STAGE5_CURRICULUM_ALLOW_CROSS_MODEL_ONLY_ANSWERS",
    "0",
).strip().lower() in {"1", "true", "yes", "y"}
RESUME_FROM = os.environ.get("STAGE5_CURRICULUM_RESUME_FROM", "").strip()
PUSH_RESULTS = os.environ.get("STAGE5_CURRICULUM_SFT_PUSH", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
COMMIT_CHECKPOINTS = os.environ.get("STAGE5_CURRICULUM_SFT_COMMIT_CHECKPOINTS", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
ALLOW_NO_DRIVE_BACKUP = os.environ.get("STAGE5_CURRICULUM_ALLOW_NO_DRIVE_BACKUP", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
CURRICULUM_INPUT_BACKUP_DIR = os.environ.get(
    "STAGE5_CURRICULUM_INPUT_BACKUP_DIR",
    "/content/drive/MyDrive/recurrent-qwen-svgd/curriculum_runs",
)


def resolve_path(value: str | Path, *, base: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


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


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no} is not a JSON object")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


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


def summary_path_for_work_dir(work_dir: Path = WORK_DIR) -> Path:
    if SUMMARY_JSON:
        return resolve_path(SUMMARY_JSON)
    return resolve_path(work_dir) / "summary.json"


def curriculum_input_backup_root() -> Path:
    return Path(CURRICULUM_INPUT_BACKUP_DIR)


def curriculum_work_dir_backup_candidates(work_dir: Path) -> list[Path]:
    root = curriculum_input_backup_root()
    local = resolve_path(work_dir)
    candidates = [root / local.name]
    try:
        relative = local.relative_to(ROOT)
    except ValueError:
        relative = Path(local.name)
    candidates.append(root / relative)
    # Preserve order while removing duplicates.
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def restore_work_dir_if_needed(work_dir: Path = WORK_DIR) -> dict[str, Any]:
    local = resolve_path(work_dir)
    summary = summary_path_for_work_dir(work_dir)
    if local.exists() and summary.exists():
        return {"restored": False, "work_dir": path_for_cli(local), "summary_json": path_for_cli(summary)}

    mount_drive_if_possible()
    candidates = curriculum_work_dir_backup_candidates(work_dir)
    for candidate in candidates:
        if (candidate / "summary.json").exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(candidate, local, dirs_exist_ok=True)
            print(f"restored_curriculum_work_dir={candidate} -> {local}", flush=True)
            return {
                "restored": True,
                "source": str(candidate),
                "work_dir": path_for_cli(local),
                "summary_json": path_for_cli(summary),
            }

    searched = [str(path) for path in candidates]
    if not local.exists() or not summary.exists():
        raise FileNotFoundError(
            f"Missing curriculum work dir or summary before GPU training: work_dir={local}, summary={summary}. "
            f"Searched Drive backups under {curriculum_input_backup_root()}: {searched}. "
            "Run the CPU/API curriculum artifact cell first, or set STAGE5_CURRICULUM_INPUT_BACKUP_DIR."
        )
    return {"restored": False, "work_dir": path_for_cli(local), "summary_json": path_for_cli(summary)}


def run_sft_gate() -> dict[str, Any]:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    work_dir = resolve_path(WORK_DIR)
    summary_json = summary_path_for_work_dir(WORK_DIR)
    output_json = RUN_DIR / "curriculum_sft_gate.json"
    output_md = RUN_DIR / "curriculum_sft_gate.md"
    args = [
        "--work_dir",
        path_for_cli(work_dir),
        "--summary_json",
        path_for_cli(summary_json),
        "--output_json",
        path_for_cli(output_json),
        "--output_md",
        path_for_cli(output_md),
        "--min_positive_rows",
        str(MIN_POSITIVE_ROWS),
        "--max_loop_target",
        str(MAX_LOOPS),
        "--fail_on_no_go",
    ]
    if MIN_MODE_ROWS:
        args.extend(["--min_mode_rows", MIN_MODE_ROWS])
    if MIN_TARGET_LOOP_ROWS:
        args.extend(["--min_target_loop_rows", MIN_TARGET_LOOP_ROWS])
    if ALLOW_ANSWER_LINE_VERIFICATION:
        args.append("--allow_answer_line_verification")
    if ALLOW_CROSS_MODEL_ONLY_ANSWERS:
        args.append("--allow_cross_model_only_answers")
    gate_args = parse_gate_args(args)
    payload = build_gate_payload(gate_args)
    write_gate_outputs(payload, output_json=path_for_cli(output_json), output_md=path_for_cli(output_md))
    if not payload.get("go"):
        raise RuntimeError(
            "Curriculum SFT gate failed; refusing GPU training. "
            f"See {path_for_cli(output_json)} and {path_for_cli(output_md)}."
        )
    return payload


def positive_sft_path_from_gate(gate: dict[str, Any]) -> Path:
    artifacts = gate.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts.get("positive_sft"):
        raise KeyError("SFT gate payload is missing artifacts.positive_sft")
    return resolve_path(str(artifacts["positive_sft"]))


def depth_hint_for_row(row: dict[str, Any]) -> str:
    mode = str(row.get("curriculum_mode") or row.get("routing_type") or "")
    if mode == "direct":
        return "Depth hint: use shallow direct reasoning and answer in one short pass."
    if mode == "deep_narrow":
        return "Depth hint: use deeper multi-step reasoning before answering."
    return "Depth hint: choose the reasoning depth that fits the problem."


def insert_depth_hint(prompt: str, hint: str) -> str:
    user_marker = "<|im_start|>user\n"
    if user_marker in prompt:
        return prompt.replace(user_marker, user_marker + hint + "\n\n", 1)
    return hint + "\n\n" + prompt


def maybe_apply_depth_hints(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if DEPTH_HINT_STYLE in {"", "none", "off", "0"}:
        return rows
    if DEPTH_HINT_STYLE not in {"natural", "natural_language"}:
        raise ValueError("STAGE5_CURRICULUM_DEPTH_HINT_STYLE must be one of: none, natural")
    hinted: list[dict[str, Any]] = []
    for row in rows:
        updated = dict(row)
        updated["prompt"] = insert_depth_hint(str(row.get("prompt") or ""), depth_hint_for_row(row))
        updated["depth_hint_style"] = DEPTH_HINT_STYLE
        hinted.append(updated)
    return hinted


def split_train_val(
    rows: list[dict[str, Any]],
    *,
    val_fraction: float = VAL_FRACTION,
    val_min_rows: int = VAL_MIN_ROWS,
    seed: int = SPLIT_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(rows) < 2:
        raise ValueError("At least two positive SFT rows are required for a held-out split.")
    val_count = max(val_min_rows, int(round(len(rows) * val_fraction)))
    val_count = min(max(val_count, 1), len(rows) - 1)

    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        group = str(row.get("curriculum_mode") or row.get("routing_type") or "ungrouped")
        grouped.setdefault(group, []).append(row)
    for group_rows in grouped.values():
        rng.shuffle(group_rows)

    val_targets: dict[str, int] = {group: 0 for group in grouped}
    eligible_groups = sorted(
        (group for group, group_rows in grouped.items() if len(group_rows) > 1),
        key=lambda group: (-len(grouped[group]), group),
    )
    remaining = val_count
    if remaining >= len(eligible_groups):
        for group in eligible_groups:
            val_targets[group] = 1
        remaining -= len(eligible_groups)
    else:
        for group in eligible_groups[:remaining]:
            val_targets[group] = 1
        remaining = 0

    # D'Hondt-style proportional allocation keeps validation mode coverage
    # stable even when one mode slightly outnumbers another.
    while remaining > 0:
        candidates = [
            group
            for group in eligible_groups
            if val_targets[group] < len(grouped[group]) - 1
        ]
        if not candidates:
            break
        group = sorted(
            candidates,
            key=lambda item: (-(len(grouped[item]) / (val_targets[item] + 1)), item),
        )[0]
        val_targets[group] += 1
        remaining -= 1

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for group in sorted(grouped):
        target = val_targets[group]
        group_rows = grouped[group]
        val.extend(group_rows[:target])
        train.extend(group_rows[target:])

    if not train or not val:
        shuffled = list(rows)
        rng.shuffle(shuffled)
        return shuffled[:-val_count], shuffled[-val_count:]
    return train, val


def prepare_train_val(gate: dict[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    positive_sft = positive_sft_path_from_gate(gate)
    rows = read_jsonl(positive_sft)
    if len(rows) < MIN_POSITIVE_ROWS:
        raise RuntimeError(f"positive_sft has {len(rows)} rows < required {MIN_POSITIVE_ROWS}.")
    rows = maybe_apply_depth_hints(rows)
    train_rows, val_rows = split_train_val(
        rows,
        val_fraction=VAL_FRACTION,
        val_min_rows=VAL_MIN_ROWS,
        seed=SPLIT_SEED,
    )
    train_jsonl = DATA_DIR / "curriculum_positive_train.jsonl"
    val_jsonl = DATA_DIR / "curriculum_positive_val.jsonl"
    write_jsonl(train_jsonl, train_rows)
    write_jsonl(val_jsonl, val_rows)
    train_mode_counts: dict[str, int] = {}
    val_mode_counts: dict[str, int] = {}
    train_target_loop_counts: dict[str, int] = {}
    val_target_loop_counts: dict[str, int] = {}
    for row in train_rows:
        mode = str(row.get("curriculum_mode") or row.get("routing_type") or "ungrouped")
        train_mode_counts[mode] = train_mode_counts.get(mode, 0) + 1
        target_loop = row.get("target_loop_count")
        if isinstance(target_loop, int):
            key = str(target_loop)
            train_target_loop_counts[key] = train_target_loop_counts.get(key, 0) + 1
    for row in val_rows:
        mode = str(row.get("curriculum_mode") or row.get("routing_type") or "ungrouped")
        val_mode_counts[mode] = val_mode_counts.get(mode, 0) + 1
        target_loop = row.get("target_loop_count")
        if isinstance(target_loop, int):
            key = str(target_loop)
            val_target_loop_counts[key] = val_target_loop_counts.get(key, 0) + 1
    return train_jsonl, val_jsonl, {
        "source_positive_sft": path_for_cli(positive_sft),
        "rows": len(rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "val_fraction": VAL_FRACTION,
        "val_min_rows": VAL_MIN_ROWS,
        "split_seed": SPLIT_SEED,
        "depth_hint_style": DEPTH_HINT_STYLE,
        "train_mode_counts": train_mode_counts,
        "val_mode_counts": val_mode_counts,
        "train_target_loop_counts": dict(sorted(train_target_loop_counts.items(), key=lambda item: int(item[0]))),
        "val_target_loop_counts": dict(sorted(val_target_loop_counts.items(), key=lambda item: int(item[0]))),
    }


def drive_backup_root() -> Path:
    return Path(os.environ.get("DRIVE_BACKUP_DIR", "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts"))


def mydrive_root() -> Path:
    return Path("/content/drive/MyDrive")


def mount_drive_if_possible() -> None:
    if mydrive_root().exists():
        return
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")
    except Exception as exc:  # pragma: no cover - Colab only
        print(f"Drive mount skipped/failed: {exc}", flush=True)


def validate_drive_backup(
    *,
    drive_root: Path | None = None,
    allow_no_backup: bool = ALLOW_NO_DRIVE_BACKUP,
) -> dict[str, Any]:
    is_default_root = drive_root is None
    root = drive_root or drive_backup_root()
    if not root.exists() and not allow_no_backup and is_default_root:
        mount_drive_if_possible()
        if mydrive_root().exists() and root.parent.exists():
            root.mkdir(parents=True, exist_ok=True)
    payload = {
        "drive_root": str(root),
        "available": root.exists(),
        "allow_no_backup": allow_no_backup,
    }
    if root.exists() or allow_no_backup:
        return payload
    raise RuntimeError(
        f"Stage 5 curriculum SFT requires a mounted Drive backup directory before A100 training: {root}. "
        "Mount/authorize Google Drive first, or set STAGE5_CURRICULUM_ALLOW_NO_DRIVE_BACKUP=1 for a deliberate "
        "non-default smoke run."
    )


def resolve_resume_from() -> Path | None:
    if not RESUME_FROM:
        return None
    path = resolve_path(RESUME_FROM)
    if not path.exists():
        raise FileNotFoundError(f"Missing STAGE5_CURRICULUM_RESUME_FROM checkpoint: {path}")
    return path


def phase1_config(train_output_dir: Path, resume_from: Path | None) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "model_name": MODEL_NAME,
        "dtype": DTYPE,
        "adapter_dtype": ADAPTER_DTYPE,
        "layer_split": "6,18",
        "max_length": MAX_LENGTH,
        "max_loops": MAX_LOOPS,
        "initial_halt_prob": 0.15,
        "beta": BETA,
        "halt_target_nll_weight": HALT_TARGET_NLL_WEIGHT,
        "optimizer_modules": OPTIMIZER_MODULES,
        "use_target_loop_control": USE_TARGET_LOOP_CONTROL,
        "use_learned_loop_control": USE_LEARNED_LOOP_CONTROL,
        "loop_control_ce_weight": LOOP_CONTROL_CE_WEIGHT,
        "reentry_rescale_mode": REENTRY_RESCALE_MODE,
        "use_reentry_adapter": USE_REENTRY_ADAPTER,
        "batch_size": 1,
        "learning_rate": LEARNING_RATE,
        "weight_decay": 0.0,
        "max_grad_norm": MAX_GRAD_NORM,
        "max_steps": MAX_STEPS,
        "save_every": SAVE_EVERY,
        "log_every": 25,
        "train_on_prompt": False,
        "output_dir": path_for_cli(train_output_dir),
        "lora": {"enabled": True, "rank": 8, "alpha": 16, "dropout": 0.0},
    }
    if resume_from is not None:
        cfg["resume_from"] = path_for_cli(resume_from)
    return cfg


def train_phase1(train_jsonl: Path, *, resume_from: Path | None) -> Path:
    train_output_dir = RUN_DIR / "phase1"
    cfg_path = RUN_DIR / "phase1_curriculum_sft.yaml"
    write_yaml(cfg_path, phase1_config(train_output_dir, resume_from))
    run(
        [
            sys.executable,
            "training/train_phase1_ponder.py",
            "--config",
            path_for_cli(cfg_path),
            "--train_jsonl",
            path_for_cli(train_jsonl),
            "--device",
            DEVICE,
        ],
        log_name="phase1_curriculum_sft_train.log",
    )
    checkpoints = sorted(train_output_dir.glob("phase1_step_*.pt"), key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints produced under {train_output_dir}")
    return checkpoints[-1]


def summarize_jsonl_eval(stdout: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        try:
            metrics[key.strip()] = float(value.strip())
        except ValueError:
            continue
    return metrics


def grouped_eval_metrics(metrics: dict[str, float], *, group_field: str) -> dict[str, dict[str, float]]:
    prefix = f"group/{group_field}/"
    grouped: dict[str, dict[str, float]] = {}
    for key, value in metrics.items():
        if not key.startswith(prefix):
            continue
        remainder = key[len(prefix):]
        group, separator, metric = remainder.partition("/")
        if not separator or not group or not metric:
            continue
        grouped.setdefault(group, {})[metric] = value
    return grouped


def validation_checks(
    phase1_val: dict[str, float],
    phase1_val_by_mode: dict[str, dict[str, float]],
    phase1_val_by_target_loop: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    nonfinite = sorted(
        key
        for key, value in phase1_val.items()
        if isinstance(value, (int, float)) and not math.isfinite(float(value))
    )
    mean_loops = phase1_val.get("mean_expected_loops")
    loop_collapse = (
        isinstance(mean_loops, (int, float))
        and math.isfinite(float(mean_loops))
        and float(mean_loops) < MIN_MEAN_EXPECTED_LOOPS
    )
    direct_loops = phase1_val_by_mode.get("direct", {}).get("mean_expected_loops")
    deep_loops = phase1_val_by_mode.get("deep_narrow", {}).get("mean_expected_loops")
    depth_gradient: dict[str, Any] = {
        "available": isinstance(direct_loops, (int, float)) and isinstance(deep_loops, (int, float)),
        "direct_mean_expected_loops": direct_loops,
        "deep_narrow_mean_expected_loops": deep_loops,
        "required_margin": DEPTH_GRADIENT_MARGIN,
        "observed": None,
    }
    if depth_gradient["available"]:
        depth_gradient["observed"] = float(deep_loops) >= float(direct_loops) + DEPTH_GRADIENT_MARGIN
    target_loop_metrics = phase1_val_by_target_loop or {}
    loop_points: list[dict[str, float]] = []
    for target_loop, metrics in sorted(
        target_loop_metrics.items(),
        key=lambda item: int(item[0]) if str(item[0]).isdigit() else 999,
    ):
        try:
            target_loop_value = int(target_loop)
        except ValueError:
            continue
        expected_loops = metrics.get("mean_expected_loops")
        examples = metrics.get("examples")
        if isinstance(expected_loops, (int, float)) and math.isfinite(float(expected_loops)):
            loop_points.append(
                {
                    "target_loop_count": float(target_loop_value),
                    "mean_expected_loops": float(expected_loops),
                    "examples": float(examples) if isinstance(examples, (int, float)) else 0.0,
                }
            )
    adjacent_margins = [
        loop_points[index + 1]["mean_expected_loops"] - loop_points[index]["mean_expected_loops"]
        for index in range(len(loop_points) - 1)
    ]
    target_loop_gradient: dict[str, Any] = {
        "available": len(loop_points) >= 2,
        "points": loop_points,
        "required_margin": TARGET_LOOP_GRADIENT_MARGIN,
        "adjacent_margins": adjacent_margins,
        "observed": None,
    }
    if target_loop_gradient["available"]:
        target_loop_gradient["observed"] = all(
            margin >= TARGET_LOOP_GRADIENT_MARGIN for margin in adjacent_margins
        )
    issues: list[str] = []
    if nonfinite:
        issues.append("nonfinite_validation_metrics")
    if mean_loops is None:
        issues.append("missing_mean_expected_loops")
    elif loop_collapse:
        issues.append("mean_expected_loops_collapsed")
    if REQUIRE_DEPTH_GRADIENT:
        if not depth_gradient["available"]:
            issues.append("missing_depth_gradient_metrics")
        elif depth_gradient["observed"] is False:
            issues.append("depth_gradient_not_observed")
    if REQUIRE_TARGET_LOOP_GRADIENT:
        if not target_loop_gradient["available"]:
            issues.append("missing_target_loop_gradient_metrics")
        elif target_loop_gradient["observed"] is False:
            issues.append("target_loop_gradient_not_observed")
    status = "validation_sane" if not issues else "validation_needs_review"
    return {
        "status": status,
        "issues": issues,
        "nonfinite_metrics": nonfinite,
        "min_mean_expected_loops": MIN_MEAN_EXPECTED_LOOPS,
        "mean_expected_loops": mean_loops,
        "require_depth_gradient": REQUIRE_DEPTH_GRADIENT,
        "depth_gradient": depth_gradient,
        "require_target_loop_gradient": REQUIRE_TARGET_LOOP_GRADIENT,
        "target_loop_gradient": target_loop_gradient,
    }


def eval_jsonl(label: str, data_jsonl: Path, checkpoint: Path) -> dict[str, float]:
    command = [
        sys.executable,
        "eval/eval_jsonl.py",
        "--model_name",
        MODEL_NAME,
        "--data_jsonl",
        path_for_cli(data_jsonl),
        "--checkpoint",
        path_for_cli(checkpoint),
        "--split",
        "6,18",
        "--max_loops",
        str(MAX_LOOPS),
        "--max_length",
        str(MAX_LENGTH),
        "--beta",
        str(BETA),
        "--dtype",
        DTYPE,
        "--adapter_dtype",
        ADAPTER_DTYPE,
        "--device",
        DEVICE,
    ]
    for group_field in ("curriculum_mode", "target_loop_count"):
        command.extend(["--group_by_field", group_field])
    if USE_TARGET_LOOP_CONTROL:
        command.append("--use_target_loop_control")
    if USE_LEARNED_LOOP_CONTROL:
        command.append("--use_learned_loop_control")
        command.extend(["--loop_control_ce_weight", str(LOOP_CONTROL_CE_WEIGHT)])
    command.extend(["--reentry_rescale_mode", REENTRY_RESCALE_MODE])
    if USE_REENTRY_ADAPTER:
        command.append("--use_reentry_adapter")
    proc = run(
        command,
        log_name=f"{label}_val.log",
    )
    return summarize_jsonl_eval(proc.stdout)


def backup_to_drive(train_jsonl: Path, val_jsonl: Path) -> dict[str, Any]:
    root = drive_backup_root()
    if ALLOW_NO_DRIVE_BACKUP and not root.exists():
        return {"backed_up": False, "drive_root": str(root), "reason": "allow_no_drive_backup"}
    mount_drive_if_possible()
    if not root.exists():
        return {"backed_up": False, "drive_root": str(root)}
    backup = root / RUN_ID
    backup.mkdir(parents=True, exist_ok=True)
    shutil.copytree(RUN_DIR, backup / "run_dir", dirs_exist_ok=True)
    for source in [train_jsonl, val_jsonl]:
        if source.exists():
            target = backup / "data" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    print(f"backed_up_to={backup}", flush=True)
    return {"backed_up": True, "path": str(backup)}


def git_commit_results() -> None:
    run(["git", "config", "user.email", "colab-runner@local"], check=False)
    run(["git", "config", "user.name", "Colab Runner"], check=False)
    safe_patterns = [
        "curriculum_sft_gate.json",
        "curriculum_sft_gate.md",
        "phase1_curriculum_sft.yaml",
        "run_metadata.json",
        "summary.json",
        "summary.md",
        "*.log",
    ]
    for pattern in safe_patterns:
        for path in RUN_DIR.glob(pattern):
            run(["git", "add", "-f", path_for_cli(path)], check=False)
    if COMMIT_CHECKPOINTS:
        checkpoints = sorted(
            (RUN_DIR / "phase1").glob("phase1_step_*.pt"),
            key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
        )
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            run(["git", "add", "-f", path_for_cli(latest_checkpoint)], check=False)
            print(f"staged_checkpoint={path_for_cli(latest_checkpoint)}", flush=True)
        else:
            print(f"checkpoint_commit_enabled_but_missing={path_for_cli(RUN_DIR / 'phase1')}", flush=True)
    pointer = current_source_summary_file()
    if pointer.exists():
        run(["git", "add", "-f", path_for_cli(pointer)], check=False)
    if run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("No Stage 5 curriculum SFT summary outputs changed.", flush=True)
        return
    run(["git", "commit", "-m", f"Record Stage 5 curriculum SFT {RUN_ID} [skip ci]"])
    run(["git", "push", "origin", "main"], check=False)


def write_summary(payload: dict[str, Any]) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RUN_DIR / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    update_current_source_summary(summary_path)
    lines = [
        f"# Stage 5 Curriculum SFT - {RUN_ID}",
        "",
        "## Question",
        "Can a strict generated depth/width curriculum improve the deterministic recurrent model without leaking unsafe traces?",
        "",
        "## Safety",
        f"- SFT gate: `{payload['sft_gate']['status']}`",
        f"- Input restore: `{payload['input_restore']}`",
        f"- Positive rows: `{payload['dataset']['rows']}`",
        f"- Train / validation rows: `{payload['dataset']['train_rows']}` / `{payload['dataset']['val_rows']}`",
        f"- Train mode counts: `{payload['dataset'].get('train_mode_counts')}`",
        f"- Validation mode counts: `{payload['dataset'].get('val_mode_counts')}`",
        f"- Train target-loop counts: `{payload['dataset'].get('train_target_loop_counts')}`",
        f"- Validation target-loop counts: `{payload['dataset'].get('val_target_loop_counts')}`",
        f"- Depth hint style: `{payload['dataset'].get('depth_hint_style')}`",
        f"- Target loop control: `{payload['config'].get('use_target_loop_control')}`",
        f"- Learned loop control: `{payload['config'].get('use_learned_loop_control')}`",
        f"- Re-entry rescale: `{payload['config'].get('reentry_rescale_mode')}`",
        f"- Re-entry adapter: `{payload['config'].get('use_reentry_adapter')}`",
        f"- Drive preflight: `{payload['drive_preflight']}`",
        f"- Validation status: `{payload.get('validation_checks', {}).get('status')}`",
        f"- Validation issues: `{payload.get('validation_checks', {}).get('issues', [])}`",
        "",
        "## Training",
        f"- Resume from: `{payload['resume_from']}`",
        f"- Checkpoint: `{payload['phase1_checkpoint']}`",
        f"- Steps: `{payload['config']['max_steps']}`",
        f"- Max loops: `{payload['config']['max_loops']}`",
        "",
        "## Validation",
        "```json",
        json.dumps(payload["phase1_val"], indent=2),
        "```",
        "",
        "## Validation By Curriculum Mode",
        "```json",
        json.dumps(payload.get("phase1_val_by_mode", {}), indent=2),
        "```",
        "",
        "## Validation By Target Loop",
        "```json",
        json.dumps(payload.get("phase1_val_by_target_loop", {}), indent=2),
        "```",
        "",
        "## Validation Checks",
        "```json",
        json.dumps(payload.get("validation_checks", {}), indent=2),
        "```",
        "",
        "## Next Decision",
        "If validation is finite and loop depth remains non-collapsed, run the paired benchmark suite before any Phase 2/SVGD training.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    input_restore = restore_work_dir_if_needed()
    metadata = {
        "run_id": RUN_ID,
        "model_name": MODEL_NAME,
        "work_dir": path_for_cli(resolve_path(WORK_DIR)),
        "summary_json": path_for_cli(summary_path_for_work_dir(WORK_DIR)),
        "curriculum_input_backup_dir": str(curriculum_input_backup_root()),
        "input_restore": input_restore,
        "min_positive_rows": MIN_POSITIVE_ROWS,
        "min_mode_rows": MIN_MODE_ROWS,
        "min_target_loop_rows": MIN_TARGET_LOOP_ROWS,
        "max_length": MAX_LENGTH,
        "max_loops": MAX_LOOPS,
        "max_steps": MAX_STEPS,
        "learning_rate": LEARNING_RATE,
        "beta": BETA,
        "halt_target_nll_weight": HALT_TARGET_NLL_WEIGHT,
        "optimizer_modules": OPTIMIZER_MODULES,
        "use_target_loop_control": USE_TARGET_LOOP_CONTROL,
        "use_learned_loop_control": USE_LEARNED_LOOP_CONTROL,
        "loop_control_ce_weight": LOOP_CONTROL_CE_WEIGHT,
        "reentry_rescale_mode": REENTRY_RESCALE_MODE,
        "use_reentry_adapter": USE_REENTRY_ADAPTER,
        "require_target_loop_gradient": REQUIRE_TARGET_LOOP_GRADIENT,
        "depth_hint_style": DEPTH_HINT_STYLE,
        "dtype": DTYPE,
        "adapter_dtype": ADAPTER_DTYPE,
        "commit_checkpoints": COMMIT_CHECKPOINTS,
    }
    (RUN_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)

    gate = run_sft_gate()
    train_jsonl, val_jsonl, dataset_summary = prepare_train_val(gate)
    if not ALLOW_NO_DRIVE_BACKUP:
        mount_drive_if_possible()
    drive_preflight = validate_drive_backup()
    resume_from = resolve_resume_from()
    checkpoint = train_phase1(train_jsonl, resume_from=resume_from)
    phase1_val = eval_jsonl("phase1_curriculum_sft", val_jsonl, checkpoint)
    phase1_val_by_mode = grouped_eval_metrics(phase1_val, group_field="curriculum_mode")
    phase1_val_by_target_loop = grouped_eval_metrics(phase1_val, group_field="target_loop_count")
    checks = validation_checks(phase1_val, phase1_val_by_mode, phase1_val_by_target_loop)
    backup = backup_to_drive(train_jsonl, val_jsonl)

    summary = {
        "run_id": RUN_ID,
        "kind": "stage5_curriculum_sft",
        "status": checks["status"],
        "config": metadata,
        "sft_gate": {
            "status": gate.get("status"),
            "go": gate.get("go"),
            "issues": gate.get("issues", []),
            "checks": gate.get("checks", {}),
        },
        "input_restore": input_restore,
        "dataset": dataset_summary,
        "drive_preflight": drive_preflight,
        "backup": backup,
        "resume_from": None if resume_from is None else path_for_cli(resume_from),
        "phase1_checkpoint": path_for_cli(checkpoint),
        "phase1_val": phase1_val,
        "phase1_val_by_mode": phase1_val_by_mode,
        "phase1_val_by_target_loop": phase1_val_by_target_loop,
        "validation_checks": checks,
    }
    write_summary(summary)
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"), flush=True)
    if PUSH_RESULTS:
        git_commit_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
