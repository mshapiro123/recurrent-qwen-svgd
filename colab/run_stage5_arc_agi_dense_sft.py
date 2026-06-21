"""Train a standard dense Qwen LoRA ARC-AGI SFT control.

This runner is the non-recurrent counterpart to ``run_stage5_arc_agi_sft.py``.
It uses the same ARC-AGI JSONL preparation path, trains a dense base-model LoRA
adapter, then evaluates:

* untuned base Qwen;
* dense LoRA-tuned Qwen;
* the current deterministic recurrent Phase1 start checkpoint.

The goal is not to replace the recurrent experiment. It is to make the
standard-vs-recurrent recipe comparison explicit before claiming architecture
lift.
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

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.compare_arc_agi_runs import compare_payloads, write_markdown as write_comparison_markdown  # noqa: E402
from training.arc_agi_training_signal import summarize_training_signal, write_training_signal_report  # noqa: E402
from colab.stage5_model_metadata import model_metadata  # noqa: E402


RUN_ID = os.environ.get("STAGE5_ARC_AGI_DENSE_SFT_RUN_ID") or time.strftime("stage5_arc_agi_dense_sft_%Y%m%d_%H%M%S")
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

BASE_RUN_ID = os.environ.get("STAGE5_BASE_RUN_ID", "stage4_opus_a100_20260620")
BASE_RUN_DIR = ROOT / "outputs" / "stage4" / BASE_RUN_ID
PHASE1_CKPT = Path(os.environ.get("STAGE5_PHASE1_CKPT", str(BASE_RUN_DIR / "phase1" / "phase1_step_500.pt")))
if not PHASE1_CKPT.is_absolute():
    PHASE1_CKPT = ROOT / PHASE1_CKPT

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
DATA_ROOT = ROOT / "data" / "arc_agi"
ARC_AGI_1_REPO = os.environ.get("ARC_AGI_1_REPO", "https://github.com/fchollet/ARC-AGI.git")
ARC_AGI_2_REPO = os.environ.get("ARC_AGI_2_REPO", "https://github.com/arcprize/ARC-AGI-2.git")
ARC_VERSION = os.environ.get("STAGE5_ARC_AGI_VERSION", "1")
TRAIN_SPLIT = os.environ.get("STAGE5_ARC_AGI_TRAIN_SPLIT", "training")
EVAL_SPLIT = os.environ.get("STAGE5_ARC_AGI_EVAL_SPLIT", "evaluation")
TRAIN_TASK_LIMIT = int(os.environ.get("STAGE5_ARC_AGI_TRAIN_TASK_LIMIT", "100"))
EVAL_TASK_LIMIT = int(os.environ.get("STAGE5_ARC_AGI_EVAL_TASK_LIMIT", "10"))
COLOR_AUGS = int(os.environ.get("STAGE5_ARC_AGI_COLOR_AUGS", "2"))
GEOMETRY_AUGS = os.environ.get("STAGE5_ARC_AGI_GEOMETRY_AUGS", "all")
TRACE_MODE = os.environ.get("STAGE5_ARC_AGI_TRACE_MODE", "symbolic_program")
TRACE_FILTER = os.environ.get("STAGE5_ARC_AGI_TRACE_FILTER", "covered")
MAX_LENGTH = int(os.environ.get("STAGE5_ARC_AGI_MAX_LENGTH", "1024"))
MAX_TOTAL_TOKENS = int(os.environ.get("STAGE5_ARC_AGI_MAX_TOTAL_TOKENS", "2048"))
TRAIN_STEPS = int(os.environ.get("STAGE5_ARC_AGI_DENSE_TRAIN_STEPS", os.environ.get("STAGE5_ARC_AGI_TRAIN_STEPS", "300")))
SAVE_EVERY = int(os.environ.get("STAGE5_ARC_AGI_DENSE_SAVE_EVERY", os.environ.get("STAGE5_ARC_AGI_SAVE_EVERY", "150")))
LEARNING_RATE = float(os.environ.get("STAGE5_ARC_AGI_DENSE_LR", os.environ.get("STAGE5_ARC_AGI_LR", "8e-6")))
DISTILL_ENABLED = os.environ.get("STAGE5_ARC_AGI_DENSE_DISTILL", os.environ.get("STAGE5_ARC_AGI_DISTILL", "0")).strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
DISTILL_WEIGHT = float(os.environ.get("STAGE5_ARC_AGI_DISTILL_WEIGHT", "0.1"))
DISTILL_TEMPERATURE = float(os.environ.get("STAGE5_ARC_AGI_DISTILL_TEMPERATURE", "1.0"))
DISTILL_ON = os.environ.get("STAGE5_ARC_AGI_DISTILL_ON", "response")
MAX_NEW_TOKENS = int(os.environ.get("STAGE5_ARC_AGI_MAX_NEW_TOKENS", "512"))
GRID_FORMAT = os.environ.get("STAGE5_ARC_AGI_GRID_FORMAT", "compact")
PROGRAM_PARSE_MODE = os.environ.get("STAGE5_ARC_AGI_PROGRAM_PARSE_MODE", "fallback")
SELECTION_STRATEGY = os.environ.get("STAGE5_ARC_AGI_SELECTION_STRATEGY", "heuristic")
DENSE_LORA_LAYER_RANGE = os.environ.get("STAGE5_ARC_AGI_DENSE_LORA_LAYER_RANGE", "6,18")
DTYPE = os.environ.get("DTYPE", "bfloat16")
ADAPTER_DTYPE = os.environ.get("ADAPTER_DTYPE", "float32")
DEVICE = os.environ.get("DEVICE", "cuda")
PUSH_RESULTS = os.environ.get("STAGE5_ARC_AGI_DENSE_SFT_PUSH", "1").strip().lower() in {"1", "true", "yes", "y"}

TRAIN_JSONL = ROOT / "data" / f"{RUN_ID}_train.jsonl"
VAL_JSONL = ROOT / "data" / f"{RUN_ID}_val.jsonl"


def run(cmd: list[str], *, check: bool = True, log_name: str | None = None) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(map(str, cmd)))
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


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mount_drive_if_possible() -> None:
    if Path("/content/drive/MyDrive").exists():
        return
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")
    except Exception as exc:  # pragma: no cover - Colab only
        print(f"Drive mount skipped/failed: {exc}")


def restore_phase1_checkpoint() -> None:
    if PHASE1_CKPT.exists():
        return
    mount_drive_if_possible()
    drive_root = Path(os.environ.get("DRIVE_BACKUP_DIR", "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts"))
    candidates = sorted(drive_root.rglob("phase1_step_500.pt")) if drive_root.exists() else []
    for candidate in candidates:
        if BASE_RUN_ID in str(candidate):
            PHASE1_CKPT.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, PHASE1_CKPT)
            print(f"restored_phase1_checkpoint={candidate} -> {PHASE1_CKPT}")
            return
    if candidates:
        PHASE1_CKPT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidates[0], PHASE1_CKPT)
        print(f"restored_phase1_checkpoint={candidates[0]} -> {PHASE1_CKPT}")
        return
    raise FileNotFoundError(f"Missing Phase1 checkpoint: {PHASE1_CKPT}")


def clone_or_update(repo_url: str, target: Path) -> None:
    if target.exists() and (target / ".git").exists():
        run(["git", "-C", str(target), "pull", "--ff-only"], check=False)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", repo_url, str(target)])


def resolve_repo_dir() -> Path:
    if ARC_VERSION == "2":
        repo_dir = DATA_ROOT / "ARC-AGI-2"
        clone_or_update(ARC_AGI_2_REPO, repo_dir)
    else:
        repo_dir = DATA_ROOT / "ARC-AGI"
        clone_or_update(ARC_AGI_1_REPO, repo_dir)
    return repo_dir


def resolve_split_path(repo_dir: Path, split: str) -> Path:
    if user_path := os.environ.get(f"STAGE5_ARC_AGI_{split.upper()}_PATH"):
        return Path(user_path)
    candidates = [
        repo_dir / "data" / split,
        repo_dir / split,
        repo_dir / "data" / f"{split}_challenges",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find ARC-AGI split {split!r} under {repo_dir}")


def prepare_sft(train_path: Path) -> None:
    run(
        [
            sys.executable,
            "training/prepare_arc_agi_sft_jsonl.py",
            "--tasks_path",
            str(train_path),
            "--tokenizer_name",
            MODEL_NAME,
            "--output_jsonl",
            path_for_cli(TRAIN_JSONL),
            "--val_jsonl",
            path_for_cli(VAL_JSONL),
            "--limit",
            str(TRAIN_TASK_LIMIT),
            "--augment_color_permutations",
            str(COLOR_AUGS),
            "--augment_geometries",
            GEOMETRY_AUGS,
            "--grid_format",
            GRID_FORMAT,
            "--trace_mode",
            TRACE_MODE,
            "--trace_filter",
            TRACE_FILTER,
            "--max_total_tokens",
            str(MAX_TOTAL_TOKENS),
        ],
        log_name="prepare_arc_agi_sft.log",
    )


def train_dense_lora() -> Path:
    cfg = {
        "model_name": MODEL_NAME,
        "dtype": DTYPE,
        "adapter_dtype": ADAPTER_DTYPE,
        "layer_split": DENSE_LORA_LAYER_RANGE,
        "max_length": MAX_LENGTH,
        "max_loops": 4,
        "batch_size": 1,
        "learning_rate": LEARNING_RATE,
        "weight_decay": 0.0,
        "max_grad_norm": 0.3,
        "max_steps": TRAIN_STEPS,
        "save_every": SAVE_EVERY,
        "log_every": 25,
        "train_on_prompt": False,
        "output_dir": path_for_cli(RUN_DIR / "dense_lora_arc_agi"),
        "lora": {"enabled": True, "rank": 8, "alpha": 16, "dropout": 0.0, "layer_range": DENSE_LORA_LAYER_RANGE},
        "distillation": {
            "enabled": DISTILL_ENABLED,
            "weight": DISTILL_WEIGHT,
            "temperature": DISTILL_TEMPERATURE,
            "on": DISTILL_ON,
        },
    }
    cfg_path = RUN_DIR / "dense_lora_arc_agi.yaml"
    write_yaml(cfg_path, cfg)
    run(
        [
            sys.executable,
            "training/train_dense_lora.py",
            "--config",
            path_for_cli(cfg_path),
            "--train_jsonl",
            path_for_cli(TRAIN_JSONL),
            "--device",
            DEVICE,
        ],
        log_name="dense_lora_arc_agi_train.log",
    )
    checkpoint = RUN_DIR / "dense_lora_arc_agi" / f"dense_lora_step_{TRAIN_STEPS}.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def eval_arc(label: str, mode: str, tasks_path: Path, checkpoint: Path | None = None) -> dict[str, Any]:
    summary_json = RUN_DIR / f"{label}_summary.json"
    cmd = [
        sys.executable,
        "eval/eval_arc_agi.py",
        "--tasks_path",
        str(tasks_path),
        "--limit",
        str(EVAL_TASK_LIMIT),
        "--mode",
        mode,
        "--max_new_tokens",
        str(MAX_NEW_TOKENS),
        "--grid_format",
        GRID_FORMAT,
        "--program_parse_mode",
        PROGRAM_PARSE_MODE,
        "--selection_strategy",
        SELECTION_STRATEGY,
        "--dtype",
        DTYPE,
        "--adapter_dtype",
        ADAPTER_DTYPE,
        "--device",
        DEVICE,
        "--output_jsonl",
        path_for_cli(RUN_DIR / f"{label}_candidates.jsonl"),
        "--summary_json",
        path_for_cli(summary_json),
        "--summary_md",
        path_for_cli(RUN_DIR / f"{label}_summary.md"),
    ]
    if checkpoint is not None:
        cmd += ["--checkpoint", path_for_cli(checkpoint)]
    if mode == "base" and checkpoint is not None:
        cmd += ["--base_lora_layer_range", DENSE_LORA_LAYER_RANGE]
    if mode != "base":
        cmd += ["--max_loops", "4", "--num_candidates", "1"]
    run(cmd, log_name=f"{label}_eval.log")
    return read_json(summary_json)


def summary_delta(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    cand = candidate["summary"]
    ref = reference["summary"]
    return {
        "selected_exact_delta": int(cand.get("selected_exact", 0)) - int(ref.get("selected_exact", 0)),
        "best_of_k_exact_delta": int(cand.get("best_of_k_exact", 0)) - int(ref.get("best_of_k_exact", 0)),
        "first_exact_delta": int(cand.get("first_exact", 0)) - int(ref.get("first_exact", 0)),
        "valid_candidate_rate_delta": float(cand.get("valid_candidate_rate", 0.0)) - float(ref.get("valid_candidate_rate", 0.0)),
    }


def compare_eval_payloads(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    reference_label: str,
    candidate_label: str,
) -> dict[str, Any]:
    payload = compare_payloads(
        reference,
        candidate,
        reference_label=reference_label,
        candidate_label=candidate_label,
        bootstrap_samples=1000,
        seed=0,
    )
    name = f"{candidate_label}_vs_{reference_label}"
    (RUN_DIR / f"{name}_paired_comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_comparison_markdown(RUN_DIR / f"{name}_paired_comparison.md", payload)
    return payload


def paired_comparisons(
    *,
    base: dict[str, Any],
    dense_tuned: dict[str, Any],
    phase1_start: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dense_tuned_vs_base": compare_eval_payloads(
            base,
            dense_tuned,
            reference_label="base",
            candidate_label="dense_tuned",
        ),
        "dense_tuned_vs_phase1_start": compare_eval_payloads(
            phase1_start,
            dense_tuned,
            reference_label="phase1_start",
            candidate_label="dense_tuned",
        ),
        "phase1_start_vs_base": compare_eval_payloads(
            base,
            phase1_start,
            reference_label="base",
            candidate_label="phase1_start",
        ),
    }


def paired_selected_line(name: str, comparison: dict[str, Any]) -> str:
    stats = (comparison.get("metrics") or {}).get("selected_exact") or {}
    return (
        f"- {name}: selected delta `{stats.get('delta_exact')}` "
        f"({stats.get('wins')}/{stats.get('losses')}/{stats.get('ties')} W/L/T, "
        f"p `{stats.get('sign_test_p_value')}`)"
    )


def audit_training_signal(metadata: dict[str, Any]) -> dict[str, Any]:
    payload = summarize_training_signal(TRAIN_JSONL, val_jsonl=VAL_JSONL, metadata=metadata)
    write_training_signal_report(payload, RUN_DIR / "training_signal.json", RUN_DIR / "training_signal.md")
    print((RUN_DIR / "training_signal.md").read_text(encoding="utf-8"))
    return payload


def backup_to_drive() -> None:
    mount_drive_if_possible()
    drive_root = Path(os.environ.get("DRIVE_BACKUP_DIR", "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts"))
    if not drive_root.exists():
        print(f"Drive backup skipped; missing {drive_root}")
        return
    backup = drive_root / RUN_ID
    backup.mkdir(parents=True, exist_ok=True)
    shutil.copytree(RUN_DIR, backup / "run_dir", dirs_exist_ok=True)
    for source in [TRAIN_JSONL, VAL_JSONL]:
        if source.exists():
            target = backup / "data" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    print(f"backed_up_to={backup}")


def git_commit_results() -> None:
    paths: list[str] = []
    for pattern in ["*.yaml", "*.json", "*.md", "*.log", "*.jsonl"]:
        paths.extend(str(path.relative_to(ROOT)) for path in RUN_DIR.glob(pattern))
    if paths:
        run(["git", "add", "-f", *paths], check=False)
    if run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("No dense SFT outputs changed.")
        return
    run(["git", "commit", "-m", "Record Stage 5 dense ARC-AGI SFT control"])
    run(["git", "push", "origin", "main"], check=False)


def write_report(payload: dict[str, Any]) -> None:
    (RUN_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"# Stage 5 Dense ARC-AGI SFT Control - {RUN_ID}",
        "",
        f"- ARC version: `{ARC_VERSION}`",
        f"- Train limit: `{TRAIN_TASK_LIMIT}`",
        f"- Eval limit: `{EVAL_TASK_LIMIT}`",
        f"- Trace mode/filter: `{TRACE_MODE}` / `{TRACE_FILTER}`",
        f"- Dense LoRA layer range: `{DENSE_LORA_LAYER_RANGE}`",
        f"- Dense checkpoint: `{payload['dense_checkpoint']}`",
        "",
        "## Results",
        "",
        f"- Base: `{payload['base']['summary']}`",
        f"- Dense tuned: `{payload['dense_tuned']['summary']}`",
        f"- Phase1 recurrent start: `{payload['phase1_start']['summary']}`",
        "",
        "## Deltas",
        "",
        f"- Dense tuned vs base: `{payload['deltas']['dense_tuned_vs_base']}`",
        f"- Dense tuned vs Phase1 start: `{payload['deltas']['dense_tuned_vs_phase1_start']}`",
        f"- Phase1 start vs base: `{payload['deltas']['phase1_start_vs_base']}`",
        "",
        "## Paired Selected-Answer Evidence",
        "",
    ]
    for name, comparison in payload.get("paired_comparisons", {}).items():
        lines.append(paired_selected_line(name, comparison))
    lines.extend(
        [
            "",
            f"Training-signal audit: `{path_for_cli(RUN_DIR / 'training_signal.md')}`",
        ]
    )
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))


def main() -> int:
    if not (Path("/content").exists() or os.environ.get("ALLOW_LOCAL_STAGE5_RUN") == "1"):
        print("Run from Colab. Configure with STAGE5_ARC_AGI_* environment variables.")
        return 2
    restore_phase1_checkpoint()
    repo_dir = resolve_repo_dir()
    train_path = resolve_split_path(repo_dir, TRAIN_SPLIT)
    eval_path = resolve_split_path(repo_dir, EVAL_SPLIT)
    prepare_sft(train_path)
    metadata = {
        "run_id": RUN_ID,
        **model_metadata(MODEL_NAME),
        "arc_version": ARC_VERSION,
        "train_split": TRAIN_SPLIT,
        "eval_split": EVAL_SPLIT,
        "train_task_limit": TRAIN_TASK_LIMIT,
        "eval_task_limit": EVAL_TASK_LIMIT,
        "color_augmentations": COLOR_AUGS,
        "geometry_augmentations": GEOMETRY_AUGS,
        "trace_mode": TRACE_MODE,
        "trace_filter": TRACE_FILTER,
        "grid_format": GRID_FORMAT,
        "program_parse_mode": PROGRAM_PARSE_MODE,
        "selection_strategy": SELECTION_STRATEGY,
        "train_steps": TRAIN_STEPS,
        "save_every": SAVE_EVERY,
        "learning_rate": LEARNING_RATE,
        "dense_lora_layer_range": DENSE_LORA_LAYER_RANGE,
    }
    training_signal = audit_training_signal(metadata)
    dense_checkpoint = train_dense_lora()

    base = eval_arc("base", "base", eval_path)
    dense_tuned = eval_arc("dense_tuned", "base", eval_path, dense_checkpoint)
    phase1_start = eval_arc("phase1_start", "phase1", eval_path, PHASE1_CKPT)
    paired = paired_comparisons(base=base, dense_tuned=dense_tuned, phase1_start=phase1_start)

    payload = {
        "run_id": RUN_ID,
        "kind": "dense_sft_control",
        "metadata": metadata,
        "training_signal": training_signal,
        "dense_checkpoint": path_for_cli(dense_checkpoint),
        "base": base,
        "dense_tuned": dense_tuned,
        "phase1_start": phase1_start,
        "deltas": {
            "dense_tuned_vs_base": summary_delta(dense_tuned, base),
            "dense_tuned_vs_phase1_start": summary_delta(dense_tuned, phase1_start),
            "phase1_start_vs_base": summary_delta(phase1_start, base),
        },
        "paired_comparisons": paired,
    }
    write_report(payload)
    backup_to_drive()
    if PUSH_RESULTS:
        git_commit_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
