"""Continue deterministic recurrent recovery and evaluate checkpoint ladder.

Stage 4 showed that the recurrent surgery is recoverable:

    base Qwen 0.5B ARC-128:       72/128
    trained Phase1 recurrent:     70/128
    trained Phase2/SVGD:          69/128

Stage 5A therefore prioritizes deterministic recurrent competence before more
particle/SVGD training. This runner continues Phase1 from the best Stage 4
checkpoint, evaluates every saved checkpoint on ARC-Challenge proxy slices and
the Opus validation split, and backs checkpoints up to Drive.

This is still an ARC-Challenge proxy, not ARC-AGI. The purpose is to establish
that the architecture surgery can be repaired and improved before building the
ARC-AGI-specific training/evaluation loop.
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

from colab.stage5_model_metadata import model_metadata  # noqa: E402

BASE_RUN_ID = os.environ.get("STAGE5_BASE_RUN_ID", "stage4_opus_a100_20260620")
RUN_ID = os.environ.get("STAGE5_RUN_ID") or time.strftime("stage5_phase1_recovery_%Y%m%d_%H%M%S")
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
DATASET_ID = os.environ.get("OPUS_DATASET_ID", "lordx64/reasoning-distill-opus-4-7-max-sft")
DATASET_LIMIT = int(os.environ.get("STAGE5_OPUS_LIMIT", os.environ.get("OPUS_LIMIT", "6000")))
VAL_FRACTION = float(os.environ.get("STAGE5_OPUS_VAL_FRACTION", "0.05"))
MAX_TOTAL_TOKENS = int(os.environ.get("STAGE5_OPUS_MAX_TOTAL_TOKENS", "1024"))
MAX_LENGTH = int(os.environ.get("STAGE5_MAX_LENGTH", "512"))
EXTRA_STEPS = int(os.environ.get("STAGE5_PHASE1_EXTRA_STEPS", "1000"))
SAVE_EVERY = int(os.environ.get("STAGE5_PHASE1_SAVE_EVERY", "250"))
LEARNING_RATE = float(os.environ.get("STAGE5_PHASE1_LR", "8e-6"))
BETA = float(os.environ.get("STAGE5_PHASE1_BETA", "0.08"))
MAX_GRAD_NORM = float(os.environ.get("STAGE5_PHASE1_MAX_GRAD_NORM", "0.3"))
ARC_LIMIT = int(os.environ.get("STAGE5_ARC_LIMIT", "256"))
ARC_SPLIT = os.environ.get("STAGE5_ARC_SPLIT", "validation")
ARC_SEED = int(os.environ.get("STAGE5_ARC_SEED", "0"))
RUN_FULL_ARC_FINAL = os.environ.get("STAGE5_RUN_FULL_ARC_FINAL", "0").strip().lower() in {"1", "true", "yes", "y"}
DTYPE = os.environ.get("DTYPE", "bfloat16")
ADAPTER_DTYPE = os.environ.get("ADAPTER_DTYPE", "float32")
DEVICE = os.environ.get("DEVICE", "cuda")
PUSH_RESULTS = os.environ.get("STAGE5_PUSH", "1").strip().lower() in {"1", "true", "yes", "y"}
DISTILL_ENABLED = os.environ.get("STAGE5_PHASE1_DISTILL", "0").strip().lower() in {"1", "true", "yes", "y"}
DISTILL_WEIGHT = float(os.environ.get("STAGE5_PHASE1_DISTILL_WEIGHT", "0.1"))
DISTILL_TEMPERATURE = float(os.environ.get("STAGE5_PHASE1_DISTILL_TEMPERATURE", "2.0"))
DISTILL_ON = os.environ.get("STAGE5_PHASE1_DISTILL_ON", "response")
DISTILL_TEACHER_MODEL_NAME = os.environ.get("STAGE5_PHASE1_DISTILL_TEACHER_MODEL_NAME", MODEL_NAME)
DISTILL_DTYPE = os.environ.get("STAGE5_PHASE1_DISTILL_DTYPE", DTYPE)

BASE_RUN_DIR = ROOT / "outputs" / "stage4" / BASE_RUN_ID
DEFAULT_RESUME = BASE_RUN_DIR / "phase1" / "phase1_step_500.pt"
RESUME_FROM = Path(os.environ.get("STAGE5_RESUME_FROM", str(DEFAULT_RESUME)))
if not RESUME_FROM.is_absolute():
    RESUME_FROM = ROOT / RESUME_FROM

TRAIN_JSONL = ROOT / "data" / f"{RUN_ID}_opus_train.jsonl"
VAL_JSONL = ROOT / "data" / f"{RUN_ID}_opus_val.jsonl"
ARC_JSONL = ROOT / "data" / f"{RUN_ID}_arc{ARC_LIMIT}.jsonl"
FULL_ARC_JSONL = ROOT / "data" / f"{RUN_ID}_arc_full.jsonl"


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


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def phase1_distillation_config() -> dict[str, Any]:
    return {
        "enabled": DISTILL_ENABLED,
        "weight": DISTILL_WEIGHT,
        "temperature": DISTILL_TEMPERATURE,
        "on": DISTILL_ON,
        "teacher_model_name": DISTILL_TEACHER_MODEL_NAME,
        "dtype": DISTILL_DTYPE,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def path_for_cli(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def mount_drive_if_possible() -> None:
    if Path("/content/drive/MyDrive").exists():
        return
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")
    except Exception as exc:  # pragma: no cover - Colab only
        print(f"Drive mount skipped/failed: {exc}")


def find_drive_checkpoint() -> Path | None:
    drive_root = Path(os.environ.get("DRIVE_BACKUP_DIR", "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts"))
    candidates = [
        drive_root / BASE_RUN_ID / "run_dir" / "phase1" / "phase1_step_500.pt",
        drive_root / BASE_RUN_ID / "phase1" / "phase1_step_500.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if not drive_root.exists():
        return None
    matches = sorted(drive_root.rglob("phase1_step_500.pt"), key=lambda p: len(str(p)))
    for match in matches:
        if BASE_RUN_ID in str(match):
            return match
    return matches[0] if matches else None


def restore_resume_checkpoint() -> None:
    if RESUME_FROM.exists():
        return
    mount_drive_if_possible()
    source = find_drive_checkpoint()
    if source is None or not source.exists():
        raise FileNotFoundError(f"Missing resume checkpoint {RESUME_FROM} and no Drive fallback was found.")
    RESUME_FROM.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, RESUME_FROM)
    print(f"restored_resume_checkpoint={source} -> {RESUME_FROM}")


def prepare_opus() -> None:
    if TRAIN_JSONL.exists() and VAL_JSONL.exists():
        return
    run(
        [
            sys.executable,
            "training/prepare_hf_reasoning_jsonl.py",
            "--dataset_id",
            DATASET_ID,
            "--tokenizer_name",
            MODEL_NAME,
            "--output_jsonl",
            path_for_cli(TRAIN_JSONL),
            "--val_jsonl",
            path_for_cli(VAL_JSONL),
            "--limit",
            str(DATASET_LIMIT),
            "--val_fraction",
            str(VAL_FRACTION),
            "--max_total_tokens",
            str(MAX_TOTAL_TOKENS),
        ],
        log_name="prepare_opus.log",
    )


def prepare_arc(path: Path, *, limit: int | None) -> None:
    cmd = [
        sys.executable,
        "eval/prepare_arc_mcq.py",
        "--config",
        "ARC-Challenge",
        "--split",
        ARC_SPLIT,
        "--seed",
        str(ARC_SEED),
        "--output_jsonl",
        path_for_cli(path),
    ]
    if limit and limit > 0:
        cmd += ["--limit", str(limit)]
    run(cmd, log_name=f"prepare_{path.stem}.log")


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


def summarize_mcq(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    by_aggregate: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_aggregate.setdefault(str(row["aggregate"]), []).append(row)
    return {
        aggregate: {
            "correct": sum(1 for row in aggregate_rows if row["hit"]),
            "total": len(aggregate_rows),
            "accuracy": sum(1 for row in aggregate_rows if row["hit"]) / max(len(aggregate_rows), 1),
        }
        for aggregate, aggregate_rows in sorted(by_aggregate.items())
    }


def mean_accuracy(summary: dict[str, Any]) -> float | None:
    metric = summary.get("mean")
    return None if metric is None else float(metric["accuracy"])


def helped_hurt(variant_path: Path, reference_path: Path) -> dict[str, int]:
    reference = {row["id"]: row for row in read_jsonl(reference_path) if row["aggregate"] == "mean"}
    helped = hurt = tied = changed = 0
    for row in read_jsonl(variant_path):
        if row["aggregate"] != "mean":
            continue
        base = reference[row["id"]]
        if row["prediction"] != base["prediction"]:
            changed += 1
        if row["hit"] and not base["hit"]:
            helped += 1
        elif base["hit"] and not row["hit"]:
            hurt += 1
        else:
            tied += 1
    return {"helped": helped, "hurt": hurt, "tied": tied, "prediction_changes": changed}


def eval_jsonl(label: str, checkpoint: Path) -> dict[str, float]:
    proc = run(
        [
            sys.executable,
            "eval/eval_jsonl.py",
            "--model_name",
            MODEL_NAME,
            "--data_jsonl",
            path_for_cli(VAL_JSONL),
            "--checkpoint",
            path_for_cli(checkpoint),
            "--split",
            "6,18",
            "--max_loops",
            "4",
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
        ],
        log_name=f"{label}_val.log",
    )
    return summarize_jsonl_eval(proc.stdout)


def eval_arc(label: str, mode: str, data_jsonl: Path, checkpoint: Path | None = None) -> Path:
    output = RUN_DIR / f"{data_jsonl.stem}_{label}.jsonl"
    if output.exists():
        output.unlink()
    cmd = [
        sys.executable,
        "eval/eval_mcq.py",
        "--data_jsonl",
        path_for_cli(data_jsonl),
        "--prompt_style",
        "with_options",
        "--score_target",
        "label",
        "--mode",
        mode,
        "--dtype",
        DTYPE,
        "--adapter_dtype",
        ADAPTER_DTYPE,
        "--device",
        DEVICE,
        "--seed",
        "0",
        "--aggregate",
        "mean",
    ]
    if mode == "phase1":
        assert checkpoint is not None
        cmd += [
            "--checkpoint",
            path_for_cli(checkpoint),
            "--max_loops",
            "4",
            "--num_trajectories",
            "1",
        ]
    run(cmd + ["--output_jsonl", path_for_cli(output)], log_name=f"{data_jsonl.stem}_{label}.log")
    return output


def train_phase1() -> list[Path]:
    cfg = {
        "model_name": MODEL_NAME,
        "dtype": DTYPE,
        "adapter_dtype": ADAPTER_DTYPE,
        "layer_split": "6,18",
        "max_length": MAX_LENGTH,
        "max_loops": 4,
        "initial_halt_prob": 0.15,
        "beta": BETA,
        "batch_size": 1,
        "learning_rate": LEARNING_RATE,
        "weight_decay": 0.0,
        "max_grad_norm": MAX_GRAD_NORM,
        "max_steps": EXTRA_STEPS,
        "save_every": SAVE_EVERY,
        "log_every": 25,
        "train_on_prompt": False,
        "output_dir": path_for_cli(RUN_DIR / "phase1"),
        "resume_from": path_for_cli(RESUME_FROM),
        "lora": {"enabled": True, "rank": 8, "alpha": 16, "dropout": 0.0},
    }
    if DISTILL_ENABLED:
        cfg["distillation"] = phase1_distillation_config()
    cfg_path = RUN_DIR / "phase1_continue.yaml"
    write_yaml(cfg_path, cfg)
    run(
        [
            sys.executable,
            "training/train_phase1_ponder.py",
            "--config",
            path_for_cli(cfg_path),
            "--train_jsonl",
            path_for_cli(TRAIN_JSONL),
            "--device",
            DEVICE,
        ],
        log_name="phase1_continue_train.log",
    )
    checkpoints = sorted((RUN_DIR / "phase1").glob("phase1_step_*.pt"), key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints produced under {RUN_DIR / 'phase1'}")
    return checkpoints


def backup_to_drive() -> None:
    mount_drive_if_possible()
    drive_root = Path(os.environ.get("DRIVE_BACKUP_DIR", "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts"))
    if not drive_root.exists():
        print(f"Drive backup skipped; missing {drive_root}")
        return
    backup = drive_root / RUN_ID
    backup.mkdir(parents=True, exist_ok=True)
    shutil.copytree(RUN_DIR, backup / "run_dir", dirs_exist_ok=True)
    for source in [TRAIN_JSONL, VAL_JSONL, ARC_JSONL, FULL_ARC_JSONL]:
        if source.exists():
            target = backup / "data" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    print(f"backed_up_to={backup}")


def git_commit_results() -> None:
    for pattern in ["*.yaml", "*.json", "*.md", "*.log", "*.jsonl"]:
        run(["git", "add", "-f", str((RUN_DIR / pattern).relative_to(ROOT))], check=False)
    if run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("No Stage 5 summary outputs changed.")
        return
    run(["git", "commit", "-m", "Record Stage 5 Phase1 recovery ladder"])
    run(["git", "push", "origin", "main"], check=False)


def main() -> int:
    metadata = {
        "run_id": RUN_ID,
        "base_run_id": BASE_RUN_ID,
        **model_metadata(MODEL_NAME),
        "dataset_id": DATASET_ID,
        "dataset_limit": DATASET_LIMIT,
        "val_fraction": VAL_FRACTION,
        "max_total_tokens": MAX_TOTAL_TOKENS,
        "max_length": MAX_LENGTH,
        "extra_steps": EXTRA_STEPS,
        "save_every": SAVE_EVERY,
        "learning_rate": LEARNING_RATE,
        "beta": BETA,
        "max_grad_norm": MAX_GRAD_NORM,
        "distillation": phase1_distillation_config(),
        "arc_limit": ARC_LIMIT,
        "arc_split": ARC_SPLIT,
        "arc_seed": ARC_SEED,
        "run_full_arc_final": RUN_FULL_ARC_FINAL,
        "dtype": DTYPE,
        "adapter_dtype": ADAPTER_DTYPE,
        "resume_from": path_for_cli(RESUME_FROM),
    }
    (RUN_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))

    restore_resume_checkpoint()
    prepare_opus()
    prepare_arc(ARC_JSONL, limit=ARC_LIMIT)

    base_arc_path = eval_arc("base_label", "base", ARC_JSONL)
    start_arc_path = eval_arc("phase1_start_label", "phase1", ARC_JSONL, RESUME_FROM)
    start_val = eval_jsonl("phase1_start", RESUME_FROM)

    checkpoints = train_phase1()
    checkpoint_results: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        label = checkpoint.stem
        val_metrics = eval_jsonl(label, checkpoint)
        arc_path = eval_arc(f"{label}_label", "phase1", ARC_JSONL, checkpoint)
        checkpoint_results.append(
            {
                "checkpoint": path_for_cli(checkpoint),
                "val": val_metrics,
                "arc_path": path_for_cli(arc_path),
                "arc": summarize_mcq(arc_path),
                "comparison_to_start": helped_hurt(arc_path, start_arc_path),
                "comparison_to_base": helped_hurt(arc_path, base_arc_path),
            }
        )

    full_arc_result: dict[str, Any] | None = None
    if RUN_FULL_ARC_FINAL:
        prepare_arc(FULL_ARC_JSONL, limit=None)
        best_result = max(checkpoint_results, key=lambda item: mean_accuracy(item["arc"]) or -1.0)
        best_ckpt = ROOT / best_result["checkpoint"]
        full_base_path = eval_arc("full_base_label", "base", FULL_ARC_JSONL)
        full_start_path = eval_arc("full_phase1_start_label", "phase1", FULL_ARC_JSONL, RESUME_FROM)
        full_best_path = eval_arc("full_phase1_best_label", "phase1", FULL_ARC_JSONL, best_ckpt)
        full_arc_result = {
            "base": summarize_mcq(full_base_path),
            "phase1_start": summarize_mcq(full_start_path),
            "phase1_best": summarize_mcq(full_best_path),
            "best_checkpoint": path_for_cli(best_ckpt),
            "best_vs_start": helped_hurt(full_best_path, full_start_path),
            "best_vs_base": helped_hurt(full_best_path, full_base_path),
        }

    base_arc = summarize_mcq(base_arc_path)
    start_arc = summarize_mcq(start_arc_path)
    best = max(checkpoint_results, key=lambda item: mean_accuracy(item["arc"]) or -1.0)
    summary = {
        "metadata": metadata,
        "base_arc": base_arc,
        "phase1_start": {"checkpoint": path_for_cli(RESUME_FROM), "val": start_val, "arc": start_arc},
        "checkpoints": checkpoint_results,
        "best_checkpoint": best,
        "full_arc_final": full_arc_result,
    }
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    base_acc = mean_accuracy(base_arc)
    start_acc = mean_accuracy(start_arc)
    best_acc = mean_accuracy(best["arc"])
    lines = [
        f"# Stage 5 Phase1 Recovery Ladder - {RUN_ID}",
        "",
        "## Question",
        "Can continued deterministic recurrent training close or reverse the remaining gap to base Qwen before further particle/SVGD work?",
        "",
        "## ARC-Challenge Proxy",
        f"- Base Qwen: `{base_arc}`",
        f"- Phase1 start ({path_for_cli(RESUME_FROM)}): `{start_arc}`",
        f"- Best checkpoint: `{best['checkpoint']}`",
        f"- Best checkpoint ARC: `{best['arc']}`",
        f"- Start lift: `{None if best_acc is None or start_acc is None else best_acc - start_acc}`",
        f"- Gap to base: `{None if best_acc is None or base_acc is None else best_acc - base_acc}`",
        "",
        "## Checkpoint Ladder",
    ]
    for item in checkpoint_results:
        lines.append(
            f"- {item['checkpoint']}: arc={item['arc']} val={item['val']} "
            f"vs_start={item['comparison_to_start']} vs_base={item['comparison_to_base']}"
        )
    if full_arc_result is not None:
        lines.extend(["", "## Full ARC-Challenge Final", json.dumps(full_arc_result, indent=2)])
    lines.extend(
        [
            "",
            "## Decision Rule",
            "If the best checkpoint improves over Phase1 start, keep extending deterministic recovery. "
            "If it regresses while validation improves, add base-logit distillation before more Opus training.",
            "",
            "Note: this is not ARC-AGI. It is the competence recovery proxy before building the ARC-AGI-specific solver/eval loop.",
        ]
    )
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))

    backup_to_drive()
    if PUSH_RESULTS:
        git_commit_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
