"""Bounded max_loops=1 direct-route preservation probe.

This runner is intentionally narrower than the ARC-mix recovery runners. It
tests whether the recurrent checkpoint can preserve base Qwen behavior on
base-correct ARC-Easy direct examples when forced through the identity-shaped
``max_loops=1`` path. If the loop-1 path already matches base, it exits before
training. Otherwise it trains one tiny base-correct, label-balanced direct SFT
probe with strong frozen-base KL on the answer position.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get("STAGE5_DIRECT_PRESERVE_RUN_ID") or time.strftime(
    "stage5_direct_preservation_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
DTYPE = os.environ.get("TRAIN_DTYPE", "bfloat16")
ADAPTER_DTYPE = os.environ.get("ADAPTER_DTYPE", "float32")
DEVICE = os.environ.get("DEVICE", "cuda")
SOURCE_SUMMARY = os.environ.get(
    "STAGE5_DIRECT_PRESERVE_SOURCE_SUMMARY",
    os.environ.get("STAGE5_ARC_AGI_CURRENT_SOURCE_SUMMARY", "config/stage5_current_source_summary.txt"),
)
ARC_TRAIN_LIMIT = os.environ.get("STAGE5_DIRECT_PRESERVE_ARC_TRAIN_LIMIT", "512")
ARC_EVAL_LIMIT = int(os.environ.get("STAGE5_DIRECT_PRESERVE_ARC_EVAL_LIMIT", "128"))
MIN_BASE_MARGIN = float(os.environ.get("STAGE5_DIRECT_PRESERVE_MIN_BASE_MARGIN", "1.0"))
MAX_ROWS_PER_LABEL = os.environ.get("STAGE5_DIRECT_PRESERVE_MAX_ROWS_PER_LABEL", "")
PROMPT_STYLE = os.environ.get("STAGE5_DIRECT_PRESERVE_PROMPT_STYLE", "with_options").strip()
SCORE_TARGET = os.environ.get("STAGE5_DIRECT_PRESERVE_SCORE_TARGET", "label").strip()
MAX_STEPS = int(os.environ.get("STAGE5_DIRECT_PRESERVE_MAX_STEPS", "75"))
SAVE_EVERY = int(os.environ.get("STAGE5_DIRECT_PRESERVE_SAVE_EVERY", "25"))
LEARNING_RATE = float(os.environ.get("STAGE5_DIRECT_PRESERVE_LR", "5e-7"))
BETA = float(os.environ.get("STAGE5_DIRECT_PRESERVE_BETA", "0.02"))
DISTILL_WEIGHT = float(os.environ.get("STAGE5_DIRECT_PRESERVE_DISTILL_WEIGHT", "1.0"))
DISTILL_TEMPERATURE = float(os.environ.get("STAGE5_DIRECT_PRESERVE_DISTILL_TEMPERATURE", "2.0"))
MIX_SEED = int(os.environ.get("STAGE5_DIRECT_PRESERVE_SEED", "0"))
MAX_PREDICTION_SHIFT = int(os.environ.get("STAGE5_DIRECT_PRESERVE_MAX_PREDICTION_SHIFT", "8"))
MIN_MARGIN_DELTA = float(os.environ.get("STAGE5_DIRECT_PRESERVE_MIN_MARGIN_DELTA", "0.0"))
SKIP_TRAIN_IF_LOOP1_MATCHES_BASE = os.environ.get(
    "STAGE5_DIRECT_PRESERVE_SKIP_TRAIN_IF_LOOP1_MATCHES_BASE",
    "1",
) not in {"0", "false", "False"}
PRECHECK_ONLY = os.environ.get("STAGE5_DIRECT_PRESERVE_PRECHECK_ONLY", "0") in {"1", "true", "True"}


def path_for_cli(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def resolve_path(value: str | Path) -> Path:
    raw = str(value).strip()
    if raw == "config/stage5_current_source_summary.txt":
        pointer = ROOT / raw
        if pointer.exists():
            for line in pointer.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    raw = stripped
                    break
    path = Path(raw.replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in resolve_path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def run(cmd: list[str], *, log_name: str | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(map(str, cmd)), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(proc.stdout, flush=True)
    if log_name:
        (RUN_DIR / log_name).write_text(proc.stdout, encoding="utf-8")
    if check and proc.returncode:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}")
    return proc


def source_payload_and_checkpoint() -> tuple[Path, dict[str, Any], Path]:
    source_summary = resolve_path(SOURCE_SUMMARY)
    source_payload = read_json(source_summary)
    if source_payload.get("kind") == "stage5_arc_mix_answer_prior_diagnosis":
        nested = str(source_payload.get("source_summary") or "")
        if nested:
            source_summary = resolve_path(nested)
            source_payload = read_json(source_summary)
    checkpoint = str(source_payload.get("resume_checkpoint") or "")
    if not checkpoint:
        checkpoint = str(source_payload.get("checkpoint") or "")
    if not checkpoint:
        best = source_payload.get("best_arm", {}).get("phase1_start", {})
        checkpoint = str(best.get("checkpoint") or "")
    if not checkpoint:
        raise ValueError("Could not resolve recurrent start checkpoint from source summary.")
    return source_summary, source_payload, resolve_path(checkpoint)


def restore_checkpoint_if_needed(checkpoint: Path) -> None:
    if checkpoint.exists():
        return
    run_id = infer_stage5_run_id(checkpoint)
    candidates = candidate_drive_checkpoints(run_id, path_for_cli(checkpoint), checkpoint.name)
    existing = [path for path in candidates if path.exists()]
    if not existing:
        searched = "\n".join(str(path) for path in candidates[:16])
        raise FileNotFoundError(
            f"Missing checkpoint and no Drive backup candidate found: {path_for_cli(checkpoint)}\n"
            f"Searched run_id={run_id!r} across project Drive roots.\n"
            f"Candidate paths:\n{searched}"
        )
    source = existing[0]
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, checkpoint)
    print(f"restored_checkpoint={source} -> {checkpoint}", flush=True)


def split_drive_roots(value: str) -> list[Path]:
    return [Path(item.strip()) for item in value.split(os.pathsep) if item.strip()]


def drive_roots() -> list[Path]:
    roots: list[Path] = []
    if os.environ.get("DRIVE_BACKUP_DIRS"):
        roots.extend(split_drive_roots(os.environ["DRIVE_BACKUP_DIRS"]))
    if os.environ.get("DRIVE_BACKUP_DIR"):
        roots.append(Path(os.environ["DRIVE_BACKUP_DIR"]))
    else:
        roots.extend(
            [
                Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts"),
                Path("/content/drive/MyDrive/recurrent-qwen-svgd"),
                Path("/content/drive/MyDrive/recurrent-qwen-svgd-fresh"),
                Path("/content/drive/MyDrive/gram-recurrent-qwen-outputs"),
            ]
        )
    if os.environ.get("STAGE5_DRIVE_BACKUP_DIR"):
        roots.append(Path(os.environ["STAGE5_DRIVE_BACKUP_DIR"]))

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def infer_stage5_run_id(checkpoint: Path) -> str:
    parts = list(checkpoint.parts)
    for marker in ("stage5", "outputs"):
        if marker in parts:
            index = parts.index(marker)
            if marker == "outputs" and index + 2 < len(parts) and parts[index + 1] == "stage5":
                return parts[index + 2]
            if marker == "stage5" and index + 1 < len(parts):
                return parts[index + 1]
    return checkpoint.parent.parent.name or checkpoint.stem


def append_unique(paths: list[Path], seen: set[str], candidate: Path) -> None:
    key = str(candidate)
    if key not in seen:
        seen.add(key)
        paths.append(candidate)


def candidate_drive_checkpoints(run_id: str, checkpoint_rel: str, filename: str) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()
    rel_path = Path(checkpoint_rel)
    rel_parts = list(rel_path.parts)
    phase_suffix = Path("phase1") / filename
    if "phase1" in rel_parts:
        phase_suffix = Path(*rel_parts[rel_parts.index("phase1") :])

    for root in drive_roots():
        for candidate in [
            root / checkpoint_rel,
            root / "outputs" / "stage5" / run_id / phase_suffix,
            root / "stage5" / run_id / phase_suffix,
            root / run_id / phase_suffix,
            root / run_id / "run_dir" / phase_suffix,
            root / "outputs" / "stage5" / run_id / "run_dir" / phase_suffix,
            root / "stage5" / run_id / "run_dir" / phase_suffix,
        ]:
            append_unique(candidates, seen, candidate)
        if not root.exists():
            continue
        for pattern in [
            f"outputs/stage5/{run_id}*/{phase_suffix.as_posix()}",
            f"outputs/stage5/{run_id}*/run_dir/{phase_suffix.as_posix()}",
            f"stage5/{run_id}*/{phase_suffix.as_posix()}",
            f"stage5/{run_id}*/run_dir/{phase_suffix.as_posix()}",
            f"{run_id}*/{phase_suffix.as_posix()}",
            f"{run_id}*/run_dir/{phase_suffix.as_posix()}",
        ]:
            for candidate in sorted(root.glob(pattern)):
                append_unique(candidates, seen, candidate)

    exact_existing = [path for path in candidates if path.exists()]
    if exact_existing:
        return exact_existing

    broad: list[Path] = []
    broad_seen: set[str] = set()
    for root in drive_roots():
        if not root.exists():
            continue
        for candidate in sorted(root.rglob(filename)):
            if run_id in candidate.as_posix():
                append_unique(broad, broad_seen, candidate)
    if broad:
        return broad

    ambiguous: list[Path] = []
    ambiguous_seen: set[str] = set()
    for root in drive_roots():
        if not root.exists():
            continue
        for candidate in sorted(root.rglob(filename)):
            append_unique(ambiguous, ambiguous_seen, candidate)
            if len(ambiguous) > 1:
                raise FileNotFoundError(
                    "Multiple same-name checkpoint backups exist, but none matched "
                    f"expected run_id={run_id!r}. Refusing ambiguous restore for {filename}.\n"
                    + "\n".join(str(path) for path in ambiguous[:8])
                )
    return ambiguous or candidates


def prepare_arc_mcq(config: str, split: str, output: Path, *, limit: str | int) -> None:
    cmd = [
        sys.executable,
        "eval/prepare_arc_mcq.py",
        "--config",
        config,
        "--split",
        split,
        "--seed",
        str(MIX_SEED),
        "--output_jsonl",
        path_for_cli(output),
    ]
    if str(limit).strip().lower() not in {"", "0", "none", "all", "full"}:
        cmd.extend(["--limit", str(limit)])
    run(cmd, log_name=f"prepare_{output.stem}.log")


def eval_mcq(
    label: str,
    mode: str,
    data_jsonl: Path,
    *,
    checkpoint: Path | None = None,
    max_loops: int = 4,
) -> Path:
    output = RUN_DIR / f"{data_jsonl.stem}_{label}.jsonl"
    if output.exists():
        output.unlink()
    cmd = [
        sys.executable,
        "eval/eval_mcq.py",
        "--data_jsonl",
        path_for_cli(data_jsonl),
        "--prompt_style",
        PROMPT_STYLE,
        "--score_target",
        SCORE_TARGET,
        "--mode",
        mode,
        "--dtype",
        DTYPE,
        "--adapter_dtype",
        ADAPTER_DTYPE,
        "--device",
        DEVICE,
        "--seed",
        str(MIX_SEED),
        "--aggregate",
        "mean",
        "--quiet_rows",
        "--output_jsonl",
        path_for_cli(output),
    ]
    if mode == "phase1":
        if checkpoint is None:
            raise ValueError("checkpoint is required for phase1 eval")
        cmd.extend(["--checkpoint", path_for_cli(checkpoint), "--max_loops", str(max_loops), "--num_trajectories", "1"])
        cmd.append("--include_loop_diagnostics")
    run(cmd, log_name=f"{label}.log")
    return output


def score_margin(row: dict[str, Any]) -> float | None:
    scores = row.get("scores") or {}
    answer = row.get("answer")
    if answer not in scores:
        return None
    others = [float(score) for label, score in scores.items() if label != answer]
    if not others:
        return None
    return float(scores[answer]) - max(others)


def summarize_mcq(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    return {
        "correct": sum(1 for row in rows if row.get("hit")),
        "total": len(rows),
        "accuracy": sum(1 for row in rows if row.get("hit")) / max(len(rows), 1),
        "prediction_counts": dict(Counter(str(row.get("prediction")) for row in rows)),
        "answer_counts": dict(Counter(str(row.get("answer")) for row in rows)),
    }


def compare(candidate_path: Path, reference_path: Path) -> dict[str, Any]:
    candidate = {str(row["id"]): row for row in read_jsonl(candidate_path)}
    reference = {str(row["id"]): row for row in read_jsonl(reference_path)}
    helped = hurt = tied = prediction_changes = 0
    margin_deltas: list[float] = []
    candidate_predictions = Counter()
    reference_predictions = Counter()
    for row_id in sorted(set(candidate) & set(reference)):
        cand = candidate[row_id]
        ref = reference[row_id]
        helped += int(bool(cand.get("hit")) and not bool(ref.get("hit")))
        hurt += int(bool(ref.get("hit")) and not bool(cand.get("hit")))
        tied += int(bool(ref.get("hit")) == bool(cand.get("hit")))
        prediction_changes += int(cand.get("prediction") != ref.get("prediction"))
        candidate_predictions[str(cand.get("prediction"))] += 1
        reference_predictions[str(ref.get("prediction"))] += 1
        cand_margin = score_margin(cand)
        ref_margin = score_margin(ref)
        if cand_margin is not None and ref_margin is not None:
            margin_deltas.append(cand_margin - ref_margin)
    prediction_delta = {
        label: candidate_predictions.get(label, 0) - reference_predictions.get(label, 0)
        for label in sorted(set(candidate_predictions) | set(reference_predictions))
    }
    max_shift = max([abs(value) for value in prediction_delta.values()] or [0])
    mean_margin_delta = sum(margin_deltas) / len(margin_deltas) if margin_deltas else None
    return {
        "helped": helped,
        "hurt": hurt,
        "tied": tied,
        "prediction_changes": prediction_changes,
        "mean_margin_delta": mean_margin_delta,
        "reference_prediction_counts": dict(reference_predictions),
        "candidate_prediction_counts": dict(candidate_predictions),
        "prediction_count_delta": prediction_delta,
        "max_abs_prediction_count_delta": max_shift,
        "calibration_ok": (mean_margin_delta is None or mean_margin_delta >= MIN_MARGIN_DELTA)
        and max_shift <= MAX_PREDICTION_SHIFT,
    }


def filter_direct_sft(train_mcq: Path, base_train_eval: Path) -> tuple[Path, dict[str, Any]]:
    out = RUN_DIR / "direct_base_correct_sft.jsonl"
    summary = RUN_DIR / "direct_base_correct_sft_summary.json"
    cmd = [
        sys.executable,
        "training/filter_mcq_sft_by_eval.py",
        "--mcq_jsonl",
        path_for_cli(train_mcq),
        "--base_eval_jsonl",
        path_for_cli(base_train_eval),
        "--output_jsonl",
        path_for_cli(out),
        "--summary_json",
        path_for_cli(summary),
        "--target_loop_count",
        "1",
        "--routing_type",
        "direct_base_preserve",
        "--prompt_style",
        PROMPT_STYLE,
        "--score_target",
        SCORE_TARGET,
        "--min_base_margin",
        str(MIN_BASE_MARGIN),
        "--seed",
        str(MIX_SEED),
    ]
    if MAX_ROWS_PER_LABEL.strip():
        cmd.extend(["--max_rows_per_label", MAX_ROWS_PER_LABEL.strip()])
    run(cmd, log_name="filter_direct_sft.log")
    return out, read_json(summary)


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def train_direct(checkpoint: Path, train_jsonl: Path) -> list[Path]:
    output_dir = RUN_DIR / "phase1_direct_preserve"
    cfg = {
        "model_name": MODEL_NAME,
        "dtype": DTYPE,
        "adapter_dtype": ADAPTER_DTYPE,
        "layer_split": "6,18",
        "max_length": 512,
        "max_loops": 1,
        "initial_halt_prob": 0.15,
        "beta": BETA,
        "batch_size": 1,
        "learning_rate": LEARNING_RATE,
        "weight_decay": 0.0,
        "max_grad_norm": 0.3,
        "max_steps": MAX_STEPS,
        "save_every": SAVE_EVERY,
        "log_every": 10,
        "train_on_prompt": False,
        "output_dir": path_for_cli(output_dir),
        "resume_from": path_for_cli(checkpoint),
        "lora": {"enabled": True, "rank": 8, "alpha": 16, "dropout": 0.0},
        "distillation": {
            "enabled": True,
            "weight": DISTILL_WEIGHT,
            "temperature": DISTILL_TEMPERATURE,
            "on": "response",
            "teacher_model_name": MODEL_NAME,
            "dtype": DTYPE,
        },
    }
    cfg_path = RUN_DIR / "direct_preserve.yaml"
    write_yaml(cfg_path, cfg)
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
        log_name="train_direct_preserve.log",
    )
    return sorted(output_dir.glob("phase1_step_*.pt"), key=lambda path: int(path.stem.rsplit("_", 1)[-1]))


def build_status(base: dict[str, Any], start: dict[str, Any], best: dict[str, Any], calibration: dict[str, Any]) -> str:
    if best["correct"] >= base["correct"] and calibration.get("calibration_ok", True):
        return "direct_route_matches_base"
    if best["correct"] > start["correct"] and calibration.get("calibration_ok", True):
        return "direct_route_lift"
    if best["correct"] > start["correct"]:
        return "direct_route_lift_calibration_warning"
    return "direct_route_no_lift"


def write_summary(payload: dict[str, Any]) -> None:
    (RUN_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    best = payload.get("best_checkpoint") or {}
    lines = [
        f"# Stage 5 Direct Preservation Probe - {RUN_ID}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Passed: `{payload['passed']}`",
        f"- Source summary: `{payload['source_summary']}`",
        f"- Resume checkpoint: `{payload['resume_checkpoint']}`",
        f"- Train rows: `{payload['data']['direct_sft'].get('selected_rows')}`",
        f"- Base proxy: `{payload['base_eval']['correct']}/{payload['base_eval']['total']}`",
        f"- Start loop1 proxy: `{payload['start_loop1_eval']['correct']}/{payload['start_loop1_eval']['total']}`",
        f"- Start loop4 proxy: `{payload['start_loop4_eval']['correct']}/{payload['start_loop4_eval']['total']}`",
        f"- Best loop1 proxy: `{best.get('loop1_eval', {}).get('correct')}/{best.get('loop1_eval', {}).get('total')}`",
        f"- Best loop4 proxy: `{best.get('loop4_eval', {}).get('correct')}/{best.get('loop4_eval', {}).get('total')}`",
        f"- Next step: {payload['next_step']}",
        "",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"), flush=True)


def main() -> int:
    source_summary, source_payload, resume_checkpoint = source_payload_and_checkpoint()
    restore_checkpoint_if_needed(resume_checkpoint)

    train_mcq = RUN_DIR / "arc_easy_train_mcq.jsonl"
    eval_mcq_path = RUN_DIR / "arc_easy_eval_mcq.jsonl"
    prepare_arc_mcq("ARC-Easy", "train", train_mcq, limit=ARC_TRAIN_LIMIT)
    prepare_arc_mcq("ARC-Easy", "validation", eval_mcq_path, limit=ARC_EVAL_LIMIT)

    base_train_eval = eval_mcq("base_train_label", "base", train_mcq)
    direct_sft, direct_sft_summary = filter_direct_sft(train_mcq, base_train_eval)
    if direct_sft_summary["selected_rows"] == 0:
        raise RuntimeError("No direct-preservation rows selected; lower min_base_margin or increase ARC_TRAIN_LIMIT.")

    base_eval_path = eval_mcq("base_eval_label", "base", eval_mcq_path)
    start_loop1_path = eval_mcq("start_loop1_label", "phase1", eval_mcq_path, checkpoint=resume_checkpoint, max_loops=1)
    start_loop4_path = eval_mcq("start_loop4_label", "phase1", eval_mcq_path, checkpoint=resume_checkpoint, max_loops=4)
    base_eval = summarize_mcq(base_eval_path)
    start_loop1_eval = summarize_mcq(start_loop1_path)
    start_loop4_eval = summarize_mcq(start_loop4_path)
    start_loop1_to_base = compare(start_loop1_path, base_eval_path)

    checkpoint_rows: list[dict[str, Any]] = []
    loop1_matches_base = (
        SKIP_TRAIN_IF_LOOP1_MATCHES_BASE
        and start_loop1_eval["correct"] >= base_eval["correct"]
        and start_loop1_to_base["calibration_ok"]
    )
    if loop1_matches_base:
        status = "direct_route_loop1_matches_base_without_training"
        passed = True
        next_step = "Use max_loops=1 routing for base-confident direct examples; confirm on larger ARC slices."
        best_checkpoint = {
            "checkpoint": path_for_cli(resume_checkpoint),
            "loop1_eval": start_loop1_eval,
            "loop4_eval": start_loop4_eval,
            "comparison_to_base": start_loop1_to_base,
            "trained": False,
        }
    elif PRECHECK_ONLY:
        status = "direct_route_precheck_needs_training"
        passed = False
        next_step = (
            "Loop-1 direct route does not match base on the proxy. "
            "Run the full bounded direct-preservation sweep if GPU budget permits."
        )
        best_checkpoint = {
            "checkpoint": path_for_cli(resume_checkpoint),
            "loop1_eval": start_loop1_eval,
            "loop4_eval": start_loop4_eval,
            "comparison_to_base": start_loop1_to_base,
            "trained": False,
        }
    else:
        checkpoints = train_direct(resume_checkpoint, direct_sft)
        for checkpoint in checkpoints:
            loop1_path = eval_mcq(f"{checkpoint.stem}_loop1_label", "phase1", eval_mcq_path, checkpoint=checkpoint, max_loops=1)
            loop4_path = eval_mcq(f"{checkpoint.stem}_loop4_label", "phase1", eval_mcq_path, checkpoint=checkpoint, max_loops=4)
            checkpoint_rows.append(
                {
                    "checkpoint": path_for_cli(checkpoint),
                    "loop1_path": path_for_cli(loop1_path),
                    "loop4_path": path_for_cli(loop4_path),
                    "loop1_eval": summarize_mcq(loop1_path),
                    "loop4_eval": summarize_mcq(loop4_path),
                    "comparison_to_base": compare(loop1_path, base_eval_path),
                    "comparison_to_start_loop1": compare(loop1_path, start_loop1_path),
                }
            )
        best_checkpoint = max(
            checkpoint_rows,
            key=lambda row: (
                row["loop1_eval"]["correct"],
                -row["comparison_to_base"]["max_abs_prediction_count_delta"],
                row["checkpoint"],
            ),
        )
        status = build_status(base_eval, start_loop1_eval, best_checkpoint["loop1_eval"], best_checkpoint["comparison_to_base"])
        passed = status in {"direct_route_matches_base", "direct_route_lift"}
        next_step = (
            "Confirm the direct-route preservation checkpoint on a larger ARC-Easy/Challenge slice."
            if passed
            else "Do not extend this probe; direct-route preservation still fails on the proxy."
        )

    payload = {
        "kind": "stage5_direct_preservation_probe",
        "run_id": RUN_ID,
        "status": status,
        "passed": passed,
        "source_summary": path_for_cli(source_summary),
        "source_status": source_payload.get("status"),
        "resume_checkpoint": path_for_cli(resume_checkpoint),
        "config": {
            "arc_train_limit": ARC_TRAIN_LIMIT,
            "arc_eval_limit": ARC_EVAL_LIMIT,
            "min_base_margin": MIN_BASE_MARGIN,
            "prompt_style": PROMPT_STYLE,
            "score_target": SCORE_TARGET,
            "max_steps": MAX_STEPS,
            "learning_rate": LEARNING_RATE,
            "distill_weight": DISTILL_WEIGHT,
            "precheck_only": PRECHECK_ONLY,
            "max_loops": 1,
        },
        "data": {
            "train_mcq": path_for_cli(train_mcq),
            "eval_mcq": path_for_cli(eval_mcq_path),
            "direct_sft": direct_sft_summary,
        },
        "base_eval": base_eval,
        "start_loop1_eval": start_loop1_eval,
        "start_loop4_eval": start_loop4_eval,
        "start_loop1_comparison_to_base": start_loop1_to_base,
        "checkpoints": checkpoint_rows,
        "best_checkpoint": best_checkpoint,
        "next_step": next_step,
    }
    write_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
