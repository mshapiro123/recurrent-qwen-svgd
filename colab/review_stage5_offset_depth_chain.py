"""Review an ARC-mix offset-then-depth chain and print the next action.

This is intentionally CPU-only. The offset/depth chain can consume a long GPU
session, so the review step should be cheap, deterministic, and explicit about
whether another paid run is actually ready.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
KIND = "stage5_arc_mix_offset_then_depth_chain"
REVIEW_KIND = "stage5_arc_mix_offset_depth_review"


def resolve_path(value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def latest_summary(kind: str = KIND) -> Path:
    candidates: list[tuple[float, Path]] = []
    for path in (ROOT / "outputs" / "stage5").rglob("summary.json"):
        try:
            payload = read_json(path)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if payload.get("kind") == kind:
            candidates.append((path.stat().st_mtime, path))
    if not candidates:
        raise FileNotFoundError(f"No outputs/stage5/**/summary.json with kind={kind!r}")
    return max(candidates, key=lambda item: item[0])[1]


def optional_payload(path_value: str | None) -> tuple[Path | None, dict[str, Any] | None]:
    if not path_value:
        return None, None
    path = resolve_path(path_value)
    if not path.exists():
        return path, None
    return path, read_json(path)


def positive_sft_locator(payload: dict[str, Any], *, source_summary: Path | None = None, seen: set[Path] | None = None) -> dict[str, Any]:
    """Find a source summary that exposes dense-control-compatible positive_sft rows.

    The dense MCQ control needs a summary path, not just a JSONL path, because it
    inherits default train settings from that source. Return both when possible.
    """

    seen = seen if seen is not None else set()
    source_summary = source_summary.resolve() if source_summary else None
    if source_summary is not None:
        if source_summary in seen:
            return {}
        seen.add(source_summary)

    dataset = payload.get("dataset") if isinstance(payload.get("dataset"), dict) else {}
    if dataset.get("source_positive_sft"):
        return {
            "source_summary": None if source_summary is None else path_for_cli(source_summary),
            "positive_sft": path_for_cli(resolve_path(str(dataset["source_positive_sft"]))),
            "source": "dataset.source_positive_sft",
        }

    gate = payload.get("gate") if isinstance(payload.get("gate"), dict) else {}
    artifacts = gate.get("artifacts") if isinstance(gate.get("artifacts"), dict) else {}
    if artifacts.get("positive_sft"):
        return {
            "source_summary": None if source_summary is None else path_for_cli(source_summary),
            "positive_sft": path_for_cli(resolve_path(str(artifacts["positive_sft"]))),
            "source": "gate.artifacts.positive_sft",
        }

    for key in ("curriculum", "config"):
        section = payload.get(key) if isinstance(payload.get(key), dict) else {}
        if section.get("work_dir"):
            return {
                "source_summary": None if source_summary is None else path_for_cli(source_summary),
                "positive_sft": path_for_cli(resolve_path(str(section["work_dir"])) / "positive_sft.jsonl"),
                "source": f"{key}.work_dir",
            }

    for key in ("source_summary", "nested_source_summary", "benchmark_source_summary"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            candidate = resolve_path(value)
            if candidate.exists():
                found = positive_sft_locator(read_json(candidate), source_summary=candidate, seen=seen)
                if found:
                    return found

    source_summaries = payload.get("source_summaries")
    if isinstance(source_summaries, dict):
        for value in source_summaries.values():
            if not isinstance(value, str) or not value.strip():
                continue
            candidate = resolve_path(value)
            if candidate.exists():
                found = positive_sft_locator(read_json(candidate), source_summary=candidate, seen=seen)
                if found:
                    return found

    return {}


def mixed_train_jsonl(depth_payload: dict[str, Any] | None) -> str | None:
    if not depth_payload:
        return None
    data = depth_payload.get("data") if isinstance(depth_payload.get("data"), dict) else {}
    value = data.get("mixed_train_jsonl")
    return str(value) if value else None


def classify(summary_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    depth_summary_path, depth_payload = optional_payload(payload.get("depth_summary"))
    post_depth_path, _post_payload = optional_payload(payload.get("post_depth_debiased_summary"))
    offset_assessment = payload.get("offset_assessment") if isinstance(payload.get("offset_assessment"), dict) else {}
    post_assessment = (
        payload.get("post_depth_debiased_assessment")
        if isinstance(payload.get("post_depth_debiased_assessment"), dict)
        else None
    )

    mixed_train = mixed_train_jsonl(depth_payload)
    positive_sft = positive_sft_locator(payload, source_summary=summary_path)
    recurrent_benchmark_summary = payload.get("post_depth_debiased_summary") or payload.get("depth_summary")

    if not offset_assessment.get("passed"):
        action = "stop_offset_not_confirmed"
        next_step = "Do not launch depth or dense control. Inspect the offset readouts first."
    elif not payload.get("depth_launched"):
        action = "run_depth_probe"
        next_step = "Offset passed, but depth did not launch; rerun the chain with STAGE5_ARC_MIX_CHAIN_EXECUTE_DEPTH=1."
    elif payload.get("depth_returncode") not in (0, None):
        action = "inspect_depth_failure"
        next_step = "Depth process returned non-zero; inspect depth_routing.log and avoid extending training."
    elif not payload.get("post_depth_debiased_summary"):
        action = "run_post_depth_debiased_gate"
        next_step = "Depth completed, but the post-depth debiased benchmark is missing."
    elif post_assessment and not post_assessment.get("passed"):
        action = "inspect_post_depth_warning"
        next_step = "Post-depth gate did not pass; inspect cyclic-debiased regressions before more training."
    elif positive_sft.get("source_summary") and mixed_train and recurrent_benchmark_summary:
        action = "run_dense_mcq_trace_sft_control"
        next_step = "Run the dense control using the same source positive_sft plus the ARC-mix extra rows."
    elif mixed_train:
        action = "dense_control_blocked_missing_positive_sft_source"
        next_step = "Mixed ARC rows exist, but no upstream positive_sft source summary was found for the dense control."
    else:
        action = "review_depth_result_manually"
        next_step = "Depth completed, but the reviewer could not locate the mixed ARC training rows."

    dense_env: dict[str, str] = {}
    if action == "run_dense_mcq_trace_sft_control":
        dense_env = {
            "STAGE5_CURRENT_A100_TARGET": "dense_mcq_trace_sft_control",
            "STAGE5_DENSE_MCQ_SOURCE_SUMMARY": str(positive_sft["source_summary"]),
            "STAGE5_DENSE_MCQ_RECURRENT_BENCHMARK_SUMMARY": str(recurrent_benchmark_summary),
            "STAGE5_DENSE_MCQ_EXTRA_TRAIN_JSONL": str(mixed_train),
            "STAGE5_DENSE_MCQ_RUN_ID": f"stage5_dense_control_after_{payload.get('run_id', summary_path.parent.name)}",
        }

    return {
        "kind": REVIEW_KIND,
        "reviewed_summary": path_for_cli(summary_path),
        "reviewed_status": payload.get("status"),
        "action": action,
        "next_step": next_step,
        "offset_passed": bool(offset_assessment.get("passed")),
        "depth_launched": bool(payload.get("depth_launched")),
        "depth_returncode": payload.get("depth_returncode"),
        "depth_summary": None if depth_summary_path is None else path_for_cli(depth_summary_path),
        "depth_summary_visible": bool(depth_payload),
        "post_depth_debiased_summary": None if post_depth_path is None else path_for_cli(post_depth_path),
        "post_depth_debiased_passed": None if post_assessment is None else bool(post_assessment.get("passed")),
        "mixed_train_jsonl": mixed_train,
        "positive_sft_locator": positive_sft,
        "dense_control_ready": bool(dense_env),
        "dense_control_env": dense_env,
    }


def report_lines(payload: dict[str, Any]) -> list[str]:
    lines = [
        f"# Stage 5 Offset-Depth Chain Review",
        "",
        f"- Reviewed summary: `{payload['reviewed_summary']}`",
        f"- Reviewed status: `{payload['reviewed_status']}`",
        f"- Action: `{payload['action']}`",
        f"- Next step: {payload['next_step']}",
        f"- Offset passed: `{payload['offset_passed']}`",
        f"- Depth launched: `{payload['depth_launched']}`",
        f"- Depth return code: `{payload['depth_returncode']}`",
        f"- Depth summary visible: `{payload['depth_summary_visible']}`",
        f"- Post-depth passed: `{payload['post_depth_debiased_passed']}`",
        f"- Mixed train JSONL: `{payload.get('mixed_train_jsonl') or 'not_found'}`",
        f"- Positive SFT source summary: `{(payload.get('positive_sft_locator') or {}).get('source_summary') or 'not_found'}`",
        f"- Dense control ready: `{payload['dense_control_ready']}`",
    ]
    if payload["dense_control_env"]:
        lines.extend(["", "## Dense Control Env", ""])
        for key, value in payload["dense_control_env"].items():
            lines.append(f"- `{key}={value}`")
    return lines


def write_review(payload: dict[str, Any], *, run_id: str | None = None) -> Path:
    run_name = run_id or f"stage5_offset_depth_chain_review_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = ROOT / "outputs" / "stage5" / run_name
    summary_path = run_dir / "summary.json"
    write_json(summary_path, payload)
    (run_dir / "summary.md").write_text("\n".join(report_lines(payload)) + "\n", encoding="utf-8")
    pointer = ROOT / "config" / "stage5_current_source_summary.txt"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(path_for_cli(summary_path) + "\n", encoding="utf-8")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="", help="Offset-depth chain summary. Defaults to latest chain summary.")
    parser.add_argument("--run_id", default="", help="Optional review output run id.")
    parser.add_argument("--no_write", action="store_true", help="Print only; do not write outputs/stage5 review files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_path = resolve_path(args.summary) if args.summary else latest_summary()
    payload = read_json(summary_path)
    if payload.get("kind") != KIND:
        raise ValueError(f"Expected {KIND}, got {payload.get('kind')!r}")
    review = classify(summary_path, payload)
    print("\n".join(report_lines(review)), flush=True)
    if not args.no_write:
        output = write_review(review, run_id=args.run_id or None)
        print(f"\nsaved_review={path_for_cli(output)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
