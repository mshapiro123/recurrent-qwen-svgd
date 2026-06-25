"""Pure helpers for recovering Stage 2 re-entry norm artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RAW_REQUIRED_FILES = [
    "reentry_norm/reentry_drift_none.json",
    "reentry_norm/reentry_drift_none.jsonl",
    "reentry_norm/effective_pathways_none.json",
    "reentry_norm/effective_pathways_none.jsonl",
    "reentry_norm/candidate_conversion_none.jsonl",
    "reentry_norm/reentry_drift_entry_rms.json",
    "reentry_norm/reentry_drift_entry_rms.jsonl",
    "reentry_norm/effective_pathways_entry_rms.json",
    "reentry_norm/effective_pathways_entry_rms.jsonl",
    "reentry_norm/candidate_conversion_entry_rms.jsonl",
]

FINAL_REQUIRED_FILES = ["summary.json", "summary.md", *RAW_REQUIRED_FILES]


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path} contains a non-object JSONL row")
            rows.append(payload)
    return rows


def has_valid_json(path: Path) -> bool:
    try:
        read_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    return True


def has_valid_jsonl(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return bool(read_jsonl(path))
    except (OSError, json.JSONDecodeError, ValueError):
        return False


def missing_reasons(path: Path, required_files: list[str] | None = None) -> list[str]:
    reasons: list[str] = []
    for rel in required_files or FINAL_REQUIRED_FILES:
        candidate = path / rel
        if rel.endswith(".jsonl"):
            if not has_valid_jsonl(candidate):
                reasons.append(f"{rel}: missing, empty, or invalid jsonl")
        elif rel.endswith(".json"):
            if not has_valid_json(candidate):
                reasons.append(f"{rel}: missing or invalid json")
        elif not candidate.exists():
            reasons.append(f"{rel}: missing")
    return reasons


def raw_stage2_complete(path: Path) -> bool:
    return not missing_reasons(path, RAW_REQUIRED_FILES)


def final_stage2_complete(path: Path) -> bool:
    reasons = missing_reasons(path, FINAL_REQUIRED_FILES)
    if reasons:
        return False
    summary = read_json(path / "summary.json")
    return summary.get("kind") == "stage5_reentry_norm_eval_only"


def recoverable_stage2(path: Path) -> bool:
    """Return true when raw GPU outputs are enough to rebuild the summary."""
    if final_stage2_complete(path):
        return True
    return raw_stage2_complete(path)


def summarize_candidate_jsonl(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    grouped: dict[tuple[object, object, object, object], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row.get("reentry_rescale_mode", "none"),
            row.get("max_loops"),
            row.get("particle_init_noise"),
            row.get("task"),
        )
        grouped.setdefault(key, []).append(row)
    by_mode: dict[str, dict[str, float | int]] = {}
    for (mode, _loops, _noise, _task), group in grouped.items():
        mode_key = str(mode)
        current = by_mode.setdefault(
            mode_key,
            {
                "task_groups": 0,
                "best_hits": 0,
                "candidate_hits": 0,
                "total_candidates": 0,
                "unique_sum": 0,
            },
        )
        current["task_groups"] += 1
        current["best_hits"] += int(any(bool(row.get("hit")) for row in group))
        current["candidate_hits"] += sum(int(bool(row.get("hit"))) for row in group)
        current["total_candidates"] += len(group)
        current["unique_sum"] += len({str(row.get("candidate", "")).strip() for row in group})
    for current in by_mode.values():
        groups = max(int(current["task_groups"]), 1)
        current["mean_unique"] = float(current["unique_sum"]) / groups
    return {"rows": len(rows), "by_mode": by_mode}


def build_summary_payload(run_dir: Path, *, cell_version: str) -> dict[str, Any]:
    existing = read_json(run_dir / "summary.json") if has_valid_json(run_dir / "summary.json") else {}
    paths = {
        mode: {
            "drift": f"reentry_norm/reentry_drift_{mode}.json",
            "effective_pathways": f"reentry_norm/effective_pathways_{mode}.json",
            "candidate_conversion": f"reentry_norm/candidate_conversion_{mode}.jsonl",
        }
        for mode in ("none", "entry_rms")
    }
    drift = {mode: read_json(run_dir / rels["drift"]) for mode, rels in paths.items()}
    effective = {mode: read_json(run_dir / rels["effective_pathways"]) for mode, rels in paths.items()}
    candidate = {
        mode: summarize_candidate_jsonl(run_dir / rels["candidate_conversion"])
        for mode, rels in paths.items()
    }
    return {
        "kind": "stage5_reentry_norm_eval_only",
        "run_id": existing.get("run_id") or run_dir.name,
        "cell_version": existing.get("cell_version") or cell_version,
        "checkpoint": existing.get("checkpoint"),
        "prompts": existing.get("prompts"),
        "paths": paths,
        "drift": {mode: payload.get("aggregate", {}) for mode, payload in drift.items()},
        "effective_pathways": {
            mode: payload.get("aggregate", payload)
            for mode, payload in effective.items()
        },
        "candidate_conversion": candidate,
        "recovered_summary": not has_valid_json(run_dir / "summary.json"),
    }


def summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# Stage 5 Re-entry Norm - {summary.get('run_id')}",
        "",
        f"- Checkpoint: `{summary.get('checkpoint')}`",
        f"- Prompts/tasks: `{summary.get('prompts')}`",
        f"- Cell version: `{summary.get('cell_version')}`",
        f"- Recovered summary: `{summary.get('recovered_summary', False)}`",
        "",
        "## Candidate Conversion",
        "| mode | task groups | best | candidates | mean unique |",
        "|---|---:|---:|---:|---:|",
    ]
    candidate = summary.get("candidate_conversion") if isinstance(summary.get("candidate_conversion"), dict) else {}
    for mode in ("none", "entry_rms"):
        payload = candidate.get(mode, {}) if isinstance(candidate.get(mode), dict) else {}
        stats = (payload.get("by_mode") or {}).get(mode, {}) if isinstance(payload, dict) else {}
        lines.append(
            f"| {mode} | {stats.get('task_groups')} | {stats.get('best_hits')} | "
            f"{stats.get('candidate_hits')}/{stats.get('total_candidates')} | {stats.get('mean_unique')} |"
        )
    lines.append("")
    return "\n".join(lines)
