"""Batch-rescore saved ARC-AGI candidate files with alternate selectors.

Run this after a Stage 5 ARC evaluation that emitted ``*_candidates.jsonl`` and
``*_summary.json`` files. It does not load a model or use the GPU.
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

try:
    from eval.compare_arc_agi_runs import compare_payloads
except ModuleNotFoundError:  # pragma: no cover - direct ``python colab/script.py`` execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from eval.compare_arc_agi_runs import compare_payloads


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get("STAGE5_ARC_AGI_RESCORE_RUN_ID") or time.strftime("stage5_arc_agi_rescore_%Y%m%d_%H%M%S")
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

STRATEGIES = os.environ.get(
    "STAGE5_ARC_AGI_RESCORE_STRATEGIES",
    "heuristic,self_consistency,reliability_vote,symbolic_priority",
)
SOURCE_RUN_DIR = os.environ.get("STAGE5_ARC_AGI_RESCORE_SOURCE_RUN_DIR", "")
SOURCE_GLOB = os.environ.get("STAGE5_ARC_AGI_RESCORE_SOURCE_GLOB", "*_candidates.jsonl")
WRITE_RESCORED_JSONL = os.environ.get("STAGE5_ARC_AGI_RESCORE_WRITE_JSONL", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
PUSH_RESULTS = os.environ.get("STAGE5_ARC_AGI_RESCORE_PUSH", "1").strip().lower() in {"1", "true", "yes", "y"}


@dataclass(frozen=True)
class CandidatePair:
    label: str
    candidates_jsonl: Path
    summary_json: Path | None


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


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def requested_strategies(value: str = STRATEGIES) -> list[str]:
    strategies = [item.strip() for item in value.split(",") if item.strip()]
    valid = {"heuristic", "self_consistency", "reliability_vote", "symbolic_priority"}
    unknown = set(strategies) - valid
    if unknown:
        raise ValueError(f"Unknown selector strategies: {sorted(unknown)}")
    return strategies


def latest_stage5_run_with_candidates() -> Path:
    candidates = sorted(
        {path.parent for path in (ROOT / "outputs" / "stage5").glob("**/*_candidates.jsonl")},
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No Stage 5 candidate files found. Set STAGE5_ARC_AGI_RESCORE_SOURCE_RUN_DIR."
        )
    return candidates[0]


def candidate_label(path: Path) -> str:
    name = path.name
    suffix = "_candidates.jsonl"
    return name[: -len(suffix)] if name.endswith(suffix) else path.stem


def find_candidate_pairs(source_run_dir: Path, pattern: str = SOURCE_GLOB) -> list[CandidatePair]:
    paths = sorted(source_run_dir.glob(pattern))
    pairs: list[CandidatePair] = []
    for candidate_path in paths:
        label = candidate_label(candidate_path)
        summary_path = candidate_path.with_name(f"{label}_summary.json")
        pairs.append(
            CandidatePair(
                label=label,
                candidates_jsonl=candidate_path,
                summary_json=summary_path if summary_path.exists() else None,
            )
        )
    if not pairs:
        raise FileNotFoundError(f"No candidate files matching {pattern!r} under {source_run_dir}")
    return pairs


def summary_row(
    *,
    label: str,
    strategy: str,
    payload: dict[str, Any],
    source_summary_json: Path | None,
    output_summary_json: Path | None = None,
) -> dict[str, Any]:
    summary = payload["summary"]
    source_summary = payload.get("source_summary") or {}
    return {
        "label": label,
        "selection_strategy": strategy,
        "source_summary_json": path_for_cli(source_summary_json) if source_summary_json is not None else None,
        "output_summary_json": path_for_cli(output_summary_json) if output_summary_json is not None else None,
        "examples": summary["examples_with_targets"],
        "selected_exact": summary["selected_exact"],
        "best_of_k_exact": summary["best_of_k_exact"],
        "first_exact": summary["first_exact"],
        "selected_accuracy": summary["selected_accuracy"],
        "best_of_k_accuracy": summary["best_of_k_accuracy"],
        "tasks_solved_best_of_k": summary["tasks_solved_best_of_k"],
        "tasks_with_targets": summary["tasks_with_targets"],
        "valid_candidate_rate": summary["valid_candidate_rate"],
        "source_selected_exact": source_summary.get("selected_exact"),
        "selected_delta_vs_source": (
            summary["selected_exact"] - source_summary["selected_exact"]
            if isinstance(source_summary.get("selected_exact"), int)
            else None
        ),
    }


def original_row(pair: CandidatePair) -> dict[str, Any] | None:
    if pair.summary_json is None:
        return None
    payload = read_json(pair.summary_json)
    strategy = f"original:{payload.get('selection_strategy', 'unknown')}"
    return summary_row(
        label=pair.label,
        strategy=strategy,
        payload=payload,
        source_summary_json=pair.summary_json,
        output_summary_json=pair.summary_json,
    )


def rescore_pair(pair: CandidatePair, strategy: str) -> dict[str, Any]:
    output_prefix = RUN_DIR / f"{pair.label}__selector_{strategy}"
    output_summary_json = output_prefix.with_name(f"{output_prefix.name}_summary.json")
    output_summary_md = output_prefix.with_name(f"{output_prefix.name}_summary.md")
    cmd = [
        sys.executable,
        "eval/rescore_arc_agi_candidates.py",
        "--candidates_jsonl",
        path_for_cli(pair.candidates_jsonl),
        "--selection_strategy",
        strategy,
        "--output_summary_json",
        path_for_cli(output_summary_json),
        "--output_summary_md",
        path_for_cli(output_summary_md),
    ]
    if pair.summary_json is not None:
        cmd += ["--summary_json", path_for_cli(pair.summary_json)]
    if WRITE_RESCORED_JSONL:
        cmd += ["--output_jsonl", path_for_cli(output_prefix.with_name(f"{output_prefix.name}_candidates.jsonl"))]
    run(cmd, log_name=f"{pair.label}__selector_{strategy}.log")
    payload = read_json(output_summary_json)
    return summary_row(
        label=pair.label,
        strategy=strategy,
        payload=payload,
        source_summary_json=pair.summary_json,
        output_summary_json=output_summary_json,
    )


def best_rows_by_label(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        label = str(row["label"])
        current = best.get(label)
        if current is None or (
            int(row["selected_exact"]),
            int(row["best_of_k_exact"]),
            float(row["valid_candidate_rate"]),
        ) > (
            int(current["selected_exact"]),
            int(current["best_of_k_exact"]),
            float(current["valid_candidate_rate"]),
        ):
            best[label] = row
    return best


def paired_selector_comparisons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for row in rows:
        strategy = str(row["selection_strategy"])
        if strategy.startswith("original:"):
            continue
        source_summary_json = row.get("source_summary_json")
        output_summary_json = row.get("output_summary_json")
        if not source_summary_json or not output_summary_json:
            continue
        label = str(row["label"])
        key = f"{label}__selector_{strategy}_vs_source"
        comparisons[key] = compare_payloads(
            read_json(resolve_path(str(source_summary_json))),
            read_json(resolve_path(str(output_summary_json))),
            reference_label=f"{label}:source",
            candidate_label=f"{label}:{strategy}",
            bootstrap_samples=0,
            seed=0,
        )
    return comparisons


def write_comparison(
    *,
    source_run_dir: Path,
    strategies: list[str],
    pairs: list[CandidatePair],
    rows: list[dict[str, Any]],
) -> None:
    best_by_label = best_rows_by_label([row for row in rows if not str(row["selection_strategy"]).startswith("original:")])
    paired_comparisons = paired_selector_comparisons(rows)
    payload = {
        "run_id": RUN_ID,
        "source_run_dir": path_for_cli(source_run_dir),
        "strategies": strategies,
        "candidate_files": [path_for_cli(pair.candidates_jsonl) for pair in pairs],
        "rows": rows,
        "best_by_label": best_by_label,
        "paired_comparisons": paired_comparisons,
    }
    write_json(RUN_DIR / "summary.json", payload)

    lines = [
        f"# Stage 5 ARC Selector Rescore - {RUN_ID}",
        "",
        f"- Source run dir: `{path_for_cli(source_run_dir)}`",
        f"- Candidate files: `{len(pairs)}`",
        f"- Strategies: `{', '.join(strategies)}`",
        f"- Wrote rescored candidate JSONL: `{WRITE_RESCORED_JSONL}`",
        "",
        "## Comparison",
        "",
        "| Label | Strategy | Selected | Best-of-K | Tasks best | Valid rate | Delta vs source selected |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        examples = int(row["examples"])
        tasks = int(row["tasks_with_targets"])
        delta = row["selected_delta_vs_source"]
        delta_text = "" if delta is None else f"{int(delta):+d}"
        lines.append(
            f"| `{row['label']}` | `{row['selection_strategy']}` | "
            f"{row['selected_exact']}/{examples} | {row['best_of_k_exact']}/{examples} | "
            f"{row['tasks_solved_best_of_k']}/{tasks} | {row['valid_candidate_rate']:.4f} | {delta_text} |"
        )

    lines += ["", "## Best Rescored Selector Per Candidate File", ""]
    for label, row in sorted(best_by_label.items()):
        examples = int(row["examples"])
        lines.append(
            f"- `{label}`: `{row['selection_strategy']}` selected `{row['selected_exact']}/{examples}`, "
            f"best `{row['best_of_k_exact']}/{examples}`, valid `{row['valid_candidate_rate']:.4f}`"
        )
    if paired_comparisons:
        lines += ["", "## Paired Selector Evidence", ""]
        for name, comparison in sorted(paired_comparisons.items()):
            selected = comparison["metrics"]["selected_exact"]
            lines.append(
                f"- `{name}` selected delta `{selected['delta_exact']}` "
                f"({selected['wins']}/{selected['losses']}/{selected['ties']} W/L/T)"
            )
        lines += [
            "",
            "## Paired Selector Evidence By Task Family",
            "",
            "| Comparison | Family | Candidate | Reference | Delta | Win/Loss/Tie | Sign p |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for name, comparison in sorted(paired_comparisons.items()):
            family_rows = comparison.get("task_family_metrics", {}).get("selected_exact", {})
            for family, stats in sorted(family_rows.items()):
                lines.append(
                    f"| `{name}` | `{family}` | "
                    f"{stats['candidate_exact']}/{stats['paired_examples']} | "
                    f"{stats['reference_exact']}/{stats['paired_examples']} | "
                    f"{stats['delta_exact']} | "
                    f"{stats['wins']}/{stats['losses']}/{stats['ties']} | "
                    f"{stats['sign_test_p_value']} |"
                )
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))


def mount_drive_if_possible() -> None:
    try:
        from google.colab import drive  # type: ignore
    except Exception:
        return
    drive.mount("/content/drive", force_remount=False)


def backup_to_drive() -> None:
    mount_drive_if_possible()
    drive_root = Path(os.environ.get("DRIVE_BACKUP_DIR", "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts"))
    if not drive_root.exists():
        print(f"Drive backup skipped; missing {drive_root}")
        return
    backup = drive_root / RUN_ID
    backup.mkdir(parents=True, exist_ok=True)
    shutil.copytree(RUN_DIR, backup / "run_dir", dirs_exist_ok=True)
    print(f"backed_up_to={backup}")


def git_commit_results() -> None:
    if not PUSH_RESULTS:
        return
    for pattern in ["*.json", "*.md", "*.log", "*.jsonl"]:
        run(["git", "add", "-f", str((RUN_DIR / pattern).relative_to(ROOT))], check=False)
    run(["git", "status", "-sb"], check=False)
    message = f"Record ARC selector rescore {RUN_ID}"
    commit = run(["git", "commit", "-m", message], check=False)
    if commit.returncode == 0:
        run(["git", "push", "origin", "main"], check=False)


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print(
            "Run after a Stage 5 ARC eval with saved *_candidates.jsonl files. "
            "Configure STAGE5_ARC_AGI_RESCORE_SOURCE_RUN_DIR and "
            "STAGE5_ARC_AGI_RESCORE_STRATEGIES."
        )
        return 0

    source_run_dir = resolve_path(SOURCE_RUN_DIR) if SOURCE_RUN_DIR else latest_stage5_run_with_candidates()
    strategies = requested_strategies()
    pairs = find_candidate_pairs(source_run_dir)
    rows: list[dict[str, Any]] = []

    for pair in pairs:
        original = original_row(pair)
        if original is not None:
            rows.append(original)
        for strategy in strategies:
            rows.append(rescore_pair(pair, strategy))

    write_comparison(source_run_dir=source_run_dir, strategies=strategies, pairs=pairs, rows=rows)
    backup_to_drive()
    git_commit_results()
    print(json.dumps({"run_id": RUN_ID, "run_dir": path_for_cli(RUN_DIR)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
