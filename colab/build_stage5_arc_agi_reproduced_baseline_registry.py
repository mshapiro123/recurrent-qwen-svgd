"""Build a same-size ARC-AGI baseline registry from reproduced eval summaries.

This utility turns local, audited Stage 5 summary artifacts into
``reproduced_eval`` baseline rows accepted by
``validate_arc_agi_baseline_registry.py``. It is meant for reproduced controls
such as unmodified base Qwen or dense same-recipe runs. It should not replace
external sources when making a public SOTA claim.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "config" / "arc_agi_same_size_baselines.json"
DEFAULT_BENCHMARK = "ARC-AGI public evaluation"
DEFAULT_METRIC = "selected_accuracy"
DEFAULT_LABELS = "base"
DEFAULT_MIN_PARAMS_B = 0.3
DEFAULT_MAX_PARAMS_B = 1.0

try:
    from colab.validate_arc_agi_baseline_registry import validate_registry_payload
except ModuleNotFoundError:  # pragma: no cover - used when run from colab/
    sys.path.insert(0, str(ROOT))
    from colab.validate_arc_agi_baseline_registry import validate_registry_payload


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def path_for_cli(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path_for_cli(path)} must contain a JSON object.")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def metadata_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def params_b_from_metadata(metadata: dict[str, Any]) -> float | None:
    for key in ("params_b", "candidate_params_b", "model_params_b", "base_model_params_b", "parameter_count_b"):
        value = finite_float(metadata.get(key))
        if value is not None:
            return value
    return None


def model_name_from_metadata(metadata: dict[str, Any]) -> str:
    for key in ("model_name", "base_model_name", "base_model", "tokenizer_name"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown-model"


def arc_split_from_metadata(metadata: dict[str, Any]) -> str | None:
    value = metadata.get("arc_split") or metadata.get("eval_split")
    return str(value) if value else None


def split_csv(value: str) -> list[str]:
    labels = [item.strip() for item in value.split(",") if item.strip()]
    if not labels:
        raise ValueError("At least one label is required.")
    return labels


def summary_from_label(payload: dict[str, Any], label: str) -> dict[str, Any]:
    if label == "summary":
        block = payload.get("summary")
    else:
        block = payload.get(label)
    if not isinstance(block, dict):
        raise ValueError(f"Summary artifact is missing label {label!r}.")
    nested = block.get("summary")
    if isinstance(nested, dict):
        block = nested
    if not isinstance(block, dict) or not examples_from_summary(block):
        raise ValueError(f"Label {label!r} does not contain an eval summary with examples.")
    return block


def examples_from_summary(summary: dict[str, Any]) -> int:
    return int(summary.get("examples_with_targets", 0) or summary.get("examples", 0) or 0)


def exact_key_for_metric(metric: str) -> str:
    if metric == "selected_accuracy":
        return "selected_exact"
    if metric == "best_of_k_accuracy":
        return "best_of_k_exact"
    if metric == "first_accuracy":
        return "first_exact"
    raise ValueError(f"Unsupported metric {metric!r}.")


def accuracy_from_summary(summary: dict[str, Any], metric: str) -> float:
    value = finite_float(summary.get(metric))
    if value is not None:
        return value
    examples = examples_from_summary(summary)
    if examples <= 0:
        raise ValueError(f"Cannot compute {metric}: summary has no examples.")
    exact = int(summary.get(exact_key_for_metric(metric), 0) or 0)
    return exact / examples


def current_git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception:
        return "unknown"
    return proc.stdout.strip()


def infer_reproduction_command(payload: dict[str, Any]) -> str:
    kind = str(payload.get("kind") or "")
    if kind == "dense_sft_control":
        return "python colab/run_stage5_arc_agi_dense_sft.py"
    if kind == "stage5_benchmark_suite":
        return "python colab/run_stage5_benchmark_suite.py"
    if "phase1_arc_agi_tuned" in payload:
        return "python colab/run_stage5_arc_agi_sft.py"
    return "python colab/run_stage5_colab_continue.py"


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug or "unknown"


def baseline_name(*, name_prefix: str, model_name: str, label: str) -> str:
    return f"{name_prefix}-{slugify(model_name)}-{slugify(label)}"


def make_row(
    *,
    summary_path: Path,
    payload: dict[str, Any],
    label: str,
    metric: str,
    name_prefix: str,
    params_b: float,
    arc_version: str,
    arc_split: str,
    reproduction_command: str,
    git_commit: str,
    accessed_date: str,
) -> dict[str, Any]:
    metadata = metadata_from_payload(payload)
    summary = summary_from_label(payload, label)
    accuracy = accuracy_from_summary(summary, metric)
    examples = examples_from_summary(summary)
    exact_key = exact_key_for_metric(metric)
    exact = int(summary.get(exact_key, 0) or 0)
    model_name = model_name_from_metadata(metadata)
    return {
        "name": baseline_name(name_prefix=name_prefix, model_name=model_name, label=label),
        "params_b": params_b,
        "arc_version": arc_version,
        "arc_split": arc_split,
        "metric": metric,
        "accuracy": accuracy,
        "evidence_type": "reproduced_eval",
        "source": "",
        "source_artifact": path_for_cli(summary_path),
        "reproduction_command": reproduction_command,
        "git_commit": git_commit,
        "accessed_date": accessed_date,
        "notes": (
            f"Generated from `{label}` in `{path_for_cli(summary_path)}`; "
            f"{exact_key}={exact}, examples={examples}."
        ),
    }


def build_registry(
    *,
    summary_paths: list[Path],
    labels: list[str],
    metric: str,
    benchmark: str,
    min_params_b: float,
    max_params_b: float,
    name_prefix: str,
    reproduction_command: str | None,
    git_commit: str | None,
    accessed_date: str,
    arc_version_override: str | None = None,
    arc_split_override: str | None = None,
    params_b_override: float | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    registry_arc_version: str | None = None
    registry_arc_split: str | None = None

    for summary_path in summary_paths:
        payload = read_json(summary_path)
        metadata = metadata_from_payload(payload)
        arc_version = arc_version_override or metadata.get("arc_version")
        arc_split = arc_split_override or arc_split_from_metadata(metadata)
        params_b = params_b_override if params_b_override is not None else params_b_from_metadata(metadata)
        if not arc_version or not arc_split:
            raise ValueError(
                f"{path_for_cli(summary_path)} needs metadata.arc_version and metadata.arc_split/eval_split."
            )
        if params_b is None:
            raise ValueError(f"{path_for_cli(summary_path)} needs metadata.params_b or --params_b.")
        if registry_arc_version is None:
            registry_arc_version = str(arc_version)
            registry_arc_split = str(arc_split)
        elif str(arc_version) != registry_arc_version or str(arc_split) != registry_arc_split:
            raise ValueError("All generated baselines must share the same ARC-AGI version and split.")

        command = reproduction_command or infer_reproduction_command(payload)
        commit = git_commit or current_git_commit()
        for label in labels:
            rows.append(
                make_row(
                    summary_path=summary_path,
                    payload=payload,
                    label=label,
                    metric=metric,
                    name_prefix=name_prefix,
                    params_b=float(params_b),
                    arc_version=str(arc_version),
                    arc_split=str(arc_split),
                    reproduction_command=command,
                    git_commit=commit,
                    accessed_date=accessed_date,
                )
            )

    return {
        "benchmark": benchmark,
        "arc_version": registry_arc_version,
        "arc_split": registry_arc_split,
        "metric": metric,
        "same_size_band": {
            "min_params_b": min_params_b,
            "max_params_b": max_params_b,
        },
        "source_notes": (
            "Generated from local reproduced ARC-AGI eval artifacts. Use this for internal same-size "
            "controls; prefer external official/paper/model-card sources for public SOTA claims."
        ),
        "baselines": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_json", action="append", required=True)
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--min_params_b", type=float, default=DEFAULT_MIN_PARAMS_B)
    parser.add_argument("--max_params_b", type=float, default=DEFAULT_MAX_PARAMS_B)
    parser.add_argument("--params_b", type=float)
    parser.add_argument("--arc_version")
    parser.add_argument("--arc_split")
    parser.add_argument("--name_prefix", default="reproduced")
    parser.add_argument("--reproduction_command")
    parser.add_argument("--git_commit")
    parser.add_argument("--accessed_date", default=date.today().isoformat())
    parser.add_argument("--output_json", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--validation_json")
    args = parser.parse_args()

    summary_paths = [resolve_path(path) for path in args.summary_json]
    labels = split_csv(args.labels)
    registry = build_registry(
        summary_paths=summary_paths,
        labels=labels,
        metric=args.metric,
        benchmark=args.benchmark,
        min_params_b=args.min_params_b,
        max_params_b=args.max_params_b,
        name_prefix=args.name_prefix,
        reproduction_command=args.reproduction_command,
        git_commit=args.git_commit,
        accessed_date=args.accessed_date,
        arc_version_override=args.arc_version,
        arc_split_override=args.arc_split,
        params_b_override=args.params_b,
    )
    output_json = resolve_path(args.output_json)
    validation = validate_registry_payload(registry, source_path=output_json)
    if not validation.get("passed"):
        print(json.dumps(validation, indent=2))
        raise SystemExit("Generated registry did not pass validation.")
    write_json(output_json, registry)
    if args.validation_json:
        write_json(resolve_path(args.validation_json), validation)
    print(f"wrote_registry={path_for_cli(output_json)}")
    print(f"baseline_count={len(registry['baselines'])}")
    print(f"best_accuracy={validation.get('best_baseline', {}).get('accuracy')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
