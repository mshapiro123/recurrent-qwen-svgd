"""Diagnose ARC-Easy content regression before choosing a repair.

This runner is intentionally CPU-cheap.  It inspects an existing debiased
benchmark-suite output and decides whether an ARC-Easy content drop looks like:

* option-order sensitivity,
* content-vs-cyclic scoring surface mismatch, or
* true direct/content erosion that warrants direct-preservation distillation.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.analyze_mcq_order_sensitivity import analyze as analyze_order_sensitivity  # noqa: E402
from eval.analyze_mcq_order_sensitivity import markdown_report as order_markdown_report  # noqa: E402
from eval.analyze_mcq_surface_mismatch import analyze as analyze_surface_mismatch  # noqa: E402
from eval.analyze_mcq_surface_mismatch import markdown_report as surface_markdown_report  # noqa: E402


RUN_ID = os.environ.get("STAGE5_ARC_EASY_REGRESSION_DIAG_RUN_ID") or time.strftime(
    "stage5_arc_easy_regression_diagnostic_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
SOURCE_SUMMARY = os.environ.get(
    "STAGE5_ARC_EASY_REGRESSION_DIAG_SOURCE_SUMMARY",
    os.environ.get("STAGE5_ARC_AGI_CURRENT_SOURCE_SUMMARY", "config/stage5_current_source_summary.txt"),
)
PUSH_RESULTS = os.environ.get("STAGE5_ARC_EASY_REGRESSION_DIAG_PUSH", "1").strip().lower() in {
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


def current_source_summary_file() -> Path:
    return ROOT / "config" / "stage5_current_source_summary.txt"


def update_current_source_summary(summary_path: Path) -> None:
    pointer = current_source_summary_file()
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(path_for_cli(summary_path) + "\n", encoding="utf-8")


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(map(str, cmd)), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(proc.stdout, flush=True)
    if check and proc.returncode:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}")
    return proc


def existing_jsonl(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def resolve_benchmark_summary(source_summary: Path, payload: dict[str, Any]) -> Path:
    if payload.get("kind") == "stage5_benchmark_suite":
        return source_summary
    for key in ("benchmark_source_summary", "benchmark_summary", "source_summary"):
        raw = str(payload.get(key) or "").strip()
        if not raw:
            continue
        candidate = resolve_path(raw)
        if candidate.exists() and read_json(candidate).get("kind") == "stage5_benchmark_suite":
            return candidate
    raise ValueError(f"Could not resolve stage5_benchmark_suite summary from {path_for_cli(source_summary)}")


def choose_status(order_summary: dict[str, Any], surface_summary: dict[str, Any]) -> tuple[str, str, str]:
    order_rec = str(order_summary.get("recommendation") or "")
    surface_rec = str(surface_summary.get("recommendation") or "")
    if "conditional_invariance" in {order_rec, surface_rec} or (
        order_rec == "prioritize_conditional_invariance_repair"
        or surface_rec == "prioritize_conditional_invariance_repair"
    ):
        return (
            "order_sensitivity_likely",
            "conditional_invariance_repair",
            "Run conditional-invariance repair on ARC-Easy direct rows before any base-distillation repair.",
        )
    if (
        surface_rec == "prioritize_content_cyclic_surface_alignment"
        or order_rec == "diagnose_content_route_scoring_or_prompt_alignment_before_more_distillation"
    ):
        return (
            "surface_mismatch_likely",
            "content_cyclic_surface_alignment",
            "Run content-vs-cyclic surface-alignment repair before direct distillation.",
        )
    if (
        order_rec == "prioritize_direct_distillation_or_data_repair"
        or surface_rec == "prioritize_direct_distillation_or_data_repair"
    ):
        return (
            "content_erosion_likely",
            "direct_preservation_repair",
            "Run bounded max_loops=1 direct-preservation repair; cyclic/order diagnostics did not explain the gap.",
        )
    return (
        "inspect_arc_easy_regression",
        "manual_inspection",
        "Inspect the order and surface reports before choosing a repair objective.",
    )


def write_summary(payload: dict[str, Any]) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    summary = RUN_DIR / "summary.json"
    summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    update_current_source_summary(summary)
    lines = [
        f"# Stage 5 ARC-Easy Regression Diagnostic - {RUN_ID}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Repair action: `{payload['repair_action']}`",
        f"- Source summary: `{payload['source_summary']}`",
        f"- Benchmark summary: `{payload['benchmark_source_summary']}`",
        f"- Order recommendation: `{payload['order_sensitivity_recommendation']}`",
        f"- Surface recommendation: `{payload['surface_recommendation']}`",
        f"- Content delta: `{payload['content_delta']}`",
        f"- Content losses: `{payload['content_losses']}`",
        f"- Order-sensitive content losses: `{payload['content_losses_order_sensitive']}`",
        f"- Stable cyclic rescues: `{payload['content_losses_stably_rescued_by_cyclic']}`",
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
        print("No ARC-Easy regression diagnostic outputs changed.")
        return
    run(["git", "commit", "-m", f"Record ARC-Easy regression diagnostic {RUN_ID} [skip ci]"])
    run(["git", "pull", "--rebase", "origin", "main"], check=False)
    run(["git", "push", "origin", "main"], check=False)


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    source_summary = resolve_path(SOURCE_SUMMARY)
    if not source_summary.exists():
        raise FileNotFoundError(source_summary)
    source_payload = read_json(source_summary)
    benchmark_summary = resolve_benchmark_summary(source_summary, source_payload)
    benchmark_dir = benchmark_summary.parent

    base_content = existing_jsonl(benchmark_dir / "arc_easy_base_content_question_only.jsonl")
    candidate_content = existing_jsonl(benchmark_dir / "arc_easy_recurrent_content_question_only.jsonl")
    candidate_cyclic = existing_jsonl(benchmark_dir / "arc_easy_recurrent_cyclic_label_aggregated.jsonl")
    base_cyclic = existing_jsonl(benchmark_dir / "arc_easy_base_cyclic_label_aggregated.jsonl")

    order_payload = analyze_order_sensitivity(
        benchmark=f"arc_easy_{benchmark_dir.name}",
        base_content_path=base_content,
        candidate_content_path=candidate_content,
        candidate_cyclic_path=candidate_cyclic,
        base_cyclic_path=base_cyclic,
    )
    surface_payload = analyze_surface_mismatch(
        benchmark=f"arc_easy_{benchmark_dir.name}",
        base_content_path=base_content,
        candidate_content_path=candidate_content,
        candidate_cyclic_path=candidate_cyclic,
        base_cyclic_path=base_cyclic,
    )
    order_json = RUN_DIR / "arc_easy_order_sensitivity_diagnosis.json"
    order_md = RUN_DIR / "arc_easy_order_sensitivity_diagnosis.md"
    surface_json = RUN_DIR / "arc_easy_surface_mismatch_diagnosis.json"
    surface_md = RUN_DIR / "arc_easy_surface_mismatch_diagnosis.md"
    order_json.write_text(json.dumps(order_payload, indent=2), encoding="utf-8")
    order_md.write_text(order_markdown_report(order_payload), encoding="utf-8")
    surface_json.write_text(json.dumps(surface_payload, indent=2), encoding="utf-8")
    surface_md.write_text(surface_markdown_report(surface_payload), encoding="utf-8")

    order_summary = order_payload["summary"]
    surface_summary = surface_payload["summary"]
    status, repair_action, next_step = choose_status(order_summary, surface_summary)
    payload = {
        "kind": "stage5_arc_easy_regression_diagnostic",
        "run_id": RUN_ID,
        "status": status,
        "repair_action": repair_action,
        "source_summary": path_for_cli(source_summary),
        "source_kind": source_payload.get("kind"),
        "benchmark_source_summary": path_for_cli(benchmark_summary),
        "order_sensitivity_diagnosis": path_for_cli(order_json),
        "surface_diagnosis": path_for_cli(surface_json),
        "order_sensitivity_recommendation": order_summary.get("recommendation"),
        "surface_recommendation": surface_summary.get("recommendation"),
        "content_delta": order_summary.get("content_delta"),
        "content_losses": order_summary.get("content_losses"),
        "content_losses_order_sensitive": order_summary.get("content_losses_order_sensitive"),
        "content_losses_order_sensitive_fraction": order_summary.get("content_losses_order_sensitive_fraction"),
        "content_losses_rescued_by_cyclic": order_summary.get("content_losses_rescued_by_cyclic"),
        "content_losses_stably_rescued_by_cyclic": surface_summary.get("content_losses_stably_rescued_by_cyclic"),
        "order_sensitivity_loss_rate_lift": order_summary.get("order_sensitivity_loss_rate_lift"),
        "next_step": next_step,
    }
    write_summary(payload)
    commit_results([order_json, order_md, surface_json, surface_md])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
