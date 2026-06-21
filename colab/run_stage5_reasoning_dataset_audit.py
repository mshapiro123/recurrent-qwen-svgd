"""Audit candidate reasoning-trace datasets from the registry in Colab.

This runner is deliberately CPU/data oriented. It does not fine-tune. It answers
which Hugging Face trace corpora are compatible with our recurrent training
format, which should feed immediate competence recovery, and which should wait
for agent/tool-specific filtering.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RUN_ID = os.environ.get("STAGE5_DATASET_AUDIT_RUN_ID") or time.strftime("stage5_dataset_audit_%Y%m%d_%H%M%S")
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
REGISTRY_PATH = ROOT / os.environ.get("STAGE5_DATASET_REGISTRY", "config/reasoning_dataset_registry.yaml")
DEFAULT_KEYS = "opus47_sft,opus47_raw,fable5_pi_agent,fable5_flat,jackrong_opus47_trace_inversion"
SELECTED_KEYS = [
    key.strip()
    for key in os.environ.get("STAGE5_DATASET_AUDIT_KEYS", DEFAULT_KEYS).split(",")
    if key.strip()
]
LIMIT = int(os.environ.get("STAGE5_DATASET_AUDIT_LIMIT", "1000"))
TOKENIZER_NAME = os.environ.get("TOKENIZER_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
PUSH_RESULTS = os.environ.get("STAGE5_DATASET_AUDIT_PUSH", "1").strip().lower() in {"1", "true", "yes", "y"}


def path_for_cli(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def audit_command(key: str, spec: dict[str, Any], output: Path) -> list[str]:
    cmd = [
        sys.executable,
        "training/inspect_hf_reasoning_dataset.py",
        "--dataset_id",
        str(spec["dataset_id"]),
        "--split",
        str(spec.get("split", "train")),
        "--adapter",
        str(spec.get("adapter", "auto")),
        "--tokenizer_name",
        TOKENIZER_NAME,
        "--limit",
        str(int(spec.get("audit_limit", LIMIT) or LIMIT)),
        "--output_json",
        path_for_cli(output),
    ]
    if spec.get("name"):
        cmd += ["--name", str(spec["name"])]
    if spec.get("hf_file"):
        cmd += ["--hf_file", str(spec["hf_file"])]
    if spec.get("streaming"):
        cmd += ["--streaming"]
    return cmd


def priority_score(priority: str) -> int:
    order = {
        "immediate": 0,
        "immediate_candidate": 0,
        "audit": 1,
        "later": 2,
        "immediate_audit_candidate": 1,
        "later_audit": 3,
        "later_audit_only": 4,
    }
    return order.get(priority, 9)


def recommendation_for(key: str, spec: dict[str, Any], report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {
            "key": key,
            "dataset_id": spec.get("dataset_id"),
            "status": "audit_failed",
            "recommendation": "Do not train until the audit error is fixed.",
            "priority": spec.get("priority", "unknown"),
            "converted_rows": 0,
            "conversion_rate": 0.0,
        }

    role = report.get("training_role", {})
    conversion_rate = float(report.get("conversion_rate", 0.0) or 0.0)
    registry_priority = str(spec.get("priority", "unknown"))
    role_priority = str(role.get("priority", registry_priority))
    converted_rows = int(report.get("converted_rows", 0) or 0)
    token_stats = report.get("token_stats", {}).get("total_tokens", {})
    p90_tokens = token_stats.get("p90")

    if role_priority in {"immediate", "immediate_candidate"} and conversion_rate >= 0.5:
        status = "promote_to_small_train_mix"
        recommendation = "Use a filtered subset in the next modified-Opus/recurrent SFT experiment."
    elif "fable" in key and converted_rows:
        status = "hold_for_agent_tool_filter"
        recommendation = "Keep for coding/tool/trajectory diversity; do not mix into ARC/GPQA recovery yet."
    elif conversion_rate >= 0.2:
        status = "audit_deeper"
        recommendation = "Schema is usable enough for a filtered pilot, but inspect samples first."
    else:
        status = "do_not_train"
        recommendation = "Low compatibility or unclear role; avoid GPU training for now."

    return {
        "key": key,
        "dataset_id": spec.get("dataset_id"),
        "name": spec.get("name"),
        "registry_priority": registry_priority,
        "role_priority": role_priority,
        "status": status,
        "recommendation": recommendation,
        "converted_rows": converted_rows,
        "conversion_rate": conversion_rate,
        "adapter_success_counts": report.get("adapter_success_counts", {}),
        "primary_role": role.get("primary_role"),
        "total_tokens_p90": p90_tokens,
        "cot_tokens_p90": report.get("token_stats", {}).get("cot_tokens", {}).get("p90"),
        "license": spec.get("license"),
        "avoid_for_now": spec.get("avoid_for_now") or role.get("avoid_for_now", []),
    }


def sort_recommendations(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        items,
        key=lambda item: (
            priority_score(str(item.get("role_priority") or item.get("registry_priority") or "")),
            str(item.get("status")),
            str(item.get("key")),
        ),
    )


def write_summary(
    registry: dict[str, Any],
    audit_reports: dict[str, dict[str, Any] | None],
    errors: dict[str, str],
) -> dict[str, Any]:
    recommendations = sort_recommendations(
        [
            recommendation_for(key, registry["datasets"][key], audit_reports.get(key))
            for key in SELECTED_KEYS
        ]
    )
    promote = [item for item in recommendations if item["status"] == "promote_to_small_train_mix"]
    hold = [item for item in recommendations if item["status"] == "hold_for_agent_tool_filter"]
    payload = {
        "run_id": RUN_ID,
        "kind": "stage5_reasoning_dataset_audit",
        "registry": path_for_cli(REGISTRY_PATH),
        "selected_keys": SELECTED_KEYS,
        "limit": LIMIT,
        "tokenizer_name": TOKENIZER_NAME,
        "status": "ok" if not errors else "completed_with_errors",
        "errors": errors,
        "recommendations": recommendations,
        "next_step": next_step(promote, hold, errors),
    }
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (RUN_DIR / "summary.md").write_text(summary_markdown(payload), encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))
    return payload


def next_step(
    promote: list[dict[str, Any]],
    hold: list[dict[str, Any]],
    errors: dict[str, str],
) -> str:
    if promote:
        keys = ", ".join(item["key"] for item in promote)
        return f"Prepare filtered small-train mix from {keys}; keep competence-preserving ARC objective active."
    if errors:
        return "Fix failed audits before expanding the training mix."
    if hold:
        return "Build agent/tool-specific filters before using Fable-style traces."
    return "No dataset is ready for training; inspect audit reports manually."


def summary_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Stage 5 Reasoning Dataset Audit - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Registry: `{payload['registry']}`",
        f"- Limit per dataset: `{payload['limit']}`",
        f"- Next step: {payload['next_step']}",
        "",
        "| Dataset | Status | Converted | P90 total tokens | Role | Recommendation |",
        "|---|---:|---:|---:|---|---|",
    ]
    for item in payload["recommendations"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{item['key']}`",
                    f"`{item['status']}`",
                    f"{item['converted_rows']} ({item['conversion_rate']:.1%})",
                    str(item.get("total_tokens_p90")),
                    str(item.get("primary_role")),
                    str(item.get("recommendation")),
                ]
            )
            + " |"
        )
    if payload["errors"]:
        lines += ["", "## Errors"]
        for key, error in payload["errors"].items():
            lines.append(f"- `{key}`: {error}")
    return "\n".join(lines) + "\n"


def commit_results() -> None:
    if not PUSH_RESULTS:
        return
    run(["git", "add", "-f", path_for_cli(RUN_DIR)], check=False)
    if run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("No dataset audit outputs changed.")
        return
    run(["git", "commit", "-m", f"Record Stage 5 reasoning dataset audit {RUN_ID}"])
    run(["git", "push", "origin", "main"], check=False)


def main() -> int:
    registry = load_registry()
    unknown = [key for key in SELECTED_KEYS if key not in registry.get("datasets", {})]
    if unknown:
        raise KeyError(f"Unknown dataset registry keys: {unknown}")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    audit_reports: dict[str, dict[str, Any] | None] = {}
    errors: dict[str, str] = {}
    for key in SELECTED_KEYS:
        spec = registry["datasets"][key]
        output = RUN_DIR / "audits" / f"{key}.json"
        try:
            run(audit_command(key, spec, output), log_name=f"audit_{key}.log")
            audit_reports[key] = read_json(output)
        except Exception as exc:  # noqa: BLE001 - keep long Colab batch moving.
            errors[key] = str(exc)
            audit_reports[key] = None

    write_summary(registry, audit_reports, errors)
    commit_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
