"""Evaluate the pre-rehearsal cap-3 source on the natural-surface canary."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colab.run_stage5_inverse_composition_staircase import (
    _prepare_guardrail_data,
    _publish,
    _run_diagonal,
    write_json,
)
from colab.run_stage5_natural_surface_transfer import restore_checkpoint, sha256_file


ROOT = Path(os.environ.get("STAGE5_ROOT", REPO_ROOT))
RUN_ID = "stage5_inverse_table_cap3_rehearsal_20260714"
SOURCE_SHA256 = "83767ebff2c2a13a2f15fe8266f605fb8485985c3289c1f1720cd70c122a9ac5"
SOURCE_DRIVE = Path(
    "/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/"
    "stage5_inverse_table_rebase_caps3_4_20260713/"
    "C_rebase_cap_3_selected/unfrozen_recurrent_step_250.pt"
)
LOCKED_BASELINE = ROOT / (
    "outputs/stage5/stage5_inverse_composition_staircase_20260713/"
    "guardrails/tier1_natural_surface_keeper/summary.json"
)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare_natural_canaries(
    *,
    locked: dict[str, Any],
    source: dict[str, Any],
    repaired: dict[str, Any],
) -> dict[str, Any]:
    depths = sorted(set(locked["by_depth"]) & set(source["by_depth"]) & set(repaired["by_depth"]), key=int)
    by_depth = {}
    for depth in depths:
        locked_accuracy = float(locked["by_depth"][depth]["accuracy"])
        source_accuracy = float(source["by_depth"][depth]["accuracy"])
        repaired_accuracy = float(repaired["by_depth"][depth]["accuracy"])
        by_depth[depth] = {
            "locked_accuracy": locked_accuracy,
            "source_accuracy": source_accuracy,
            "repaired_accuracy": repaired_accuracy,
            "source_minus_locked": source_accuracy - locked_accuracy,
            "repaired_minus_source": repaired_accuracy - source_accuracy,
            "repaired_minus_locked": repaired_accuracy - locked_accuracy,
        }
    return {
        "locked_accuracy": float(locked["accuracy"]),
        "source_accuracy": float(source["accuracy"]),
        "repaired_accuracy": float(repaired["accuracy"]),
        "source_minus_locked": float(source["accuracy"]) - float(locked["accuracy"]),
        "repaired_minus_source": float(repaired["accuracy"]) - float(source["accuracy"]),
        "repaired_minus_locked": float(repaired["accuracy"]) - float(locked["accuracy"]),
        "by_depth": by_depth,
    }


def main() -> int:
    run_dir = ROOT / "outputs" / "stage5" / RUN_ID
    repaired_summary = read_json(run_dir / "guardrails" / "cap3_rehearsal_natural" / "summary.json")
    locked_summary = read_json(LOCKED_BASELINE)
    source_checkpoint, receipt = restore_checkpoint(
        [run_dir / "restored" / "C_cap3.pt", SOURCE_DRIVE],
        run_dir / "restored" / "C_cap3_attribution.pt",
        label="rehearsal_attribution_C_cap3",
    )
    actual_sha = sha256_file(source_checkpoint)
    if actual_sha != SOURCE_SHA256:
        raise RuntimeError(f"Source cap-3 SHA mismatch: expected={SOURCE_SHA256}, actual={actual_sha}")
    guardrail_paths = _prepare_guardrail_data(run_dir)
    source_summary = _run_diagonal(
        run_dir,
        label="source_cap3_natural_attribution",
        checkpoint=source_checkpoint,
        data_jsonl=guardrail_paths["natural"],
        max_depth=8,
        value_prefix="name:",
    )
    comparison = compare_natural_canaries(
        locked=locked_summary,
        source=source_summary,
        repaired=repaired_summary,
    )
    payload = {
        "kind": "stage5_inverse_rehearsal_natural_attribution",
        "run_id": RUN_ID,
        "source_checkpoint_sha256": actual_sha,
        "restore_receipt": receipt,
        "comparison": comparison,
        "cap4_authorized": False,
        "status": "attribution_complete_successor_still_blocked",
    }
    output = run_dir / "natural_canary_attribution.json"
    write_json(output, payload)
    lines = [
        "# Natural Canary Attribution",
        "",
        f"- Locked keeper: `{comparison['locked_accuracy']:.6f}`",
        f"- Pre-rehearsal cap-3 source: `{comparison['source_accuracy']:.6f}`",
        f"- Rehearsal checkpoint: `{comparison['repaired_accuracy']:.6f}`",
        f"- Source minus locked: `{comparison['source_minus_locked']:+.6f}`",
        f"- Rehearsal minus source: `{comparison['repaired_minus_source']:+.6f}`",
        "- Cap 4 remains unauthorized.",
    ]
    (run_dir / "natural_canary_attribution.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    _publish(run_dir, f"Attribute cap-3 rehearsal natural regression {RUN_ID} [skip ci]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
