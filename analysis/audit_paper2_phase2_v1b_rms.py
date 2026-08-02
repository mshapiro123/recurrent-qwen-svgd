"""Audit V1b state-RMS tails and recommend a bounded-write cap without inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from transformers import AutoTokenizer

from training.speculative_depth_d0_spec import DRAFTER_MODEL, DRAFTER_MODEL_REVISION
from training.speculative_depth_d0_corpus import sha256_file


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def deduplicate_position_records(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    invariant_fields = (
        "state_rms",
        "gradient_l2",
        "margin_before",
        "stratum",
        "scored_positions",
    )
    grouped: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["cohort"]), int(row["row_index"]), int(row["position"]))
        current = grouped.get(key)
        if current is None:
            grouped[key] = dict(row)
            continue
        for field in invariant_fields:
            if field in row or field in current:
                if row.get(field) != current.get(field):
                    raise RuntimeError(f"V1b invariant drift for {key}: {field}")
    return [grouped[key] for key in sorted(grouped)]


def sequence_position_bucket(position: int, scored_positions: int) -> str:
    if position == 0:
        return "position_0"
    if position <= 3:
        return "positions_1_3"
    denominator = max(1, scored_positions - 1)
    fraction = position / denominator
    if fraction < 0.25:
        return "early_quartile"
    if fraction < 0.75:
        return "middle_half"
    return "late_quartile"


def recommend_rms_cap(
    *,
    median: float,
    p99: float,
    high_rms_positions: Sequence[int],
    tail_hurt_rate: float,
    body_hurt_rate: float,
) -> dict[str, Any]:
    sink_fraction = (
        sum(int(position) <= 3 for position in high_rms_positions)
        / len(high_rms_positions)
        if high_rms_positions
        else 0.0
    )
    disproportionate_harm = tail_hurt_rate > 0 and tail_hurt_rate > 2 * body_hurt_rate
    if sink_fraction >= 0.5 or disproportionate_harm:
        return {
            "form": "p99_state_rms_cap",
            "value": float(p99),
            "attention_sink_fraction_in_top_1pct": sink_fraction,
            "reason": "top RMS tail is position-concentrated or has disproportionate collateral harm",
        }
    multiple = math.ceil((p99 / max(median, 1e-12)) * 10.0) / 10.0
    return {
        "form": "fixed_multiple_of_median_rms",
        "multiple": multiple,
        "value": float(multiple * median),
        "attention_sink_fraction_in_top_1pct": sink_fraction,
        "reason": "top RMS tail is diffuse and does not show disproportionate collateral harm",
    }


def _quantile_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        name: float(np.quantile(array, quantile))
        for name, quantile in (
            ("min", 0.0),
            ("p25", 0.25),
            ("median", 0.5),
            ("p75", 0.75),
            ("p90", 0.9),
            ("p95", 0.95),
            ("p99", 0.99),
            ("max", 1.0),
        )
    }


def _quantile_cells(
    records: Sequence[dict[str, Any]], field: str
) -> list[dict[str, Any]]:
    values = np.asarray([abs(float(row[field])) for row in records], dtype=np.float64)
    quantiles = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    edges = np.quantile(values, quantiles)
    cells = []
    for index in range(len(quantiles) - 1):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        if index + 1 == len(quantiles) - 1:
            selected = [row for row in records if lower <= abs(float(row[field])) <= upper]
        else:
            selected = [row for row in records if lower <= abs(float(row[field])) < upper]
        cells.append(
            {
                "quantile_range": [quantiles[index], quantiles[index + 1]],
                "value_range": [lower, upper],
                "positions": len(selected),
                "cohorts": dict(Counter(str(row["cohort"]) for row in selected)),
                "strata": dict(Counter(str(row["stratum"]) for row in selected)),
            }
        )
    return cells


def _rates(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    selected = list(rows)
    collateral_positions = sum(int(row["collateral_positions"]) for row in selected)
    hurts = sum(int(row["collateral_hurts"]) for row in selected)
    return {
        "records": len(selected),
        "pair_cross_rate": (
            sum(bool(row["realized_pair_cross"]) for row in selected) / len(selected)
            if selected
            else None
        ),
        "teacher_flip_rate": (
            sum(bool(row["target_correct_after"]) and not bool(row["target_correct_before"]) for row in selected)
            / len(selected)
            if selected
            else None
        ),
        "collateral_positions": collateral_positions,
        "collateral_hurts": hurts,
        "collateral_hurt_rate": hurts / collateral_positions if collateral_positions else 0.0,
    }


def _load_private_rows(private_dir: Path) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    paths = sorted((private_dir / "rows").glob("row_*.json"))
    if not paths:
        raise FileNotFoundError(f"No V1b private rows under {private_dir}")
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise RuntimeError(f"Unexpected V1b row payload: {path}")
        outputs.extend(payload)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private_dir", required=True)
    parser.add_argument("--v1b_summary", required=True)
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--v1_config", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--private_detail", required=True)
    args = parser.parse_args()

    private_dir = Path(args.private_dir)
    private_config = json.loads((private_dir / "config.json").read_text(encoding="utf-8"))
    if private_config.get("comparison_baseline") != "same_shape_same_batch_index_neutral_v2":
        raise RuntimeError("RMS audit requires corrected matched-neutral V1b records")
    v1b = json.loads(Path(args.v1b_summary).read_text(encoding="utf-8"))
    if v1b.get("kind") != "paper2_phase2_v1b_finite_perturbation":
        raise RuntimeError("RMS audit requires canonical V1b receipt")

    rows = _load_private_rows(private_dir)
    expected = int(private_config["sample_size_per_cohort"]) * 2 * len(private_config["c_values"])
    if len(rows) != expected:
        raise RuntimeError(f"V1b private record count {len(rows)} != expected {expected}")
    unique = deduplicate_position_records(rows)
    all_data = read_jsonl(args.data_jsonl)
    v1_config = json.loads(Path(args.v1_config).read_text(encoding="utf-8"))
    selected_indices = sorted(
        range(len(all_data)),
        key=lambda index: hashlib.sha256(
            f"phase2:{all_data[index]['row_id']}:-1".encode("utf-8")
        ).hexdigest(),
    )[: int(v1_config["selected_rows"])]
    selected_rows = [all_data[index] for index in selected_indices]
    tokenizer = AutoTokenizer.from_pretrained(
        DRAFTER_MODEL, revision=DRAFTER_MODEL_REVISION
    )

    rms = _quantile_summary([float(row["state_rms"]) for row in unique])
    p99 = rms["p99"]
    top = [row for row in unique if float(row["state_rms"]) >= p99]
    top_keys = {
        (str(row["cohort"]), int(row["row_index"]), int(row["position"]))
        for row in top
    }
    max_c = max(float(value) for value in private_config["c_values"])
    max_c_rows = [row for row in rows if math.isclose(float(row["c_value"]), max_c)]
    tail_rates = _rates(
        row
        for row in max_c_rows
        if (str(row["cohort"]), int(row["row_index"]), int(row["position"])) in top_keys
    )
    body_rates = _rates(
        row
        for row in max_c_rows
        if (str(row["cohort"]), int(row["row_index"]), int(row["position"])) not in top_keys
    )
    recommendation = recommend_rms_cap(
        median=rms["median"],
        p99=p99,
        high_rms_positions=[int(row["position"]) for row in top],
        tail_hurt_rate=float(tail_rates["collateral_hurt_rate"]),
        body_hurt_rate=float(body_rates["collateral_hurt_rate"]),
    )

    private_outliers = []
    for record in sorted(top, key=lambda row: float(row["state_rms"]), reverse=True):
        source = selected_rows[int(record["row_index"])]
        position = int(record["position"])
        input_ids = [int(value) for value in source["input_ids"]]
        token_id = input_ids[position]
        next_id = input_ids[position + 1] if position + 1 < len(input_ids) else None
        private_outliers.append(
            {
                **{key: record[key] for key in ("cohort", "row_id", "row_index", "position", "stratum", "state_rms", "gradient_l2", "margin_before", "scored_positions")},
                "position_bucket": sequence_position_bucket(position, int(record["scored_positions"])),
                "token_id": token_id,
                "token_text": tokenizer.convert_ids_to_tokens(token_id),
                "next_token_id": next_id,
                "next_token_text": tokenizer.convert_ids_to_tokens(next_id) if next_id is not None else None,
            }
        )
    _write_json(
        Path(args.private_detail),
        {
            "kind": "paper2_phase2_v1b_rms_private_outliers",
            "source_summary_sha256": sha256_file(args.v1b_summary),
            "outliers": private_outliers,
        },
    )

    public = {
        "kind": "paper2_phase2_v1b_rms_audit",
        "status": "complete_cpu_only_existing_records",
        "training_started": False,
        "model_inference_started": False,
        "records": {"finite_perturbation": len(rows), "unique_positions": len(unique)},
        "state_rms": rms,
        "gradient_l2": _quantile_summary([float(row["gradient_l2"]) for row in unique]),
        "absolute_original_margin": _quantile_summary([abs(float(row["margin_before"])) for row in unique]),
        "quantile_strata": {
            field: _quantile_cells(unique, field)
            for field in ("state_rms", "gradient_l2", "margin_before")
        },
        "content_strata": {
            stratum: _rates(row for row in max_c_rows if str(row["stratum"]) == stratum)
            for stratum in sorted({str(row["stratum"]) for row in unique})
        },
        "sequence_position": {
            bucket: _rates(
                row
                for row in max_c_rows
                if sequence_position_bucket(int(row["position"]), int(row["scored_positions"])) == bucket
            )
            for bucket in ("position_0", "positions_1_3", "early_quartile", "middle_half", "late_quartile")
        },
        "top_1pct_rms": {
            "positions": len(top),
            "cohorts": dict(Counter(str(row["cohort"]) for row in top)),
            "strata": dict(Counter(str(row["stratum"]) for row in top)),
            "position_buckets": dict(
                Counter(sequence_position_bucket(int(row["position"]), int(row["scored_positions"])) for row in top)
            ),
            "token_ids": dict(Counter(str(row["token_id"]) for row in private_outliers).most_common(20)),
            "token_text": dict(Counter(str(row["token_text"]) for row in private_outliers).most_common(20)),
            "max_c_rates": tail_rates,
        },
        "non_tail_max_c_rates": body_rates,
        "cap_recommendation": recommendation,
        "sources": {
            "v1b_summary_sha256": sha256_file(args.v1b_summary),
            "v1b_private_config_sha256": sha256_file(private_dir / "config.json"),
            "data_jsonl_sha256": sha256_file(args.data_jsonl),
            "v1_config_sha256": sha256_file(args.v1_config),
            "private_detail_sha256": sha256_file(args.private_detail),
        },
        "method_notes": [
            "The audit uses existing corrected matched-neutral V1b records and performs no model inference.",
            "RMS, margin, and gradient strata are descriptive and do not alter V1c constants.",
            "Token-level outlier rows remain private; the public receipt contains aggregate token counts only.",
        ],
    }
    _write_json(Path(args.output_summary), public)
    print(json.dumps(public, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
