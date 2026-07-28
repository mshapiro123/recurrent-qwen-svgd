"""Contracts shared by the fresh EVAL-B and DC0 depth-by-append jobs."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable


EVAL_B_SEED = 20260728
EVAL_B_TOKENS = 200_000
EVAL_B_STRATUM_TOKENS = {"general": 100_000, "code": 100_000}
DC0_APPEND_STEPS = (0, 1, 2, 3)
DC0_INPLACE_DEPTHS = (1, 2, 3, 4)


def _document_ids(rows: Iterable[dict[str, Any]]) -> list[str]:
    return sorted({str(row["document_id"]) for row in rows})


def eval_b_document_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    document_ids = _document_ids(rows)
    payload = ("\n".join(document_ids) + "\n").encode("utf-8")
    return {
        "rows": len(rows),
        "tokens": sum(int(row.get("token_count", len(row.get("input_ids", [])))) for row in rows),
        "documents": len(document_ids),
        "document_id_list_sha256": hashlib.sha256(payload).hexdigest(),
    }


def assert_eval_b_document_disjoint(
    rows: list[dict[str, Any]], *, prior_document_ids: set[str]
) -> dict[str, Any]:
    current = set(_document_ids(rows))
    overlap = sorted(current & set(prior_document_ids))
    if overlap:
        raise RuntimeError(
            "EVAL-B overlaps prior D0 documents: " + ", ".join(overlap[:10])
        )
    return {
        "document_disjoint": True,
        "prior_documents": len(prior_document_ids),
        "eval_b_documents": len(current),
        "overlap_count": 0,
    }


def layer_application_costs(*, recurrent_layers: int = 12, outer_layers: int = 12) -> dict[str, Any]:
    return {
        "loop_one_total_per_position": outer_layers + recurrent_layers,
        "inplace_extra_per_loop": recurrent_layers,
        "append_extra_per_slot": outer_layers + recurrent_layers,
        "append_to_inplace_first_marginal_ratio": (outer_layers + recurrent_layers)
        / recurrent_layers,
        "attention_overhead_included": False,
        "inplace_formula": "(depth - 1) * recurrent_layers",
        "append_formula": "slots * (outer_layers + recurrent_layers) plus attention overhead",
    }
