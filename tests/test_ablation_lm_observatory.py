from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib

import pytest

from models.ablation_lm.config import RATIFIED_TARGET_AUTHORITY_SHA256
from models.ablation_lm.observatory import (
    ObservatoryBlocked,
    ObservatoryGuard,
    T14bReceipt,
)


GRAPH_SHA = hashlib.sha256(b"weft1-test-graph").hexdigest()
CONFIG_SHA = hashlib.sha256(b"weft1-test-config").hexdigest()
STALE_GRAPH_SHA = hashlib.sha256(b"weft1-stale-graph").hexdigest()
K_VALUES = (1, 2, 4, 8)
STAGES = (
    "token_embedding",
    "prelude",
    "recurrent_core",
    "scratch_lanes",
    "callosum",
    "sidecar",
    "coda",
    "lm_head",
)


def _receipt(**changes: object) -> T14bReceipt:
    values: dict[str, object] = {
        "authority_sha256": RATIFIED_TARGET_AUTHORITY_SHA256,
        "graph_fingerprint": GRAPH_SHA,
        "config_fingerprint": CONFIG_SHA,
        "tested_k_values": K_VALUES,
        "sequence_axis_stages": STAGES,
        "packed": True,
        "padded": True,
        "max_future_gradient": 0.0,
        "passed": True,
    }
    values.update(changes)
    return T14bReceipt(**values)  # type: ignore[arg-type]


def _guard(receipt: T14bReceipt) -> ObservatoryGuard:
    return ObservatoryGuard(
        expected_graph_fingerprint=GRAPH_SHA,
        expected_config_fingerprint=CONFIG_SHA,
        expected_receipt_sha256=receipt.receipt_sha256,
        required_k_values=K_VALUES,
        required_sequence_axis_stages=STAGES,
    )


def _counting_callback(calls: list[str]) -> str:
    calls.append("called")
    return "statistic-result"


def test_t14b_receipt_is_immutable_and_hashes_a_canonical_payload() -> None:
    first = _receipt()
    second = _receipt()

    assert first.receipt_sha256 == second.receipt_sha256
    assert first.receipt_sha256 == hashlib.sha256(first.canonical_payload()).hexdigest()
    assert first.canonical_payload() == second.canonical_payload()
    with pytest.raises(FrozenInstanceError):
        first.passed = False  # type: ignore[misc]


def test_missing_and_failed_receipts_block_before_callback() -> None:
    passing = _receipt()
    guard = _guard(passing)
    calls: list[str] = []

    with pytest.raises(ObservatoryBlocked, match="required before observation"):
        guard.invoke(None, _counting_callback, calls)
    failed = replace(passing, passed=False, max_future_gradient=1e-6)
    with pytest.raises(ObservatoryBlocked, match="did not pass"):
        guard.invoke(failed, _counting_callback, calls)

    assert calls == []


def test_stale_graph_receipt_blocks_before_callback() -> None:
    passing = _receipt()
    stale = _receipt(graph_fingerprint=STALE_GRAPH_SHA)
    guard = _guard(passing)
    calls: list[str] = []

    with pytest.raises(ObservatoryBlocked, match="graph fingerprint is stale"):
        guard.invoke(stale, _counting_callback, calls)

    assert calls == []


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"tested_k_values": (1, 2, 4)}, "K coverage"),
        ({"sequence_axis_stages": STAGES[:-1]}, "stage coverage"),
        ({"packed": False}, "packed-sequence coverage"),
        ({"padded": False}, "padded-sequence coverage"),
    ],
)
def test_missing_coverage_blocks_before_callback(
    changes: dict[str, object],
    message: str,
) -> None:
    passing = _receipt()
    incomplete = _receipt(**changes)
    guard = _guard(passing)
    calls: list[str] = []

    with pytest.raises(ObservatoryBlocked, match=message):
        guard.invoke(incomplete, _counting_callback, calls)

    assert calls == []


def test_valid_exact_receipt_allows_one_callback() -> None:
    passing = _receipt()
    guard = _guard(passing)
    calls: list[str] = []

    result = guard.invoke(passing, _counting_callback, calls)

    assert result == "statistic-result"
    assert calls == ["called"]


def test_receipt_rejects_inexact_public_types_and_nonzero_pass() -> None:
    with pytest.raises(TypeError, match="exact tuple"):
        _receipt(tested_k_values=[1, 2, 4, 8])
    with pytest.raises(TypeError, match="exact int"):
        _receipt(tested_k_values=(1, True))
    with pytest.raises(TypeError, match="exact bool"):
        _receipt(packed=1)
    with pytest.raises(TypeError, match="exact float"):
        _receipt(max_future_gradient=0)
    with pytest.raises(ValueError, match="exact zero"):
        _receipt(max_future_gradient=1e-12)
