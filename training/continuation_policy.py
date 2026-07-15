"""Mechanical continuation and lineage policy for post-Part-1 experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class GuardrailFloor:
    metric: str
    floor: float


def _nested_value(payload: dict[str, Any], dotted_path: str) -> float:
    value: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise RuntimeError(f"Launch receipt is missing guardrail metric {dotted_path!r}")
        value = value[part]
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Guardrail metric {dotted_path!r} is not numeric: {value!r}") from exc


def assert_launch_guardrail_floors(
    receipt: dict[str, Any],
    floors: Iterable[GuardrailFloor],
) -> dict[str, Any]:
    """Fail before a continuation when any resolved source metric is below floor."""

    checks: list[dict[str, Any]] = []
    failures: list[str] = []
    for requirement in floors:
        value = _nested_value(receipt, requirement.metric)
        passed = value >= float(requirement.floor)
        checks.append({**asdict(requirement), "value": value, "passed": passed})
        print(
            "[launch-floor] "
            f"metric={requirement.metric} value={value:.6f} floor={requirement.floor:.6f} passed={passed}",
            flush=True,
        )
        if not passed:
            failures.append(f"{requirement.metric}={value:g} below floor={requirement.floor:g}")
    if failures:
        raise RuntimeError("Continuation source failed launch-time floor assertion: " + "; ".join(failures))
    return {"status": "green", "checks": checks}


def assert_training_lineage(
    *,
    regime: str,
    full_block_trainable: bool,
    checkpoint_promotable: bool,
    successor_source_allowed: bool,
    detachable_adapter: bool = False,
) -> dict[str, Any]:
    """Enforce the frozen-asset regime and disposable measurement carve-out."""

    allowed_regimes = {"frozen_asset", "disposable_measurement"}
    if regime not in allowed_regimes:
        raise RuntimeError(f"Unknown lineage regime {regime!r}; expected {sorted(allowed_regimes)}")
    if full_block_trainable and regime != "disposable_measurement":
        raise RuntimeError("full-block training is allowed only in a disposable measurement branch")
    if regime == "disposable_measurement" and checkpoint_promotable:
        raise RuntimeError("Disposable measurement checkpoints must never be promoted")
    if regime == "disposable_measurement" and successor_source_allowed:
        raise RuntimeError("Disposable measurement checkpoints must never be used as successor sources")
    if regime == "frozen_asset" and full_block_trainable:
        raise RuntimeError("A frozen asset cannot permit full-block training")
    result = {
        "status": "allowed",
        "regime": regime,
        "full_block_trainable": bool(full_block_trainable),
        "detachable_adapter": bool(detachable_adapter),
        "checkpoint_promotable": bool(checkpoint_promotable),
        "successor_source_allowed": bool(successor_source_allowed),
        "keeper_successor": bool(checkpoint_promotable and successor_source_allowed),
    }
    print(f"[lineage-policy] {result}", flush=True)
    return result
