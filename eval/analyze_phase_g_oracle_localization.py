"""Post-hoc localization for the terminal Phase G oracle-interface probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        raise ValueError("Cannot average an empty row collection")
    return sum(float(row[key]) for row in rows) / len(rows)


def analyze_route(
    route: str,
    arm: dict[str, Any],
    trace: list[dict[str, Any]],
    *,
    window: int = 100,
) -> dict[str, Any]:
    if len(trace) < window:
        raise ValueError(f"{route} trace has {len(trace)} rows, fewer than window={window}")
    if any(str(row["route"]) != route for row in trace):
        raise ValueError(f"{route} trace contains rows from another route")

    steps = [int(row["step"]) for row in trace]
    if steps != list(range(1, len(trace) + 1)):
        raise ValueError(f"{route} trace steps must be contiguous from 1")

    transition = arm["transition_control"]
    loop_one = transition["by_loop_index"]["1"]
    loop_four = transition["by_loop_index"]["4"]
    default = transition["default"]
    nondefault = transition["nondefault"]
    first = trace[:window]
    last = trace[-window:]

    return {
        "route": route,
        "registered_gate_status": arm["gate_status"],
        "registered_passed": bool(arm["passed"]),
        "heldout": {
            "overall_transition_control": transition["overall"],
            "default_transition_control": default,
            "nondefault_transition_control": nondefault,
            "default_minus_nondefault_control_rate": (
                float(default["control_rate"]) - float(nondefault["control_rate"])
            ),
            "loop_1_transition_control": loop_one,
            "loop_4_transition_control": loop_four,
            "loop_1_minus_loop_4_control_rate": (
                float(loop_one["control_rate"]) - float(loop_four["control_rate"])
            ),
            "loop_1_minus_loop_4_legality_rate": (
                float(loop_one["legality_rate"]) - float(loop_four["legality_rate"])
            ),
        },
        "training_liveness": {
            "steps": len(trace),
            "window": window,
            "first_window_mean_loss": mean(first, "loss"),
            "last_window_mean_loss": mean(last, "loss"),
            "loss_change": mean(last, "loss") - mean(first, "loss"),
            "first_window_mean_gradient_norm": mean(first, "gradient_norm"),
            "last_window_mean_gradient_norm": mean(last, "gradient_norm"),
            "last_window_mean_residual_rms_ratio": mean(
                last, "oracle_reentry_residual_rms_ratio"
            ),
        },
    }


def analyze(
    gate: dict[str, Any],
    traces: dict[str, list[dict[str, Any]]],
    *,
    window: int = 100,
) -> dict[str, Any]:
    routes = ("additive", "film")
    if tuple(gate["arms"].keys()) != routes:
        raise ValueError("Gate must contain additive and film arms in locked order")
    if set(traces) != set(routes):
        raise ValueError("Traces must contain exactly additive and film routes")

    analyzed = {
        route: analyze_route(route, gate["arms"][route], traces[route], window=window)
        for route in routes
    }
    return {
        "kind": "phase_g_oracle_interface_posthoc_localization",
        "analysis_status": "post_hoc_localization_not_preregistered_gate",
        "registered_reading": gate["measured_reading"],
        "registered_interpretation": gate["interpretation"],
        "automatic_successor_authorized": bool(gate["automatic_successor_authorized"]),
        "routes": analyzed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate_json", required=True)
    parser.add_argument("--additive_trace", required=True)
    parser.add_argument("--film_trace", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--window", type=int, default=100)
    args = parser.parse_args()

    payload = analyze(
        read_json(args.gate_json),
        {
            "additive": read_jsonl(args.additive_trace),
            "film": read_jsonl(args.film_trace),
        },
        window=args.window,
    )
    payload["sources"] = {
        "gate_json": args.gate_json,
        "additive_trace": args.additive_trace,
        "film_trace": args.film_trace,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
