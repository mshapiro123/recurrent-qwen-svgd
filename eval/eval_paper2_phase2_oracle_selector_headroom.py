"""Compute perfect-selector accepted-length headroom from banked matched-alpha rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


EXPECTED_ARMS = {
    "alpha_0p0_seed_0": 500,
    "alpha_0p0_seed_1": 600,
    "alpha_0p5_seed_0": 193,
    "alpha_0p5_seed_1": 184,
    "alpha_1p0_seed_0": 146,
    "alpha_1p0_seed_1": 146,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selector_headroom(rows: dict[str, torch.Tensor]) -> dict[str, Any]:
    accepted = rows["accepted_length"].float().reshape(-1)
    baseline = rows["base_accepted_length"].float().reshape(-1)
    delta = accepted - baseline
    recorded_delta = rows.get("acceptance_delta")
    if recorded_delta is not None and not torch.equal(delta, recorded_delta.float().reshape(-1)):
        raise RuntimeError("banked accepted lengths disagree with the recorded row deltas")
    if not bool(torch.isfinite(delta).all()):
        raise RuntimeError("banked accepted-length deltas contain non-finite values")
    selected = delta > 0
    base_correct = rows["base_correct_by_horizon"].bool()
    bridge_correct = rows["bridge_correct_by_horizon"].bool()
    if base_correct.shape != bridge_correct.shape or base_correct.shape[0] != delta.numel():
        raise RuntimeError("banked quality horizons do not align with accepted-length rows")
    quality_loss = (base_correct & ~bridge_correct).any(dim=1)
    safe_selected = selected & ~quality_loss

    acceptance_oracle_delta = torch.where(selected, delta, torch.zeros_like(delta))
    safe_oracle_delta = torch.where(safe_selected, delta, torch.zeros_like(delta))
    always_on_delta = float(delta.mean())
    oracle_delta = float(acceptance_oracle_delta.mean())
    safe_delta = float(safe_oracle_delta.mean())

    oracle_correct = torch.where(selected.unsqueeze(1), bridge_correct, base_correct)
    safe_oracle_correct = torch.where(safe_selected.unsqueeze(1), bridge_correct, base_correct)
    baseline_correct_count = int(base_correct.sum())
    oracle_retained = int((oracle_correct & base_correct).sum())
    safe_oracle_retained = int((safe_oracle_correct & base_correct).sum())
    positive_mass = float(delta.clamp_min(0).sum())
    negative_mass = float((-delta.clamp_max(0)).sum())

    return {
        "rows": int(delta.numel()),
        "always_on_acceptance_delta": always_on_delta,
        "oracle_selected_fraction": float(selected.float().mean()),
        "oracle_selected_rows": int(selected.sum()),
        "oracle_acceptance_delta": oracle_delta,
        "oracle_gain_over_always_on": oracle_delta - always_on_delta,
        "quality_safe_selected_fraction": float(safe_selected.float().mean()),
        "quality_safe_selected_rows": int(safe_selected.sum()),
        "quality_safe_oracle_acceptance_delta": safe_delta,
        "quality_safe_oracle_gain_over_always_on": safe_delta - always_on_delta,
        "positive_accepted_length_mass": positive_mass,
        "negative_accepted_length_mass": negative_mass,
        "positive_to_negative_mass_ratio": (
            positive_mass / negative_mass if negative_mass else None
        ),
        "quality_loss_rows_among_acceptance_oracle": int((selected & quality_loss).sum()),
        "baseline_correct_horizon_decisions": baseline_correct_count,
        "acceptance_oracle_retention": oracle_retained / max(1, baseline_correct_count),
        "quality_safe_oracle_retention": safe_oracle_retained / max(1, baseline_correct_count),
    }


def run(private_dir: Path, output_dir: Path) -> dict[str, Any]:
    arms = []
    for arm, step in EXPECTED_ARMS.items():
        path = private_dir / f"{arm}_exact_step_{step:04d}_rows.pt"
        if not path.is_file():
            raise FileNotFoundError(f"missing banked exact-row tensor: {path}")
        rows = torch.load(path, map_location="cpu", weights_only=False)
        result = {
            "arm": arm,
            "alpha": float(arm.split("_seed_")[0].removeprefix("alpha_").replace("p", ".")),
            "seed": int(arm.rsplit("_", 1)[1]),
            "terminal_step": step,
            "source": {"path": str(path), "sha256": sha256_file(path)},
            **selector_headroom(rows),
        }
        arms.append(result)

    by_alpha = {}
    for alpha in (0.0, 0.5, 1.0):
        selected = [arm for arm in arms if arm["alpha"] == alpha]
        by_alpha[str(alpha)] = {
            "seeds": [arm["seed"] for arm in selected],
            "mean_always_on_acceptance_delta": sum(
                arm["always_on_acceptance_delta"] for arm in selected
            )
            / len(selected),
            "mean_oracle_acceptance_delta": sum(
                arm["oracle_acceptance_delta"] for arm in selected
            )
            / len(selected),
            "mean_quality_safe_oracle_acceptance_delta": sum(
                arm["quality_safe_oracle_acceptance_delta"] for arm in selected
            )
            / len(selected),
            "mean_oracle_selected_fraction": sum(
                arm["oracle_selected_fraction"] for arm in selected
            )
            / len(selected),
        }

    summary = {
        "kind": "paper2_phase2_oracle_selector_headroom",
        "status": "complete_cpu_only_existing_rows",
        "model_inference_runs": 0,
        "optimizer_steps": 0,
        "frozen_evaluation_partitions_touched": [],
        "selection_rule": "use sidecar iff its banked row accepted length exceeds zero-loop",
        "quality_safe_rule": (
            "use sidecar iff accepted length improves and no baseline-correct horizon becomes wrong"
        ),
        "arms": arms,
        "by_alpha": by_alpha,
        "scope": (
            "DEV-only cached teacher-forced accepted length; perfect hindsight selector; ceiling, "
            "not a deployable router result"
        ),
        "do_not_claim": [
            "oracle headroom is achievable by available inference-time features",
            "cached teacher-forced accepted length is serving throughput",
            "row-level hindsight selection is confirmation evidence",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args.private_dir, args.output_dir)
    print(json.dumps({"status": result["status"], "by_alpha": result["by_alpha"]}, indent=2))


if __name__ == "__main__":
    main()
