"""Read-only P3.4 A_r pricing audit over banked oracle caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


REALIZED_PI_DIR = 0.14901016586409846


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def numerical_column_basis(matrix: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    if matrix.ndim != 2 or not bool(torch.isfinite(matrix).all()):
        raise ValueError("readout matrix must be a finite rank-two tensor")
    u, singular, _vh = torch.linalg.svd(matrix.double(), full_matrices=False)
    tolerance = torch.finfo(torch.float64).eps * max(matrix.shape) * float(singular.max())
    rank = int((singular > tolerance).sum())
    if rank <= 0:
        raise RuntimeError("readout projection has zero numerical rank")
    return u[:, :rank], {
        "rank": rank,
        "shape": list(matrix.shape),
        "rank_tolerance": tolerance,
        "largest_singular_value": float(singular[0]),
        "smallest_retained_singular_value": float(singular[rank - 1]),
    }


def leading_covariance_basis(states: torch.Tensor, *, rank: int) -> tuple[torch.Tensor, dict[str, Any]]:
    if states.ndim != 2 or rank <= 0 or rank > states.shape[1]:
        raise ValueError("state matrix or requested covariance rank is invalid")
    centered = states.double() - states.double().mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    selected_values = eigenvalues.index_select(0, order[:rank]).clamp_min(0)
    total = eigenvalues.clamp_min(0).sum()
    return eigenvectors.index_select(1, order[:rank]), {
        "rank": rank,
        "rows": int(states.shape[0]),
        "dimension": int(states.shape[1]),
        "variance_fraction": float(selected_values.sum() / total.clamp_min(1e-30)),
        "largest_eigenvalue": float(selected_values[0]),
        "smallest_retained_eigenvalue": float(selected_values[-1]),
    }


def projected_energy(directions: torch.Tensor, basis: torch.Tensor) -> dict[str, float]:
    values = directions.double()
    if values.ndim != 2 or basis.ndim != 2 or values.shape[1] != basis.shape[0]:
        raise ValueError("directions and basis dimensions do not align")
    total = values.square().sum(dim=1).clamp_min(1e-30)
    captured = (values @ basis).square().sum(dim=1)
    fractions = captured / total
    return {
        "aggregate_energy_fraction": float(captured.sum() / total.sum()),
        "mean_row_energy_fraction": float(fractions.mean()),
        "median_row_energy_fraction": float(fractions.median()),
        "minimum_row_energy_fraction": float(fractions.min()),
        "maximum_row_energy_fraction": float(fractions.max()),
    }


def _checkpoint_projection(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("trainable_state")
    if not isinstance(state, Mapping):
        raise RuntimeError(f"P3.4 endpoint lacks trainable_state: {path}")
    key = "bridge.output_projection.weight"
    if key not in state:
        raise RuntimeError(f"P3.4 endpoint lacks {key}: {path}")
    seed = payload.get("seed", payload.get("source_seed"))
    return state[key].float(), {
        "path": str(path),
        "sha256": sha256_file(path),
        "seed": None if seed is None else int(seed),
        "optimizer_steps": int(payload.get("step", payload.get("optimizer_steps", -1))),
    }


def _load_oracle(path: Path) -> tuple[list[str], torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") != "paper2_phase3_agreement_oracle_direction_cache_v1":
        raise RuntimeError("P3.4 A_r received the wrong oracle cache kind")
    ids = [str(value) for value in payload["record_ids"]]
    directions = payload["directions"].float()
    if directions.shape != (len(ids), 896) or len(ids) != len(set(ids)):
        raise RuntimeError("P3.4 oracle cache shape or record IDs changed")
    return ids, directions, {"path": str(path), "sha256": sha256_file(path), "rows": len(ids)}


def _load_states(path: Path, record_ids: list[str]) -> tuple[torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") != "paper2_phase3_agreement_forecast_feature_cache_v1":
        raise RuntimeError("P3.4 A_r received the wrong feature cache kind")
    observed_ids = [str(value) for value in payload["record_ids"]]
    if observed_ids != record_ids:
        raise RuntimeError("P3.4 A_r oracle and feature rows are not aligned")
    features = payload["features"].float()
    if features.shape != (len(record_ids), 1920):
        raise RuntimeError("P3.4 forecast feature shape changed")
    return features[:, :896], {
        "path": str(path),
        "sha256": sha256_file(path),
        "seed": int(payload["source_seed"]),
        "loop_index": int(payload["loop_index"]),
        "rows": len(record_ids),
    }


def audit(
    *,
    direction_cache: Path,
    feature_caches: Iterable[Path],
    checkpoints: Iterable[Path],
    output: Path,
) -> dict[str, Any]:
    record_ids, directions, oracle_receipt = _load_oracle(direction_cache)
    states = [_load_states(path, record_ids) for path in feature_caches]
    endpoints = [_checkpoint_projection(path) for path in checkpoints]
    if len(states) != len(endpoints):
        raise RuntimeError("P3.4 A_r requires one feature cache per endpoint")
    rows = []
    for (hidden, hidden_receipt), (weight, checkpoint_receipt) in zip(states, endpoints):
        if hidden_receipt["seed"] != checkpoint_receipt["seed"]:
            raise RuntimeError("P3.4 A_r seed ordering differs across inputs")
        readout_basis, readout_spectrum = numerical_column_basis(weight)
        state_basis, state_spectrum = leading_covariance_basis(hidden, rank=readout_spectrum["rank"])
        readout_energy = projected_energy(directions, readout_basis)
        state_energy = projected_energy(directions, state_basis)
        rows.append(
            {
                "seed": hidden_receipt["seed"],
                "checkpoint": checkpoint_receipt,
                "feature_cache": hidden_receipt,
                "readout_span": {**readout_spectrum, **readout_energy},
                "matched_state_covariance_span": {**state_spectrum, **state_energy},
                "outside_readout_aggregate_energy_fraction": 1.0
                - readout_energy["aggregate_energy_fraction"],
                "state_minus_readout_aggregate_energy_fraction": state_energy[
                    "aggregate_energy_fraction"
                ]
                - readout_energy["aggregate_energy_fraction"],
                "realized_pi_dir": REALIZED_PI_DIR,
                "readout_energy_to_realized_capture_ratio": readout_energy[
                    "aggregate_energy_fraction"
                ]
                / REALIZED_PI_DIR,
            }
        )
    result = {
        "kind": "paper2_phase3_p34_ar_pricing_audit_v1",
        "status": "complete_strategy_fork_unbound",
        "oracle_cache": oracle_receipt,
        "rows": rows,
        "interpretation_contract": {
            "automatic_fork_decision": False,
            "reason": "the charter does not bind a high-versus-low A_r threshold",
            "strategy_confirmation_required": True,
            "reported_separately": [
                "oracle energy in learned readout column span",
                "oracle energy in matched-rank state covariance span",
                "energy outside learned readout span",
                "realized pi_dir",
            ],
        },
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    write_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction_cache", type=Path, required=True)
    parser.add_argument("--feature_cache", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        direction_cache=args.direction_cache,
        feature_caches=args.feature_cache,
        checkpoints=args.checkpoint,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
