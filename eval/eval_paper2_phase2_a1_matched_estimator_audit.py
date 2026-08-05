"""Audit A1 gradient shares with the population that calibrated the static weights."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from models.paper2_dc2_student import Phase2StudentModules
from training.paper2_phase2_matched_alpha import document_partition
from training.paper2_phase2_staged_repilot import realized_gradient_shares
from training.run_paper2_phase2_matched_alpha import (
    _batch,
    _decoder_for_alpha,
    _local_source,
    _tensor_digest,
    build_pilot_cache,
    sha256_file,
    write_json,
)
from training.run_paper2_phase2_staged_a1 import (
    CALIBRATION_KIND,
    LOSS_MASK_CONTRACT,
    _a1_losses,
    _frozen_hash,
    _gradient_norms,
    _loss_gradients,
    _seed_everything,
    _set_a1_trainable,
)


AUDIT_KIND = "paper2_phase2_a1_matched_estimator_audit_v1"
STRATEGY_DRIVE_ID = "1C-4h5v1OksmYVY5HR9IJnJVNv55R1JqA"
DEV_BATCH_SEED = 20260805
BOOTSTRAP_SEED = 2026080501
BOOTSTRAP_DRAWS = 10_000
EXPECTED_CHECKPOINT_SHA256 = {
    0: "9815592e5358fbde535bec27d102717f4f9fe4a0beb9f649f0d0879f88db2c58",
    1: "f3538465223c2f09f286bbb276631b3ce9e60a7c3ecd43bf677d4d4c4dfb6e4e",
}


def calibration_measurement_batches(
    train_indices: torch.Tensor,
    *,
    seed: int,
    batch_size: int,
    sampled_batches: int,
    first_batch: int,
    last_batch: int,
) -> list[torch.Tensor]:
    """Reproduce the original calibration batches, including its one-based slice."""
    generator = torch.Generator().manual_seed(seed + 34001)
    batches = [
        train_indices.index_select(
            0,
            torch.randint(train_indices.numel(), (batch_size,), generator=generator),
        )
        for _ in range(sampled_batches)
    ]
    return batches[first_batch - 1 : last_batch]


def fixed_dev_batches(
    eval_indices: torch.Tensor, *, batch_size: int, count: int
) -> list[torch.Tensor]:
    """Use one common DEV sample for both seeds; DEV never owns the hard verdict."""
    generator = torch.Generator().manual_seed(DEV_BATCH_SEED)
    return [
        eval_indices.index_select(
            0,
            torch.randint(eval_indices.numel(), (batch_size,), generator=generator),
        )
        for _ in range(count)
    ]


def share_contract(shares: dict[str, float]) -> dict[str, bool]:
    flow = float(shares["flow"]) >= 0.50
    probe = float(shares["functional_probe_kl"]) <= 0.25
    return {"flow_at_least_0p50": flow, "probe_at_most_0p25": probe, "joint": flow and probe}


def summarize_norm_rows(
    norm_rows: list[dict[str, float]],
    *,
    weights: dict[str, float],
    bootstrap_seed: int,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    if not norm_rows:
        raise ValueError("gradient audit requires at least one batch")
    names = list(norm_rows[0])
    if any(set(row) != set(names) for row in norm_rows):
        raise ValueError("gradient norm rows disagree on loss names")
    matrix = np.asarray([[row[name] for name in names] for row in norm_rows], dtype=np.float64)
    mean_norms = dict(zip(names, matrix.mean(axis=0).tolist()))
    aggregate_shares = realized_gradient_shares(mean_norms, weights)
    batch_shares = [realized_gradient_shares(row, weights) for row in norm_rows]
    batch_contracts = [share_contract(row) for row in batch_shares]

    generator = np.random.default_rng(bootstrap_seed)
    bootstrap_indices = generator.integers(0, len(norm_rows), size=(bootstrap_draws, len(norm_rows)))
    bootstrap_means = matrix[bootstrap_indices].mean(axis=1)
    weight_vector = np.asarray([weights[name] for name in names], dtype=np.float64)
    weighted = bootstrap_means * weight_vector
    bootstrap_shares = weighted / weighted.sum(axis=1, keepdims=True)
    intervals = {
        name: {
            "lower_2p5": float(np.quantile(bootstrap_shares[:, index], 0.025)),
            "upper_97p5": float(np.quantile(bootstrap_shares[:, index], 0.975)),
        }
        for index, name in enumerate(names)
    }
    return {
        "batches": len(norm_rows),
        "batch_size": None,
        "mean_gradient_norms": mean_norms,
        "aggregate_shares": aggregate_shares,
        "aggregate_contract": share_contract(aggregate_shares),
        "bootstrap": {
            "method": "nonparametric_batch_resampling",
            "draws": bootstrap_draws,
            "seed": bootstrap_seed,
            "share_intervals_95": intervals,
        },
        "batch_contract_fraction": {
            name: sum(int(row[name]) for row in batch_contracts) / len(batch_contracts)
            for name in batch_contracts[0]
        },
        "batch_shares": batch_shares,
    }


def measure_population(
    *,
    module: Phase2StudentModules,
    batches: list[torch.Tensor],
    cache: dict[str, Any],
    embedding: nn.Embedding,
    teacher_embedding: nn.Embedding,
    decoder: torch.Tensor,
    decoder_bias: torch.Tensor,
    huber_delta: float,
    weights: dict[str, float],
    device: str,
    bootstrap_seed: int,
    label: str,
) -> dict[str, Any]:
    module.train()
    parameter_hash_before = _tensor_digest(dict(module.named_parameters()))
    frozen_hash_before = _frozen_hash(module)
    parameters = [value for value in module.flow.parameters() if value.requires_grad]
    norm_rows: list[dict[str, float]] = []
    for batch_number, indices in enumerate(batches, start=1):
        batch = _batch(cache, indices, alpha=0.5, device=device)
        losses, _metrics = _a1_losses(
            module=module,
            batch=batch,
            embedding=embedding,
            teacher_embedding=teacher_embedding,
            decoder=decoder,
            decoder_bias=decoder_bias,
            huber_delta=huber_delta,
        )
        norm_rows.append(_gradient_norms(_loss_gradients(losses, parameters)))
        if batch_number == 1 or batch_number % 10 == 0 or batch_number == len(batches):
            print(f"a1_matched_audit label={label} batch={batch_number}/{len(batches)}", flush=True)
        del batch, losses
    result = summarize_norm_rows(
        norm_rows,
        weights=weights,
        bootstrap_seed=bootstrap_seed,
    )
    result["batch_size"] = int(batches[0].numel())
    result["parameter_hash_before"] = parameter_hash_before
    result["parameter_hash_after"] = _tensor_digest(dict(module.named_parameters()))
    result["frozen_hash_before"] = frozen_hash_before
    result["frozen_hash_after"] = _frozen_hash(module)
    result["optimizer_updates"] = 0
    result["model_mutated"] = (
        result["parameter_hash_before"] != result["parameter_hash_after"]
        or result["frozen_hash_before"] != result["frozen_hash_after"]
    )
    if result["model_mutated"]:
        raise RuntimeError(f"read-only audit mutated model state for {label}")
    return result


def _load_flow_checkpoint(
    module: Phase2StudentModules, checkpoint_path: Path, *, seed: int, initial_hash: str
) -> dict[str, Any]:
    observed_sha = sha256_file(checkpoint_path)
    if observed_sha != EXPECTED_CHECKPOINT_SHA256[seed]:
        raise RuntimeError(f"seed {seed} checkpoint SHA mismatch: {observed_sha}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("kind") != "paper2_phase2_staged_a1_resume_v1":
        raise RuntimeError("unexpected A1 resume checkpoint kind")
    if int(payload["seed"]) != seed or int(payload["step"]) != 200:
        raise RuntimeError("audit requires the exact seed-matched step-200 checkpoint")
    if payload["initial_trainable_hash"] != initial_hash:
        raise RuntimeError("saved A1 initialization differs from reconstructed initialization")
    current = dict(module.named_parameters())
    with torch.no_grad():
        for name, value in payload["flow_state"].items():
            current[name].copy_(value.to(device=current[name].device, dtype=current[name].dtype))
    return {
        "path": str(checkpoint_path),
        "sha256": observed_sha,
        "kind": payload["kind"],
        "step": int(payload["step"]),
        "seed": int(payload["seed"]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    registration = json.loads(args.protocol.read_text(encoding="utf-8"))
    cache = build_pilot_cache(
        stage0a_summary_path=args.stage0a_summary,
        stage0a_private=args.stage0a_private,
        canonicalizer_path=args.canonicalizer,
        output_path=args.cache,
    )
    eval_mask = document_partition(cache["documents"], evaluation_fraction=0.2, seed=20260804)
    eval_indices = torch.where(eval_mask)[0]
    train_indices = torch.where(~eval_mask)[0]

    student_summary = json.loads(
        (args.stage0a_private / "model_cache/student_0p5b/summary.json").read_text(encoding="utf-8")
    )
    teacher_summary = json.loads(
        (args.stage0a_private / "model_cache/teacher_14b/summary.json").read_text(encoding="utf-8")
    )
    student_head = _local_source(student_summary["lm_head"]["path"], args.stage0a_private)
    teacher_head = _local_source(teacher_summary["lm_head"]["path"], args.stage0a_private)
    embedding_weight = torch.load(student_head, map_location="cpu", weights_only=False)[
        "weight_bfloat16"
    ]
    teacher_weight = torch.load(teacher_head, map_location="cpu", weights_only=False)[
        "weight_bfloat16"
    ]
    teacher_embedding = nn.Embedding.from_pretrained(teacher_weight.float(), freeze=True).to(
        args.device
    )

    measurement_count = (
        int(registration["calibration"]["measurement_last_batch"])
        - int(registration["calibration"]["measurement_first_batch"])
        + 1
    )
    dev_batches = fixed_dev_batches(
        eval_indices,
        batch_size=int(registration["batch_size"]),
        count=measurement_count,
    )
    arm_results = []
    for seed in (0, 1):
        _seed_everything(seed)
        embedding = nn.Embedding.from_pretrained(embedding_weight.float(), freeze=True).to(
            args.device
        )
        module = Phase2StudentModules(
            tied_embedding=embedding,
            hidden_size=896,
            rms_cap=float(registration["constants"]["state_rms_cap"]),
        ).to(device=args.device, dtype=torch.float32)
        _set_a1_trainable(module)
        initial_trainable_hash = _tensor_digest(
            {name: value for name, value in module.named_parameters() if value.requires_grad}
        )
        calibration_path = args.prior_private / f"alpha_0p5_seed_{seed}/a1_calibration.json"
        checkpoint_path = args.prior_private / f"alpha_0p5_seed_{seed}/a1_resume.pt"
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibration.get("kind") != CALIBRATION_KIND or int(calibration["seed"]) != seed:
            raise RuntimeError("calibration receipt does not match audit seed")
        if calibration.get("loss_mask_contract") != LOSS_MASK_CONTRACT:
            raise RuntimeError("calibration loss-mask contract mismatch")
        if calibration["parameter_hash_before"] != initial_trainable_hash:
            raise RuntimeError("calibration initialization cannot be reconstructed")
        weights = {name: float(value) for name, value in calibration["static_loss_weights"].items()}
        huber_delta = float(calibration["huber_delta"])
        decoder, decoder_bias = _decoder_for_alpha(cache, alpha=0.5, device=args.device)
        train_batches = calibration_measurement_batches(
            train_indices,
            seed=seed,
            batch_size=int(registration["batch_size"]),
            sampled_batches=int(registration["calibration"]["batches"]),
            first_batch=int(registration["calibration"]["measurement_first_batch"]),
            last_batch=int(registration["calibration"]["measurement_last_batch"]),
        )

        checkpoints: dict[str, Any] = {}
        for checkpoint_index, checkpoint_name in enumerate(("initialization", "step_200")):
            checkpoint_receipt = None
            if checkpoint_name == "step_200":
                checkpoint_receipt = _load_flow_checkpoint(
                    module, checkpoint_path, seed=seed, initial_hash=initial_trainable_hash
                )
            training = measure_population(
                module=module,
                batches=train_batches,
                cache=cache,
                embedding=embedding,
                teacher_embedding=teacher_embedding,
                decoder=decoder,
                decoder_bias=decoder_bias,
                huber_delta=huber_delta,
                weights=weights,
                device=args.device,
                bootstrap_seed=BOOTSTRAP_SEED + seed * 100 + checkpoint_index * 10,
                label=f"seed_{seed}_{checkpoint_name}_training",
            )
            dev = measure_population(
                module=module,
                batches=dev_batches,
                cache=cache,
                embedding=embedding,
                teacher_embedding=teacher_embedding,
                decoder=decoder,
                decoder_bias=decoder_bias,
                huber_delta=huber_delta,
                weights=weights,
                device=args.device,
                bootstrap_seed=BOOTSTRAP_SEED + seed * 100 + checkpoint_index * 10 + 1,
                label=f"seed_{seed}_{checkpoint_name}_dev",
            )
            checkpoints[checkpoint_name] = {
                "checkpoint": checkpoint_receipt,
                "training_matched_primary": training,
                "dev_population_shift_descriptive": dev,
            }

        calibration_shares = {
            name: float(value) for name, value in calibration["calibration_realized_shares"].items()
        }
        reconstructed = checkpoints["initialization"]["training_matched_primary"][
            "aggregate_shares"
        ]
        reconstruction_max_abs_difference = max(
            abs(reconstructed[name] - calibration_shares[name]) for name in calibration_shares
        )
        if reconstruction_max_abs_difference > 1e-5:
            raise RuntimeError(
                "matched estimator failed to reconstruct calibration shares: "
                f"{reconstruction_max_abs_difference}"
            )
        arm_results.append(
            {
                "seed": seed,
                "calibration_path": str(calibration_path),
                "calibration_sha256": sha256_file(calibration_path),
                "calibration_realized_shares": calibration_shares,
                "calibration_reconstruction_max_abs_difference": reconstruction_max_abs_difference,
                "static_loss_weights": weights,
                "huber_delta": huber_delta,
                "checkpoints": checkpoints,
            }
        )

    resume_authorized = all(
        arm["checkpoints"]["step_200"]["training_matched_primary"]["aggregate_contract"][
            "joint"
        ]
        for arm in arm_results
    )
    summary = {
        "kind": AUDIT_KIND,
        "status": "complete",
        "strategy_drive_id": STRATEGY_DRIVE_ID,
        "mode": "read_only_gradient_estimator_audit",
        "optimizer_updates": 0,
        "model_training": False,
        "alpha": 0.5,
        "seeds": [0, 1],
        "primary_population": "exact_seed_specific_calibration_training_batches_50_through_100",
        "dev_role": "descriptive_population_shift_only",
        "dev_batch_seed": DEV_BATCH_SEED,
        "contract": {"flow_minimum": 0.50, "functional_probe_kl_maximum": 0.25},
        "arms": arm_results,
        "decision": "resume_saved_step_200" if resume_authorized else "fresh_rerun_required",
        "resume_authorized": resume_authorized,
        "prior_run_classification": "protocol_bug_not_registered_attempt",
        "a2_launched": False,
        "frozen_confirmatory_partitions_touched": [],
        "source_hashes": {
            "protocol_sha256": sha256_file(args.protocol),
            "canonicalizer_sha256": sha256_file(args.canonicalizer),
            "student_lm_head_sha256": sha256_file(student_head),
            "teacher_lm_head_sha256": sha256_file(teacher_head),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage0a_summary", type=Path, required=True)
    parser.add_argument("--stage0a_private", type=Path, required=True)
    parser.add_argument("--canonicalizer", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--prior_private", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result["decision"],
                "resume_authorized": result["resume_authorized"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
