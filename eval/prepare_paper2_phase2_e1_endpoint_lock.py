"""Prepare immutable endpoint identities for the Phase-2 E1 lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from training.paper2_phase2_e1_confirmation import sha256_file
from training.run_paper2_phase2_a2 import _tensor_digest


KIND = "paper2_phase2_e1_endpoint_lock_preparation_v1"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare_endpoint_lock(
    *, registration: Path, checkpoints: dict[str, Path], output: Path
) -> dict[str, Any]:
    draft = json.loads(registration.read_text(encoding="utf-8"))
    expected = draft["checkpoints"]
    if set(checkpoints) != set(expected):
        raise RuntimeError("E1 endpoint-lock checkpoint set differs from draft")
    endpoints: dict[str, Any] = {}
    for name, path in sorted(checkpoints.items()):
        expected_row = expected[name]
        observed_sha = sha256_file(path)
        if observed_sha != expected_row["sha256"]:
            raise RuntimeError(f"E1 endpoint file hash mismatch: {name}")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        seed = int(name.split("_")[1])
        arm = name.removeprefix(f"seed_{seed}_")
        if (
            saved.get("kind") != "paper2_phase2_option_b_arm_v1"
            or saved.get("name") != name
            or int(saved.get("seed", -1)) != seed
            or saved.get("arm") != arm
            or int(saved.get("step", -1)) != 20_000
            or saved.get("abort_reason") is not None
        ):
            raise RuntimeError(f"E1 endpoint identity mismatch: {name}")
        trainable_state = saved.get("trainable_state")
        if not isinstance(trainable_state, dict) or not trainable_state:
            raise RuntimeError(f"E1 endpoint lacks trainable state: {name}")
        endpoints[name] = {
            "path": str(path),
            "sha256": observed_sha,
            "semantic_trainable_state_digest": _tensor_digest(trainable_state),
            "semantic_digest_algorithm": "sorted_name_dtype_shape_tensor_bytes_sha256",
            "kind": saved["kind"],
            "name": saved["name"],
            "seed": seed,
            "arm": arm,
            "step": 20_000,
            "abort_reason": None,
            "frozen_parameter_hash_before": saved.get("frozen_parameter_hash_before"),
            "frozen_parameter_hash_after": saved.get("frozen_parameter_hash_after"),
        }
        del saved

    result = {
        "kind": KIND,
        "version": "paper2_phase2_e1_endpoint_lock_preparation_v1_20260808",
        "status": "complete_integrity_only",
        "registration_draft": str(registration),
        "registration_draft_sha256": sha256_file(registration),
        "endpoints": endpoints,
        "endpoint_checkpoints_loaded_as_containers_only": True,
        "model_instantiated": False,
        "eval_d_touched": False,
        "outcome_scores_computed": False,
        "model_quality_scores_computed": False,
        "read_once_scoring_spent": False,
        "training_started": False,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "ready_for_lock_transcription": True,
    }
    write_json(output, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    for seed in (0, 1):
        for arm in ("full_a2", "draft_only_control"):
            parser.add_argument(f"--seed_{seed}_{arm}", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoints = {
        f"seed_{seed}_{arm}": getattr(args, f"seed_{seed}_{arm}")
        for seed in (0, 1)
        for arm in ("full_a2", "draft_only_control")
    }
    prepare_endpoint_lock(
        registration=args.registration,
        checkpoints=checkpoints,
        output=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
