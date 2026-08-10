"""Migrate both frozen E1 full-system endpoints into Phase 3 without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from models.paper2_dc2_student import Phase2StudentModules, Phase3StudentModules
from training.paper2_phase3_migration import (
    migrate_phase2_trainable_state,
    sha256_file,
    tensor_state_digest,
    trainable_state,
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _new_embedding(seed: int) -> nn.Embedding:
    generator = torch.Generator().manual_seed(20260810 + seed)
    embedding_weight = torch.randn(257, 896, generator=generator)
    return nn.Embedding.from_pretrained(embedding_weight, freeze=True)


def _reconstruct_phase2_state(
    *,
    a1_payload: Mapping[str, Any],
    endpoint_state: Mapping[str, torch.Tensor],
    seed: int,
    rms_cap: float,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    phase2 = Phase2StudentModules(
        tied_embedding=_new_embedding(seed), hidden_size=896, rms_cap=rms_cap
    ).float()
    parameters = dict(phase2.named_parameters())
    flow_state = a1_payload.get("flow_state")
    if not isinstance(flow_state, Mapping):
        raise RuntimeError("A1 source checkpoint lacks flow_state")
    with torch.no_grad():
        for name, value in flow_state.items():
            if name not in parameters or not name.startswith("flow."):
                raise RuntimeError(f"unexpected A1 flow key during migration: {name}")
            parameters[name].copy_(value.to(parameters[name]))
        for name, value in endpoint_state.items():
            if name not in parameters:
                raise RuntimeError(f"unexpected E1 endpoint key during migration: {name}")
            parameters[name].copy_(value.to(parameters[name]))
    return trainable_state(phase2)


def _module_pair(
    source: Mapping[str, torch.Tensor], *, seed: int, rms_cap: float
) -> tuple[Phase2StudentModules, Phase3StudentModules]:
    torch.manual_seed(seed)
    phase2_embedding = _new_embedding(seed)
    phase3_embedding = _new_embedding(seed)
    phase2 = Phase2StudentModules(
        tied_embedding=phase2_embedding, hidden_size=896, rms_cap=rms_cap
    ).float().eval()
    phase3 = Phase3StudentModules(
        tied_embedding=phase3_embedding, hidden_size=896, rms_cap=rms_cap
    ).float().eval()

    phase2_parameters = dict(phase2.named_parameters())
    expected = {
        name for name, parameter in phase2_parameters.items() if parameter.requires_grad
    }
    if set(source) != expected:
        raise RuntimeError(
            "E1 Phase 2 state mismatch "
            f"missing={sorted(expected - set(source))} extra={sorted(set(source) - expected)}"
        )
    with torch.no_grad():
        for name, value in source.items():
            phase2_parameters[name].copy_(value.to(phase2_parameters[name]))
    migrate_phase2_trainable_state(phase3, source)
    return phase2, phase3


@torch.no_grad()
def checkpoint_equivalence(
    phase2: Phase2StudentModules,
    phase3: Phase3StudentModules,
    *,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(20260820 + seed)
    hidden = torch.randn(2, 12, 896, generator=generator)
    previous_logits = torch.randn(2, 4, 257, generator=generator)
    rows = []
    for steps in range(5):
        old = phase2(hidden=hidden, previous_logits=previous_logits, steps=steps)
        new = phase3(hidden=hidden, previous_logits=previous_logits, steps=steps)
        rows.append(
            {
                "steps": steps,
                "hidden_max_abs_difference": float((old.hidden - new.hidden).abs().max()),
                "logits_max_abs_difference": float((old.logits - new.logits).abs().max()),
                "scratch_max_abs_difference": float((old.scratch - new.scratch).abs().max()),
                "hidden_bit_exact": torch.equal(old.hidden, new.hidden),
                "logits_bit_exact": torch.equal(old.logits, new.logits),
                "scratch_bit_exact": torch.equal(old.scratch, new.scratch),
            }
        )
    return {
        "kind": "paper2_phase3_checkpoint_integrated_equivalence_v1",
        "seed": seed,
        "rows": rows,
        "all_outputs_bit_exact": all(
            row["hidden_bit_exact"]
            and row["logits_bit_exact"]
            and row["scratch_bit_exact"]
            for row in rows
        ),
        "zero_loop_bit_exact": all(
            rows[0][field]
            for field in ("hidden_bit_exact", "logits_bit_exact", "scratch_bit_exact")
        ),
    }


def migrate_registration(
    registration: Mapping[str, Any],
    *,
    migration_sources: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    rms_cap = float(migration_sources["state_rms_cap"])
    checkpoint_specs = registration["checkpoints"]
    selected = {
        key: checkpoint_specs[key]
        for key in ("seed_0_full_a2", "seed_1_full_a2")
    }
    receipts = {}
    for key, spec in selected.items():
        lineage = migration_sources["seeds"][str(key.split("_")[1])]
        a1_spec = lineage["a1"]
        a1_path = Path(a1_spec["path"])
        if sha256_file(a1_path) != a1_spec["sha256"]:
            raise RuntimeError(f"A1 source checkpoint SHA mismatch: {key}")
        a1_payload = torch.load(a1_path, map_location="cpu", weights_only=False)
        expected_seed = int(key.split("_")[1])
        if (
            int(a1_payload.get("seed", -1)) != expected_seed
            or int(a1_payload.get("step", -1)) != int(a1_spec["expected_step"])
        ):
            raise RuntimeError(f"A1 source checkpoint metadata mismatch: {key}")
        source_path = Path(spec["path"])
        expected_sha = str(spec["sha256"])
        if sha256_file(source_path) != expected_sha:
            raise RuntimeError(f"E1 source checkpoint SHA mismatch: {key}")
        source_payload = torch.load(source_path, map_location="cpu", weights_only=False)
        seed = int(spec.get("seed", source_payload.get("seed", -1)))
        if seed not in (0, 1) or int(source_payload.get("step", -1)) != int(spec["expected_step"]):
            raise RuntimeError(f"E1 source checkpoint metadata mismatch: {key}")
        endpoint_state = source_payload.get("trainable_state")
        if not isinstance(endpoint_state, Mapping):
            raise RuntimeError(f"E1 source lacks a trainable state: {key}")
        expected_state_digest = spec.get("semantic_trainable_state_digest")
        observed_state_digest = tensor_state_digest(endpoint_state)
        if expected_state_digest and observed_state_digest != expected_state_digest:
            raise RuntimeError(f"E1 source semantic state digest mismatch: {key}")

        source_state = _reconstruct_phase2_state(
            a1_payload=a1_payload,
            endpoint_state=endpoint_state,
            seed=seed,
            rms_cap=rms_cap,
        )

        phase2, phase3 = _module_pair(source_state, seed=seed, rms_cap=rms_cap)
        equivalence = checkpoint_equivalence(phase2, phase3, seed=seed)
        if not equivalence["all_outputs_bit_exact"]:
            raise RuntimeError(f"E1 checkpoint migration is not bit-exact: {key}")

        destination = output_dir / f"{key}_phase3_migrated.pt"
        migration = migrate_phase2_trainable_state(phase3, source_state)
        migration.update(
            {
                "a1_checkpoint_sha256": a1_spec["sha256"],
                "e1_endpoint_checkpoint_sha256": expected_sha,
                "combined_phase2_state_sha256": tensor_state_digest(source_state),
                "state_rms_cap": rms_cap,
            }
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(
            {
                "kind": "paper2_phase3_migrated_checkpoint_v2",
                "source_seed": seed,
                "source_step": source_payload.get("step"),
                "source_lineage": {
                    "a1_checkpoint_sha256": a1_spec["sha256"],
                    "e1_endpoint_checkpoint_sha256": expected_sha,
                },
                "trainable_state": trainable_state(phase3),
                "optimizer_state": None,
                "migration_receipt": migration,
            },
            temporary,
        )
        temporary.replace(destination)
        migration["destination_checkpoint_sha256"] = sha256_file(destination)
        receipts[key] = {
            "source": dict(spec),
            "a1_source": dict(a1_spec),
            "observed_semantic_trainable_state_digest": observed_state_digest,
            "migration": migration,
            "equivalence": equivalence,
            "destination": str(destination),
        }

    source_hashes = {value["source"]["sha256"] for value in receipts.values()}
    destination_hashes = {
        value["migration"]["destination_checkpoint_sha256"] for value in receipts.values()
    }
    if len(source_hashes) != 2 or len(destination_hashes) != 2:
        raise RuntimeError("Phase 3 migration did not preserve distinct seed lineages")
    return {
        "kind": "paper2_phase3_e1_checkpoint_migration_summary_v1",
        "status": "complete_both_seeds_no_training",
        "receipts": receipts,
        "assertions": {
            "both_full_system_seeds_migrated": set(receipts) == {
                "seed_0_full_a2",
                "seed_1_full_a2",
            },
            "all_source_hashes_asserted": True,
            "all_migrations_bit_exact_steps_0_through_4": all(
                value["equivalence"]["all_outputs_bit_exact"]
                for value in receipts.values()
            ),
            "new_optimizer_state_absent": all(
                torch.load(value["destination"], map_location="cpu", weights_only=False)[
                    "optimizer_state"
                ]
                is None
                for value in receipts.values()
            ),
            "p33_training_unauthorized": True,
        },
        "training_started": False,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--output_summary", type=Path, required=True)
    parser.add_argument("--migration_sources", type=Path, required=True)
    args = parser.parse_args()
    registration = json.loads(args.registration.read_text(encoding="utf-8"))
    migration_sources = json.loads(args.migration_sources.read_text(encoding="utf-8"))
    summary = migrate_registration(
        registration,
        migration_sources=migration_sources,
        output_dir=args.output_dir,
    )
    write_json(args.output_summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
