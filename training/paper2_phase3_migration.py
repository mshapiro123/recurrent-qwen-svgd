"""One-way Phase 2 to Phase 3 per-position gate checkpoint migration."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


PHASE2_TRAINABLE_PARAMETER_COUNT = 1_184_917
PHASE3_NEW_GATE_PARAMETERS = (
    "bridge.gate_control.weight",
    "bridge.gate_hidden.weight",
    "bridge.gate_scratch.weight",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_state_digest(payload: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(payload):
        value = payload[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def trainable_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def phase3_trainable_parameter_count(module: nn.Module) -> int:
    tied_ids = {
        id(parameter)
        for child in module.modules()
        if isinstance(child, nn.Embedding)
        for parameter in child.parameters(recurse=False)
        if not parameter.requires_grad
    }
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad and id(parameter) not in tied_ids
    )


def migrate_phase2_trainable_state(
    module: nn.Module,
    source: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Load a Phase 2 trainable state while creating only zero-init gate projections."""

    target = {
        name: parameter
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    expected_source = set(target) - set(PHASE3_NEW_GATE_PARAMETERS)
    if set(source) != expected_source:
        missing = sorted(expected_source - set(source))
        extra = sorted(set(source) - expected_source)
        raise RuntimeError(f"Phase 2 migration state mismatch missing={missing} extra={extra}")
    if "bridge.gate_logits" not in source:
        raise RuntimeError("Phase 2 migration requires the trained scalar gate bank")

    with torch.no_grad():
        for name in sorted(expected_source):
            parameter = target[name]
            value = source[name]
            if tuple(value.shape) != tuple(parameter.shape):
                raise RuntimeError(f"Phase 2 migration shape mismatch for {name}")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
        for name in PHASE3_NEW_GATE_PARAMETERS:
            target[name].zero_()

    migrated = trainable_state(module)
    return {
        "kind": "paper2_phase3_scalar_gate_migration_receipt_v1",
        "source_trainable_state_sha256": tensor_state_digest(source),
        "migrated_trainable_state_sha256": tensor_state_digest(migrated),
        "scalar_gate_source": "bridge.gate_logits",
        "scalar_gate_destination": "bridge.gate_logits",
        "new_zero_initialized_parameters": list(PHASE3_NEW_GATE_PARAMETERS),
        "new_parameters_are_zero": all(
            bool(torch.count_nonzero(migrated[name]) == 0)
            for name in PHASE3_NEW_GATE_PARAMETERS
        ),
        "phase2_trainable_parameter_count": PHASE2_TRAINABLE_PARAMETER_COUNT,
        "phase3_trainable_parameter_count": phase3_trainable_parameter_count(module),
    }


def write_migrated_checkpoint(
    *,
    source_checkpoint: Path,
    destination_checkpoint: Path,
    module: nn.Module,
    expected_source_sha256: str,
) -> dict[str, Any]:
    """Write a fresh-state Phase 3 checkpoint without mutating the Phase 2 source."""

    observed_source_sha256 = sha256_file(source_checkpoint)
    if observed_source_sha256 != expected_source_sha256:
        raise RuntimeError("Phase 3 migration source checkpoint SHA mismatch")
    source_payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if "trainable_state" not in source_payload:
        raise RuntimeError("Phase 3 migration source lacks trainable_state")
    receipt = migrate_phase2_trainable_state(module, source_payload["trainable_state"])
    receipt["source_checkpoint_sha256"] = observed_source_sha256

    destination_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_checkpoint.with_suffix(destination_checkpoint.suffix + ".tmp")
    torch.save(
        {
            "kind": "paper2_phase3_migrated_checkpoint_v1",
            "source_checkpoint_sha256": observed_source_sha256,
            "source_kind": source_payload.get("kind"),
            "source_seed": source_payload.get("seed"),
            "source_step": source_payload.get("step"),
            "trainable_state": trainable_state(module),
            "optimizer_state": None,
            "migration_receipt": receipt,
        },
        temporary,
    )
    temporary.replace(destination_checkpoint)
    receipt["destination_checkpoint_sha256"] = sha256_file(destination_checkpoint)
    return receipt
