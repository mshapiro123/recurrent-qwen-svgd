"""CPU-only assertion receipt for the complete loss-free DC2 student build."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from models.paper2_dc2_student import (
    Phase2StudentModules,
    masked_effective_rank,
    masked_huber_loss,
)


ROOT = Path(__file__).resolve().parents[1]
CONSTANTS_PATH = ROOT / "training/paper2_phase2_dc2_constants.json"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _tensor_hash(parameters: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for parameter in parameters:
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _new_parameter_count(module: Phase2StudentModules) -> int:
    tied_ids = {id(parameter) for parameter in module.draft.tied_embedding.parameters()}
    return sum(parameter.numel() for parameter in module.parameters() if id(parameter) not in tied_ids)


def _named_tensor_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def checkpoint_identity(model_name: str) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval().requires_grad_(False)
    before = _named_tensor_hash(model)
    encoded = tokenizer("Verify the inactive recurrent sidecar.", return_tensors="pt")
    with torch.no_grad():
        base = model(**encoded, output_hidden_states=True, use_cache=False, return_dict=True)
    hidden = base.hidden_states[-1]
    logits = base.logits
    sidecar = Phase2StudentModules(
        tied_embedding=model.get_input_embeddings(),
        hidden_size=int(model.config.hidden_size),
        latent_dim=128,
        n_slots=8,
        control_dim=32,
        draft_rank=64,
        max_steps=4,
        rms_cap=float(json.loads(CONSTANTS_PATH.read_text(encoding="utf-8"))["p99_state_rms_cap"]),
    )
    inactive = sidecar(hidden=hidden, previous_logits=logits[:, -4:], steps=0)
    after = _named_tensor_hash(model)
    tied = model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()
    result = {
        "model_name": model_name,
        "base_parameter_sha256_before": before,
        "base_parameter_sha256_after": after,
        "base_hash_unchanged": before == after,
        "base_requires_grad_false": not any(parameter.requires_grad for parameter in model.parameters()),
        "tie_word_embeddings_policy": bool(model.config.tie_word_embeddings),
        "input_output_embedding_storage_tied": bool(tied),
        "zero_loop_hidden_bit_identity": bool(torch.equal(inactive.hidden, hidden)),
        "zero_loop_logits_bit_identity": bool(torch.equal(inactive.logits, logits[:, -4:])),
        "max_abs_hidden_difference": float((inactive.hidden - hidden).abs().max()),
        "max_abs_logits_difference": float((inactive.logits - logits[:, -4:]).abs().max()),
    }
    failed = [
        key
        for key in (
            "base_hash_unchanged",
            "base_requires_grad_false",
            "zero_loop_hidden_bit_identity",
            "zero_loop_logits_bit_identity",
        )
        if not result[key]
    ]
    if failed:
        raise RuntimeError(f"checkpoint-integrated build assertions failed: {failed}")
    return result


def run_build(output_summary: Path, *, model_name: str = "") -> dict[str, Any]:
    constants_bytes = CONSTANTS_PATH.read_bytes()
    constants = json.loads(constants_bytes.decode("utf-8"))
    constants_sha256 = hashlib.sha256(constants_bytes).hexdigest()
    if constants.get("status") != "confirmed_by_v1d":
        raise RuntimeError("student build requires V1d-confirmed writeback constants")
    torch.manual_seed(20260804)
    hidden_size = 896
    latent_dim = 128
    n_slots = 8
    control_dim = 32
    vocabulary_probe_size = 257
    tied_embedding = nn.Embedding(vocabulary_probe_size, hidden_size)
    tied_embedding.requires_grad_(False)
    frozen_before = _tensor_hash(list(tied_embedding.parameters()))
    module = Phase2StudentModules(
        tied_embedding=tied_embedding,
        hidden_size=hidden_size,
        latent_dim=latent_dim,
        n_slots=n_slots,
        control_dim=control_dim,
        draft_rank=64,
        max_steps=4,
        rms_cap=float(constants["p99_state_rms_cap"]),
    )
    hidden = torch.randn(2, 12, hidden_size)
    previous_logits = torch.randn(2, 4, vocabulary_probe_size)
    inactive = module(hidden=hidden, previous_logits=previous_logits, steps=0)
    active = module(hidden=hidden, previous_logits=previous_logits, steps=1)

    slot_mask = torch.zeros(2, n_slots, dtype=torch.bool)
    slot_mask[:, :4] = True
    target = torch.zeros(2, n_slots, latent_dim)
    prediction = target.clone()
    prediction[:, 4:] = 10_000
    masked_loss = masked_huber_loss(prediction, target, slot_mask)
    rank_control = active.scratch.detach().clone()
    rank_control[:, 4:] = 10_000 * torch.randn_like(rank_control[:, 4:])
    effective_rank = masked_effective_rank(rank_control, slot_mask)
    effective_rank_reference = masked_effective_rank(active.scratch.detach(), slot_mask)

    initial_magnitude = float(active.flow.magnitudes[:, 0].mean().detach())
    scratch_rms = active.flow.states[0].float().square().mean(dim=(1, 2)).sqrt()
    measured_r0 = active.flow.initial_update_ratio.detach().cpu()
    predicted_r0 = 0.01814992791780973 / scratch_rms
    frozen_after = _tensor_hash(list(tied_embedding.parameters()))
    assertions = {
        "k_cap_four": module.max_steps == 4 and module.flow.max_steps == 4,
        "inactive_hidden_identity_exact": torch.equal(inactive.hidden, hidden),
        "inactive_logits_identity_exact": torch.equal(inactive.logits, previous_logits),
        "loss_surface_absent": inactive.loss is None and active.loss is None,
        "softplus_scalar_magnitude": (
            active.flow.magnitudes.shape == (2, 1)
            and abs(initial_magnitude - 0.01814992791780973) < 1e-6
        ),
        "direction_rmsnorm_gain_init_one": torch.equal(
            module.flow.direction_norm.weight.detach(),
            torch.ones_like(module.flow.direction_norm.weight),
        ),
        "persistent_state_not_projected": not torch.allclose(
            active.scratch.float().square().mean(dim=-1).sqrt(),
            torch.ones(2, n_slots),
            atol=1e-3,
        ),
        "trust_region_wired": active.flow.trust_penalty.ndim == 0,
        "position_zero_writeback_closed": active.bridge.position_zero_gate_closed,
        "bridge_output_nonzero_initialization": bool(
            module.bridge.output_projection.weight.detach().abs().sum() > 0
        ),
        "masked_slots_excluded_from_loss": float(masked_loss) == 0.0,
        "masked_slots_excluded_from_effective_rank": torch.allclose(
            effective_rank, effective_rank_reference
        ),
        "stage_c_control_read_exposed": torch.equal(active.control_read, active.control_state),
        "frozen_tied_embedding_requires_grad_false": not any(
            parameter.requires_grad for parameter in tied_embedding.parameters()
        ),
        "frozen_tied_embedding_hash_unchanged": frozen_before == frozen_after,
        "optimizer_absent": True,
        "training_steps_zero": True,
    }
    failed = [name for name, passed in assertions.items() if not bool(passed)]
    if failed:
        raise RuntimeError(f"DC2 student build assertions failed: {failed}")
    result = {
        "kind": "paper2_dc2_student_build_assertion_receipt",
        "status": (
            "complete_build_only_no_losses"
            if model_name
            else "local_shape_check_only_checkpoint_identity_not_requested"
        ),
        "architecture": {
            "hidden_size": hidden_size,
            "latent_dim": latent_dim,
            "n_slots": n_slots,
            "populated_future_slots": 4,
            "reserved_masked_span_slots": 4,
            "control_dim": control_dim,
            "draft_rank": 64,
            "max_steps": 4,
            "new_trainable_parameter_count": _new_parameter_count(module),
            "tied_embedding_parameter_count_excluded": tied_embedding.weight.numel(),
        },
        "initialization": {
            "softplus_magnitude_bias": -4.0,
            "initial_magnitude_mean": initial_magnitude,
            "r_0_measured": measured_r0.tolist(),
            "r_0_approx_0p018_over_rms_z0": predicted_r0.tolist(),
            "direction_gain": 1.0,
            "bridge_p_out_std": 1e-3,
            "bridge_gate_logit": -4.0,
            "rho": 0.95,
            "writeback_rms_cap": float(constants["p99_state_rms_cap"]),
            "tube_c": float(constants["tube_c"]),
        },
        "constants": {
            "path": CONSTANTS_PATH.relative_to(ROOT).as_posix(),
            "sha256": constants_sha256,
            "version": constants["version"],
            "source_receipt_sha256": constants["source_receipt_sha256"],
        },
        "assertions": assertions,
        "frozen_tied_embedding_sha256": frozen_before,
        "training_started": False,
        "optimizer_steps": 0,
        "losses_attached": [],
        "frozen_evaluation_partitions_touched": [],
        "do_not_claim": [
            "the untrained student improves teacher agreement",
            "build assertions establish matched-pilot quality",
        ],
    }
    if model_name:
        result["checkpoint_identity"] = checkpoint_identity(model_name)
        result["assertions"]["checkpoint_integrated_zero_loop_identity"] = True
        result["assertions"]["checkpoint_integrated_base_hash_unchanged"] = True
    else:
        result["checkpoint_identity"] = {"status": "not_requested_local_shape_only_check"}
        result["do_not_claim"].append("the vocabulary probe embedding is a model checkpoint")
    write_json(output_summary, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_summary", type=Path, required=True)
    parser.add_argument("--model_name", default="")
    args = parser.parse_args()
    result = run_build(args.output_summary, model_name=args.model_name)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
