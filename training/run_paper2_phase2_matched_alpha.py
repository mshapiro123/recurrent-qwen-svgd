"""Run the six locked DEV-only Phase-2 matched-alpha pilots."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from eval.cache_paper2_phase2_stage0a import _load_flat_shard
from models.paper2_dc2_student import Phase2StudentModules, masked_effective_rank
from training.paper2_phase2_matched_alpha import (
    PROTOCOL_LOCK_COMMIT,
    alpha_transform,
    build_adamw_groups,
    clip_module_groups,
    distribution_overlap,
    document_partition,
    expected_accepted_length,
    masked_sparse_kl,
    normalize_sparse_with_tail,
    quality_noninferior,
    reconstruct_sparse_residual_seed,
    summarize_clip_fractions,
    trust_saturated,
    wilson_lower,
)


BETAS = (0.5,)
PROTOCOL_RELATIVE = Path("training/paper2_phase2_matched_alpha_preregistration.json")
LOSS_MASK_CONTRACT = "masked_sparse_candidates_plus_tail_v3_finite_residual_seed"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_lf_file(path: Path) -> str:
    """Hash text bytes after canonicalizing checkout line endings to LF."""
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _assert_lock(root: Path) -> dict[str, Any]:
    protocol = json.loads((root / PROTOCOL_RELATIVE).read_text(encoding="utf-8"))
    if protocol.get("status") != "locked_before_training":
        raise RuntimeError("matched-alpha protocol is not locked")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", PROTOCOL_LOCK_COMMIT, "HEAD"], cwd=root
    ).returncode:
        raise RuntimeError(f"launcher HEAD does not descend from protocol lock {PROTOCOL_LOCK_COMMIT}")
    return protocol


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _tensor_digest(payload: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(payload):
        value = payload[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _load_trainable_state(module: nn.Module, payload: dict[str, torch.Tensor]) -> None:
    trainable = {name: value for name, value in module.named_parameters() if value.requires_grad}
    if set(payload) != set(trainable):
        missing = sorted(set(trainable) - set(payload))
        extra = sorted(set(payload) - set(trainable))
        raise RuntimeError(f"resume trainable-state mismatch missing={missing} extra={extra}")
    with torch.no_grad():
        for name, value in trainable.items():
            value.copy_(payload[name].to(device=value.device, dtype=value.dtype))


def _resume_has_no_optimizer_update(saved: dict[str, Any]) -> bool:
    optimizer_state = saved.get("optimizer", {}).get("state", {})
    history = list(saved.get("history", []))
    return (
        int(saved.get("step", -1)) <= 1
        and not optimizer_state
        and not saved.get("trust_history")
        and not saved.get("clip_events")
        and len(history) == 1
        and int(history[0].get("step", -1)) == 0
    )


def _parallel_receipts(summary: dict[str, Any], key: str) -> list[dict[str, Any]]:
    return list(summary["model_caches"][key]["shards"])


def _local_source(path: str, stage0a_private: Path) -> Path:
    normalized = str(path).replace("\\", "/")
    for marker in ("/private/stage0a/", "/private/full/"):
        if marker in normalized:
            return stage0a_private / normalized.split(marker, 1)[1]
    raise RuntimeError(f"unrecognized Phase-2 private cache path: {path}")


def _grow_last_dim(value: torch.Tensor, width: int, fill: int | float | bool) -> torch.Tensor:
    if width <= value.shape[-1]:
        return value
    grown = torch.full(
        (*value.shape[:-1], width),
        fill,
        dtype=value.dtype,
        device=value.device,
    )
    grown[..., : value.shape[-1]] = value
    return grown


def build_pilot_cache(
    *,
    stage0a_summary_path: Path,
    stage0a_private: Path,
    canonicalizer_path: Path,
    output_path: Path,
    expected_samples: int = 200_000,
) -> dict[str, Any]:
    if output_path.is_file():
        cached = torch.load(output_path, map_location="cpu", weights_only=False)
        if (
            cached.get("kind") == "paper2_phase2_matched_alpha_cache_v1"
            and len(cached.get("documents", [])) * 4 == int(expected_samples)
        ):
            print(f"matched_alpha_cache_resume={output_path}", flush=True)
            return cached
    summary = json.loads(stage0a_summary_path.read_text(encoding="utf-8"))
    samples_path = stage0a_private / "sample_manifest.jsonl"
    samples = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines() if line]
    if len(samples) != int(expected_samples):
        raise RuntimeError(
            "matched-alpha cache sample count differs from the explicit contract: "
            f"observed={len(samples)} expected={int(expected_samples)}"
        )
    if sha256_file(samples_path) != summary["manifest"]["sample_manifest_sha256"]:
        raise RuntimeError("Stage 0A manifest hash mismatch")
    canonicalizer = torch.load(canonicalizer_path, map_location="cpu", weights_only=False)
    if canonicalizer.get("arm") != "learned_mixture_rrr" or int(canonicalizer.get("seed", -1)) != 20260814:
        raise RuntimeError("wrong frozen canonicalizer")

    anchors = max(int(row["anchor_index"]) for row in samples) + 1
    sample_anchor = torch.tensor([int(row["anchor_index"]) for row in samples], dtype=torch.long)
    sample_horizon = torch.tensor([int(row["horizon"]) for row in samples], dtype=torch.long)
    documents_by_anchor = [""] * anchors
    strata_by_anchor = [""] * anchors
    positions = torch.zeros(anchors, dtype=torch.long)
    for row in samples:
        anchor = int(row["anchor_index"])
        documents_by_anchor[anchor] = str(row["document_id"])
        strata_by_anchor[anchor] = str(row["stratum"])
        if int(row["horizon"]) == 1:
            positions[anchor] = int(row["prediction_position"])

    lattice_receipts = list(summary["lattice"]["shards"])
    student_receipts = _parallel_receipts(summary, "student_0p5b")
    teacher_receipts = _parallel_receipts(summary, "teacher_14b")
    if not (len(lattice_receipts) == len(student_receipts) == len(teacher_receipts)):
        raise RuntimeError("Stage 0A shard ledgers do not align")

    first_lattice = torch.load(
        _local_source(lattice_receipts[0]["path"], stage0a_private),
        map_location="cpu",
        weights_only=False,
    )
    candidates = int(first_lattice["union_ids"].shape[1])
    student_hidden = torch.empty((anchors, 4, 896), dtype=torch.bfloat16)
    teacher_h4_states = torch.empty((anchors, 3, 5120), dtype=torch.bfloat16)
    candidate_ids = torch.full((anchors, 4, candidates), -1, dtype=torch.int32)
    candidate_mask = torch.zeros((anchors, 4, candidates), dtype=torch.bool)
    base_log_probs = torch.full(
        (anchors, 4, candidates), float("-inf"), dtype=torch.bfloat16
    )
    base_tail = torch.empty((anchors, 4), dtype=torch.bfloat16)
    teacher_log_probs = torch.full(
        (anchors, 4, candidates), float("-inf"), dtype=torch.bfloat16
    )
    teacher_tail = torch.empty((anchors, 4), dtype=torch.bfloat16)
    teacher_topk_ids = torch.empty((anchors, 4, 128), dtype=torch.int32)
    teacher_topk_log_probs = torch.empty((anchors, 4, 128), dtype=torch.bfloat16)
    seen = torch.zeros((anchors, 4), dtype=torch.bool)

    for shard_number, (lr, sr, tr) in enumerate(
        zip(lattice_receipts, student_receipts, teacher_receipts), start=1
    ):
        paths = [
            _local_source(lr["path"], stage0a_private),
            _local_source(sr["path"], stage0a_private),
            _local_source(tr["path"], stage0a_private),
        ]
        receipts = [lr, sr, tr]
        for path, receipt in zip(paths, receipts):
            if sha256_file(path) != receipt["sha256"]:
                raise RuntimeError(f"pilot source shard hash mismatch: {path}")
        lattice = torch.load(paths[0], map_location="cpu", weights_only=False)
        student = _load_flat_shard(paths[1])
        teacher = _load_flat_shard(paths[2])
        shard_candidates = int(lattice["union_ids"].shape[1])
        if shard_candidates > candidates:
            print(
                f"matched_alpha_cache_grow old_width={candidates} new_width={shard_candidates} "
                f"shard={shard_number}",
                flush=True,
            )
            candidate_ids = _grow_last_dim(candidate_ids, shard_candidates, -1)
            candidate_mask = _grow_last_dim(candidate_mask, shard_candidates, False)
            base_log_probs = _grow_last_dim(base_log_probs, shard_candidates, float("-inf"))
            teacher_log_probs = _grow_last_dim(
                teacher_log_probs, shard_candidates, float("-inf")
            )
            candidates = shard_candidates
        indices = lattice["sample_indices"].long()
        if not torch.equal(indices, student["sample_indices"].long()) or not torch.equal(
            indices, teacher["sample_indices"].long()
        ):
            raise RuntimeError("pilot source sample alignment failed")
        anchor = sample_anchor.index_select(0, indices)
        horizon = sample_horizon.index_select(0, indices) - 1
        student_hidden[anchor, horizon] = student["final_hidden_bfloat16"]
        candidate_ids[anchor, horizon, :shard_candidates] = lattice["union_ids"].to(torch.int32)
        candidate_mask[anchor, horizon, :shard_candidates] = lattice["union_mask"]
        base_log_probs[anchor, horizon, :shard_candidates] = lattice[
            "model_candidate_log_probs"
        ]["student_0p5b"].to(torch.bfloat16)
        base_tail[anchor, horizon] = lattice["model_tail_log_probs"]["student_0p5b"].to(
            torch.bfloat16
        )
        teacher_log_probs[anchor, horizon, :shard_candidates] = lattice[
            "model_candidate_log_probs"
        ]["teacher_14b"].to(torch.bfloat16)
        teacher_tail[anchor, horizon] = lattice["model_tail_log_probs"]["teacher_14b"].to(
            torch.bfloat16
        )
        teacher_topk_ids[anchor, horizon] = teacher["topk_ids"].to(torch.int32)
        teacher_topk_log_probs[anchor, horizon] = teacher["topk_log_probs"].to(torch.bfloat16)
        fourth = horizon.eq(3)
        teacher_h4_states[anchor[fourth]] = teacher["teacher_states_bfloat16"][fourth]
        seen[anchor, horizon] = True
        if shard_number == 1 or shard_number % 64 == 0 or shard_number == len(lattice_receipts):
            print(
                f"matched_alpha_cache_progress shard={shard_number}/{len(lattice_receipts)} "
                f"anchors_complete={int(seen.all(dim=1).sum())}",
                flush=True,
            )
    if not bool(seen.all()):
        raise RuntimeError("pilot cache is missing anchor horizons")

    # Exact learned mixture banked in the source arbitration receipt. The
    # selected seed artifact intentionally stores only the fitted RRR tensors.
    weights = torch.tensor(
        [0.5840215682983398, 0.3323609232902527, 0.08361753076314926]
    )
    weights = weights / weights.sum()
    normalized = teacher_h4_states.float() * torch.rsqrt(
        teacher_h4_states.float().square().mean(dim=-1, keepdim=True) + 1e-6
    )
    pooled = (normalized * weights.view(1, 3, 1)).sum(dim=1)
    raw = ((pooled - canonicalizer["teacher_mean"].float()) @ canonicalizer["projector_weight"].float()).view(
        anchors, 8, 128
    )
    centered_raw = raw - canonicalizer["canonical_mean"].float()
    payload = {
        "kind": "paper2_phase2_matched_alpha_cache_v1",
        "documents": documents_by_anchor,
        "strata": strata_by_anchor,
        "positions": positions,
        "student_hidden": student_hidden,
        "target_centered_raw": centered_raw.to(torch.bfloat16),
        "candidate_ids": candidate_ids,
        "candidate_mask": candidate_mask,
        "base_log_probs": base_log_probs,
        "base_tail": base_tail,
        "teacher_log_probs": teacher_log_probs,
        "teacher_tail": teacher_tail,
        "teacher_topk_ids": teacher_topk_ids,
        "teacher_topk_log_probs": teacher_topk_log_probs,
        "whiten_basis": canonicalizer["whiten_basis"].float(),
        "whiten_eigenvalues": canonicalizer["whiten_eigenvalues"].float(),
        "decoder_weight_alpha_0p5": canonicalizer["decoder_weight"].float(),
        "decoder_bias": canonicalizer["decoder_bias"].float(),
        "source": {
            "stage0a_summary_sha256": sha256_file(stage0a_summary_path),
            "manifest_sha256": sha256_file(samples_path),
            "canonicalizer_sha256": sha256_file(canonicalizer_path),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(output_path)
    return payload


def _position_buckets(positions: torch.Tensor) -> torch.Tensor:
    output = torch.zeros_like(positions)
    output[(positions >= 1) & (positions <= 3)] = 1
    output[(positions >= 4) & (positions <= 31)] = 2
    output[(positions >= 32) & (positions <= 127)] = 3
    output[positions >= 128] = 4
    return output


def _losses(
    *,
    module: Phase2StudentModules,
    batch: dict[str, torch.Tensor],
    embedding: nn.Embedding,
    teacher_embedding: nn.Embedding,
    decoder: torch.Tensor,
    decoder_bias: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    hidden4 = batch["hidden"].float()
    dummy = torch.zeros_like(hidden4[:, :1])
    hidden = torch.cat([dummy, hidden4], dim=1)
    attention = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
    attention[:, 0] = False
    candidate_embeddings = embedding(batch["candidate_ids"].clamp_min(0)).float()
    raw_base_candidate_logits = torch.einsum(
        "bhd,bhcd->bhc", hidden4, candidate_embeddings
    )
    residual_seed_candidates = reconstruct_sparse_residual_seed(
        batch["base_candidates"],
        raw_base_candidate_logits,
        batch["candidate_mask"],
    ).detach()
    output = module(
        hidden=hidden,
        previous_logits=residual_seed_candidates,
        steps=1,
        attention_mask=attention,
        position_bucket=batch["position_bucket"],
        target_scratch=batch["target_scratch"].float(),
        apply_trust_penalty=False,
        candidate_ids=batch["candidate_ids"],
    )
    draft_log = normalize_sparse_with_tail(output.logits, batch["base_tail"], batch["candidate_mask"])
    target_log = normalize_sparse_with_tail(
        batch["teacher_candidates"], batch["teacher_tail"], batch["candidate_mask"]
    ).detach()
    base_log = normalize_sparse_with_tail(
        batch["base_candidates"], batch["base_tail"], batch["candidate_mask"]
    ).detach()

    bridge_delta = torch.einsum(
        "bhd,bhcd->bhc", output.hidden[:, 1:].float() - hidden4, candidate_embeddings
    )
    bridge_log = normalize_sparse_with_tail(
        residual_seed_candidates + bridge_delta,
        batch["base_tail"],
        batch["candidate_mask"],
    )
    teacher_argmax = target_log.argmax(dim=-1)
    final_ce = F.nll_loss(bridge_log.reshape(-1, bridge_log.shape[-1]), teacher_argmax.reshape(-1))
    cumulative_kl = masked_sparse_kl(
        target_log, draft_log, batch["candidate_mask"]
    ).mean()
    base_top = base_log.argmax(dim=-1)
    preserve = base_top.eq(teacher_argmax)
    preserve_kl_rows = masked_sparse_kl(
        base_log, bridge_log, batch["candidate_mask"]
    )
    preserve_kl = preserve_kl_rows[preserve].mean() if bool(preserve.any()) else preserve_kl_rows.mean() * 0

    start = output.flow.states[0]
    desired = start.detach() + BETAS[0] * (batch["target_scratch"].float() - start.detach())
    slot_mask = torch.zeros(start.shape[:2], dtype=torch.bool, device=start.device)
    slot_mask[:, :4] = True
    selected = slot_mask.unsqueeze(-1).expand_as(output.flow.state)
    flow_huber = F.huber_loss(output.flow.state[selected], desired[selected])
    flow_cosine = 1.0 - F.cosine_similarity(
        output.flow.state[:, :4].reshape(start.shape[0], -1),
        desired[:, :4].reshape(start.shape[0], -1),
        dim=1,
    ).mean()
    flow = flow_huber + 0.1 * flow_cosine

    predicted_hidden = torch.einsum("bhd,df->bhf", output.scratch[:, :4].float(), decoder) + decoder_bias
    topk_embeddings = teacher_embedding(batch["teacher_topk_ids"].long()).float()
    probe_logits = torch.einsum("bhf,bhkf->bhk", predicted_hidden, topk_embeddings)
    probe_log = torch.log_softmax(probe_logits, dim=-1)
    probe_target = torch.log_softmax(batch["teacher_topk_log_probs"].float(), dim=-1).detach()
    functional = (probe_target.exp() * (probe_target - probe_log)).sum(dim=-1).mean()
    base_predicted_hidden = (
        torch.einsum("bhd,df->bhf", output.flow.states[0][:, :4].float(), decoder)
        + decoder_bias
    )
    base_probe_logits = torch.einsum("bhf,bhkf->bhk", base_predicted_hidden, topk_embeddings)
    base_probe_log = torch.log_softmax(base_probe_logits, dim=-1)
    update = output.flow.updates[0][:, :4].float()
    prior_state = output.flow.states[0][:, :4].float()
    target_state = batch["target_scratch"][:, :4].float().detach()
    update_rms = update.square().mean(dim=(1, 2)).sqrt()
    state_rms = prior_state.square().mean(dim=(1, 2)).sqrt()
    endpoint_rms = target_state.square().mean(dim=(1, 2)).sqrt()
    state_ratio = update_rms / state_rms.clamp_min(1e-6)
    endpoint_ratio = update_rms / torch.maximum(state_rms, endpoint_rms).add(1e-6)
    trust = F.relu(endpoint_ratio - 0.5).square().mean()
    losses = {
        "final_ce": final_ce,
        "preserve_kl": preserve_kl,
        "flow": flow,
        "functional_probe_kl": functional,
        "cumulative_kl": cumulative_kl,
        "trust": trust,
    }
    metrics = {
        "draft_log": draft_log,
        "bridge_log": bridge_log,
        "target_log": target_log,
        "base_log": base_log,
        "flow_state": output.flow.state,
        "flow_start": output.flow.states[0],
        "target_scratch": batch["target_scratch"],
        "draft_gate": output.draft.write_gates,
        "bridge_gate": output.bridge.gate.expand(hidden4.shape[0]),
        "state_ratios": state_ratio.unsqueeze(1),
        "endpoint_ratios": endpoint_ratio.unsqueeze(1),
        "trust_penalty": 0.01 * trust,
        "slot_mask": slot_mask,
        "probe_log": probe_log,
        "probe_target": probe_target,
        "base_probe_log": base_probe_log,
        "scratch": output.scratch,
    }
    return losses, metrics


def _weighted_total(losses: dict[str, torch.Tensor]) -> torch.Tensor:
    weights = {
        "final_ce": 1.0,
        "preserve_kl": 0.1,
        "flow": 1.0,
        "functional_probe_kl": 0.5,
        "cumulative_kl": 1.0,
        "trust": 0.01,
    }
    return sum(weights[name] * value for name, value in losses.items())


def _parameter_groups(module: Phase2StudentModules) -> dict[str, list[nn.Parameter]]:
    return {
        "refiner": [value for value in module.flow.parameters() if value.requires_grad],
        "bridge": [value for value in module.bridge.parameters() if value.requires_grad],
        "heads": [
            value
            for child in (module.initializer, module.control, module.draft)
            for value in child.parameters()
            if value.requires_grad
        ],
    }


def _gradient_atlas(
    *, module: Phase2StudentModules, batch: dict[str, torch.Tensor],
    embedding: nn.Embedding, teacher_embedding: nn.Embedding,
    decoder: torch.Tensor, decoder_bias: torch.Tensor, seed: int
) -> dict[str, Any]:
    module.train()
    module.zero_grad(set_to_none=True)
    losses, _metrics = _losses(
        module=module, batch=batch, embedding=embedding,
        teacher_embedding=teacher_embedding, decoder=decoder, decoder_bias=decoder_bias,
    )
    groups = _parameter_groups(module)
    flat: dict[str, dict[str, torch.Tensor]] = {}
    norms: dict[str, dict[str, float]] = {}
    all_parameters = [value for values in groups.values() for value in values]
    for loss_name, loss in losses.items():
        gradients = torch.autograd.grad(
            loss, all_parameters, retain_graph=True, allow_unused=True
        )
        cursor = 0
        flat[loss_name] = {}
        norms[loss_name] = {}
        for group_name, parameters in groups.items():
            pieces = []
            for parameter in parameters:
                gradient = gradients[cursor]
                cursor += 1
                pieces.append(
                    torch.zeros_like(parameter).reshape(-1)
                    if gradient is None else gradient.detach().reshape(-1)
                )
            vector = torch.cat(pieces).float()
            flat[loss_name][group_name] = vector.cpu()
            norms[loss_name][group_name] = float(torch.linalg.vector_norm(vector))

    conflicts: dict[str, dict[str, float]] = {}
    names = list(losses)
    for group_name in groups:
        local: dict[str, float] = {}
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                a = flat[left][group_name]
                b = flat[right][group_name]
                denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
                local[f"{left}__{right}"] = float(torch.dot(a, b) / denominator.clamp_min(1e-30))
        conflicts[group_name] = local
    gradient_cv = {}
    for group_name in groups:
        values = torch.tensor([norms[name][group_name] for name in names])
        gradient_cv[group_name] = float(values.std(unbiased=False) / values.mean().clamp_min(1e-30))

    hidden = batch["hidden"][:4].float()
    attention = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
    scratch = module.initializer(hidden, attention).detach().requires_grad_(True)
    context = hidden.mean(dim=1).detach()
    generator = torch.Generator(device=scratch.device).manual_seed(int(seed))
    gains = []
    for _ in range(4):
        direction = torch.randn(scratch.shape, generator=generator, device=scratch.device)
        direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-12)
        _value, tangent = torch.autograd.functional.jvp(
            lambda state: module.flow.step(state, context, 0)[0],
            scratch,
            direction,
            create_graph=False,
            strict=True,
        )
        gains.append(float(torch.linalg.vector_norm(tangent)))
    module.zero_grad(set_to_none=True)
    return {
        "loop": 1,
        "loss_module_gradient_norms": norms,
        "module_loss_conflict_cosines": conflicts,
        "module_gradient_norm_cv": gradient_cv,
        "flow_step_jvp_gains": gains,
        "flow_step_jvp_mean": sum(gains) / len(gains),
    }


@torch.no_grad()
def _assert_zero_loop_identity(
    *, module: Phase2StudentModules, batch: dict[str, torch.Tensor]
) -> dict[str, Any]:
    hidden4 = batch["hidden"].float()
    dummy = torch.zeros_like(hidden4[:, :1])
    hidden = torch.cat([dummy, hidden4], dim=1)
    attention = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
    attention[:, 0] = False
    output = module(
        hidden=hidden,
        previous_logits=batch["base_candidates"].float(),
        steps=0,
        attention_mask=attention,
        position_bucket=batch["position_bucket"],
        candidate_ids=batch["candidate_ids"],
    )
    logits_identical = torch.equal(output.logits, batch["base_candidates"].float())
    hidden_identical = torch.equal(output.hidden, hidden)
    if not logits_identical or not hidden_identical:
        raise RuntimeError("zero-loop identity contract failed")
    return {
        "bit_exact": True,
        "logits_bit_exact": logits_identical,
        "hidden_bit_exact": hidden_identical,
        "executed_steps": 0,
    }


def _batch(cache: dict[str, Any], indices: torch.Tensor, *, alpha: float, device: str) -> dict[str, torch.Tensor]:
    basis = cache["whiten_basis"].to(device)
    eigenvalues = cache["whiten_eigenvalues"].to(device)
    raw = cache["target_centered_raw"].index_select(0, indices).to(device)
    target = alpha_transform(raw, basis, eigenvalues, alpha)
    positions = cache["positions"].index_select(0, indices).to(device)
    return {
        "hidden": cache["student_hidden"].index_select(0, indices).to(device),
        "target_scratch": target,
        "candidate_ids": cache["candidate_ids"].index_select(0, indices).long().to(device),
        "candidate_mask": cache["candidate_mask"].index_select(0, indices).to(device),
        "base_candidates": cache["base_log_probs"].index_select(0, indices).to(device),
        "base_tail": cache["base_tail"].index_select(0, indices).to(device),
        "teacher_candidates": cache["teacher_log_probs"].index_select(0, indices).to(device),
        "teacher_tail": cache["teacher_tail"].index_select(0, indices).to(device),
        "teacher_topk_ids": cache["teacher_topk_ids"].index_select(0, indices).long().to(device),
        "teacher_topk_log_probs": cache["teacher_topk_log_probs"].index_select(0, indices).to(device),
        "position_bucket": _position_buckets(positions),
    }


def _decoder_for_alpha(cache: dict[str, Any], *, alpha: float, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    scale = cache["whiten_eigenvalues"].float().pow(0.5 * (float(alpha) - 0.5))
    decoder = scale.unsqueeze(1) * cache["decoder_weight_alpha_0p5"].float()
    return decoder.to(device), cache["decoder_bias"].float().to(device)


@torch.no_grad()
def evaluate(
    *, module: Phase2StudentModules, cache: dict[str, Any], indices: torch.Tensor,
    alpha: float, embedding: nn.Embedding, teacher_embedding: nn.Embedding,
    decoder: torch.Tensor, decoder_bias: torch.Tensor,
    device: str, batch_size: int = 64
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    module.eval()
    accum: dict[str, list[torch.Tensor]] = {}
    loss_accum: dict[str, list[float]] = {}
    for start in range(0, indices.numel(), batch_size):
        local = indices[start : start + batch_size]
        losses, metrics = _losses(
            module=module,
            batch=_batch(cache, local, alpha=alpha, device=device),
            embedding=embedding,
            teacher_embedding=teacher_embedding,
            decoder=decoder,
            decoder_bias=decoder_bias,
        )
        for key, value in losses.items():
            loss_accum.setdefault(key, []).append(float(value.detach()))
        for key in (
            "draft_log", "bridge_log", "target_log", "base_log", "draft_gate",
            "bridge_gate", "state_ratios", "endpoint_ratios", "probe_log",
            "probe_target", "base_probe_log", "scratch", "flow_state", "flow_start",
            "target_scratch",
        ):
            accum.setdefault(key, []).append(metrics[key].detach().cpu())
    values = {key: torch.cat(parts) for key, parts in accum.items()}
    overlap = distribution_overlap(values["target_log"], values["draft_log"])
    accepted = expected_accepted_length(overlap)
    base_overlap = distribution_overlap(values["target_log"], values["base_log"])
    base_accepted = expected_accepted_length(base_overlap)
    target_top = values["target_log"].argmax(-1)
    base_top = values["base_log"].argmax(-1)
    bridge_top = values["bridge_log"].argmax(-1)
    baseline_correct = base_top.eq(target_top)
    retained = bridge_top.eq(target_top) & baseline_correct
    baseline_count = int(baseline_correct.sum())
    retained_count = int(retained.sum())
    probe_kl_rows = (
        values["probe_target"].exp() * (values["probe_target"] - values["probe_log"])
    ).sum(-1).mean(-1)
    base_probe_kl_rows = (
        values["probe_target"].exp() * (values["probe_target"] - values["base_probe_log"])
    ).sum(-1).mean(-1)
    probe_top1_rows = values["probe_target"].argmax(-1).eq(
        values["probe_log"].argmax(-1)
    ).float().mean(-1)
    improved_rows = accepted > base_accepted
    quality_rows = bridge_top.eq(target_top).float().mean(-1)
    base_quality_rows = base_top.eq(target_top).float().mean(-1)
    gate_rows = values["draft_gate"].mean(-1)
    draft_gate_initial = float(torch.sigmoid(torch.tensor(-3.5)))
    bridge_gate_initial = float(torch.sigmoid(torch.tensor(-4.0)))
    gate_open = gate_rows > draft_gate_initial
    populated_mask = torch.zeros(values["scratch"].shape[:2], dtype=torch.bool)
    populated_mask[:, :4] = True
    endpoint_error_rows = (
        values["flow_state"][:, :4].float() - values["target_scratch"][:, :4].float()
    ).square().mean(dim=(1, 2)).sqrt()
    state_rms_rows = values["flow_state"][:, :4].float().square().mean(dim=(1, 2)).sqrt()
    target_rms_rows = values["target_scratch"][:, :4].float().square().mean(dim=(1, 2)).sqrt()
    radial_drift_rows = (state_rms_rows - target_rms_rows).abs()

    def correlation(left: torch.Tensor, right: torch.Tensor) -> float:
        left = left.float() - left.float().mean()
        right = right.float() - right.float().mean()
        denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
        return float((left * right).sum() / denominator.clamp_min(1e-12))

    stratified = {}
    for name, mask in (("gate_open", gate_open), ("gate_closed", ~gate_open)):
        selected = improved_rows & mask
        stratified[name] = {
            "accepted_improvement_rows": int(selected.sum()),
            "probe_kl_worsened_fraction": float(
                (probe_kl_rows[selected] > base_probe_kl_rows[selected]).float().mean()
            ) if bool(selected.any()) else None,
            "final_quality_worsened_fraction": float(
                (quality_rows[selected] < base_quality_rows[selected]).float().mean()
            ) if bool(selected.any()) else None,
        }
    summary = {
        "mean_accepted_length": float(accepted.mean()),
        "zero_loop_mean_accepted_length": float(base_accepted.mean()),
        "acceptance_delta": float(accepted.mean() - base_accepted.mean()),
        "mean_overlap_by_horizon": overlap.mean(0).tolist(),
        "baseline_correct": baseline_count,
        "retained_correct": retained_count,
        "retention": retained_count / max(1, baseline_count),
        "retention_wilson_95_lower": wilson_lower(retained_count, baseline_count),
        "quality_noninferior": quality_noninferior(retained_count, baseline_count),
        "flow_validation_loss": sum(loss_accum["flow"]) / len(loss_accum["flow"]),
        "mean_endpoint_error": float(endpoint_error_rows.mean()),
        "mean_radial_drift": float(radial_drift_rows.mean()),
        "mean_draft_gate": float(values["draft_gate"].mean()),
        "mean_bridge_gate": float(values["bridge_gate"].mean()),
        "draft_gate_open_rate": float((values["draft_gate"] > draft_gate_initial).float().mean()),
        "bridge_gate_open_rate": float((values["bridge_gate"] > bridge_gate_initial).float().mean()),
        "mean_state_ratio": float(values["state_ratios"].mean()),
        "mean_endpoint_ratio": float(values["endpoint_ratios"].mean()),
        "state_ratio_by_loop": values["state_ratios"].mean(0).tolist(),
        "endpoint_ratio_by_loop": values["endpoint_ratios"].mean(0).tolist(),
        "trust_rent": 0.01 * sum(loss_accum["trust"]) / len(loss_accum["trust"]),
        "mean_probe_kl": float(probe_kl_rows.mean()),
        "probe_top1": float(probe_top1_rows.mean()),
        "scratch_effective_rank_populated_slots": float(
            masked_effective_rank(values["scratch"], populated_mask)
        ),
        "correlations": {
            "probe_kl_vs_accepted_length": correlation(probe_kl_rows, accepted),
            "probe_top1_vs_accepted_length": correlation(probe_top1_rows, accepted),
            "probe_kl_vs_probe_top1": correlation(probe_kl_rows, probe_top1_rows),
        },
        "trained_monotonicity": stratified,
        "losses": {key: sum(parts) / len(parts) for key, parts in loss_accum.items()},
    }
    rows = {
        "accepted_length": accepted,
        "base_accepted_length": base_accepted,
        "acceptance_delta": accepted - base_accepted,
        "quality_correct": bridge_top.eq(target_top).float().mean(-1),
        "base_correct_by_horizon": base_top.eq(target_top),
        "bridge_correct_by_horizon": bridge_top.eq(target_top),
        "draft_gate": values["draft_gate"],
        "bridge_gate": values["bridge_gate"],
        "state_ratio": values["state_ratios"],
        "endpoint_ratio": values["endpoint_ratios"],
        "probe_kl": probe_kl_rows,
        "base_probe_kl": base_probe_kl_rows,
        "probe_top1": probe_top1_rows,
        "endpoint_error": endpoint_error_rows,
        "radial_drift": radial_drift_rows,
        "flow_state": values["flow_state"][:, :4],
        "flow_start": values["flow_start"][:, :4],
        "target_scratch": values["target_scratch"][:, :4],
    }
    module.train()
    return summary, rows


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng(payload: dict[str, Any]) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch"])
    torch.cuda.set_rng_state_all(payload["cuda"])


def run_arm(
    *, cache: dict[str, Any], embedding_weight: torch.Tensor,
    teacher_embedding: nn.Embedding, alpha: float, seed: int,
    train_indices: torch.Tensor, eval_indices: torch.Tensor, output_dir: Path,
    private_dir: Path, target_steps: int, device: str, source_hashes: dict[str, str],
    rms_cap: float,
) -> dict[str, Any]:
    arm = f"alpha_{str(alpha).replace('.', 'p')}_seed_{seed}"
    arm_dir = private_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = arm_dir / "resume.pt"
    summary_path = output_dir / f"{arm}.json"
    _seed_everything(seed)
    embedding = nn.Embedding.from_pretrained(embedding_weight.float(), freeze=True).to(device)
    module = Phase2StudentModules(
        tied_embedding=embedding, hidden_size=896, rms_cap=float(rms_cap)
    ).to(device=device, dtype=torch.float32)
    trainable = {name: value for name, value in module.named_parameters() if value.requires_grad}
    initial_hash = _tensor_digest(trainable)
    optimizer = torch.optim.AdamW(build_adamw_groups(module, weight_decay=0.01), lr=3e-4)
    history: list[dict[str, Any]] = []
    gradient_atlases: list[dict[str, Any]] = []
    trust_history: list[bool] = []
    clip_events: list[tuple[str, bool]] = []
    step = 0
    consecutive_quality_failures = 0
    generator = torch.Generator().manual_seed(seed + 9173)
    if checkpoint.is_file():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["alpha"] != alpha or saved["seed"] != seed or saved["initial_hash"] != initial_hash:
            raise RuntimeError(f"resume metadata mismatch for {arm}")
        if saved.get("loss_mask_contract") != LOSS_MASK_CONTRACT:
            if not _resume_has_no_optimizer_update(saved):
                raise RuntimeError(f"incompatible resume with possible optimizer update for {arm}")
            print(f"matched_alpha_recompute_invalid_step_zero arm={arm}", flush=True)
        else:
            _load_trainable_state(module, saved["trainable_state"])
            optimizer.load_state_dict(saved["optimizer"])
            step = int(saved["step"])
            history = list(saved["history"])
            gradient_atlases = list(saved.get("gradient_atlases", []))
            trust_history = list(saved["trust_history"])
            clip_events = list(saved["clip_events"])
            generator.set_state(saved["batch_generator_state"])
            _restore_rng(saved["rng"])
            print(f"matched_alpha_resume arm={arm} step={step}", flush=True)
    decoder, decoder_bias = _decoder_for_alpha(cache, alpha=alpha, device=device)

    def save() -> None:
        payload = {
            "kind": "paper2_phase2_matched_alpha_resume",
            "protocol_lock_commit": PROTOCOL_LOCK_COMMIT,
            "loss_mask_contract": LOSS_MASK_CONTRACT,
            "alpha": alpha,
            "seed": seed,
            "step": step,
            "initial_hash": initial_hash,
            "trainable_state": {
                key: value.detach().cpu()
                for key, value in module.named_parameters()
                if value.requires_grad
            },
            "optimizer": optimizer.state_dict(),
            "history": history,
            "gradient_atlases": gradient_atlases,
            "trust_history": trust_history,
            "clip_events": clip_events,
            "batch_generator_state": generator.get_state(),
            "rng": _rng_state(),
        }
        temporary = checkpoint.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(checkpoint)

    identity_batch = _batch(cache, eval_indices[:16], alpha=alpha, device=device)
    zero_loop_identity = _assert_zero_loop_identity(module=module, batch=identity_batch)
    if sum(value.numel() for value in trainable.values()) != 1_184_917:
        raise RuntimeError("student trainable parameter count differs from the locked value")
    if any(value.requires_grad for value in embedding.parameters()) or any(
        value.requires_grad for value in teacher_embedding.parameters()
    ):
        raise RuntimeError("a frozen LM head unexpectedly requires gradients")
    smoke_losses, _smoke_metrics = _losses(
        module=module,
        batch=identity_batch,
        embedding=embedding,
        teacher_embedding=teacher_embedding,
        decoder=decoder,
        decoder_bias=decoder_bias,
    )
    nonfinite_smoke = [
        name for name, value in smoke_losses.items() if not bool(torch.isfinite(value))
    ]
    if nonfinite_smoke:
        raise RuntimeError(
            f"matched-alpha finite-loss preflight failed for {arm}: {nonfinite_smoke}"
        )
    reconstructed_support = int(
        (
            identity_batch["candidate_mask"]
            & ~torch.isfinite(identity_batch["base_candidates"])
        ).sum()
    )
    print(
        f"matched_alpha_loss_preflight arm={arm} finite=1 "
        f"reconstructed_support={reconstructed_support}",
        flush=True,
    )
    if step == 0 and not history:
        evaluation, rows = evaluate(
            module=module, cache=cache, indices=eval_indices, alpha=alpha,
            embedding=embedding, teacher_embedding=teacher_embedding,
            decoder=decoder, decoder_bias=decoder_bias,
            device=device,
        )
        history.append({"step": 0, **evaluation})
        gradient_atlases.append({
            "step": 0,
            **_gradient_atlas(
                module=module, batch=identity_batch, embedding=embedding,
                teacher_embedding=teacher_embedding, decoder=decoder,
                decoder_bias=decoder_bias, seed=seed + 1000,
            ),
        })
        torch.save(rows, arm_dir / "rows_step_0000.pt")
        save()
    abort_reason = None
    while step < target_steps:
        step += 1
        sampled = train_indices.index_select(
            0, torch.randint(train_indices.numel(), (128,), generator=generator)
        )
        batch = _batch(cache, sampled, alpha=alpha, device=device)
        losses, metrics = _losses(
            module=module, batch=batch, embedding=embedding,
            teacher_embedding=teacher_embedding, decoder=decoder, decoder_bias=decoder_bias
        )
        total = _weighted_total(losses)
        if not bool(torch.isfinite(total)):
            abort_reason = "nonfinite_loss"
            break
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        if any(value.grad is not None for value in embedding.parameters()) or any(
            value.grad is not None for value in teacher_embedding.parameters()
        ):
            abort_reason = "frozen_lm_head_received_gradient"
            break
        for name, (parameters, ceiling) in clip_module_groups(module).items():
            active = [value for value in parameters if value.requires_grad and value.grad is not None]
            norm = torch.nn.utils.clip_grad_norm_(active, ceiling) if active else torch.tensor(0.0)
            clip_events.append((name, float(norm) > ceiling))
        learning_rate = 3e-4 * min(1.0, step / 100.0)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.step()
        nonzero_trust = bool(float(metrics["trust_penalty"].detach()) > 0)
        trust_history.append(nonzero_trust)
        if step > 100 and trust_saturated(trust_history, window=100, maximum_nonzero=50):
            abort_reason = "trust_saturation"
            break
        if step % 100 == 0:
            evaluation, rows = evaluate(
                module=module, cache=cache, indices=eval_indices, alpha=alpha,
                embedding=embedding, teacher_embedding=teacher_embedding,
                decoder=decoder, decoder_bias=decoder_bias,
                device=device,
            )
            evaluation.update({"step": step, "learning_rate": learning_rate})
            history.append(evaluation)
            if step in {100, target_steps}:
                gradient_atlases.append({
                    "step": step,
                    **_gradient_atlas(
                        module=module,
                        batch=_batch(cache, eval_indices[:16], alpha=alpha, device=device),
                        embedding=embedding,
                        teacher_embedding=teacher_embedding,
                        decoder=decoder,
                        decoder_bias=decoder_bias,
                        seed=seed + 1000 + step,
                    ),
                })
            torch.save(rows, arm_dir / f"rows_step_{step:04d}.pt")
            consecutive_quality_failures = (
                consecutive_quality_failures + 1 if not evaluation["quality_noninferior"] else 0
            )
            print(
                f"matched_alpha_eval arm={arm} step={step} loss={float(total):.6f} "
                f"eal={evaluation['mean_accepted_length']:.6f} "
                f"quality={evaluation['retention']:.6f}",
                flush=True,
            )
            save()
            if consecutive_quality_failures >= 2:
                abort_reason = "quality_noninferiority_two_consecutive_evaluations"
                break
    save()
    final = history[-1]
    result = {
        "kind": "paper2_phase2_matched_alpha_arm",
        "status": "aborted" if abort_reason else "complete",
        "abort_reason": abort_reason,
        "alpha": alpha,
        "seed": seed,
        "step": step,
        "target_steps": target_steps,
        "protocol_lock_commit": PROTOCOL_LOCK_COMMIT,
        "loss_mask_contract": LOSS_MASK_CONTRACT,
        "launcher_commit": _git_head(Path(__file__).resolve().parents[1]),
        "source_hashes": source_hashes,
        "non_alpha_initialization_sha256": initial_hash,
        "history": history,
        "gradient_atlases": gradient_atlases,
        "zero_loop_identity": zero_loop_identity,
        "frozen_lm_heads": {
            "requires_grad_false": True,
            "gradient_none_after_training": all(
                value.grad is None
                for value in list(embedding.parameters()) + list(teacher_embedding.parameters())
            ),
        },
        "final": final,
        "clip_events": summarize_clip_fractions(clip_events),
        "trust_nonzero_steps": sum(trust_history),
        "trust_observed_steps": len(trust_history),
        "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        "final_rows": {
            "path": str(arm_dir / f"rows_step_{int(final['step']):04d}.pt"),
            "sha256": sha256_file(arm_dir / f"rows_step_{int(final['step']):04d}.pt"),
        },
    }
    write_json(summary_path, result)
    return result


def _adequate(arms: list[dict[str, Any]]) -> bool:
    count = 0
    for arm in arms:
        by_step = {int(row["step"]): row for row in arm["history"]}
        if 100 not in by_step or int(arm["step"]) < 1000:
            continue
        final = arm["final"]
        improved = final["flow_validation_loss"] <= 0.8 * by_step[100]["flow_validation_loss"]
        gate_opened = final["draft_gate_open_rate"] > by_step[0]["draft_gate_open_rate"]
        count += int(improved and gate_opened)
    return count >= 4


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    protocol = _assert_lock(root)
    constants_path = root / "training/paper2_phase2_dc2_constants.json"
    if sha256_lf_file(constants_path) != protocol["constants_lf_sha256"]:
        raise RuntimeError("V1d constants file does not match the preregistration")
    constants = json.loads(constants_path.read_text(encoding="utf-8"))
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 35 * 2**30:
        raise RuntimeError(
            "cached-state sparse-logit pilots require an A100-class GPU with at least 35 GiB VRAM"
        )
    cache = build_pilot_cache(
        stage0a_summary_path=args.stage0a_summary,
        stage0a_private=args.stage0a_private,
        canonicalizer_path=args.canonicalizer,
        output_path=args.cache,
    )
    if cache["source"]["canonicalizer_sha256"] != protocol["canonicalizer"]["sha256"]:
        raise RuntimeError("pilot cache canonicalizer does not match preregistration")
    stage0a_summary = json.loads(args.stage0a_summary.read_text(encoding="utf-8"))
    expected_data = protocol["frozen_data"]
    observed = {
        "data_sha256": stage0a_summary["config"]["data_sha256"],
        "sample_manifest_sha256": stage0a_summary["manifest"]["sample_manifest_sha256"],
        "position_key_sha256": stage0a_summary["manifest"]["position_key_sha256"],
    }
    for key, value in observed.items():
        if value != expected_data[key]:
            raise RuntimeError(f"Stage 0A {key} does not match the preregistration")
    eval_mask = document_partition(cache["documents"], evaluation_fraction=0.2, seed=20260804)
    eval_indices = torch.where(eval_mask)[0]
    train_indices = torch.where(~eval_mask)[0]
    if eval_indices.numel() < 1000 or train_indices.numel() < 1000:
        raise RuntimeError("fixed DEV split is unexpectedly small")
    student_summary = json.loads(
        (args.stage0a_private / "model_cache/student_0p5b/summary.json").read_text(encoding="utf-8")
    )
    teacher_summary = json.loads(
        (args.stage0a_private / "model_cache/teacher_14b/summary.json").read_text(encoding="utf-8")
    )
    head_path = _local_source(student_summary["lm_head"]["path"], args.stage0a_private)
    if sha256_file(head_path) != student_summary["lm_head"]["sha256"]:
        raise RuntimeError("student LM-head hash mismatch")
    if student_summary["lm_head"]["sha256"] != protocol["frozen_data"]["student_lm_head_sha256"]:
        raise RuntimeError("student LM-head does not match the preregistration")
    teacher_head_path = _local_source(teacher_summary["lm_head"]["path"], args.stage0a_private)
    if sha256_file(teacher_head_path) != teacher_summary["lm_head"]["sha256"]:
        raise RuntimeError("teacher LM-head hash mismatch")
    embedding_weight = torch.load(head_path, map_location="cpu", weights_only=False)["weight_bfloat16"]
    teacher_embedding_weight = torch.load(
        teacher_head_path, map_location="cpu", weights_only=False
    )["weight_bfloat16"]
    teacher_embedding = nn.Embedding.from_pretrained(
        teacher_embedding_weight.float(), freeze=True
    ).to(args.device)
    source_hashes = {
        **cache["source"],
        "student_lm_head_sha256": sha256_file(head_path),
        "teacher_lm_head_sha256": sha256_file(teacher_head_path),
        "constants_lf_sha256": protocol["constants_lf_sha256"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.private_dir.mkdir(parents=True, exist_ok=True)
    extra_alpha = args.extra_alpha
    if extra_alpha is not None and float(extra_alpha) not in {0.25, 0.75}:
        raise RuntimeError("only the preregistered alpha 0.25 or 0.75 refinement is allowed")
    alphas = [float(value) for value in protocol["alphas"]]
    if extra_alpha is not None:
        alphas.append(float(extra_alpha))
    prior_summary_path = args.output_dir / "summary.json"
    prior_extended = False
    if prior_summary_path.is_file():
        prior = json.loads(prior_summary_path.read_text(encoding="utf-8"))
        prior_extended = bool(prior.get("extended_once", False))
    initial_target_steps = int(protocol["extension_steps"] if prior_extended else protocol["steps"])
    arms = []
    for alpha in alphas:
        for seed in protocol["seeds"]:
            arms.append(
                run_arm(
                    cache=cache, embedding_weight=embedding_weight,
                    teacher_embedding=teacher_embedding, alpha=float(alpha), seed=int(seed),
                    train_indices=train_indices, eval_indices=eval_indices,
                    output_dir=args.output_dir, private_dir=args.private_dir,
                    target_steps=initial_target_steps, device=args.device,
                    source_hashes=source_hashes,
                    rms_cap=float(constants["p99_state_rms_cap"]),
                )
            )
    initial_arms = [arm for arm in arms if float(arm["alpha"]) in set(protocol["alphas"])]
    extended = prior_extended
    if (
        extra_alpha is None
        and not extended
        and not _adequate(initial_arms)
        and all(arm["status"] == "complete" for arm in initial_arms)
    ):
        extended = True
        arms = []
        for alpha in alphas:
            for seed in protocol["seeds"]:
                arms.append(
                    run_arm(
                        cache=cache, embedding_weight=embedding_weight,
                        teacher_embedding=teacher_embedding, alpha=float(alpha), seed=int(seed),
                        train_indices=train_indices, eval_indices=eval_indices,
                        output_dir=args.output_dir, private_dir=args.private_dir,
                        target_steps=int(protocol["extension_steps"]), device=args.device,
                        source_hashes=source_hashes,
                        rms_cap=float(constants["p99_state_rms_cap"]),
                    )
                )
        initial_arms = [arm for arm in arms if float(arm["alpha"]) in set(protocol["alphas"])]
    non_alpha_hashes = {}
    for arm in arms:
        non_alpha_hashes.setdefault(str(arm["seed"]), set()).add(arm["non_alpha_initialization_sha256"])
    if any(len(values) != 1 for values in non_alpha_hashes.values()):
        raise RuntimeError("non-alpha initialization differs across alpha arms")
    summary = {
        "kind": "paper2_phase2_matched_alpha_pilots",
        "status": "complete_dev_only" if all(arm["status"] == "complete" for arm in arms) else "blocked_with_receipts",
        "protocol_lock_commit": PROTOCOL_LOCK_COMMIT,
        "launcher_commit": _git_head(root),
        "extended_once": extended,
        "adequacy_precondition_met": _adequate(initial_arms),
        "train_anchors": int(train_indices.numel()),
        "evaluation_anchors": int(eval_indices.numel()),
        "document_isolated": True,
        "arms": arms,
        "conditional_refinement_alpha": extra_alpha,
        "alpha_selected": False,
        "selection_deferred_to_locked_decision_script": True,
        "frozen_evaluation_partitions_touched": [],
        "do_not_claim": [
            "DEV pilot alpha is confirmatory E1 evidence",
            "cached teacher-forced accepted length is serving throughput",
            "probe fidelity alone establishes upper-model quality",
        ],
    }
    write_json(args.output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage0a_summary", type=Path, required=True)
    parser.add_argument("--stage0a_private", type=Path, required=True)
    parser.add_argument("--canonicalizer", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--private_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--extra_alpha", type=float)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run(arguments)
    print(json.dumps({"status": result["status"], "adequacy": result["adequacy_precondition_met"]}, indent=2))
