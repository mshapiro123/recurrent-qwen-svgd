"""Run one locked, resumable P3.3 aimed-writeback seed."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import t as student_t
from torch import nn

from eval.cache_paper2_phase3_agreement_oracle import (
    _lm_head,
    _load_phase3_module,
    load_selected_anchor_hidden,
)
from eval.eval_paper2_phase3_retention_step0 import position_buckets
from training.paper2_phase2_matched_alpha import build_adamw_groups, clip_module_groups
from training.paper2_phase3_p32 import GateLabel
from training.paper2_phase3_p33 import (
    P33_ADAM_BETAS,
    P33_BATCH_SIZE,
    P33_LEARNING_RATE,
    P33_LOOK_INTERVAL,
    P33_LOOKS,
    P33_TOTAL_STEPS,
    P33_WARMUP_STEPS,
    P33_WEIGHT_DECAY,
    activate_operating_clamp,
    gate_classification,
    independent_gradient_shares,
    learning_rate_at_step,
    p33_forward_losses,
    set_p33_trainable,
    weighted_total,
)
from training.paper2_phase3_p33_prep import (
    intervene_state,
    observatory_event_rows,
    observatory_metrics,
    sha256_file,
)
from training.run_paper2_phase2_option_b import build_option_b_cache


RUN_KIND = "paper2_phase3_p33_aimed_writeback_seed_v1"
AUDIT_RADIUS = 0.15
EXPECTED_TRAINABLE = 1_185_973
POSITIVE_PER_BATCH = 32
NEGATIVE_PER_BATCH = 96
AUDIT_BATCH_SIZE = 32
BOOTSTRAP_DRAWS = 2_000
EXPECTED_PREFLIGHT_SHA256 = "9a71e3e59526383b3dd830a320a0e18ad3778571f67dac1e262ee2713ea0ffd0"
EXPECTED_CALIBRATION_SHA256 = "e46198291bdea16f3561b44eaa1a77764aa7a0fcc49a60c4c58802491aef985c"
EXPECTED_MIGRATED_SHA256 = {
    0: "d0f2b735825d29ab9801a5200493ca9aa65294778aea2fb7f728eb8e85dfc519",
    1: "3ca1cdf8dd16bf4f435e81a675d7514778144c5c881af52a70171659f7734b4f",
}
OBSERVATORY_ROWS = 128
ASTATE_ROWS = 512


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    temporary.replace(path)


def tensor_digest(values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def object_digest(value: object) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def atomic_torch_save(payload: object, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return sha256_file(path)


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(payload: Mapping[str, Any]) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch"])
    if torch.cuda.is_available() and payload.get("cuda"):
        torch.cuda.set_rng_state_all(payload["cuda"])


def _active_record_pools(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positives = [dict(row) for row in records if int(row["gate_label"]) == int(GateLabel.POSITIVE)]
    negatives = [dict(row) for row in records if int(row["gate_label"]) == int(GateLabel.NEGATIVE)]
    if len(positives) != 34_521 or len(negatives) != 103_563:
        raise RuntimeError(f"P3.3 training counts changed: {len(positives)}/{len(negatives)}")
    if any(not bool(row.get("training_eligible")) for row in (*positives, *negatives)):
        raise RuntimeError("P3.3 active training pool contains an ineligible record")
    return positives, negatives


def _direction_lookup(path: Path) -> tuple[dict[str, int], torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") != "paper2_phase3_agreement_oracle_direction_cache_v1":
        raise RuntimeError("P3.3 oracle direction cache kind changed")
    record_ids = [str(value) for value in payload["record_ids"]]
    directions = payload["directions"].float()
    if directions.shape != (len(record_ids), 896):
        raise RuntimeError("P3.3 oracle direction cache shape changed")
    return {record_id: index for index, record_id in enumerate(record_ids)}, directions, {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": len(record_ids),
    }


def _batch(
    *,
    cache: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    direction_index: Mapping[str, int],
    directions: torch.Tensor,
    device: str,
) -> dict[str, torch.Tensor]:
    anchors = torch.tensor([int(row["anchor_index"]) for row in records], dtype=torch.long)
    labels = torch.full((len(records), 4), int(GateLabel.IGNORED), dtype=torch.long)
    oracle = torch.zeros((len(records), 4, 896), dtype=torch.float32)
    for index, row in enumerate(records):
        horizon = int(row["horizon"]) - 1
        label = int(row["gate_label"])
        labels[index, horizon] = label
        if label == int(GateLabel.POSITIVE):
            record_id = str(row["record_id"])
            if record_id not in direction_index:
                raise RuntimeError(f"positive record missing oracle direction: {record_id}")
            oracle[index, horizon] = directions[direction_index[record_id]]
    positions = cache["positions"].index_select(0, anchors)
    return {
        "hidden4": cache["student_hidden"].index_select(0, anchors).to(device),
        "candidate_ids": cache["candidate_ids"].index_select(0, anchors).long().to(device),
        "candidate_mask": cache["candidate_mask"].index_select(0, anchors).to(device),
        "base_candidates": cache["base_log_probs"].index_select(0, anchors).to(device),
        "base_tail": cache["base_tail"].index_select(0, anchors).to(device),
        "gate_labels": labels.to(device),
        "oracle_directions": oracle.to(device),
        "position_bucket": position_buckets(positions.to(device)),
    }


def _losses(module: nn.Module, batch: Mapping[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    return p33_forward_losses(
        module=module,
        tied_embedding=module.draft.tied_embedding,
        steps=1,
        **batch,
    )


def fixed_audit_batches(
    positives: Sequence[Mapping[str, Any]],
    negatives: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    batches: int = 4,
) -> list[list[dict[str, Any]]]:
    generator = torch.Generator().manual_seed(20260811 + int(seed) + 50_000)
    output = []
    for _ in range(batches):
        positive = torch.randint(len(positives), (POSITIVE_PER_BATCH,), generator=generator)
        negative = torch.randint(len(negatives), (NEGATIVE_PER_BATCH,), generator=generator)
        rows = [dict(positives[int(index)]) for index in positive]
        rows.extend(dict(negatives[int(index)]) for index in negative)
        output.append(rows)
    return output


def directional_share_audit(
    *,
    module: nn.Module,
    cache: Mapping[str, Any],
    batches: Sequence[Sequence[Mapping[str, Any]]],
    direction_index: Mapping[str, int],
    directions: torch.Tensor,
    device: str,
) -> dict[str, Any]:
    was_training = module.training
    module.train()
    parameters = list(set_p33_trainable(module).values())
    rows = []
    for records in batches:
        losses, _ = _losses(
            module,
            _batch(
                cache=cache,
                records=records,
                direction_index=direction_index,
                directions=directions,
                device=device,
            ),
        )
        rows.append(independent_gradient_shares(losses, parameters))
    mean_norms = {
        name: sum(row["raw_gradient_norms"][name] for row in rows) / len(rows)
        for name in ("aim", "gate", "preserve")
    }
    denominator = sum(mean_norms.values())
    shares = {name: mean_norms[name] / max(denominator, 1e-30) for name in mean_norms}
    primary = shares["aim"] + shares["gate"]
    classification = (
        "gross"
        if primary < 0.40 or shares["preserve"] > 0.35
        else "marginal"
        if primary < 0.50 or shares["preserve"] > 0.25
        else "pass"
    )
    if not was_training:
        module.eval()
    return {
        "matched_batches": len(rows),
        "batch_size": P33_BATCH_SIZE,
        "mean_raw_gradient_norms": mean_norms,
        "shares": shares,
        "primary_share": primary,
        "preservation_share": shares["preserve"],
        "classification": classification,
        "per_batch": rows,
    }


def _model_components(module: nn.Module, hidden4: torch.Tensor, positions: torch.Tensor) -> dict[str, torch.Tensor]:
    dummy = torch.zeros_like(hidden4[:, :1])
    hidden = torch.cat([dummy, hidden4], dim=1)
    attention = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
    attention[:, 0] = False
    candidate_ids = torch.zeros((hidden.shape[0], 4, 1), dtype=torch.long, device=hidden.device)
    previous = torch.zeros((hidden.shape[0], 4, 1), dtype=hidden.dtype, device=hidden.device)
    output = module(
        hidden=hidden,
        previous_logits=previous,
        steps=1,
        attention_mask=attention,
        position_bucket=position_buckets(positions),
        candidate_ids=candidate_ids,
    )
    return {
        "delta": output.bridge.delta[:, 1:].float(),
        "deployed_hidden": output.hidden[:, 1:].float(),
        "gate_unclamped": output.bridge.position_gate_unclamped[:, 1:, 0].float(),
        "gate_deployed": output.bridge.position_gate[:, 1:, 0].float(),
    }


def _top1(states: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
    return (states.float() @ embedding.float().T).argmax(dim=-1)


def load_audit_material(
    *,
    positive_path: Path,
    negative_path: Path,
    retention_path: Path,
    sources: Mapping[str, tuple[Path, Path]],
) -> dict[str, Any]:
    positive = read_jsonl(positive_path)
    negative = read_jsonl(negative_path)
    retention = read_jsonl(retention_path)
    if (len(positive), len(negative), len(retention)) != (4096, 12288, 1024):
        raise RuntimeError("P3.3 audit population counts changed")
    material = {}
    for name, records in (("positive", positive), ("negative", negative), ("retention", retention)):
        hidden, record_anchor, _lookup, receipt = load_selected_anchor_hidden(records=records, sources=sources)
        material[name] = {
            "records": records,
            "hidden4": hidden.index_select(0, record_anchor),
            "hidden_receipt": receipt,
        }
    return material


def _ratio_bootstrap(
    rows: Sequence[Mapping[str, Any]], *, numerator: str, denominator: str, seed: int
) -> dict[str, float | None]:
    grouped: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    for row in rows:
        key = str(row["document_id"])
        left, right = grouped[key]
        grouped[key] = (left + int(bool(row[numerator])), right + int(bool(row[denominator])))
    values = list(grouped.values())
    point_denominator = sum(value[1] for value in values)
    point = sum(value[0] for value in values) / max(1, point_denominator)
    generator = np.random.default_rng(seed)
    samples = []
    for _ in range(BOOTSTRAP_DRAWS):
        chosen = generator.integers(0, len(values), len(values))
        left = sum(values[int(index)][0] for index in chosen)
        right = sum(values[int(index)][1] for index in chosen)
        if right:
            samples.append(left / right)
    return {
        "point": point,
        "ci95_low": float(np.quantile(samples, 0.025)) if samples else None,
        "ci95_high": float(np.quantile(samples, 0.975)) if samples else None,
        "documents": len(values),
    }


@torch.inference_mode()
def audit_model(
    *,
    module: nn.Module,
    material: Mapping[str, Any],
    direction_index: Mapping[str, int],
    directions: torch.Tensor,
    seed: int,
    step: int,
    device: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    module.eval()
    embedding = module.draft.tied_embedding.weight
    rows: list[dict[str, Any]] = []
    for population in ("positive", "negative"):
        records = material[population]["records"]
        hidden_all = material[population]["hidden4"]
        for start in range(0, len(records), AUDIT_BATCH_SIZE):
            stop = min(len(records), start + AUDIT_BATCH_SIZE)
            local_records = records[start:stop]
            hidden4 = hidden_all[start:stop].to(device=device, dtype=torch.float32)
            positions = torch.tensor(
                [int(row["prediction_position"]) for row in local_records],
                dtype=torch.long,
                device=device,
            )
            horizons = torch.tensor(
                [int(row["horizon"]) - 1 for row in local_records],
                dtype=torch.long,
                device=device,
            )
            index = torch.arange(stop - start, device=device)
            components = _model_components(module, hidden4, positions)
            base = hidden4[index, horizons]
            delta = components["delta"][index, horizons]
            deployed = components["deployed_hidden"][index, horizons]
            gate_unclamped = components["gate_unclamped"][index, horizons]
            gate_deployed = components["gate_deployed"][index, horizons]
            state_stack = [base, deployed]
            oracle = None
            if population == "positive":
                oracle = torch.stack(
                    [directions[direction_index[str(row["record_id"])]] for row in local_records]
                ).to(device)
                reference = base.square().mean(dim=-1).sqrt().clamp_max(module.bridge.rms_cap)
                oracle_scaled = oracle / oracle.square().mean(dim=-1).sqrt().clamp_min(1e-8).unsqueeze(-1)
                oracle_scaled = oracle_scaled * reference.unsqueeze(-1)
                state_stack.extend(
                    [
                        base + AUDIT_RADIUS * delta,
                        base + AUDIT_RADIUS * oracle_scaled,
                        base + gate_deployed.unsqueeze(-1) * oracle_scaled,
                    ]
                )
            tokens = _top1(torch.cat(state_stack, dim=0), embedding).cpu().split(stop - start)
            base_token, deployed_token = tokens[0], tokens[1]
            for offset, source in enumerate(local_records):
                row = {
                    "seed": seed,
                    "step": step,
                    "population": population,
                    "record_id": source["record_id"],
                    "document_id": source["document_id"],
                    "horizon": int(source["horizon"]),
                    "teachability_decile": source.get("teachability_decile"),
                    "base_top1": int(base_token[offset]),
                    "deployed_top1": int(deployed_token[offset]),
                    "gate_unclamped": float(gate_unclamped[offset]),
                    "gate_deployed": float(gate_deployed[offset]),
                    "collateral_change": bool(base_token[offset] != deployed_token[offset]),
                }
                if int(base_token[offset]) != int(source["student_top1"]):
                    raise RuntimeError("P3.3 audit base-reader mismatch")
                if population == "positive":
                    teacher = int(source["teacher_14b_top1"])
                    forced_trained, forced_oracle, deployed_oracle = tokens[2:5]
                    row.update(
                        {
                            "teacher_top1": teacher,
                            "direction_cosine": float(F.cosine_similarity(delta[offset], oracle[offset], dim=0)),
                            "forced_trained_flip": int(forced_trained[offset]) == teacher,
                            "forced_oracle_flip": int(forced_oracle[offset]) == teacher,
                            "deployed_trained_flip": int(deployed_token[offset]) == teacher,
                            "deployed_oracle_flip": int(deployed_oracle[offset]) == teacher,
                        }
                    )
                rows.append(row)
    positive_rows = [row for row in rows if row["population"] == "positive"]
    negative_rows = [row for row in rows if row["population"] == "negative"]
    labels = torch.tensor(
        [int(GateLabel.POSITIVE)] * len(positive_rows) + [int(GateLabel.NEGATIVE)] * len(negative_rows)
    )
    probabilities = torch.tensor(
        [float(row["gate_unclamped"]) for row in (*positive_rows, *negative_rows)]
    )
    gate = gate_classification(probabilities, labels)
    summary = {
        "seed": seed,
        "step": step,
        "pi_dir": _ratio_bootstrap(
            positive_rows,
            numerator="forced_trained_flip",
            denominator="forced_oracle_flip",
            seed=20260811 + seed + step,
        ),
        "pi_dep": _ratio_bootstrap(
            positive_rows,
            numerator="deployed_trained_flip",
            denominator="deployed_oracle_flip",
            seed=20270811 + seed + step,
        ),
        "gate": gate,
        "collateral_chi": sum(bool(row["collateral_change"]) for row in negative_rows) / len(negative_rows),
        "mean_direction_cosine": float(np.mean([row["direction_cosine"] for row in positive_rows])),
        "by_teachability_decile": {},
    }
    for decile in range(10):
        local = [row for row in positive_rows if int(row["teachability_decile"]) == decile]
        summary["by_teachability_decile"][str(decile)] = {
            "rows": len(local),
            "pi_dir": sum(bool(row["forced_trained_flip"]) for row in local)
            / max(1, sum(bool(row["forced_oracle_flip"]) for row in local)),
            "pi_dep": sum(bool(row["deployed_trained_flip"]) for row in local)
            / max(1, sum(bool(row["deployed_oracle_flip"]) for row in local)),
        }
    return summary, rows


@torch.inference_mode()
def retention_read(
    *, module: nn.Module, material: Mapping[str, Any], device: str
) -> tuple[dict[str, Any], list[bool]]:
    records = material["retention"]["records"]
    hidden_all = material["retention"]["hidden4"]
    embedding = module.draft.tied_embedding.weight
    retained: list[bool] = []
    for start in range(0, len(records), 64):
        stop = min(len(records), start + 64)
        local = records[start:stop]
        hidden4 = hidden_all[start:stop].to(device=device, dtype=torch.float32)
        positions = torch.tensor([int(row["prediction_position"]) for row in local], device=device)
        horizons = torch.tensor([int(row["horizon"]) - 1 for row in local], device=device)
        index = torch.arange(stop - start, device=device)
        components = _model_components(module, hidden4, positions)
        base = hidden4[index, horizons]
        deployed = components["deployed_hidden"][index, horizons]
        base_token, deployed_token = _top1(torch.cat([base, deployed], dim=0), embedding).cpu().split(stop - start)
        expected = torch.tensor([int(row["student_top1"]) for row in local])
        if not torch.equal(base_token, expected):
            raise RuntimeError("P3.3 retention base-reader mismatch")
        retained.extend((base_token == deployed_token).tolist())
    return {"retained": sum(retained), "positions": len(retained), "retention": sum(retained) / len(retained)}, retained


def guardrail_read(
    *, retained: Sequence[bool], calibration: Mapping[str, Any]
) -> dict[str, Any]:
    differences = np.asarray(retained, dtype=np.float64) - 1.0
    mean = float(differences.mean())
    standard_error = float(differences.std(ddof=1) / math.sqrt(len(differences)))
    output = {"mean_difference_from_init": mean, "standard_error": standard_error}
    for name in ("tier_s", "tier_w"):
        rule = calibration[name]
        critical = float(student_t.ppf(1.0 - float(rule["one_sided_alpha"]), len(differences) - 1))
        upper = mean + critical * standard_error
        output[name] = {
            "upper_bound": upper,
            "decision_margin": float(rule["decision_margin_relative_to_init"]),
            "condition_met": upper < float(rule["decision_margin_relative_to_init"]),
        }
    return output


def _zero_loop_identity(module: nn.Module, cache: Mapping[str, Any], device: str) -> dict[str, Any]:
    hidden4 = cache["student_hidden"][:2].to(device)
    dummy = torch.zeros_like(hidden4[:, :1])
    hidden = torch.cat([dummy, hidden4], dim=1)
    previous = cache["base_log_probs"][:2].to(device)
    output = module(
        hidden=hidden,
        previous_logits=previous,
        steps=0,
        attention_mask=torch.ones(hidden.shape[:2], dtype=torch.bool, device=device),
        position_bucket=torch.zeros(2, dtype=torch.long, device=device),
        candidate_ids=cache["candidate_ids"][:2].long().to(device),
    )
    return {
        "hidden_bit_exact": bool(torch.equal(output.hidden, hidden)),
        "logits_bit_exact": bool(torch.equal(output.logits, previous)),
    }


def instrumentation_nonperturbation(
    *,
    module: nn.Module,
    cache: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    direction_index: Mapping[str, int],
    directions: torch.Tensor,
    device: str,
) -> dict[str, Any]:
    """A4: detached telemetry must not change outputs, RNG, precision, or kernel policy."""

    batch = _batch(
        cache=cache,
        records=records,
        direction_index=direction_index,
        directions=directions,
        device=device,
    )
    module.eval()
    before_rng = rng_state()
    losses_a, metrics_a = _losses(module, batch)
    telemetry = {
        name: float(value.detach().float().mean())
        for name, value in metrics_a.items()
        if isinstance(value, torch.Tensor) and value.numel()
    }
    after_telemetry_rng = rng_state()
    losses_b, metrics_b = _losses(module, batch)
    after_repeat_rng = rng_state()
    loss_exact = all(torch.equal(losses_a[name], losses_b[name]) for name in losses_a)
    metric_exact = all(
        torch.equal(value, metrics_b[name])
        for name, value in metrics_a.items()
        if isinstance(value, torch.Tensor) and name in metrics_b
    )
    rng_exact = (
        object_digest(before_rng["python"])
        == object_digest(after_telemetry_rng["python"])
        == object_digest(after_repeat_rng["python"])
        and object_digest(before_rng["numpy"])
        == object_digest(after_telemetry_rng["numpy"])
        == object_digest(after_repeat_rng["numpy"])
        and torch.equal(before_rng["torch"], after_telemetry_rng["torch"])
        and torch.equal(before_rng["torch"], after_repeat_rng["torch"])
    )
    if torch.cuda.is_available():
        rng_exact = rng_exact and all(
            torch.equal(before, after_telemetry)
            and torch.equal(before, after_repeat)
            for before, after_telemetry, after_repeat in zip(
                before_rng["cuda"], after_telemetry_rng["cuda"], after_repeat_rng["cuda"]
            )
        )
    result = {
        "loss_bit_exact": loss_exact,
        "metric_bit_exact": metric_exact,
        "rng_stream_exact": rng_exact,
        "default_dtype": str(torch.get_default_dtype()),
        "autocast_enabled": bool(torch.is_autocast_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "telemetry_fields": sorted(telemetry),
    }
    result["passed"] = bool(loss_exact and metric_exact and rng_exact)
    return result


def _flow_and_writes(
    *, module: nn.Module, hidden4: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    dummy = torch.zeros_like(hidden4[:, :1])
    hidden = torch.cat([dummy, hidden4], dim=1)
    attention = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
    attention[:, 0] = False
    scratch0 = module.initializer(hidden, attention)
    context = hidden.float().mean(dim=1)
    flow = module.flow(scratch0, context, steps=4)
    writes = []
    for loop_index, scratch in enumerate(flow.states[1:]):
        update = flow.updates[loop_index]
        control = module.control(
            scratch=scratch,
            previous=None,
            innovation_norm=update.float().square().mean(dim=-1).sqrt().mean(dim=1),
            student_entropy=hidden.new_zeros((hidden.shape[0],)),
            top2_margin=hidden.new_zeros((hidden.shape[0],)),
            position_bucket=position_buckets(positions),
        )
        bridge = module.bridge(
            h0=hidden,
            previous=hidden,
            scratch=scratch,
            control_state=control,
            loop_index=loop_index,
            active=True,
        )
        writes.append(bridge.hidden - hidden)
    return torch.stack(flow.states, dim=1), torch.stack(writes, dim=1)


def tier1_observatory_read(
    *, module: nn.Module, material: Mapping[str, Any], step: int, device: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = material["positive"]["records"][:OBSERVATORY_ROWS]
    hidden4 = material["positive"]["hidden4"][:OBSERVATORY_ROWS].to(device=device, dtype=torch.float32)
    positions = torch.tensor([int(row["prediction_position"]) for row in records], device=device)
    with torch.no_grad():
        states, writes = _flow_and_writes(module=module, hidden4=hidden4, positions=positions)

    # The lock names gradient-dot-write as an acceptance-facing diagnostic.  At
    # each scored position, use the exact gradient of the local teacher-versus-
    # student token margin with respect to the bridge write.  This avoids
    # inventing a task-generation graph in P3.3 while retaining a causal,
    # position-specific quantity in the estimator P3.3 actually owns.
    teacher = torch.tensor(
        [int(row["teacher_14b_top1"]) for row in records], device=device
    )
    student = torch.tensor([int(row["student_top1"]) for row in records], device=device)
    embedding = module.draft.tied_embedding.weight.detach().float()
    margin_gradient_at_position = embedding.index_select(0, teacher) - embedding.index_select(
        0, student
    )
    loss_gradient = torch.zeros_like(writes)
    batch_index = torch.arange(len(records), device=device)
    write_position = positions + 1  # bridge writes include the prepended control slot
    for loop_index in range(writes.shape[1]):
        loss_gradient[batch_index, loop_index, write_position] = margin_gradient_at_position
    metrics = observatory_metrics(
        states=states,
        writes=writes,
        loss_gradient=loss_gradient,
    )
    rows = observatory_event_rows(
        record_ids=[str(row["record_id"]) for row in records], metrics=metrics
    )
    for row in rows:
        row["step"] = int(step)
    summary = {
        "step": int(step),
        "rows": len(records),
        "events": len(rows),
        "gradient_definition": (
            "exact gradient of teacher_14b_top1 minus student_top1 local token margin "
            "with respect to the bridge write at the prediction position"
        ),
        "by_metric": {},
    }
    for name, value in metrics.items():
        tensor = value.detach().float().cpu()
        summary["by_metric"][name] = {
            "mean": float(tensor.mean()),
            "median": float(tensor.median()),
            "maximum": float(tensor.max()),
        }
    return summary, rows


@torch.inference_mode()
def astate_battery(
    *, module: nn.Module, material: Mapping[str, Any], seed: int, device: str
) -> dict[str, Any]:
    records = material["positive"]["records"][:ASTATE_ROWS]
    hidden4 = material["positive"]["hidden4"][:ASTATE_ROWS].to(device=device, dtype=torch.float32)
    positions = torch.tensor([int(row["prediction_position"]) for row in records], device=device)
    horizons = torch.tensor([int(row["horizon"]) - 1 for row in records], device=device)
    index = torch.arange(len(records), device=device)
    dummy = torch.zeros_like(hidden4[:, :1])
    hidden = torch.cat([dummy, hidden4], dim=1)
    attention = torch.ones(hidden.shape[:2], dtype=torch.bool, device=device)
    attention[:, 0] = False
    scratch0 = module.initializer(hidden, attention)
    context = hidden.float().mean(dim=1)
    flow = module.flow(scratch0, context, steps=4)
    selected_position = horizons + 1
    teacher = torch.tensor([int(row["teacher_14b_top1"]) for row in records], device=device)
    student = torch.tensor([int(row["student_top1"]) for row in records], device=device)
    embedding = module.draft.tied_embedding.weight.float()

    def margin(scratch: torch.Tensor, bypass: bool) -> torch.Tensor:
        update = scratch - flow.states[-2]
        control = module.control(
            scratch=scratch,
            previous=None,
            innovation_norm=update.float().square().mean(dim=-1).sqrt().mean(dim=1),
            student_entropy=hidden.new_zeros((hidden.shape[0],)),
            top2_margin=hidden.new_zeros((hidden.shape[0],)),
            position_bucket=position_buckets(positions),
        )
        if bypass:
            deployed = hidden
        else:
            deployed = module.bridge(
                h0=hidden,
                previous=hidden,
                scratch=scratch,
                control_state=control,
                loop_index=3,
                active=True,
            ).hidden
        selected = deployed[index, selected_position]
        return (selected * embedding.index_select(0, teacher)).sum(dim=-1) - (
            selected * embedding.index_select(0, student)
        ).sum(dim=-1)

    baseline = margin(flow.state, False)
    no_recurrence = margin(flow.state, True)
    denominator = baseline - no_recurrence
    modes = {}
    for offset, mode in enumerate(("zero", "norm_matched_random", "cross_example", "stale", "bypass")):
        altered, bypass = intervene_state(
            flow.state,
            mode=mode,
            seed=20260811 + seed + offset,
            stale_state=flow.states[-2],
        )
        intervention = margin(altered, bypass)
        numerator = baseline - intervention
        ratio = numerator / denominator
        finite = ratio[torch.isfinite(ratio)]
        modes[mode] = {
            "rows": len(records),
            "numerator_mean": float(numerator.mean()),
            "denominator_mean": float(denominator.mean()),
            "a_state_unclipped_mean_finite": float(finite.mean()) if finite.numel() else None,
            "a_state_unclipped_median_finite": float(finite.median()) if finite.numel() else None,
            "finite_rows": int(finite.numel()),
            "paired_from_same_cached_state": True,
            "ratio_clipped": False,
        }
    return {"rows": len(records), "modes": modes}


def run(args: argparse.Namespace) -> dict[str, Any]:
    seed = int(args.seed)
    if seed not in (0, 1):
        raise ValueError("P3.3 seed must be 0 or 1")
    preflight = json.loads(args.preflight_summary.read_text(encoding="utf-8"))
    calibration = json.loads(args.guardrail_calibration.read_text(encoding="utf-8"))
    if sha256_file(args.preflight_summary) != EXPECTED_PREFLIGHT_SHA256:
        raise RuntimeError("P3.3 e2 preflight summary SHA mismatch")
    if sha256_file(args.guardrail_calibration) != EXPECTED_CALIBRATION_SHA256:
        raise RuntimeError("P3.3 retention calibration SHA mismatch")
    if preflight["status"] != "complete_e2_assertions_final_training_may_follow_in_separate_target":
        raise RuntimeError("P3.3 e2 preflight is not complete")
    if not all(preflight["assertions"].values()) or not all(calibration["assertions"].values()):
        raise RuntimeError("P3.3 preflight assertions are not all green")
    if int(calibration["looks"]) != P33_LOOKS:
        raise RuntimeError("P3.3 calibration look count changed")
    if sha256_file(args.migrated_checkpoint) != EXPECTED_MIGRATED_SHA256[seed]:
        raise RuntimeError("P3.3 migrated checkpoint SHA mismatch")

    output_dir = args.output_dir
    private_dir = args.private_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    write_json(status_path, {"kind": "paper2_phase3_p33_status_v1", "seed": seed, "status": "loading_inputs"})

    records = read_jsonl(args.staged_labels)
    positives, negatives = _active_record_pools(records)
    direction_index, directions, direction_receipt = _direction_lookup(args.direction_cache)
    old, new, cache = build_option_b_cache(
        old_summary=args.old_summary,
        old_private=args.old_private,
        new_summary=args.new_summary,
        new_private=args.new_private,
        canonicalizer=args.canonicalizer,
        old_cache_path=args.old_cache,
        new_cache_path=args.new_cache,
    )
    del old, new
    sources = {
        "old": (args.old_summary, args.old_private),
        "new": (args.new_summary, args.new_private),
    }
    lm_head, lm_head_receipt = _lm_head(sources)
    audit_material = load_audit_material(
        positive_path=args.positive_audit,
        negative_path=args.negative_audit,
        retention_path=args.retention_panel,
        sources=sources,
    )
    module, checkpoint_receipt = _load_phase3_module(
        checkpoint=args.migrated_checkpoint,
        embedding_weight=lm_head,
        device=args.device,
    )
    if int(checkpoint_receipt["source_seed"]) != seed:
        raise RuntimeError("P3.3 migrated checkpoint seed mismatch")
    zero_loop = _zero_loop_identity(module, cache, args.device)
    if not all(zero_loop.values()):
        raise RuntimeError("P3.3 zero-loop identity failed")
    clamp = activate_operating_clamp(module)
    trainable = set_p33_trainable(module)
    trainable_count = sum(parameter.numel() for parameter in trainable.values())
    if trainable_count != EXPECTED_TRAINABLE:
        raise RuntimeError(f"P3.3 trainable count changed: {trainable_count}")
    frozen_before = tensor_digest(
        {name: value for name, value in module.named_parameters() if not value.requires_grad}
    )
    fixed_batches = fixed_audit_batches(positives, negatives, seed=seed)
    instrumentation = instrumentation_nonperturbation(
        module=module,
        cache=cache,
        records=fixed_batches[0],
        direction_index=direction_index,
        directions=directions,
        device=args.device,
    )
    if not instrumentation["passed"]:
        raise RuntimeError("P3.3 A4 instrumentation non-perturbation failed")
    share_history: list[dict[str, Any]] = []
    step0_share = directional_share_audit(
        module=module,
        cache=cache,
        batches=fixed_batches,
        direction_index=direction_index,
        directions=directions,
        device=args.device,
    )
    share_history.append({"step": 0, **step0_share})
    if step0_share["classification"] == "gross":
        result = {
            "kind": RUN_KIND,
            "status": "blocked_before_optimizer",
            "seed": seed,
            "reason": "step0_directional_gross_miss",
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "step0_directional_share": step0_share,
        }
        write_json(output_dir / "summary.json", result)
        return result

    resume_path = private_dir / "resume.pt"
    optimizer = torch.optim.AdamW(
        build_adamw_groups(module, weight_decay=P33_WEIGHT_DECAY),
        lr=0.0,
        betas=P33_ADAM_BETAS,
    )
    schedule_generator = torch.Generator().manual_seed(20260811 + seed)
    step = 0
    history: list[dict[str, Any]] = []
    schedule_hashes: list[str] = []
    gradient_norms: list[float] = []
    gradient_exceedances = 0
    previous_share_marginal = False
    previous_tier_s = False
    previous_tier_w = False
    observatory_history: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    stop_reason: str | None = None
    if resume_path.is_file():
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if saved.get("kind") != RUN_KIND or int(saved.get("seed", -1)) != seed:
            raise RuntimeError("P3.3 resume identity mismatch")
        for name, value in saved["trainable_state"].items():
            dict(module.named_parameters())[name].data.copy_(value.to(args.device))
        optimizer.load_state_dict(saved["optimizer_state"])
        step = int(saved["step"])
        history = list(saved["history"])
        share_history = list(saved["share_history"])
        schedule_hashes = list(saved["schedule_hashes"])
        gradient_norms = list(saved["gradient_norms"])
        gradient_exceedances = int(saved["gradient_exceedances"])
        previous_share_marginal = bool(saved["previous_share_marginal"])
        previous_tier_s = bool(saved["previous_tier_s"])
        previous_tier_w = bool(saved["previous_tier_w"])
        observatory_history = list(saved.get("observatory_history", []))
        warnings = list(saved["warnings"])
        stop_reason = saved.get("stop_reason")
        schedule_generator.set_state(saved["schedule_generator_state"])
        restore_rng(saved["rng_state"])
        print(f"p33_resume seed={seed} step={step}", flush=True)

    def save(archive: bool) -> dict[str, Any]:
        payload = {
            "kind": RUN_KIND,
            "seed": seed,
            "step": step,
            "target_steps": P33_TOTAL_STEPS,
            "trainable_state": {
                name: value.detach().cpu() for name, value in module.named_parameters() if value.requires_grad
            },
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "share_history": share_history,
            "schedule_hashes": schedule_hashes,
            "gradient_norms": gradient_norms,
            "gradient_exceedances": gradient_exceedances,
            "previous_share_marginal": previous_share_marginal,
            "previous_tier_s": previous_tier_s,
            "previous_tier_w": previous_tier_w,
            "observatory_history": observatory_history,
            "warnings": warnings,
            "stop_reason": stop_reason,
            "schedule_generator_state": schedule_generator.get_state(),
            "rng_state": rng_state(),
            "source_checkpoint": checkpoint_receipt,
            "preflight_summary_sha256": sha256_file(args.preflight_summary),
            "guardrail_calibration_sha256": sha256_file(args.guardrail_calibration),
        }
        digest = atomic_torch_save(payload, resume_path)
        if archive:
            destination = private_dir / f"checkpoint_step_{step:04d}.pt"
            shutil.copy2(resume_path, destination)
            if sha256_file(destination) != digest:
                raise RuntimeError("P3.3 checkpoint copy mismatch")
        return {"path": str(resume_path), "sha256": digest}

    if not history:
        audit, rows = audit_model(
            module=module,
            material=audit_material,
            direction_index=direction_index,
            directions=directions,
            seed=seed,
            step=0,
            device=args.device,
        )
        retention, retained = retention_read(module=module, material=audit_material, device=args.device)
        guardrail = guardrail_read(retained=retained, calibration=calibration)
        observatory, events = tier1_observatory_read(
            module=module, material=audit_material, step=0, device=args.device
        )
        observatory_history.append(observatory)
        history.append({"step": 0, "learning_rate": 0.0, "audit": audit, "retention": retention, "guardrail": guardrail, "observatory": observatory})
        write_jsonl(private_dir / "audit_rows_step_0000.jsonl", rows)
        write_jsonl(private_dir / "observatory_events_step_0000.jsonl", events)
        save(archive=True)

    while step < P33_TOTAL_STEPS and stop_reason is None:
        pos_indices = torch.randint(len(positives), (POSITIVE_PER_BATCH,), generator=schedule_generator)
        neg_indices = torch.randint(len(negatives), (NEGATIVE_PER_BATCH,), generator=schedule_generator)
        selected = [dict(positives[int(index)]) for index in pos_indices]
        selected.extend(dict(negatives[int(index)]) for index in neg_indices)
        permutation = torch.randperm(len(selected), generator=schedule_generator).tolist()
        selected = [selected[index] for index in permutation]
        schedule_hashes.append(
            hashlib.sha256("\n".join(str(row["record_id"]) for row in selected).encode("ascii")).hexdigest()
        )
        batch = _batch(
            cache=cache,
            records=selected,
            direction_index=direction_index,
            directions=directions,
            device=args.device,
        )
        module.train()
        losses, _metrics = _losses(module, batch)
        total = weighted_total(losses)
        if not bool(torch.isfinite(total)):
            stop_reason = "non_finite_loss"
            break
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        gradients = [parameter.grad for parameter in trainable.values() if parameter.grad is not None]
        if not gradients or any(not bool(gradient.isfinite().all()) for gradient in gradients):
            stop_reason = "non_finite_gradient"
            break
        raw_norm = math.sqrt(sum(float(gradient.detach().double().square().sum()) for gradient in gradients))
        median = float(np.median(gradient_norms[-100:])) if gradient_norms else raw_norm
        if len(gradient_norms) >= 100 and raw_norm > 10.0 * max(median, 1e-12):
            gradient_exceedances += 1
        else:
            gradient_exceedances = 0
        if gradient_exceedances >= 3:
            stop_reason = "relative_gradient_explosion"
            break
        gradient_norms.append(raw_norm)
        clip_events = {}
        for name, (parameters, ceiling) in clip_module_groups(module).items():
            active = [parameter for parameter in parameters if parameter.requires_grad and parameter.grad is not None]
            if active:
                before = float(torch.nn.utils.clip_grad_norm_(active, ceiling))
                clip_events[name] = {"norm_before": before, "ceiling": ceiling, "clipped": before > ceiling}
        next_step = step + 1
        learning_rate = learning_rate_at_step(next_step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.step()
        step = next_step

        if step % P33_LOOK_INTERVAL == 0:
            share = directional_share_audit(
                module=module,
                cache=cache,
                batches=fixed_batches,
                direction_index=direction_index,
                directions=directions,
                device=args.device,
            )
            share_history.append({"step": step, **share})
            share_marginal = share["classification"] == "marginal"
            if share["classification"] == "gross":
                stop_reason = "directional_share_gross_miss"
            elif share_marginal and previous_share_marginal:
                stop_reason = "directional_share_repeated_marginal_miss"
            previous_share_marginal = share_marginal
            audit, rows = audit_model(
                module=module,
                material=audit_material,
                direction_index=direction_index,
                directions=directions,
                seed=seed,
                step=step,
                device=args.device,
            )
            retention, retained = retention_read(module=module, material=audit_material, device=args.device)
            guardrail = guardrail_read(retained=retained, calibration=calibration)
            observatory, events = tier1_observatory_read(
                module=module, material=audit_material, step=step, device=args.device
            )
            observatory_history.append(observatory)
            tier_s = bool(guardrail["tier_s"]["condition_met"])
            tier_w = bool(guardrail["tier_w"]["condition_met"])
            if tier_s and previous_tier_s:
                stop_reason = stop_reason or "tier_s_token_retention"
            if tier_w and previous_tier_w:
                warnings.append({"step": step, "kind": "tier_w_token_retention", **guardrail["tier_w"]})
            previous_tier_s = tier_s
            previous_tier_w = tier_w
            history.append(
                {
                    "step": step,
                    "learning_rate": learning_rate,
                    "train_losses": {name: float(value.detach()) for name, value in losses.items()},
                    "train_total": float(total.detach()),
                    "clip_events": clip_events,
                    "audit": audit,
                    "retention": retention,
                    "guardrail": guardrail,
                    "directional_share": share,
                    "observatory": observatory,
                    "stop_reason": stop_reason,
                }
            )
            if step in (P33_TOTAL_STEPS // 2, P33_TOTAL_STEPS) or stop_reason:
                write_jsonl(private_dir / f"audit_rows_step_{step:04d}.jsonl", rows)
                write_jsonl(private_dir / f"observatory_events_step_{step:04d}.jsonl", events)
            checkpoint = save(archive=True)
            write_json(
                status_path,
                {
                    "kind": "paper2_phase3_p33_status_v1",
                    "seed": seed,
                    "status": "stopped" if stop_reason else "training",
                    "step": step,
                    "target_steps": P33_TOTAL_STEPS,
                    "pi_dir": audit["pi_dir"]["point"],
                    "pi_dep": audit["pi_dep"]["point"],
                    "retention": retention["retention"],
                    "stop_reason": stop_reason,
                    "checkpoint": checkpoint,
                },
            )
            print(
                f"p33_look seed={seed} step={step} pi_dir={audit['pi_dir']['point']:.6f} "
                f"pi_dep={audit['pi_dep']['point']:.6f} retention={retention['retention']:.6f} "
                f"share={share['classification']} stop={stop_reason}",
                flush=True,
            )

    frozen_after = tensor_digest(
        {name: value for name, value in module.named_parameters() if not value.requires_grad}
    )
    if frozen_after != frozen_before:
        raise RuntimeError("P3.3 frozen parameter lineage changed")
    checkpoint = save(archive=not (private_dir / f"checkpoint_step_{step:04d}.pt").is_file())
    final_astate = astate_battery(
        module=module, material=audit_material, seed=seed, device=args.device
    )
    result = {
        "kind": RUN_KIND,
        "status": "stopped" if stop_reason else "complete",
        "seed": seed,
        "step": step,
        "target_steps": P33_TOTAL_STEPS,
        "looks_completed": len(history) - 1,
        "expected_looks": P33_LOOKS,
        "stop_reason": stop_reason,
        "warnings": warnings,
        "history": history,
        "directional_share_history": share_history,
        "observatory_history": observatory_history,
        "a_state_intervention_battery": final_astate,
        "checkpoint": checkpoint,
        "source_checkpoint": checkpoint_receipt,
        "direction_cache": direction_receipt,
        "lm_head": lm_head_receipt,
        "operating_clamp": clamp,
        "zero_loop_identity": zero_loop,
        "instrumentation_nonperturbation": instrumentation,
        "trainable_parameters": trainable_count,
        "frozen_parameter_digest_before": frozen_before,
        "frozen_parameter_digest_after": frozen_after,
        "optimizer": {
            "family": "AdamW",
            "learning_rate": P33_LEARNING_RATE,
            "betas": list(P33_ADAM_BETAS),
            "weight_decay": P33_WEIGHT_DECAY,
            "warmup_steps": P33_WARMUP_STEPS,
            "batch_size": P33_BATCH_SIZE,
            "positive_per_batch": POSITIVE_PER_BATCH,
            "negative_per_batch": NEGATIVE_PER_BATCH,
        },
        "look_schedule": list(range(P33_LOOK_INTERVAL, P33_TOTAL_STEPS + 1, P33_LOOK_INTERVAL)),
        "schedule_sha256": hashlib.sha256("\n".join(schedule_hashes).encode("ascii")).hexdigest(),
        "preflight": {
            "summary_sha256": sha256_file(args.preflight_summary),
            "guardrail_calibration_sha256": sha256_file(args.guardrail_calibration),
            "positive_audit_sha256": sha256_file(args.positive_audit),
            "negative_audit_sha256": sha256_file(args.negative_audit),
            "retention_panel_sha256": sha256_file(args.retention_panel),
        },
        "task_level_capability_scoring": False,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    write_json(output_dir / "summary.json", result)
    write_json(
        status_path,
        {
            "kind": "paper2_phase3_p33_status_v1",
            "seed": seed,
            "status": result["status"],
            "step": step,
            "stop_reason": stop_reason,
            "summary": str(output_dir / "summary.json"),
        },
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--old_summary", type=Path, required=True)
    parser.add_argument("--old_private", type=Path, required=True)
    parser.add_argument("--new_summary", type=Path, required=True)
    parser.add_argument("--new_private", type=Path, required=True)
    parser.add_argument("--canonicalizer", type=Path, required=True)
    parser.add_argument("--old_cache", type=Path, required=True)
    parser.add_argument("--new_cache", type=Path, required=True)
    parser.add_argument("--staged_labels", type=Path, required=True)
    parser.add_argument("--positive_audit", type=Path, required=True)
    parser.add_argument("--negative_audit", type=Path, required=True)
    parser.add_argument("--retention_panel", type=Path, required=True)
    parser.add_argument("--preflight_summary", type=Path, required=True)
    parser.add_argument("--guardrail_calibration", type=Path, required=True)
    parser.add_argument("--direction_cache", type=Path, required=True)
    parser.add_argument("--migrated_checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--private_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"complete", "stopped", "blocked_before_optimizer"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
