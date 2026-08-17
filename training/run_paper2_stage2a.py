"""Train one registered Stage 2A memory arm from frozen cached prefixes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM

from eval.eval_paper2_phase3_p34_task_trajectory import load_condition
from training.paper2_phase3_p31_completion import sha256_file
from training.paper2_stage2a_lock import assert_stage2a_training_authorized
from training.paper2_stage2a_objective import stage2a_flat_answer_objective
from training.paper2_stage2a_runtime import (
    STAGE2A_AMPLITUDE_HIGH,
    STAGE2A_AMPLITUDE_LOW,
    STAGE2A_BATCH_SIZE,
    STAGE2A_EMA_DECAY,
    STAGE2A_STEPS,
    Stage2AMemorySystem,
    assert_frozen_sidecar,
    canonical_fingerprint_query,
    exact_prefix_features,
    frozen_sidecar_digest,
    initialize_stage2a_ema,
    memory_augmented_logits,
    stage2a_learning_rate,
    tensor_digest,
    update_stage2a_ema,
)


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
CHECKPOINT_STEPS = {200, 400, 600, 800, 1_000, 1_200}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _manifest_core(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "battery",
        "item_id",
        "content_sha256",
        "owns_memory_slot",
        "owner_slot",
    )
    return [{field: row.get(field) for field in fields} for row in rows]


def _named_optimizer_state(
    optimizer: AdamW, parameters: Mapping[str, torch.nn.Parameter]
) -> dict[str, Any]:
    reverse = {id(parameter): name for name, parameter in parameters.items()}
    state: dict[str, Any] = {}
    for parameter, values in optimizer.state.items():
        name = reverse[id(parameter)]
        state[name] = {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in values.items()
        }
    groups = []
    for group in optimizer.param_groups:
        groups.append(
            {
                key: value
                for key, value in group.items()
                if key != "params"
            }
            | {"parameter_names": [reverse[id(parameter)] for parameter in group["params"]]}
        )
    return {"state_by_parameter_name": state, "parameter_groups": groups}


def _load_named_optimizer_state(
    optimizer: AdamW,
    payload: Mapping[str, Any],
    parameters: Mapping[str, torch.nn.Parameter],
) -> None:
    state = payload["state_by_parameter_name"]
    if set(state) != set(parameters):
        raise RuntimeError("Stage 2A optimizer parameter-name schema changed on resume")
    optimizer.state.clear()
    for name, parameter in parameters.items():
        optimizer.state[parameter] = {
            key: value.to(parameter.device) if torch.is_tensor(value) else value
            for key, value in state[name].items()
        }


class EpochSampler:
    def __init__(self, *, rows: int, seed: int) -> None:
        self.rows = int(rows)
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.order = torch.randperm(self.rows, generator=self.generator)
        self.cursor = 0

    def next(self, count: int) -> torch.Tensor:
        selected = []
        while sum(part.numel() for part in selected) < int(count):
            remaining = int(count) - sum(part.numel() for part in selected)
            stop = min(self.rows, self.cursor + remaining)
            selected.append(self.order[self.cursor:stop])
            self.cursor = stop
            if self.cursor == self.rows:
                self.order = torch.randperm(self.rows, generator=self.generator)
                self.cursor = 0
        return torch.cat(selected)

    def state_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "generator_state": self.generator.get_state(),
            "order": self.order,
            "cursor": self.cursor,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state["rows"]) != self.rows:
            raise RuntimeError("Stage 2A resume dataset size changed")
        self.generator.set_state(state["generator_state"])
        self.order = state["order"].long()
        self.cursor = int(state["cursor"])


def _batch_forward(
    *,
    row_indexes: Sequence[int],
    sidecar: Any,
    memory: Stage2AMemorySystem,
    geometry: Mapping[str, Any],
    teacher: Mapping[str, Any],
    features: Mapping[str, Any],
    owners: Sequence[Mapping[str, Any]],
    lm_head_weight: torch.Tensor,
    amplitude: float,
    device: torch.device,
    force_zero_memory: bool = False,
) -> tuple[Any, dict[str, float]]:
    scratch_rows = []
    context_rows = []
    current_rows = []
    layer6_rows = []
    literal_rows = []
    target_rows = []
    top_id_rows = []
    top_logit_rows = []
    example_rows = []
    memory_values = []
    compatibility = []
    retrieval_entropy = []
    retrieval_max = []
    answer_offsets = features["answer_offsets"].long()
    sequence_offsets = features["sequence_offsets"].long()
    teacher_offsets = teacher["row_offsets"].long()

    for local_index, row_index in enumerate(row_indexes):
        seq_start = int(sequence_offsets[row_index])
        seq_stop = int(sequence_offsets[row_index + 1])
        ans_start = int(answer_offsets[row_index])
        ans_stop = int(answer_offsets[row_index + 1])
        teach_start = int(teacher_offsets[row_index])
        teach_stop = int(teacher_offsets[row_index + 1])
        if ans_stop - ans_start != teach_stop - teach_start:
            raise RuntimeError("Stage 2A teacher/student answer offsets differ")
        hidden = features["final_hidden"][seq_start:seq_stop].to(device)
        positions = features["answer_prefix_positions"][ans_start:ans_stop].to(device)
        scratch, context, current = exact_prefix_features(sidecar, hidden, positions)
        scratch_rows.append(scratch)
        context_rows.append(context)
        current_rows.append(current)
        layer6 = features["layer6_queries"][ans_start:ans_stop].to(device)
        layer6_rows.append(layer6)
        token_ids = features["input_ids"][seq_start:seq_stop].long().to(device)
        literal_rows.append((token_ids, positions))
        target_rows.append(teacher["teacher_token_ids"][teach_start:teach_stop].to(device))
        top_id_rows.append(teacher["teacher_topk_token_ids"][teach_start:teach_stop].to(device))
        top_logit_rows.append(teacher["teacher_topk_logits"][teach_start:teach_stop].to(device))
        example_rows.append(
            torch.full((ans_stop - ans_start,), local_index, dtype=torch.long, device=device)
        )

        if memory.arm == "t3b":
            # One causal table pass yields every answer-prefix value without future access.
            all_values, _audit = memory.reader(token_ids[None, :])
            memory_values.append(all_values[0, positions.long()])
        else:
            query = canonical_fingerprint_query(
                layer6,
                student_mean=geometry["student_mean"].to(device),
                student_basis=geometry["student_basis"].to(device),
            )
            owner = owners[row_index]
            excluded = None
            if bool(owner["owns_memory_slot"]):
                excluded = torch.full(
                    (query.shape[0],),
                    int(owner["owner_slot"]),
                    dtype=torch.long,
                    device=device,
                )
            readout = memory.read_fingerprint(query, excluded_slot_indices=excluded)
            memory_values.append(readout.value)
            compatibility.append(readout.compatibility_gate.detach())
            retrieval_max.append(readout.slot_scores[:, 0].detach())
            entropy = -(
                readout.slot_weights.float()
                * readout.slot_weights.float().clamp_min(1e-12).log()
            ).sum(dim=-1)
            retrieval_entropy.append(entropy.detach())

    flat_memory_values = torch.cat(memory_values)
    if force_zero_memory:
        flat_memory_values = torch.zeros_like(flat_memory_values)
    logits, telemetry = memory_augmented_logits(
        sidecar=sidecar,
        memory_system=memory,
        scratch0=torch.cat(scratch_rows),
        contexts=torch.cat(context_rows),
        current_hidden=torch.cat(current_rows),
        memory_value=flat_memory_values,
        lm_head_weight=lm_head_weight,
        amplitude=float(amplitude),
    )
    objective = stage2a_flat_answer_objective(
        student_logits=logits,
        teacher_topk_token_ids=torch.cat(top_id_rows),
        teacher_topk_logits=torch.cat(top_logit_rows),
        teacher_token_ids=torch.cat(target_rows),
        example_index=torch.cat(example_rows),
        example_count=len(row_indexes),
    )
    summary = {
        "loss": float(objective.loss.detach()),
        "cross_entropy": float(objective.cross_entropy.detach()),
        "forward_kl": float(objective.forward_kl.detach()),
        "answer_positions": int(objective.answer_positions_per_example.sum()),
        "memory_write_rms": float(telemetry["memory_write_rms"].mean().detach()),
        "position_gate": float(telemetry["position_gate"].mean().detach()),
        "writeback_ratio": float(telemetry["writeback_ratio"].mean().detach()),
        "compatibility_gate": (
            float(torch.cat(compatibility).mean()) if compatibility else 1.0
        ),
        "retrieval_score_max": (
            float(torch.cat(retrieval_max).mean()) if retrieval_max else 0.0
        ),
        "retrieval_entropy": (
            float(torch.cat(retrieval_entropy).mean()) if retrieval_entropy else 0.0
        ),
    }
    return objective, summary


def _save_checkpoint(
    *,
    path: Path,
    step: int,
    arm: str,
    seed: int,
    memory: Stage2AMemorySystem,
    ema: Mapping[str, torch.Tensor],
    optimizer: AdamW,
    parameters: Mapping[str, torch.nn.Parameter],
    sampler: EpochSampler,
    amplitude_generator: torch.Generator,
    frozen_digest: str,
    source_hashes: Mapping[str, str],
    micro_batch_size: int,
) -> None:
    payload = {
        "kind": "paper2_stage2a_t3_checkpoint_v1",
        "step": int(step),
        "arm": arm,
        "seed": int(seed),
        "raw_state": {name: value.detach().cpu() for name, value in memory.state_dict().items()},
        "ema_state": {name: value.detach().cpu() for name, value in ema.items()},
        "optimizer_by_parameter_name": _named_optimizer_state(optimizer, parameters),
        "sampler": sampler.state_dict(),
        "amplitude_generator_state": amplitude_generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all(),
        "python_random_state": random.getstate(),
        "frozen_sidecar_digest": frozen_digest,
        "source_hashes": dict(source_hashes),
        "ema_decay": STAGE2A_EMA_DECAY,
        "effective_batch_size": STAGE2A_BATCH_SIZE,
        "micro_batch_size": int(micro_batch_size),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("t3a", "t3b", "shuffled", "random"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--teacher_lattice", type=Path, required=True)
    parser.add_argument("--population_manifest", type=Path, required=True)
    parser.add_argument("--owner_manifest", type=Path, required=True)
    parser.add_argument("--student_features", type=Path, required=True)
    parser.add_argument("--migrated", type=Path, required=True)
    parser.add_argument("--migrated_sha256", required=True)
    parser.add_argument("--p33", type=Path, required=True)
    parser.add_argument("--p33_sha256", required=True)
    parser.add_argument("--i1", type=Path, required=True)
    parser.add_argument("--i1_sha256", required=True)
    parser.add_argument("--p34", type=Path, required=True)
    parser.add_argument("--p34_sha256", required=True)
    parser.add_argument("--p35", type=Path, required=True)
    parser.add_argument("--p35_sha256", required=True)
    parser.add_argument("--model_cache", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--micro_batch_size", type=int, default=8)
    parser.add_argument("--control_escalation_authorized", action="store_true")
    args = parser.parse_args()

    lock = read_json(args.lock)
    assert_stage2a_training_authorized(lock)
    if (
        args.arm in ("shuffled", "random")
        and args.seed != 0
        and not args.control_escalation_authorized
    ):
        raise RuntimeError("Stage 2A control seed 1 requires the registered escalation trigger")
    source_paths = {
        "lock": args.lock,
        "geometry": args.geometry,
        "teacher_lattice": args.teacher_lattice,
        "population_manifest": args.population_manifest,
        "owner_manifest": args.owner_manifest,
        "student_features": args.student_features,
        "migrated": args.migrated,
        "p33": args.p33,
        "i1": args.i1,
        "p34": args.p34,
        "p35": args.p35,
    }
    source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    expected = {
        "geometry": lock["geometry_fit"]["artifact_sha256"],
        "teacher_lattice": lock["training_data"]["teacher_lattice_artifact_sha256"],
        "population_manifest": lock["training_data"]["population_manifest_sha256"],
        "owner_manifest": lock["training_data"]["memory_owner_manifest_sha256"],
        "student_features": lock["training_data"]["student_feature_cache_sha256"],
        "p35": lock["initialization"][f"seed_{args.seed}"]["sha256"],
    }
    for name, digest in expected.items():
        if source_hashes[name] != digest:
            raise RuntimeError(f"Stage 2A source SHA mismatch: {name}")

    population = read_jsonl(args.population_manifest)
    owners = read_jsonl(args.owner_manifest)
    if _manifest_core(population) != _manifest_core(owners):
        raise RuntimeError("Stage 2A population and owner manifest identities differ")
    if not 1 <= int(args.micro_batch_size) <= STAGE2A_BATCH_SIZE:
        raise ValueError("Stage 2A micro batch must be in 1..128")
    geometry = torch.load(args.geometry, map_location="cpu", weights_only=False)
    teacher = torch.load(args.teacher_lattice, map_location="cpu", weights_only=False)
    features = torch.load(args.student_features, map_location="cpu", weights_only=False)
    if len(population) != len(features["sequence_offsets"]) - 1:
        raise RuntimeError("Stage 2A cached feature row count changed")

    device = torch.device("cuda")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=REVISION,
        cache_dir=args.model_cache,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device).eval()
    lm_head_weight = base.get_output_embeddings().weight.detach()
    sidecar, checkpoint_receipts = load_condition(
        embedding_weight=lm_head_weight.detach().cpu(),
        migrated=args.migrated,
        migrated_sha256=args.migrated_sha256,
        p33=args.p33,
        p33_sha256=args.p33_sha256,
        i1=args.i1,
        i1_sha256=args.i1_sha256,
        p34=args.p34,
        p34_sha256=args.p34_sha256,
        p35=args.p35,
        p35_sha256=args.p35_sha256,
        control_reader="mean",
    )
    sidecar.bridge.set_gate_ceiling(0.02)
    for parameter in sidecar.parameters():
        parameter.requires_grad_(False)
    frozen_digest = frozen_sidecar_digest(sidecar)

    memory = Stage2AMemorySystem(
        arm=args.arm,
        memory_slots=int(lock["data_separation"]["memory_slots"]),
        memory_keys=geometry["memory_keys"].float(),
        teacher_values=geometry["teacher_values"].float(),
        seed=args.seed,
    ).to(device)
    parameters = memory.allowed_trainable()

    # Identity is tested before optimizer construction with the actual registered graph.
    first = [0]
    if not bool((memory.injection.gate.detach() == 0).all()):
        raise RuntimeError("Stage 2A memory injection gate is not zero at initialization")
    with torch.no_grad():
        zero_objective, _zero = _batch_forward(
            row_indexes=first,
            sidecar=sidecar,
            memory=memory,
            geometry=geometry,
            teacher=teacher,
            features=features,
            owners=owners,
            lm_head_weight=lm_head_weight,
            amplitude=0.05,
            device=device,
        )
        identity_objective, _identity = _batch_forward(
            row_indexes=first,
            sidecar=sidecar,
            memory=memory,
            geometry=geometry,
            teacher=teacher,
            features=features,
            owners=owners,
            lm_head_weight=lm_head_weight,
            amplitude=0.05,
            device=device,
            force_zero_memory=True,
        )
    if not all(
        torch.equal(left, right)
        for left, right in (
            (zero_objective.loss, identity_objective.loss),
            (zero_objective.cross_entropy, identity_objective.cross_entropy),
            (zero_objective.forward_kl, identity_objective.forward_kl),
        )
    ):
        raise RuntimeError("Stage 2A zero-gate identity is not bit exact")
    assert_frozen_sidecar(sidecar, frozen_digest)

    optimizer = AdamW(
        list(parameters.values()),
        lr=5e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )
    ema = initialize_stage2a_ema(parameters)
    sampler = EpochSampler(rows=len(population), seed=20_260_817 + 101 * args.seed)
    amplitude_generator = torch.Generator(device="cpu").manual_seed(
        20_260_817 + 10_007 * args.seed
    )
    start_step = 0
    if args.resume is not None:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume["arm"] != args.arm or int(resume["seed"]) != args.seed:
            raise RuntimeError("Stage 2A resume arm identity changed")
        if resume["source_hashes"] != source_hashes:
            raise RuntimeError("Stage 2A resume source hashes changed")
        if int(resume.get("micro_batch_size", -1)) != int(args.micro_batch_size):
            raise RuntimeError("Stage 2A resume micro-batch partition changed")
        memory.load_state_dict(resume["raw_state"])
        ema = {name: value.float().to(device) for name, value in resume["ema_state"].items()}
        _load_named_optimizer_state(optimizer, resume["optimizer_by_parameter_name"], parameters)
        sampler.load_state_dict(resume["sampler"])
        amplitude_generator.set_state(resume["amplitude_generator_state"])
        torch.set_rng_state(resume["torch_rng_state"])
        torch.cuda.set_rng_state_all(resume["cuda_rng_states"])
        random.setstate(resume["python_random_state"])
        start_step = int(resume["step"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "training_metrics.jsonl"
    for step in range(start_step + 1, STAGE2A_STEPS + 1):
        row_indexes = sampler.next(STAGE2A_BATCH_SIZE).tolist()
        amplitude = float(
            torch.empty((), dtype=torch.float32).uniform_(
                STAGE2A_AMPLITUDE_LOW,
                STAGE2A_AMPLITUDE_HIGH,
                generator=amplitude_generator,
            )
        )
        learning_rate = stage2a_learning_rate(step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        telemetry_totals: dict[str, float] = {}
        for micro_start in range(0, STAGE2A_BATCH_SIZE, int(args.micro_batch_size)):
            micro_rows = row_indexes[
                micro_start : micro_start + int(args.micro_batch_size)
            ]
            objective, micro_telemetry = _batch_forward(
                row_indexes=micro_rows,
                sidecar=sidecar,
                memory=memory,
                geometry=geometry,
                teacher=teacher,
                features=features,
                owners=owners,
                lm_head_weight=lm_head_weight,
                amplitude=amplitude,
                device=device,
            )
            if not bool(torch.isfinite(objective.loss)):
                raise RuntimeError("Stage 2A non-finite loss")
            weight = len(micro_rows) / STAGE2A_BATCH_SIZE
            (objective.loss * weight).backward()
            for name, value in micro_telemetry.items():
                telemetry_totals[name] = telemetry_totals.get(name, 0.0) + value * weight
        if any(
            parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
            for parameter in parameters.values()
        ):
            raise RuntimeError("Stage 2A non-finite gradient")
        optimizer.step()
        update_stage2a_ema(ema, parameters)
        assert_frozen_sidecar(sidecar, frozen_digest)
        metric = {
            "step": step,
            "arm": args.arm,
            "seed": args.seed,
            "learning_rate": learning_rate,
            "amplitude": amplitude,
            "effective_batch_size": STAGE2A_BATCH_SIZE,
            "micro_batch_size": int(args.micro_batch_size),
            **telemetry_totals,
        }
        with metrics_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(metric, sort_keys=True) + "\n")
        if step == 1 or step % 25 == 0:
            print("stage2a_train " + json.dumps(metric, sort_keys=True), flush=True)
        if step in CHECKPOINT_STEPS:
            checkpoint = args.output_dir / f"checkpoint_step_{step}.pt"
            _save_checkpoint(
                path=checkpoint,
                step=step,
                arm=args.arm,
                seed=args.seed,
                memory=memory,
                ema=ema,
                optimizer=optimizer,
                parameters=parameters,
                sampler=sampler,
                amplitude_generator=amplitude_generator,
                frozen_digest=frozen_digest,
                source_hashes=source_hashes,
                micro_batch_size=int(args.micro_batch_size),
            )
            write_json(
                args.output_dir / f"checkpoint_step_{step}.receipt.json",
                {
                    "kind": "paper2_stage2a_checkpoint_receipt_v1",
                    "arm": args.arm,
                    "seed": args.seed,
                    "step": step,
                    "path": str(checkpoint),
                    "sha256": sha256_file(checkpoint),
                    "raw_trainable_digest": tensor_digest(
                        {name: value for name, value in memory.state_dict().items()}
                    ),
                    "ema_trainable_digest": tensor_digest(ema),
                    "frozen_sidecar_digest": frozen_digest,
                    "optimizer_parameter_names": sorted(parameters),
                    "confirm_scored": False,
                    "eval_e_scored": False,
                },
            )

    final_checkpoint = args.output_dir / "checkpoint_step_1200.pt"
    summary = {
        "kind": "paper2_stage2a_training_summary_v1",
        "status": "complete_dev_evaluation_pending",
        "arm": args.arm,
        "seed": args.seed,
        "steps": STAGE2A_STEPS,
        "effective_batch_size": STAGE2A_BATCH_SIZE,
        "micro_batch_size": int(args.micro_batch_size),
        "checkpoint": str(final_checkpoint),
        "checkpoint_sha256": sha256_file(final_checkpoint),
        "source_hashes": source_hashes,
        "checkpoint_receipts": checkpoint_receipts,
        "trainable_parameters": sum(value.numel() for value in parameters.values()),
        "trainable_parameter_names": sorted(parameters),
        "frozen_sidecar_digest": frozen_digest,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
