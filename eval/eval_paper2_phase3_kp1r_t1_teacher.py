"""Run KP-1R teacher-forced probes and gauge-invariant teacher fingerprints."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.eval_paper2_phase3_kp1_t1 import canonical_prompt, state_cells
from eval.eval_paper2_phase3_p34_task_trajectory import load_condition
from training.paper2_phase3_kp1r_t1_teacher import (
    KP1R_BOOTSTRAP_DRAWS,
    KP1R_BOOTSTRAP_SEED,
    KP1R_PRIMARY_BATTERIES,
    T1_TEACHER_PCA_DIM,
    answer_token_ids,
    battery_frequency_predictions,
    centered_gram,
    fit_orthogonal_procrustes,
    fit_teacher_pca,
    knowledge_margin_rows,
    linear_cka_from_grams,
    principal_angle_metrics_from_bases,
    probe_token_logits,
    sample_space_basis,
    stratified_alignment_split,
    summarize_margin,
    target_entropy_audit,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=True) + "\n")


def parameter_fingerprint(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def parameter_sentinel_fingerprint(module: torch.nn.Module) -> str:
    """Hash bounded samples across a large frozen model without copying all weights."""

    state = sorted(module.state_dict().items())
    indexes = sorted({0, len(state) // 4, len(state) // 2, 3 * len(state) // 4, len(state) - 1})
    digest = hashlib.sha256()
    for index in indexes:
        name, value = state[index]
        flat = value.detach().reshape(-1)
        sample = torch.cat([flat[:1024], flat[-1024:]]).cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(sample.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def fixed_prediction_permutation(
    *,
    predictions: Sequence[int],
    controls: Sequence[int],
    targets: Sequence[int],
    batteries: Sequence[str],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    observed = sum(knowledge_margin_rows(predictions, controls, targets)) / len(targets)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, battery in enumerate(batteries):
        grouped[str(battery)].append(index)
    generator = random.Random(int(seed))
    exceed = 0
    null_sum = 0.0
    for _ in range(int(draws)):
        permuted = list(int(value) for value in targets)
        for indexes in grouped.values():
            values = [permuted[index] for index in indexes]
            generator.shuffle(values)
            for index, value in zip(indexes, values):
                permuted[index] = value
        value = sum(knowledge_margin_rows(predictions, controls, permuted)) / len(permuted)
        null_sum += value
        exceed += int(value >= observed)
    return {
        "kind": "fixed-prediction_within-battery_eval-label_permutation",
        "draws": int(draws),
        "seed": int(seed),
        "observed_pooled_margin": float(observed),
        "null_mean_margin": float(null_sum / draws),
        "one_sided_p_value": float((1 + exceed) / (draws + 1)),
    }


def benjamini_hochberg(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 1.0
    total = len(ordered)
    for reverse_index in range(total - 1, -1, -1):
        name, p_value = ordered[reverse_index]
        rank = reverse_index + 1
        running = min(running, float(p_value) * total / rank)
        adjusted[name] = min(1.0, running)
    return adjusted


def aggregate_token_rows(
    *,
    token_records: Sequence[Mapping[str, Any]],
    predictions: Sequence[int],
    controls: Sequence[int],
    probe_log_probabilities: Sequence[float],
) -> tuple[list[dict[str, Any]], list[float], list[str]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(token_records):
        grouped[str(row["item_id"])].append(index)
    rows: list[dict[str, Any]] = []
    margins: list[float] = []
    batteries: list[str] = []
    for item_id, indexes in grouped.items():
        probe_accuracy = sum(
            int(int(predictions[index]) == int(token_records[index]["target_id"]))
            for index in indexes
        ) / len(indexes)
        control_accuracy = sum(
            int(int(controls[index]) == int(token_records[index]["target_id"]))
            for index in indexes
        ) / len(indexes)
        battery = str(token_records[indexes[0]]["battery"])
        rows.append(
            {
                "item_id": item_id,
                "battery": battery,
                "tokens": len(indexes),
                "probe_token_accuracy": probe_accuracy,
                "frequency_token_accuracy": control_accuracy,
                "margin": probe_accuracy - control_accuracy,
                "probe_mean_log_probability": sum(
                    float(probe_log_probabilities[index]) for index in indexes
                )
                / len(indexes),
                "native_base_mean_log_probability": sum(
                    float(token_records[index]["native_base_log_probability"])
                    for index in indexes
                )
                / len(indexes),
            }
        )
        margins.append(probe_accuracy - control_accuracy)
        batteries.append(battery)
    return rows, margins, batteries


def score_sequence_surface(
    *,
    name: str,
    features: torch.Tensor,
    token_records: Sequence[Mapping[str, Any]],
    embedding: torch.Tensor,
    ridge: float,
    permutation_draws: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train = [index for index, row in enumerate(token_records) if row["probe_split"] == "probe_train"]
    evaluate = [index for index, row in enumerate(token_records) if row["probe_split"] == "probe_eval"]
    train_targets = [int(token_records[index]["target_id"]) for index in train]
    eval_targets = [int(token_records[index]["target_id"]) for index in evaluate]
    train_batteries = [str(token_records[index]["battery"]) for index in train]
    eval_batteries = [str(token_records[index]["battery"]) for index in evaluate]
    logits = probe_token_logits(
        train_features=features[torch.tensor(train)],
        train_target_ids=train_targets,
        eval_features=features[torch.tensor(evaluate)],
        output_embedding=embedding,
        ridge=float(ridge),
    )
    predictions = logits.argmax(dim=-1).tolist()
    top10 = logits.topk(k=min(10, logits.shape[1]), dim=-1).indices.tolist()
    probe_log_probabilities = (
        torch.log_softmax(logits.float(), dim=-1)
        .gather(1, torch.tensor(eval_targets, dtype=torch.long)[:, None])[:, 0]
        .tolist()
    )
    controls = battery_frequency_predictions(train_targets, train_batteries, eval_batteries)
    eval_records = [dict(token_records[index]) for index in evaluate]
    row_reads, margins, row_batteries = aggregate_token_rows(
        token_records=eval_records,
        predictions=predictions,
        controls=controls,
        probe_log_probabilities=probe_log_probabilities,
    )
    for record, prediction, control, local_top10, probe_logp in zip(
        eval_records, predictions, controls, top10, probe_log_probabilities
    ):
        record.update(
            {
                "surface": name,
                "probe_prediction_id": int(prediction),
                "frequency_prediction_id": int(control),
                "probe_correct": bool(prediction == int(record["target_id"])),
                "frequency_correct": bool(control == int(record["target_id"])),
                "probe_top10_ids": [int(value) for value in local_top10],
                "probe_target_log_probability": float(probe_logp),
            }
        )
    return (
        {
            "surface": name,
            "train_token_positions": len(train),
            "eval_token_positions": len(evaluate),
            "eval_rows": len(row_reads),
            "knowledge_presence_margin": summarize_margin(
                margins,
                row_batteries,
                seed=KP1R_BOOTSTRAP_SEED,
                draws=KP1R_BOOTSTRAP_DRAWS,
            ),
            "label_permutation_control": fixed_prediction_permutation(
                predictions=predictions,
                controls=controls,
                targets=eval_targets,
                batteries=eval_batteries,
                draws=int(permutation_draws),
                seed=20260818,
            ),
            "row_summary": row_reads,
        },
        eval_records,
    )


def student_surfaces(core: torch.Tensor) -> dict[str, torch.Tensor]:
    result = {"prelude_pool": core[:, :8].mean(dim=1)}
    for loop in range(1, 5):
        start = 8 + (loop - 1) * 8
        result[f"loop_{loop}_pool"] = core[:, start : start + 8].mean(dim=1)
    for offset, layer in enumerate((6, 12, 18, 24)):
        result[f"layer_{layer}_cell"] = core[:, 40 + offset]
    return result


def projected_transport_metrics(
    *,
    student_fit: torch.Tensor,
    student_eval: torch.Tensor,
    teacher_fit_projected: torch.Tensor,
    teacher_eval_projected: torch.Tensor,
) -> dict[str, Any]:
    rotation, student_mean, teacher_mean = fit_orthogonal_procrustes(
        student_fit, teacher_fit_projected
    )
    transported = (student_eval.float() - student_mean.float()) @ rotation.float() + teacher_mean.float()
    similarity = F.normalize(transported, dim=-1) @ F.normalize(teacher_eval_projected.float(), dim=-1).T
    order = similarity.argsort(dim=-1, descending=True)
    targets = torch.arange(similarity.shape[0], device=similarity.device)[:, None]
    ranks = (order == targets).nonzero(as_tuple=False)[:, 1] + 1
    relative_error = torch.linalg.norm(transported - teacher_eval_projected.float()) / torch.linalg.norm(
        teacher_eval_projected.float()
    ).clamp_min(1e-12)
    return {
        "fit_rows": int(student_fit.shape[0]),
        "eval_rows": int(student_eval.shape[0]),
        "top1_retrieval_accuracy": float(ranks.eq(1).float().mean()),
        "top10_retrieval_accuracy": float(ranks.le(10).float().mean()),
        "mean_reciprocal_rank": float((1.0 / ranks.float()).mean()),
        "median_rank": float(ranks.float().median()),
        "relative_transport_error": float(relative_error),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--gap_rows", type=Path, required=True)
    parser.add_argument("--state_cache", type=Path, required=True)
    parser.add_argument("--chain_manifest", type=Path, required=True)
    parser.add_argument("--student_cache", type=Path, required=True)
    parser.add_argument("--teacher_cache", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--private_dir", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--permutation_draws", type=int, default=10_000)
    parser.add_argument("--teacher_batch_size", type=int, default=4)
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    if lock.get("status") != "locked_score_only" or not lock.get("locked_before_scoring"):
        raise RuntimeError("KP-1R/T1 teacher lock is not active")
    panel = read_jsonl(args.panel)
    gap_rows = read_jsonl(args.gap_rows)
    panel_by_id = {str(row["item_id"]): row for row in panel}
    if len(panel) != 1024 or len(gap_rows) != 329:
        raise RuntimeError("KP-1R/T1 teacher populations changed")
    if any(str(row.get("partition")) != "dev" for row in panel):
        raise RuntimeError("KP-1R/T1 teacher may read DEV only")

    student_spec = lock["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        student_spec["id"], revision=student_spec["revision"], cache_dir=args.student_cache
    )
    gap_panel = [panel_by_id[str(row["item_id"])] for row in gap_rows]
    target_sequences = [answer_token_ids(row, tokenizer) for row in gap_panel]
    target_entropy = target_entropy_audit(
        gap_panel,
        [values[0] for values in target_sequences],
        enforce_batteries=KP1R_PRIMARY_BATTERIES,
    )
    # The target audit is complete before either model is loaded.
    pre_model_path = args.output_dir / "pre_model_target_audit.json"
    write_json(
        pre_model_path,
        {
            "kind": "paper2_phase3_kp1r_t1_teacher_pre_model_audit_v1",
            "target_entropy": target_entropy,
            "confirm_scored": False,
            "eval_e_scored": False,
            "model_loaded": False,
            "optimizer_constructed": False,
        },
    )

    device = torch.device("cuda")
    base = AutoModelForCausalLM.from_pretrained(
        student_spec["id"],
        revision=student_spec["revision"],
        cache_dir=args.student_cache,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device).eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    base_before = parameter_fingerprint(base)
    chain = json.loads(args.chain_manifest.read_text(encoding="utf-8"))
    item = chain["checkpoint"]
    module, load_receipts = load_condition(
        embedding_weight=base.get_input_embeddings().weight,
        migrated=Path(item["paths"]["migrated"]),
        migrated_sha256=item["sha256"]["migrated"],
        p33=Path(item["paths"]["p33"]),
        p33_sha256=item["sha256"]["p33"],
        i1=Path(item["paths"]["i1"]),
        i1_sha256=item["sha256"]["i1"],
        p34=Path(item["paths"]["p34"]),
        p34_sha256=item["sha256"]["p34"],
        p35=Path(item["paths"]["p35"]),
        p35_sha256=item["sha256"]["p35"],
    )
    module_before = parameter_fingerprint(module)
    embedding = base.get_output_embeddings().weight.detach().cpu()
    feature_parts: dict[str, list[torch.Tensor]] = {
        **{f"substrate_layer_{layer}": [] for layer in (6, 12, 18, 24)},
        **{f"p35_seed_0_loop_{loop}_recurrent_cell_set": [] for loop in range(1, 5)},
    }
    token_records: list[dict[str, Any]] = []
    mbpp_native_rows: list[dict[str, Any]] = []
    for row_index, (gap_row, source_row, answer_ids) in enumerate(
        zip(gap_rows, gap_panel, target_sequences)
    ):
        battery = str(gap_row["battery"])
        prompt = canonical_prompt(source_row, tokenizer)
        prompt_ids = [int(value) for value in tokenizer(prompt, add_special_tokens=True)["input_ids"]]
        if battery == "mbpp":
            capped = answer_ids[:128]
            inputs = torch.tensor([prompt_ids + capped[:-1]], device=device)
            with torch.inference_mode():
                output = base(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False)
            positions = torch.arange(len(prompt_ids) - 1, len(prompt_ids) - 1 + len(capped), device=device)
            logits = output.logits[0, positions].float()
            targets = torch.tensor(capped, device=device)
            logp = torch.log_softmax(logits, dim=-1).gather(1, targets[:, None])[:, 0]
            mbpp_native_rows.append(
                {
                    "item_id": str(gap_row["item_id"]),
                    "tokens": len(capped),
                    "truncated_to_128": len(answer_ids) > 128,
                    "base_mean_log_probability": float(logp.mean()),
                }
            )
            continue
        for token_position, target_id in enumerate(answer_ids):
            input_ids = torch.tensor(
                [prompt_ids + answer_ids[:token_position]], device=device, dtype=torch.long
            )
            mask = torch.ones_like(input_ids)
            with torch.inference_mode():
                output = base(
                    input_ids=input_ids,
                    attention_mask=mask,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                core, _deployments, _controls = state_cells(
                    module=module,
                    hidden_states=output.hidden_states,
                    attention_mask=mask,
                )
            for layer in (6, 12, 18, 24):
                feature_parts[f"substrate_layer_{layer}"].append(
                    output.hidden_states[layer][0, -1].detach().cpu().to(torch.float16)
                )
            for loop in range(1, 5):
                start = 8 + (loop - 1) * 8
                feature_parts[f"p35_seed_0_loop_{loop}_recurrent_cell_set"].append(
                    core[0, start : start + 8].detach().cpu().to(torch.float16).reshape(-1)
                )
            token_records.append(
                {
                    "item_id": str(gap_row["item_id"]),
                    "battery": battery,
                    "probe_split": str(gap_row["probe_split"]),
                    "token_position": token_position,
                    "target_id": int(target_id),
                    "native_base_log_probability": float(
                        torch.log_softmax(output.logits[0, -1].float(), dim=-1)[int(target_id)]
                    ),
                }
            )
        if (row_index + 1) % 32 == 0:
            print(f"kp1r_teacher_forced_progress rows={row_index + 1}/329", flush=True)

    feature_tensors = {name: torch.stack(values) for name, values in feature_parts.items()}
    surface_reads: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    for name, features in feature_tensors.items():
        read, rows = score_sequence_surface(
            name=name,
            features=features,
            token_records=token_records,
            embedding=embedding,
            ridge=float(args.ridge),
            permutation_draws=int(args.permutation_draws),
        )
        surface_reads[name] = read
        prediction_rows.extend(rows)
        print(f"kp1r_surface_complete name={name}", flush=True)
    secondary_p = {
        name: read["label_permutation_control"]["one_sided_p_value"]
        for name, read in surface_reads.items()
        if name not in {"substrate_layer_24", "p35_seed_0_loop_4_recurrent_cell_set"}
    }
    secondary_q = benjamini_hochberg(secondary_p)
    for name, q_value in secondary_q.items():
        surface_reads[name]["secondary_fdr_q_value"] = q_value

    if base_before != parameter_fingerprint(base) or module_before != parameter_fingerprint(module):
        raise RuntimeError("KP-1R student scoring mutated frozen parameters")
    del module, base
    gc.collect()
    torch.cuda.empty_cache()

    teacher_spec = lock["teacher_fingerprints"]["teacher"]
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        teacher_spec["id"], revision=teacher_spec["revision"], cache_dir=args.teacher_cache
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_spec["id"],
        revision=teacher_spec["revision"],
        cache_dir=args.teacher_cache,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher_before = parameter_sentinel_fingerprint(teacher)
    teacher_parts = {layer: [] for layer in teacher_spec["layer_taps"]}
    teacher_prompts = [canonical_prompt(row, teacher_tokenizer) for row in panel]
    teacher_tokenizer.padding_side = "right"
    if teacher_tokenizer.pad_token_id is None:
        teacher_tokenizer.pad_token_id = teacher_tokenizer.eos_token_id
    for start in range(0, len(panel), int(args.teacher_batch_size)):
        stop = min(len(panel), start + int(args.teacher_batch_size))
        encoded = teacher_tokenizer(
            teacher_prompts[start:stop],
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        ).to(device)
        with torch.inference_mode():
            output = teacher(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        positions = encoded["attention_mask"].sum(dim=1) - 1
        batches = torch.arange(stop - start, device=device)
        for layer in teacher_parts:
            teacher_parts[layer].extend(
                output.hidden_states[int(layer)][batches, positions]
                .detach()
                .cpu()
                .to(torch.float16)
                .unbind(dim=0)
            )
        if stop % 32 == 0 or stop == len(panel):
            print(f"teacher_fingerprint_progress rows={stop}/1024", flush=True)
    if teacher_before != parameter_sentinel_fingerprint(teacher):
        raise RuntimeError("teacher fingerprint scoring mutated frozen parameters")
    teacher_features = {int(layer): torch.stack(values) for layer, values in teacher_parts.items()}
    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    state_cache = torch.load(args.state_cache, map_location="cpu", weights_only=False)
    item_ids = [str(value) for value in state_cache["item_ids"]]
    panel_ids = [str(row["item_id"]) for row in panel]
    if item_ids != panel_ids:
        raise RuntimeError("teacher fingerprints and student cache row order differ")
    batteries = [str(row["battery"]) for row in panel]
    split = stratified_alignment_split(
        item_ids,
        batteries,
        seed=int(lock["teacher_fingerprints"]["split_seed"]),
        fit_fraction=float(lock["teacher_fingerprints"]["alignment_fit_fraction"]),
    )
    fit_mask = torch.tensor([value == "alignment_fit" for value in split])
    eval_mask = ~fit_mask
    rank = int(lock["teacher_fingerprints"]["subspace_rank"])
    teacher_grams = {}
    teacher_bases = {}
    teacher_projected = {}
    for layer, features in teacher_features.items():
        gpu = features.to(device)
        teacher_grams[layer] = centered_gram(gpu)
        teacher_bases[layer] = sample_space_basis(gpu, rank=rank)
        mean, basis = fit_teacher_pca(gpu[fit_mask.to(device)], output_dim=T1_TEACHER_PCA_DIM)
        teacher_projected[layer] = {
            "fit": (gpu[fit_mask.to(device)].float() - mean) @ basis,
            "eval": (gpu[eval_mask.to(device)].float() - mean) @ basis,
        }

    comparison_rows: list[dict[str, Any]] = []
    transport_rows: list[dict[str, Any]] = []
    for checkpoint in ("p35_seed_0_ema_step_4400", "p35_seed_1_ema_step_4400"):
        core = state_cache["core_cells"][checkpoint].to(device).float()
        for cell_index in range(core.shape[1]):
            student = core[:, cell_index]
            gram = centered_gram(student)
            basis = sample_space_basis(student, rank=rank)
            for teacher_layer in sorted(teacher_features):
                comparison_rows.append(
                    {
                        "checkpoint": checkpoint,
                        "student_cell_index": cell_index,
                        "teacher_layer": teacher_layer,
                        "linear_cka": linear_cka_from_grams(
                            gram, teacher_grams[teacher_layer]
                        ),
                        "principal_angles": principal_angle_metrics_from_bases(
                            basis, teacher_bases[teacher_layer]
                        ),
                    }
                )
        for surface_name, student in student_surfaces(core).items():
            for teacher_layer in sorted(teacher_features):
                transport_rows.append(
                    {
                        "checkpoint": checkpoint,
                        "student_surface": surface_name,
                        "teacher_layer": teacher_layer,
                        **projected_transport_metrics(
                            student_fit=student[fit_mask.to(device)],
                            student_eval=student[eval_mask.to(device)],
                            teacher_fit_projected=teacher_projected[teacher_layer]["fit"],
                            teacher_eval_projected=teacher_projected[teacher_layer]["eval"],
                        ),
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.private_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.private_dir / "kp1r_teacher_forced_row_predictions.jsonl"
    teacher_path = args.private_dir / "teacher_fingerprint_states.pt"
    comparison_path = args.private_dir / "teacher_fingerprint_comparisons.jsonl"
    transport_path = args.private_dir / "teacher_fingerprint_transport.jsonl"
    write_jsonl(prediction_path, prediction_rows)
    torch.save(
        {
            "kind": "paper2_phase3_teacher_fingerprint_states_v1",
            "item_ids": item_ids,
            "teacher_features": teacher_features,
            "teacher_layers": list(sorted(teacher_features)),
        },
        teacher_path,
    )
    write_jsonl(comparison_path, comparison_rows)
    write_jsonl(transport_path, transport_rows)
    summary = {
        "kind": "paper2_phase3_kp1r_t1_teacher_summary_v1",
        "status": "complete_score_only",
        "authority": lock["authority"],
        "pre_model_target_audit_sha256": sha256_file(pre_model_path),
        "kp1r": {
            "target_entropy_audit": target_entropy,
            "primary_surfaces": {
                name: surface_reads[name]
                for name in ("substrate_layer_24", "p35_seed_0_loop_4_recurrent_cell_set")
            },
            "secondary_surfaces": {
                name: read
                for name, read in surface_reads.items()
                if name not in {"substrate_layer_24", "p35_seed_0_loop_4_recurrent_cell_set"}
            },
            "mbpp_exploratory": {
                "rows": len(mbpp_native_rows),
                "teacher_forced_native_base_log_probability": mbpp_native_rows,
                "excluded_from_primary": True,
            },
            "row_predictions": {
                "path": str(prediction_path),
                "rows": len(prediction_rows),
                "sha256": sha256_file(prediction_path),
            },
        },
        "teacher_fingerprints": {
            "rows": len(panel),
            "alignment_fit_rows": int(fit_mask.sum()),
            "alignment_eval_rows": int(eval_mask.sum()),
            "split_sha256": hashlib.sha256(("\n".join(split) + "\n").encode("utf-8")).hexdigest(),
            "comparison_rows": len(comparison_rows),
            "transport_rows": len(transport_rows),
            "state_cache": {"path": str(teacher_path), "sha256": sha256_file(teacher_path)},
            "comparisons": {"path": str(comparison_path), "sha256": sha256_file(comparison_path)},
            "transport": {"path": str(transport_path), "sha256": sha256_file(transport_path)},
            "raw_cross_instance_cosine_primary": False,
        },
        "load_receipts": load_receipts,
        "assertions": {
            "dev_only": True,
            "confirm_scored": False,
            "eval_e_scored": False,
            "target_entropy_audited_before_model_load": True,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "student_and_teacher_frozen": True,
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
