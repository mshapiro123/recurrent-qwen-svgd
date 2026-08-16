"""Extract and score the authorized KP-1 and amended T1 DEV state audit."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.eval_paper2_phase3_p31_references import (
    _chat_prompt,
    _generation_prompt,
    _mcq,
    _mcq_prompt,
)
from eval.eval_paper2_phase3_p34_task_inference import current_position_mask, position_buckets
from eval.eval_paper2_phase3_p34_task_trajectory import load_condition
from training.paper2_phase3_kp1_t1 import (
    KP1_RIDGE,
    T1_CEILINGS,
    T1_LAYER_TAPS,
    T1_LOOPS,
    assemble_core_cells,
    core_cell_mask,
    knowledge_gap_rows,
    ridge_embedding_probe,
    row_reindex,
    token_ranks,
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


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
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


def canonical_prompt(row: Mapping[str, Any], tokenizer: Any) -> str:
    if row["battery"] in {"arc_easy", "arc_challenge", "mmlu"}:
        question, choices, _answer = _mcq(row)
        return _mcq_prompt(question, choices)
    content, _cap = _generation_prompt(row)
    return _chat_prompt(tokenizer, content)


def canonical_answer(row: Mapping[str, Any]) -> str:
    if row["battery"] in {"arc_easy", "arc_challenge", "mmlu"}:
        _question, _choices, answer = _mcq(row)
        return answer
    return str(row["answer"]).strip()


def first_gold_token(
    row: Mapping[str, Any], tokenizer: Any, prompt: str
) -> tuple[int, str, bool]:
    answer = canonical_answer(row)
    suffix = f" {answer}"
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    complete = tokenizer(prompt + suffix, add_special_tokens=True)["input_ids"]
    stable = complete[: len(prompt_ids)] == prompt_ids and len(complete) > len(prompt_ids)
    if stable:
        return int(complete[len(prompt_ids)]), answer, True
    tokens = tokenizer(suffix, add_special_tokens=False)["input_ids"]
    if not tokens:
        raise RuntimeError(f"KP-1 answer has no tokens: {row['item_id']}")
    return int(tokens[0]), answer, False


def last_active(values: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    positions = current_position_mask(attention_mask)[1]
    return values[torch.arange(values.shape[0], device=values.device), positions]


def final_norm(base: Any, hidden: torch.Tensor) -> torch.Tensor:
    owner = getattr(base, "model", None)
    norm = getattr(owner, "norm", None)
    if norm is None:
        raise RuntimeError("KP-1 cannot locate the pinned Qwen final RMSNorm")
    return norm(hidden)


def rank_metrics(
    logits: torch.Tensor, target_ids: torch.Tensor, batteries: Sequence[str]
) -> dict[str, Any]:
    ranks = token_ranks(logits.float(), target_ids)
    logp = torch.log_softmax(logits.float(), dim=-1).gather(1, target_ids[:, None])[:, 0]
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, battery in enumerate(batteries):
        grouped[str(battery)].append(index)

    def summarize(indexes: Sequence[int]) -> dict[str, Any]:
        selected = torch.tensor(indexes, device=ranks.device, dtype=torch.long)
        local_rank = ranks[selected].float()
        local_logp = logp[selected]
        return {
            "rows": len(indexes),
            "top1": int(local_rank.eq(1).sum().item()),
            "top1_accuracy": float(local_rank.eq(1).float().mean().item()),
            "top10_accuracy": float(local_rank.le(10).float().mean().item()),
            "median_rank": float(local_rank.median().item()),
            "mean_log_probability": float(local_logp.mean().item()),
        }

    return {
        "pooled": summarize(list(range(len(batteries)))),
        "by_battery": {battery: summarize(indexes) for battery, indexes in sorted(grouped.items())},
    }


def rank_value_metrics(
    ranks: Sequence[int], log_probabilities: Sequence[float], batteries: Sequence[str]
) -> dict[str, Any]:
    if not (len(ranks) == len(log_probabilities) == len(batteries)):
        raise ValueError("rank-value metrics inputs must align")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, battery in enumerate(batteries):
        grouped[str(battery)].append(index)

    def summarize(indexes: Sequence[int]) -> dict[str, Any]:
        local_rank = torch.tensor([ranks[index] for index in indexes], dtype=torch.float32)
        local_logp = torch.tensor(
            [log_probabilities[index] for index in indexes], dtype=torch.float32
        )
        return {
            "rows": len(indexes),
            "top1": int(local_rank.eq(1).sum()),
            "top1_accuracy": float(local_rank.eq(1).float().mean()),
            "top10_accuracy": float(local_rank.le(10).float().mean()),
            "median_rank": float(local_rank.median()),
            "mean_log_probability": float(local_logp.mean()),
        }

    return {
        "pooled": summarize(list(range(len(ranks)))),
        "by_battery": {
            battery: summarize(indexes) for battery, indexes in sorted(grouped.items())
        },
    }


def cosine_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    if left.shape != right.shape or left.ndim < 2:
        raise ValueError("T1 cosine inputs must have matching row-first shapes")
    left_rows = left.float().reshape(left.shape[0], -1)
    right_rows = right.float().reshape(right.shape[0], -1)
    cosine = F.cosine_similarity(left_rows, right_rows, dim=-1)
    return {
        "mean": float(cosine.mean()),
        "median": float(cosine.median()),
        "minimum": float(cosine.min()),
        "p05": float(torch.quantile(cosine, 0.05)),
    }


def embedding_probe_metrics(
    *,
    features: torch.Tensor,
    target_ids: torch.Tensor,
    splits: Sequence[str],
    batteries: Sequence[str],
    embedding: torch.Tensor,
    ridge: float,
) -> dict[str, Any]:
    train = torch.tensor([value == "probe_train" for value in splits], device=features.device)
    evaluate = ~train
    if int(train.sum()) < 2 or not bool(evaluate.any()):
        raise RuntimeError("KP-1 probe split is not estimable")
    normalized_embedding = F.normalize(embedding.float(), dim=-1)
    targets = normalized_embedding[target_ids]
    prediction = ridge_embedding_probe(
        features[train], targets[train], features[evaluate], ridge=ridge
    )
    prediction = F.normalize(prediction, dim=-1)
    logits = prediction @ normalized_embedding.T
    eval_ids = target_ids[evaluate]
    eval_batteries = [battery for battery, keep in zip(batteries, evaluate.tolist()) if keep]
    probe = rank_metrics(logits, eval_ids, eval_batteries)
    intercept = F.normalize(targets[train].mean(dim=0, keepdim=True), dim=-1)
    intercept_logits = intercept.expand(eval_ids.shape[0], -1) @ normalized_embedding.T
    return {
        "fit": {
            "kind": "affine_ridge_to_frozen_output_embedding",
            "ridge": float(ridge),
            "train_rows": int(train.sum()),
            "eval_rows": int(evaluate.sum()),
            "optimizer_constructed": False,
        },
        "probe": probe,
        "intercept_only_control": rank_metrics(intercept_logits, eval_ids, eval_batteries),
    }


def state_cells(
    *,
    module: Any,
    hidden_states: Sequence[torch.Tensor],
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[tuple[int, float], torch.Tensor], dict[int, torch.Tensor]]:
    final_hidden = hidden_states[-1]
    prelude = module.initializer(final_hidden, attention_mask.bool())
    context = final_hidden.float().mean(dim=1)
    current = prelude
    recurrent = []
    controls: dict[int, torch.Tensor] = {}
    deployments: dict[tuple[int, float], torch.Tensor] = {}
    positions = current_position_mask(attention_mask)[1]
    batch_index = torch.arange(final_hidden.shape[0], device=final_hidden.device)
    current_hidden = final_hidden[batch_index, positions]
    compact = torch.stack([torch.zeros_like(current_hidden), current_hidden], dim=1)
    write_mask = torch.zeros((compact.shape[0], 2, 1), dtype=torch.bool, device=compact.device)
    write_mask[:, 1] = True
    projected_layers = []
    projection_dtype = module.initializer.value.weight.dtype
    for layer in T1_LAYER_TAPS:
        selected = last_active(hidden_states[layer], attention_mask)
        projected_layers.append(
            module.initializer.value(
                module.initializer.hidden_norm(selected.float()).to(projection_dtype)
            )
        )
    layer_cells = torch.stack(projected_layers, dim=1)
    for loop_index in range(T1_LOOPS):
        current, update, _magnitude, _ratio = module.flow.step(current, context, loop_index)
        recurrent.append(current)
        innovation = update.float().square().mean(dim=-1).sqrt().mean(dim=1)
        control = module.control(
            scratch=current,
            previous=None,
            innovation_norm=innovation,
            student_entropy=current.new_zeros((current.shape[0],)),
            top2_margin=current.new_zeros((current.shape[0],)),
            position_bucket=position_buckets(positions),
        )
        controls[loop_index + 1] = control
        for ceiling in T1_CEILINGS:
            module.bridge.set_gate_ceiling(float(ceiling))
            bridge = module.bridge(
                h0=compact,
                previous=compact,
                scratch=current,
                control_state=control,
                loop_index=loop_index,
                active=True,
                write_position_mask=write_mask,
            )
            deployed = bridge.hidden[:, 1]
            deployments[(loop_index + 1, float(ceiling))] = module.initializer.value(
                module.initializer.hidden_norm(deployed.float()).to(projection_dtype)
            ).unsqueeze(1)
    return assemble_core_cells(prelude, recurrent, layer_cells), deployments, controls


def pooled_state(core: torch.Tensor, deployment: torch.Tensor, *, loops: int) -> torch.Tensor:
    mask = core_cell_mask(loop_count=loops, batch=core.shape[0], device=core.device)
    cells = torch.cat([core, deployment], dim=1)
    full_mask = torch.cat(
        [mask, torch.ones((core.shape[0], 1), dtype=torch.bool, device=core.device)], dim=1
    )
    return (cells * full_mask.unsqueeze(-1)).sum(dim=1) / full_mask.sum(dim=1, keepdim=True)


def pooled_core(core: torch.Tensor, *, loops: int) -> torch.Tensor:
    mask = core_cell_mask(loop_count=loops, batch=core.shape[0], device=core.device)
    return (core * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--base_scores", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--model_cache", type=Path, required=True)
    parser.add_argument("--chain_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--private_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    chain = json.loads(args.chain_manifest.read_text(encoding="utf-8"))
    if lock.get("status") != "authorized_score_only" or not lock.get("locked_before_model_access"):
        raise RuntimeError("KP-1/T1 lock is not active")
    if manifest.get("lock_sha256") != sha256_file(args.lock):
        raise RuntimeError("KP-1/T1 manifest and lock differ")
    for key, path in (
        ("panel_sha256", args.panel),
        ("base_scores_sha256", args.base_scores),
        ("reference_scores_sha256", args.references),
    ):
        expected = lock["source_files"][key]
        if sha256_file(path) != expected:
            raise RuntimeError(f"KP-1/T1 source identity changed: {key}")

    panel = read_jsonl(args.panel)
    base_scores = read_jsonl(args.base_scores)
    references = read_jsonl(args.references)
    gap = knowledge_gap_rows(panel, references)
    gap_ids = [str(row["item_id"]) for row in gap]
    if gap_ids != manifest["kp1_gap_item_ids"]:
        raise RuntimeError("KP-1 gap population changed after manifest lock")
    assignments = manifest["probe_split"]

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model_cache, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        args.model_cache,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    base_before = parameter_fingerprint(base)
    embedding = base.get_output_embeddings().weight.detach()

    modules: dict[str, Any] = {}
    module_before: dict[str, str] = {}
    if set(chain["checkpoints"]) != set(lock["t1"]["checkpoints"]):
        raise RuntimeError("KP-1/T1 checkpoint labels differ from the locked four-condition set")
    for label, item in chain["checkpoints"].items():
        paths = item["paths"]
        endpoint_name = "p35" if label.startswith("p35_") else "p34"
        if item["sha256"][endpoint_name] != lock["t1"]["checkpoints"][label]:
            raise RuntimeError(f"KP-1/T1 endpoint differs from lock: {label}")
        module, receipts = load_condition(
            embedding_weight=base.get_input_embeddings().weight,
            migrated=Path(paths["migrated"]),
            migrated_sha256=item["sha256"]["migrated"],
            p33=Path(paths["p33"]),
            p33_sha256=item["sha256"]["p33"],
            i1=Path(paths["i1"]),
            i1_sha256=item["sha256"]["i1"],
            p34=Path(paths["p34"]),
            p34_sha256=item["sha256"]["p34"],
            p35=(Path(paths["p35"]) if paths.get("p35") else None),
            p35_sha256=item["sha256"].get("p35"),
        )
        modules[label] = module
        module_before[label] = parameter_fingerprint(module)
        item["load_receipts"] = receipts

    gap_lookup = {item_id: index for index, item_id in enumerate(gap_ids)}
    target_ids: list[int] = []
    target_answers: list[str] = []
    suffix_stable: list[bool] = []
    prompts = [canonical_prompt(row, tokenizer) for row in panel]
    prompt_by_id = {
        str(row["item_id"]): prompt for row, prompt in zip(panel, prompts)
    }
    for row in gap:
        prompt = prompt_by_id[str(row["item_id"])]
        token, answer, stable = first_gold_token(row, tokenizer, prompt)
        target_ids.append(token)
        target_answers.append(answer)
        suffix_stable.append(stable)

    base_taps: dict[int, list[torch.Tensor]] = {layer: [] for layer in T1_LAYER_TAPS}
    core_cache: dict[str, list[torch.Tensor]] = {label: [] for label in modules}
    deploy_cache: dict[str, dict[tuple[int, float], list[torch.Tensor]]] = {
        label: {(k, c): [] for k in range(1, 5) for c in T1_CEILINGS}
        for label in modules
    }
    recurrent_gap_features: dict[str, dict[int, list[torch.Tensor]]] = {
        label: {k: [] for k in range(1, 5)} for label in modules
    }
    native_ranks: dict[str, dict[int, list[int]]] = {
        label: {k: [] for k in range(1, 5)} for label in modules
    }
    native_logps: dict[str, dict[int, list[float]]] = {
        label: {k: [] for k in range(1, 5)} for label in modules
    }
    panel_ids = [str(row["item_id"]) for row in panel]
    for start in range(0, len(panel), args.batch_size):
        stop = min(len(panel), start + args.batch_size)
        encoded = tokenizer(
            prompts[start:stop], return_tensors="pt", padding=True, add_special_tokens=True
        ).to(device)
        with torch.inference_mode():
            output = base(
                **encoded, output_hidden_states=True, use_cache=False, return_dict=True
            )
        local_gap = [
            local for local, item_id in enumerate(panel_ids[start:stop]) if item_id in gap_lookup
        ]
        if local_gap:
            selected = torch.tensor(local_gap, device=device, dtype=torch.long)
            for layer in T1_LAYER_TAPS:
                base_taps[layer].append(last_active(output.hidden_states[layer], encoded["attention_mask"])[selected].cpu().to(torch.float16))
        for label, module in modules.items():
            with torch.inference_mode():
                core, deployments, controls = state_cells(
                    module=module,
                    hidden_states=output.hidden_states,
                    attention_mask=encoded["attention_mask"],
                )
            core_cache[label].append(core.cpu().to(torch.float16))
            for condition, values in deployments.items():
                deploy_cache[label][condition].append(values.cpu().to(torch.float16))
            if local_gap:
                for k in range(1, 5):
                    recurrent_start = 8 + (k - 1) * 8
                    recurrent_gap_features[label][k].append(
                        core[selected, recurrent_start : recurrent_start + 8].mean(dim=1).cpu().to(torch.float16)
                    )
                    # Native draft head is a descriptive readout, not the registered task path.
                    pooled = core[selected, recurrent_start : recurrent_start + 8].mean(dim=1).to(device)
                    hidden_update = module.draft.up[k - 1](F.silu(module.draft.down[k - 1](pooled)))
                    write_gate = torch.sigmoid(module.draft.write_gate(controls[k]))[:, k - 1]
                    base_logits = output.logits[
                        torch.arange(output.logits.shape[0], device=device),
                        current_position_mask(encoded["attention_mask"])[1],
                    ][selected].float()
                    logits = base_logits + write_gate[selected, None].float() * (
                        hidden_update.float() @ embedding.float().T
                    )
                    ids = torch.tensor(
                        [target_ids[gap_lookup[panel_ids[start + local]]] for local in local_gap],
                        device=device,
                    )
                    native_ranks[label][k].extend(token_ranks(logits, ids).cpu().tolist())
                    native_logps[label][k].extend(
                        torch.log_softmax(logits.float(), dim=-1)
                        .gather(1, ids[:, None])[:, 0]
                        .cpu()
                        .tolist()
                    )
        print(f"kp1_t1_extract_progress rows={stop}/{len(panel)}", flush=True)

    extracted_gap_ids = [item_id for item_id in panel_ids if item_id in gap_lookup]
    reindex = torch.tensor(
        row_reindex(extracted_gap_ids, gap_ids), dtype=torch.long
    )
    base_feature = {
        layer: torch.cat(parts)[reindex].to(device) for layer, parts in base_taps.items()
    }
    cores = {label: torch.cat(parts) for label, parts in core_cache.items()}
    deployments = {
        label: {condition: torch.cat(parts) for condition, parts in by_condition.items()}
        for label, by_condition in deploy_cache.items()
    }
    recurrent_features = {
        label: {k: torch.cat(parts)[reindex].to(device) for k, parts in by_loop.items()}
        for label, by_loop in recurrent_gap_features.items()
    }
    native_ranks = {
        label: {
            k: [values[index] for index in reindex.tolist()] for k, values in by_loop.items()
        }
        for label, by_loop in native_ranks.items()
    }
    native_logps = {
        label: {
            k: [values[index] for index in reindex.tolist()] for k, values in by_loop.items()
        }
        for label, by_loop in native_logps.items()
    }
    gap_target_ids = torch.tensor(target_ids, device=device, dtype=torch.long)
    gap_batteries = [str(row["battery"]) for row in gap]
    gap_splits = [str(assignments[item_id]) for item_id in gap_ids]

    logit_lens = {}
    probes = {}
    base_dtype = next(base.parameters()).dtype
    for layer, features in base_feature.items():
        lens_logits = base.get_output_embeddings()(
            final_norm(base, features.to(base_dtype))
        ).float()
        logit_lens[f"layer_{layer}"] = rank_metrics(lens_logits, gap_target_ids, gap_batteries)
        probes[f"layer_{layer}"] = embedding_probe_metrics(
            features=features,
            target_ids=gap_target_ids,
            splits=gap_splits,
            batteries=gap_batteries,
            embedding=embedding,
            ridge=float(lock["kp1"]["ridge"]),
        )
    for label, by_loop in recurrent_features.items():
        for k, features in by_loop.items():
            probes[f"{label}_loop_{k}"] = embedding_probe_metrics(
                features=features,
                target_ids=gap_target_ids,
                splits=gap_splits,
                batteries=gap_batteries,
                embedding=embedding,
                ridge=float(lock["kp1"]["ridge"]),
            )

    stability = {}
    for label, core in cores.items():
        core_reference = pooled_core(core.float(), loops=4)
        core_depth = {}
        deployment_depth = {}
        deployment_ceiling = {}
        for k in range(1, 5):
            core_depth[f"k{k}"] = cosine_metrics(
                core_reference, pooled_core(core.float(), loops=k)
            )
            deployment_depth[f"k{k}_ceiling_0.05"] = cosine_metrics(
                deployments[label][(4, 0.05)], deployments[label][(k, 0.05)]
            )
            for ceiling in T1_CEILINGS:
                deployment_ceiling[f"k{k}_ceiling_{ceiling:.2f}"] = cosine_metrics(
                    deployments[label][(k, 0.05)], deployments[label][(k, ceiling)]
                )
        stability[label] = {
            "core_depth_to_k4": core_depth,
            "deployment_depth_to_k4_at_ceiling_0p05": deployment_depth,
            "deployment_ceiling_to_0p05_at_matched_k": deployment_ceiling,
        }

    cross_checkpoint = {}
    for left, right in itertools.combinations(sorted(cores), 2):
        pair = f"{left}__vs__{right}"
        condition_metrics = {}
        for k in range(1, 5):
            for ceiling in T1_CEILINGS:
                condition_metrics[f"k{k}_ceiling_{ceiling:.2f}"] = cosine_metrics(
                    deployments[left][(k, ceiling)], deployments[right][(k, ceiling)]
                )
        cross_checkpoint[pair] = {
            "core_44_cell_fingerprint": cosine_metrics(cores[left], cores[right]),
            "deployment_fingerprints": condition_metrics,
        }

    args.private_dir.mkdir(parents=True, exist_ok=True)
    state_cache = {
        "kind": "paper2_phase3_t1_state_cache_v1",
        "item_ids": panel_ids,
        "core_cells": cores,
        "deployment_cells": deployments,
        "schema": lock["t1"],
    }
    cache_path = args.private_dir / "t1_state_cache.pt"
    torch.save(state_cache, cache_path)
    gap_rows = [
        {
            "item_id": item_id,
            "battery": battery,
            "probe_split": split,
            "gold_answer": answer,
            "gold_first_token_id": token,
            "context_stable_suffix": stable,
            "base_correct": False,
            "teacher_14b_correct": True,
        }
        for item_id, battery, split, answer, token, stable in zip(
            gap_ids, gap_batteries, gap_splits, target_answers, target_ids, suffix_stable
        )
    ]
    write_jsonl(args.private_dir / "kp1_gap_rows.jsonl", gap_rows)

    base_after = parameter_fingerprint(base)
    module_after = {label: parameter_fingerprint(module) for label, module in modules.items()}
    if base_before != base_after or module_before != module_after:
        raise RuntimeError("KP-1/T1 score-only extraction mutated frozen parameters")
    by_battery = defaultdict(int)
    for battery in gap_batteries:
        by_battery[battery] += 1
    summary = {
        "kind": "paper2_phase3_kp1_t1_summary_v1",
        "status": "complete_dev_score_only",
        "authority": lock["authority"],
        "manifest_sha256": sha256_file(args.manifest),
        "chain_manifest_sha256": sha256_file(args.chain_manifest),
        "kp1": {
            "gap_rows": len(gap),
            "gap_rows_by_battery": dict(sorted(by_battery.items())),
            "target": lock["kp1"]["primary_target"],
            "context_stable_suffix_rows": sum(suffix_stable),
            "generation_accuracy_on_gap_population": 0.0,
            "logit_lens": logit_lens,
            "linear_probes": probes,
            "native_recurrent_gold_token_ranks": {
                label: {
                    f"loop_{k}": rank_value_metrics(
                        values, native_logps[label][k], gap_batteries
                    )
                    for k, values in by_loop.items()
                }
                for label, by_loop in native_ranks.items()
            },
            "scope_warning": lock["kp1"]["scope_warning"],
        },
        "t1": {
            "status": "state_extraction_complete_retrieval_pending_teacher_fingerprints",
            "checkpoint_count": len(modules),
            "rows": len(panel),
            "core_cells": 44,
            "deployment_cell_separate": True,
            "stability": stability,
            "cross_checkpoint_fingerprints": cross_checkpoint,
            "state_cache": {
                "path": str(cache_path),
                "bytes": cache_path.stat().st_size,
                "sha256": sha256_file(cache_path),
            },
        },
        "assertions": {
            "dev_only": True,
            "confirm_scored": False,
            "eval_e_scored": False,
            "main_model_training": False,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "frozen_parameter_fingerprints_unchanged": True,
            "manifest_locked_before_model_access": True,
        },
        "fingerprints": {
            "base": base_before,
            "sidecars": module_before,
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
