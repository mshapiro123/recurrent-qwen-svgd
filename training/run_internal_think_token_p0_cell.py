"""Train one preregistered P0 internal-control pilot cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_identity import model_load_kwargs
from training.dataset import JsonlCausalDataset, collate_causal_batch
from training.internal_think_token_runtime import (
    install_internal_control_tokens,
    split_internal_control_token_rows,
)
from training.internal_think_token_spec import INTERNAL_CONTROL_TOKENS
from training.internal_think_token_t1 import (
    CandidateTrieContract,
    PILOT_STEPS,
    build_candidate_trie_contract,
    candidate_trie_edges,
    class_weights_from_ratio,
    gather_control_examples,
    locate_readout_positions,
    score_control_predictions,
)
from training.stability import (
    assert_finite_trainable_gradients,
    assert_finite_trainable_parameters,
)
from training.train_unfrozen_recurrent import build_optimizer, prepare_wrapper


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class PilotDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, path: str | Path, tokenizer: Any, *, max_length: int, max_loops: int) -> None:
        self.base = JsonlCausalDataset(
            path,
            tokenizer=tokenizer,
            max_length=max_length,
            max_train_loops=max_loops,
            train_on_prompt=False,
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = dict(self.base[index])
        row = self.base.rows[index]
        item["control_active"] = torch.tensor(bool(row.get("control_active", False)))
        item["required_depth"] = torch.tensor(int(row["depth"]), dtype=torch.long)
        item["row_index"] = torch.tensor(index, dtype=torch.long)
        return item


def collate_pilot(examples: list[dict[str, torch.Tensor]], *, pad_token_id: int) -> dict[str, torch.Tensor]:
    batch = collate_causal_batch(examples, pad_token_id=pad_token_id)
    for key in ("control_active", "required_depth", "row_index"):
        batch[key] = torch.stack([example[key] for example in examples]).view(-1)
    return batch


def seed_all(seed: int) -> torch.Generator:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    return torch.Generator(device="cpu").manual_seed(int(seed))


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def logical_trainable_summary(wrapper: Any, embedding: torch.nn.Parameter, control_ids: tuple[int, ...]) -> dict[str, int]:
    embedding_id = id(embedding)
    adapter_bridge = sum(
        parameter.numel()
        for parameter in wrapper.parameters()
        if parameter.requires_grad and id(parameter) != embedding_id
    )
    control_rows = len(control_ids) * int(embedding.shape[1])
    return {
        "adapter_bridge": adapter_bridge,
        "control_token_rows": control_rows,
        "forward_active": adapter_bridge + control_rows,
        "optimizer_dense_embedding_storage": int(embedding.numel()),
    }


def compact_checkpoint_state(wrapper: Any, embedding: torch.nn.Parameter, control_ids: tuple[int, ...]) -> dict[str, Any]:
    embedding_id = id(embedding)
    return {
        "trainable_state_dict": {
            name: parameter.detach().cpu()
            for name, parameter in wrapper.named_parameters()
            if parameter.requires_grad and id(parameter) != embedding_id
        },
        # ``embedding`` is already the compact three-row control parameter.
        # ``control_ids`` remain absolute vocabulary IDs for restore metadata.
        "control_token_rows": embedding.detach().cpu(),
        "control_token_ids": list(control_ids),
    }


@torch.no_grad()
def score_candidate_trie_batch(
    wrapper: Any,
    batch: dict[str, torch.Tensor],
    *,
    root_next_logits: torch.Tensor,
    answer_starts: torch.Tensor,
    candidate_contract: CandidateTrieContract,
    pad_token_id: int,
    device: str,
    max_loops: int,
) -> torch.Tensor:
    """Return exact length-normalized candidate log likelihoods.

    Shared trie prefixes are evaluated once per batch. The empty-prefix logits
    reuse the control-evaluation forward pass, so numeric candidates such as
    0-15 require only the distinct nonempty branch prefixes as extra forwards.
    """

    batch_size = int(batch["input_ids"].shape[0])
    candidate_scores = torch.zeros(
        (batch_size, len(candidate_contract.candidate_values)),
        dtype=torch.float32,
        device=device,
    )
    depths = batch["required_depth"].to(device=device, dtype=torch.long)
    row_indices = torch.arange(batch_size, device=device)

    for prefix, edges in candidate_trie_edges(candidate_contract).items():
        if not prefix:
            next_logits = root_next_logits
        else:
            sequences: list[torch.Tensor] = []
            sequence_lengths: list[int] = []
            prefix_tensor = torch.tensor(prefix, device=device, dtype=torch.long)
            for row_index in range(batch_size):
                prompt_end = int(answer_starts[row_index].item())
                sequence = torch.cat(
                    [batch["input_ids"][row_index, :prompt_end], prefix_tensor]
                )
                sequences.append(sequence)
                sequence_lengths.append(int(sequence.numel()))
            padded_length = max(sequence_lengths)
            prefix_input_ids = torch.full(
                (batch_size, padded_length),
                int(pad_token_id),
                device=device,
                dtype=torch.long,
            )
            prefix_attention_mask = torch.zeros_like(prefix_input_ids)
            for row_index, sequence in enumerate(sequences):
                length = sequence_lengths[row_index]
                prefix_input_ids[row_index, :length] = sequence
                prefix_attention_mask[row_index, :length] = 1
            prefix_output = wrapper(
                input_ids=prefix_input_ids,
                attention_mask=prefix_attention_mask,
                labels=None,
                max_loops=max_loops,
                use_cache=False,
                return_dict=True,
                return_loop_logits=True,
            )
            if prefix_output.loop_logits is None:
                raise AssertionError("candidate trie scoring requires loop logits")
            decision_positions = torch.tensor(
                [length - 1 for length in sequence_lengths],
                device=device,
                dtype=torch.long,
            )
            next_logits = prefix_output.loop_logits[
                row_indices,
                0,
                depths - 1,
                decision_positions,
            ].detach().clone()
            del prefix_output
        next_logprobs = torch.log_softmax(next_logits.float(), dim=-1)
        for candidate_index, next_token_id in edges:
            candidate_scores[:, candidate_index] += next_logprobs[:, next_token_id]

    lengths = torch.tensor(
        [len(tokens) for tokens in candidate_contract.candidate_token_ids],
        device=device,
        dtype=torch.float32,
    )
    return candidate_scores / lengths.unsqueeze(0)


@torch.no_grad()
def evaluate_pilot(
    wrapper: Any,
    loader: DataLoader,
    *,
    readout_token_id: int,
    continue_token_id: int,
    stop_token_id: int,
    candidate_contract: CandidateTrieContract,
    pad_token_id: int,
    device: str,
    max_loops: int,
) -> dict[str, Any]:
    wrapper.eval()
    control_rows: list[dict[str, Any]] = []
    answer_correct = 0
    answer_total = 0
    sequence_to_candidate = {
        sequence: candidate_index
        for candidate_index, sequence in enumerate(candidate_contract.candidate_token_ids)
    }
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        output = wrapper(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=None,
            max_loops=max_loops,
            use_cache=False,
            return_dict=True,
            return_loop_logits=True,
        )
        if output.loop_logits is None:
            raise AssertionError("pilot evaluation requires loop logits")
        positions = locate_readout_positions(
            batch["input_ids"],
            readout_token_id=readout_token_id,
            control_active=batch["control_active"],
        )
        answer_starts: list[int] = []
        for row_index in range(batch["input_ids"].shape[0]):
            depth = int(batch["required_depth"][row_index].item())
            position = int(positions[row_index].item())
            if position >= 0:
                two_class = output.loop_logits[
                    row_index,
                    0,
                    :max_loops,
                    position,
                ].index_select(
                    dim=-1,
                    index=torch.tensor(
                        [continue_token_id, stop_token_id],
                        device=device,
                        dtype=torch.long,
                    ),
                )
                predictions = two_class.float().argmax(dim=-1).tolist()
                control_rows.append(
                    {
                        "row_id": int(batch["row_index"][row_index].item()),
                        "depth": depth,
                        "predictions": predictions,
                    }
                )

            active_labels = batch["labels"][row_index].ne(-100).nonzero(as_tuple=False).view(-1)
            if active_labels.numel() == 0:
                raise AssertionError("pilot evaluation row has no answer label")
            first_answer = int(active_labels[0].item())
            if first_answer < 1:
                raise AssertionError("pilot answer cannot begin at token zero")
            answer_starts.append(first_answer)

        answer_start_tensor = torch.tensor(answer_starts, device=device, dtype=torch.long)
        row_indices = torch.arange(batch["input_ids"].shape[0], device=device)
        depths = batch["required_depth"].to(device=device, dtype=torch.long)
        root_next_logits = output.loop_logits[
            row_indices,
            0,
            depths - 1,
            answer_start_tensor - 1,
        ].detach().clone()
        del output
        candidate_scores = score_candidate_trie_batch(
            wrapper,
            batch,
            root_next_logits=root_next_logits,
            answer_starts=answer_start_tensor,
            candidate_contract=candidate_contract,
            pad_token_id=pad_token_id,
            device=device,
            max_loops=max_loops,
        )
        predicted_indices = candidate_scores.argmax(dim=-1).tolist()
        for row_index, predicted_index in enumerate(predicted_indices):
            active_labels = batch["labels"][row_index].ne(-100).nonzero(as_tuple=False).view(-1)
            target_sequence = tuple(
                int(value)
                for value in batch["labels"][row_index, active_labels].tolist()
            )
            if target_sequence not in sequence_to_candidate:
                raise AssertionError(
                    "Pilot answer tokens are not one of the verified candidate sequences: "
                    f"observed={list(target_sequence)}"
                )
            answer_correct += int(predicted_index == sequence_to_candidate[target_sequence])
            answer_total += 1
    scored = score_control_predictions(control_rows, max_loops=max_loops)
    scored.update(
        {
            "answer_correct": answer_correct,
            "answer_total": answer_total,
            "answer_accuracy": answer_correct / max(1, answer_total),
        }
    )
    return scored


def gradient_rms(parameters: list[torch.nn.Parameter]) -> float:
    total = 0
    sum_sq = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        total += grad.numel()
        sum_sq += float(grad.square().sum().item())
    return math.sqrt(sum_sq / total) if total else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell_json", required=True)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--pilot_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="default")
    parser.add_argument("--eval_batch_size", type=int, default=8)
    args = parser.parse_args()

    cell = json.loads(Path(args.cell_json).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        if prior.get("status") == "finished":
            print(f"pilot_cell_already_finished={cell['cell_id']}", flush=True)
            return 0

    seed = 9999
    loader_generator = seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_load_kwargs(args.dtype, args.attn_implementation),
    ).to(args.device)
    resize = install_internal_control_tokens(tokenizer, model)
    split_rows = split_internal_control_token_rows(
        model,
        original_vocab_size=resize.original_vocab_size,
    )
    config = {
        "layer_split": "6,18",
        "initial_halt_prob": 0.15,
        "bridge_projection_mode": "split",
        "adapter_dtype": "float32",
        "training_mode": "frozen_lora",
        "resume_lora": {"enabled": True, "rank": 16, "alpha": 32, "dropout": 0.0},
        "merge_lora_before_unfreeze": False,
        "train_auxiliary": {
            "bridge": True,
            "halting": False,
            "reentry_adapter": False,
            "latent": False,
        },
        "optimizer": "adamw",
        "learning_rate": 1e-5,
        "weight_decay": 0.0,
        "bridge_prelude_lr_multiplier": 10.0,
        "bridge_prelude_weight_decay": 0.0,
    }
    wrapper, setup = prepare_wrapper(model, config, device=args.device)
    split_embedding = wrapper.base_model.get_input_embeddings()
    embedding = split_embedding.control_rows
    control_ids = tuple(int(value) for value in resize.control_token_ids)
    frozen_prefix_sha_start = tensor_sha256(split_embedding.old_weight)
    embedding.requires_grad_(True)
    wrapper.set_trainable_modules_dtype(torch.float32)
    optimizer = build_optimizer(wrapper, config)
    assert_finite_trainable_parameters(wrapper, step=0)

    train_dataset = PilotDataset(args.train_jsonl, tokenizer, max_length=512, max_loops=8)
    pilot_dataset = PilotDataset(args.pilot_jsonl, tokenizer, max_length=512, max_loops=8)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        generator=loader_generator,
        collate_fn=partial(collate_pilot, pad_token_id=tokenizer.pad_token_id),
    )
    pilot_loader = DataLoader(
        pilot_dataset,
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        collate_fn=partial(collate_pilot, pad_token_id=tokenizer.pad_token_id),
    )
    if not pilot_dataset.base.rows:
        raise AssertionError("P0 pilot dataset is empty")
    candidate_contract = build_candidate_trie_contract(
        tokenizer,
        prompt=str(pilot_dataset.base.rows[0]["prompt"]),
        n_symbols=16,
    )
    for row in pilot_dataset.base.rows[1:]:
        row_contract = build_candidate_trie_contract(
            tokenizer,
            prompt=str(row["prompt"]),
            n_symbols=16,
        )
        if row_contract.candidate_token_ids != candidate_contract.candidate_token_ids:
            raise AssertionError(
                "P0 candidate tokenization differs across pilot prompts; batched exact "
                "trie scoring cannot share one contract"
            )
    continue_id = int(tokenizer.convert_tokens_to_ids(INTERNAL_CONTROL_TOKENS[0]))
    stop_id = int(tokenizer.convert_tokens_to_ids(INTERNAL_CONTROL_TOKENS[1]))
    readout_id = int(tokenizer.convert_tokens_to_ids(INTERNAL_CONTROL_TOKENS[2]))
    if (continue_id, stop_id, readout_id) != control_ids:
        raise AssertionError("control token ID order drifted after resize")

    class_weights = class_weights_from_ratio(
        stop_to_continue_ratio=float(cell["stop_to_continue_ratio"]),
        continue_count=28,
        stop_count=8,
    )
    class_weight_tensor = torch.tensor(class_weights, device=args.device, dtype=torch.float32)
    loss_lambda = float(cell["control_loss_lambda"])
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(train_loader)
    trace: list[dict[str, Any]] = []
    evaluations: dict[str, Any] = {}
    max_steps = 1500
    wrapper.train()
    for step in range(1, max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = {key: value.to(args.device) for key, value in batch.items()}
        output = wrapper(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            loop_labels=batch["loop_labels"],
            target_loop_counts=batch["target_loop_counts"],
            max_loops=8,
            use_cache=False,
            return_dict=True,
            return_loop_logits=True,
            loop_loss_mode="per_loop_labels",
            beta=0.0,
            halt_target_nll_weight=0.0,
        )
        if output.loss is None or output.loop_logits is None:
            raise AssertionError("P0 requires mechanism loss and loop logits")
        positions = locate_readout_positions(
            batch["input_ids"],
            readout_token_id=readout_id,
            control_active=batch["control_active"],
        )
        if bool(batch["control_active"].any()):
            control_logits, control_targets, _, _ = gather_control_examples(
                output.loop_logits,
                readout_positions=positions,
                required_depths=batch["required_depth"],
                control_active=batch["control_active"],
                continue_token_id=continue_id,
                stop_token_id=stop_id,
            )
            control_loss = F.cross_entropy(
                control_logits.float(),
                control_targets,
                weight=class_weight_tensor,
            )
        else:
            control_loss = output.loss.new_zeros(())
        total_loss = output.loss + loss_lambda * control_loss
        if not bool(torch.isfinite(total_loss)):
            raise FloatingPointError(f"nonfinite P0 loss at step {step}")
        total_loss.backward()
        assert_finite_trainable_gradients(wrapper, step)
        if step == 1 and loss_lambda > 0.0:
            row_grad = embedding.grad
            if row_grad is None or int(row_grad.count_nonzero().item()) == 0:
                raise AssertionError("control-token rows received no gradient at step one")
        grad_receipt = {
            "adapter_bridge_rms": gradient_rms(
                [parameter for parameter in wrapper.parameters() if id(parameter) != id(embedding)]
            ),
            "control_rows_rms": (
                0.0
                if embedding.grad is None
                else float(embedding.grad.float().square().mean().sqrt().item())
            ),
        }
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in wrapper.parameters() if parameter.requires_grad],
            0.5,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 100 == 0:
            record = {
                "step": step,
                "mechanism_loss": float(output.loss.detach().cpu().item()),
                "control_loss": float(control_loss.detach().cpu().item()),
                "total_loss": float(total_loss.detach().cpu().item()),
                "gradient_rms": grad_receipt,
            }
            trace.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
        if step in PILOT_STEPS:
            metrics = evaluate_pilot(
                wrapper,
                pilot_loader,
                readout_token_id=readout_id,
                continue_token_id=continue_id,
                stop_token_id=stop_id,
                candidate_contract=candidate_contract,
                pad_token_id=tokenizer.pad_token_id,
                device=args.device,
                max_loops=8,
            )
            evaluations[f"step_{step}"] = metrics
            compact = compact_checkpoint_state(wrapper, embedding, control_ids)
            torch.save(
                {
                    "kind": "paper2_t1_p0_compact_checkpoint",
                    "cell": cell,
                    "step": step,
                    "resize": resize.to_dict(),
                    "split_control_rows": split_rows.to_dict(),
                    **compact,
                },
                output_dir / f"p0_compact_step_{step}.pt",
            )
            print(json.dumps({"step": step, "evaluation": metrics}, sort_keys=True), flush=True)
            wrapper.train()

    frozen_prefix_sha_end = tensor_sha256(split_embedding.old_weight)
    if frozen_prefix_sha_end != frozen_prefix_sha_start:
        raise AssertionError("pretrained embedding rows changed during P0")
    summary = {
        "kind": "paper2_internal_think_token_p0_cell",
        "status": "finished",
        "registered_t1_training": False,
        "citable": False,
        "cell": cell,
        "seed": seed,
        "steps": max_steps,
        "control_token_resize": resize.to_dict(),
        "split_control_rows": split_rows.to_dict(),
        "class_weights": {"continue": class_weights[0], "stop": class_weights[1]},
        "candidate_trie_contract": {
            **candidate_contract.to_dict(),
            "score": "mean_token_log_probability_exact_sequence",
        },
        "setup": setup,
        "logical_trainable_parameters": logical_trainable_summary(wrapper, embedding, control_ids),
        "frozen_embedding_prefix_sha256_start": frozen_prefix_sha_start,
        "frozen_embedding_prefix_sha256_end": frozen_prefix_sha_end,
        "frozen_embedding_prefix_unchanged": True,
        "training_trace": trace,
        "evaluations": evaluations,
        "step_1500": evaluations["step_1500"],
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
