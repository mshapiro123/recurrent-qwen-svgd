"""Audit whether per-loop chain losses reach the re-entry bridge.

This is a read-only diagnostic. It loads one checkpoint and one chain-labeled
training batch, computes each per-loop loss separately, records gradient norms
for the bridge/recurrent/coda groups, then finite-differences the bridge
prelude weights to distinguish an autograd cut from true functional
independence.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_mcq import load_recurrent_wrapper  # noqa: E402
from training.dataset import JsonlCausalDataset, collate_causal_batch  # noqa: E402


@dataclass(frozen=True)
class SignatureThresholds:
    grad_tol: float = 1e-12
    fd_tol: float = 1e-6
    tiny_grad_tol: float = 1e-8


def finite_float(value: torch.Tensor | float | int | None) -> float:
    if value is None:
        return 0.0
    if torch.is_tensor(value):
        value = float(value.detach().float().cpu())
    value = float(value)
    return value if math.isfinite(value) else 0.0


def tensor_rms(value: torch.Tensor | None) -> float:
    if value is None:
        return 0.0
    tensor = value.detach().float()
    if tensor.numel() == 0:
        return 0.0
    return finite_float(tensor.square().mean().sqrt())


def tensor_l2(value: torch.Tensor | None) -> float:
    if value is None:
        return 0.0
    tensor = value.detach().float()
    if tensor.numel() == 0:
        return 0.0
    return finite_float(torch.linalg.vector_norm(tensor.reshape(-1)))


def parameter_grad_stats(params: Iterable[torch.nn.Parameter]) -> dict[str, float]:
    grads = [param.grad.detach().float().reshape(-1) for param in params if param.grad is not None]
    if not grads:
        return {"grad_rms": 0.0, "grad_l2": 0.0, "grad_param_tensors": 0.0}
    flat = torch.cat(grads)
    return {
        "grad_rms": finite_float(flat.square().mean().sqrt()),
        "grad_l2": finite_float(torch.linalg.vector_norm(flat)),
        "grad_param_tensors": float(len(grads)),
    }


def bridge_slice_grad_stats(wrapper: torch.nn.Module) -> dict[str, float]:
    bridge = wrapper.bridge
    weight_grad = bridge.proj.weight.grad
    bias_grad = bridge.proj.bias.grad
    gate_grad = bridge.bridge_gate.grad
    norm_weight_grad = bridge.prelude_norm.weight.grad
    norm_bias_grad = bridge.prelude_norm.bias.grad
    hidden = int(getattr(bridge, "hidden_size", bridge.proj.weight.shape[0]))
    if weight_grad is not None and weight_grad.dim() == 2 and weight_grad.shape[1] == 2 * hidden:
        prelude = weight_grad[:, :hidden]
        state = weight_grad[:, hidden:]
    else:
        prelude = None
        state = weight_grad
    return {
        "bridge_prelude_weight_grad_rms": tensor_rms(prelude),
        "bridge_prelude_weight_grad_l2": tensor_l2(prelude),
        "bridge_state_weight_grad_rms": tensor_rms(state),
        "bridge_state_weight_grad_l2": tensor_l2(state),
        "bridge_bias_grad_rms": tensor_rms(bias_grad),
        "bridge_gate_grad_abs": finite_float(gate_grad.abs() if gate_grad is not None else 0.0),
        "bridge_prelude_norm_weight_grad_rms": tensor_rms(norm_weight_grad),
        "bridge_prelude_norm_bias_grad_rms": tensor_rms(norm_bias_grad),
    }


def bridge_weight_stats(wrapper: torch.nn.Module) -> dict[str, float]:
    bridge = wrapper.bridge
    weight = bridge.proj.weight.detach().float()
    hidden = int(getattr(bridge, "hidden_size", weight.shape[0]))
    if weight.dim() == 2 and weight.shape[1] == 2 * hidden:
        prelude = weight[:, :hidden]
        state = weight[:, hidden:]
        eye = torch.eye(hidden, dtype=weight.dtype, device=weight.device)
        state_identity = finite_float((state - eye).abs().max())
    else:
        prelude = weight.new_zeros(())
        state_identity = 0.0
    return {
        "bridge_gate": finite_float(bridge.bridge_gate),
        "bridge_prelude_weight_rms": tensor_rms(prelude),
        "bridge_prelude_weight_max_abs": finite_float(prelude.abs().max() if prelude.numel() else 0.0),
        "bridge_state_identity_max_abs_diff": state_identity,
        "bridge_bias_rms": tensor_rms(bridge.proj.bias),
    }


def recurrent_block_params(wrapper: torch.nn.Module) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for idx in range(wrapper.layer_split.prelude_end, wrapper.layer_split.recurrent_end):
        params.extend(wrapper.qwen.layers[idx].parameters())
    return params


def coda_params(wrapper: torch.nn.Module) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for idx in range(wrapper.layer_split.recurrent_end, len(wrapper.qwen.layers)):
        params.extend(wrapper.qwen.layers[idx].parameters())
    params.extend(wrapper.qwen.norm.parameters())
    params.extend(wrapper.lm_head.parameters())
    return params


def set_audit_requires_grad(wrapper: torch.nn.Module) -> None:
    for param in wrapper.parameters():
        param.requires_grad_(False)
    for param in wrapper.bridge.parameters():
        param.requires_grad_(True)
    for param in recurrent_block_params(wrapper):
        param.requires_grad_(True)
    for param in coda_params(wrapper):
        param.requires_grad_(True)


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def sequence_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if labels.shape != logits.shape[:2]:
        raise ValueError(f"labels/logits shape mismatch: labels={tuple(labels.shape)}, logits={tuple(logits.shape)}")
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    return torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )


def active_label_tokens(labels: torch.Tensor) -> int:
    return int(labels.ne(-100).sum().item())


def loop_logits_from_output(output: Any, loop_idx: int) -> torch.Tensor:
    if output.loop_logits is None:
        raise RuntimeError("wrapper did not return loop_logits")
    logits = output.loop_logits
    if logits.dim() != 5:
        raise RuntimeError(f"Expected loop_logits [batch, traj, loops, seq, vocab], got {tuple(logits.shape)}")
    return logits[:, 0, loop_idx, :, :]


def compute_loop_losses(
    wrapper: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    max_loops: int,
) -> list[dict[str, Any]]:
    output = wrapper(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=None,
        loop_labels=None,
        max_loops=max_loops,
        num_trajectories=1,
        particle_update_mode="none",
        use_cache=False,
        return_loop_logits=True,
        return_dict=True,
    )
    loop_labels = batch["loop_labels"]
    losses: list[dict[str, Any]] = []
    for loop_idx in range(max_loops):
        labels = loop_labels[:, loop_idx, :] if loop_idx < loop_labels.shape[1] else torch.full_like(batch["labels"], -100)
        active = active_label_tokens(labels)
        if active:
            loss = sequence_ce(loop_logits_from_output(output, loop_idx), labels)
        else:
            loss = None
        losses.append(
            {
                "loop": loop_idx + 1,
                "active_label_tokens": active,
                "loss": loss,
            }
        )
    return losses


def zero_grad(wrapper: torch.nn.Module) -> None:
    for param in wrapper.parameters():
        param.grad = None


def per_loop_gradient_matrix(
    wrapper: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    max_loops: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for loop_idx in range(max_loops):
        zero_grad(wrapper)
        losses = compute_loop_losses(wrapper, batch, max_loops=max_loops)
        loss_entry = losses[loop_idx]
        loss = loss_entry["loss"]
        row: dict[str, Any] = {
            "loop": loop_idx + 1,
            "active_label_tokens": int(loss_entry["active_label_tokens"]),
            "loss": finite_float(loss) if loss is not None else None,
        }
        if loss is None:
            row.update(
                {
                    "bridge_prelude_weight_grad_rms": 0.0,
                    "bridge_state_weight_grad_rms": 0.0,
                    "recurrent_block_grad_rms": 0.0,
                    "coda_grad_rms": 0.0,
                }
            )
            rows.append(row)
            continue
        loss.backward()
        row.update(bridge_slice_grad_stats(wrapper))
        row.update({f"recurrent_block_{key}": value for key, value in parameter_grad_stats(recurrent_block_params(wrapper)).items()})
        row.update({f"coda_{key}": value for key, value in parameter_grad_stats(coda_params(wrapper)).items()})
        rows.append(row)
    zero_grad(wrapper)
    return rows


def finite_difference_bridge_prelude(
    wrapper: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    max_loops: int,
    epsilon: float,
    seed: int,
) -> list[dict[str, Any]]:
    bridge = wrapper.bridge
    hidden = int(getattr(bridge, "hidden_size", bridge.proj.weight.shape[0]))
    weight = bridge.proj.weight
    if weight.dim() != 2 or weight.shape[1] != 2 * hidden:
        raise RuntimeError(f"Unexpected bridge projection shape: {tuple(weight.shape)}")
    generator = torch.Generator(device=weight.device)
    generator.manual_seed(int(seed))
    direction = torch.randn(weight[:, :hidden].shape, generator=generator, device=weight.device, dtype=weight.dtype)
    direction = direction / direction.float().square().mean().sqrt().clamp_min(1e-12).to(dtype=direction.dtype)

    with torch.no_grad():
        base = compute_loop_losses(wrapper, batch, max_loops=max_loops)
        base_values = [finite_float(entry["loss"]) if entry["loss"] is not None else None for entry in base]
        weight[:, :hidden].add_(float(epsilon) * direction)
        perturbed = compute_loop_losses(wrapper, batch, max_loops=max_loops)
        perturbed_values = [finite_float(entry["loss"]) if entry["loss"] is not None else None for entry in perturbed]
        weight[:, :hidden].sub_(float(epsilon) * direction)

    rows: list[dict[str, Any]] = []
    for idx, (base_loss, pert_loss) in enumerate(zip(base_values, perturbed_values)):
        if base_loss is None or pert_loss is None:
            delta = None
            delta_per_eps = None
        else:
            delta = float(pert_loss - base_loss)
            delta_per_eps = float(delta / float(epsilon))
        rows.append(
            {
                "loop": idx + 1,
                "base_loss": base_loss,
                "perturbed_loss": pert_loss,
                "delta": delta,
                "abs_delta": None if delta is None else abs(delta),
                "delta_per_epsilon": delta_per_eps,
            }
        )
    return rows


def interpret_gradient_signature(
    gradient_rows: list[dict[str, Any]],
    finite_difference_rows: list[dict[str, Any]],
    *,
    thresholds: SignatureThresholds = SignatureThresholds(),
) -> dict[str, Any]:
    fd_by_loop = {int(row["loop"]): row for row in finite_difference_rows}
    active_deep_rows = [row for row in gradient_rows if int(row.get("loop", 0)) >= 2 and int(row.get("active_label_tokens") or 0) > 0]
    issues: list[str] = []
    if not active_deep_rows:
        return {
            "status": "no_active_deep_loop_loss",
            "issues": ["no active loop >=2 labels in audited batch"],
            "deep_loops_analyzed": 0,
        }

    any_non_bridge_grad = False
    any_bridge_autograd = False
    any_bridge_tiny = False
    any_fd_dependence = False
    any_fd_missing = False
    per_loop: list[dict[str, Any]] = []
    for row in active_deep_rows:
        loop = int(row["loop"])
        bridge_grad = max(
            float(row.get("bridge_prelude_weight_grad_rms") or 0.0),
            float(row.get("bridge_state_weight_grad_rms") or 0.0),
            float(row.get("bridge_prelude_norm_weight_grad_rms") or 0.0),
            float(row.get("bridge_prelude_norm_bias_grad_rms") or 0.0),
        )
        non_bridge_grad = max(
            float(row.get("recurrent_block_grad_rms") or 0.0),
            float(row.get("coda_grad_rms") or 0.0),
        )
        fd = fd_by_loop.get(loop, {})
        fd_abs = fd.get("abs_delta")
        fd_dep = fd_abs is not None and float(fd_abs) > thresholds.fd_tol
        fd_miss = fd_abs is None or float(fd_abs) <= thresholds.fd_tol
        bridge_live = bridge_grad > thresholds.grad_tol
        bridge_tiny = thresholds.grad_tol < bridge_grad <= thresholds.tiny_grad_tol
        any_non_bridge_grad = any_non_bridge_grad or non_bridge_grad > thresholds.grad_tol
        any_bridge_autograd = any_bridge_autograd or bridge_live
        any_bridge_tiny = any_bridge_tiny or bridge_tiny
        any_fd_dependence = any_fd_dependence or fd_dep
        any_fd_missing = any_fd_missing or fd_miss
        per_loop.append(
            {
                "loop": loop,
                "bridge_grad_rms_max": bridge_grad,
                "non_bridge_grad_rms_max": non_bridge_grad,
                "finite_difference_abs_delta": fd_abs,
                "bridge_autograd_live": bridge_live,
                "finite_difference_dependence": fd_dep,
            }
        )

    if any_fd_dependence and not any_bridge_autograd:
        status = "autograd_cut_suspected"
        issues.append("finite_difference_dependence_with_zero_bridge_autograd")
    elif not any_fd_dependence and not any_bridge_autograd and any_non_bridge_grad:
        status = "structural_independence_or_decode_bypass_suspected"
        issues.append("deep_losses_do_not_functionally_depend_on_bridge_prelude")
    elif not any_non_bridge_grad:
        status = "whole_matrix_zero_or_loss_weighting_suspected"
        issues.append("no_non_bridge_gradients_from_active_deep_losses")
    elif any_bridge_autograd and any_fd_dependence:
        status = "graph_connected"
        if any_bridge_tiny:
            issues.append("bridge_gradients_connected_but_tiny")
        if any_fd_missing:
            issues.append("some_deep_loops_have_no_measurable_fd_dependence")
    else:
        status = "ambiguous_needs_review"
        issues.append("mixed_gradient_and_finite_difference_signature")

    return {
        "status": status,
        "issues": issues,
        "deep_loops_analyzed": len(active_deep_rows),
        "any_bridge_autograd": any_bridge_autograd,
        "any_finite_difference_dependence": any_fd_dependence,
        "any_non_bridge_grad": any_non_bridge_grad,
        "per_loop": per_loop,
    }


def static_source_audit() -> dict[str, Any]:
    source = (ROOT / "models/recurrent_wrapper.py").read_text(encoding="utf-8")
    loop_start = source.find("for loop_idx in range(max_loops):")
    loop_end = source.find("halt_probs_tensor = torch.stack", loop_start)
    loop_body = source[loop_start:loop_end] if loop_start >= 0 and loop_end > loop_start else ""
    return {
        "bridge_call_in_loop_body": "self.bridge(loop_input, prelude_hidden=prelude_hidden_states)" in loop_body,
        "loop_logits_after_coda": "coda_hidden" in loop_body and "loop_logits.append(logits)" in loop_body,
        "per_loop_labels_path": "loop_labels_flat[:, loop_idx, :]" in source,
        "loop_body_contains_detach": ".detach(" in loop_body or ".detach()" in loop_body,
        "loop_body_contains_data_access": ".data" in loop_body,
        "loop_body_contains_no_grad": "no_grad" in loop_body,
        "loop_body_contains_inplace_detach": "detach_" in loop_body,
    }


def select_audit_rows(
    source: Path,
    dest: Path,
    *,
    max_loops: int,
    max_scan_rows: int,
    row_id: str | None,
    min_active_loop_labels: int | None = None,
) -> dict[str, Any]:
    min_active = int(min_active_loop_labels or max_loops)
    if min_active < 1 or min_active > max_loops:
        raise ValueError("min_active_loop_labels must be in [1, max_loops]")
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected: list[dict[str, Any]] = []
    for idx, row in enumerate(rows[:max_scan_rows]):
        if row_id and str(row.get("id") or row.get("instance_id") or idx) != row_id:
            continue
        completions = list(row.get("loop_completions") or [])
        active = sum(item is not None for item in completions[:max_loops])
        if active >= min_active:
            selected.append(row)
            break
    if not selected:
        raise RuntimeError(
            f"No audit row with >={min_active} active loop_completions found in "
            f"first {max_scan_rows} rows of {source}"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in selected), encoding="utf-8")
    row = selected[0]
    return {
        "source": str(source),
        "audit_batch_jsonl": str(dest),
        "selected_rows": len(selected),
        "selected_id": str(row.get("id") or row.get("instance_id") or "0"),
        "selected_depth": row.get("depth") or row.get("synthetic_depth"),
        "min_active_loop_labels": min_active,
        "loop_completions": row.get("loop_completions"),
    }


def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def optimizer_bookkeeping(cfg: dict[str, Any], wrapper: torch.nn.Module) -> dict[str, Any]:
    train_aux = cfg.get("train_auxiliary", {}) if isinstance(cfg.get("train_auxiliary"), dict) else {}
    trainable = {name: bool(param.requires_grad) for name, param in wrapper.named_parameters() if name.startswith("bridge.")}
    return {
        "optimizer": cfg.get("optimizer"),
        "learning_rate": cfg.get("learning_rate"),
        "adamw_lr": cfg.get("adamw_lr"),
        "bridge_prelude_grad_multiplier": cfg.get("bridge_prelude_grad_multiplier"),
        "loop_loss_mode": cfg.get("loop_loss_mode"),
        "train_auxiliary_bridge": train_aux.get("bridge"),
        "bridge_params_require_grad": trainable,
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Gradient-Path Audit",
        "",
        f"status: `{summary['interpretation']['status']}`",
        f"issues: `{summary['interpretation']['issues']}`",
        "",
        "## Selected Batch",
        "",
        f"- row: `{summary['batch_selection']['selected_id']}`",
        f"- depth: `{summary['batch_selection']['selected_depth']}`",
        "",
        "## Per-Loop Gradient Matrix",
        "",
        "| loop | active_tokens | loss | bridge_prelude_rms | bridge_state_rms | recurrent_rms | coda_rms |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["gradient_matrix"]:
        loss = row.get("loss")
        lines.append(
            "| {loop} | {active} | {loss} | {bp:.3e} | {bs:.3e} | {rb:.3e} | {coda:.3e} |".format(
                loop=row["loop"],
                active=row["active_label_tokens"],
                loss="NA" if loss is None else f"{float(loss):.6f}",
                bp=float(row.get("bridge_prelude_weight_grad_rms") or 0.0),
                bs=float(row.get("bridge_state_weight_grad_rms") or 0.0),
                rb=float(row.get("recurrent_block_grad_rms") or 0.0),
                coda=float(row.get("coda_grad_rms") or 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Finite Difference",
            "",
            "| loop | base_loss | perturbed_loss | abs_delta | delta_per_epsilon |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["finite_difference"]:
        lines.append(
            "| {loop} | {base} | {pert} | {delta} | {dpe} |".format(
                loop=row["loop"],
                base="NA" if row.get("base_loss") is None else f"{float(row['base_loss']):.6f}",
                pert="NA" if row.get("perturbed_loss") is None else f"{float(row['perturbed_loss']):.6f}",
                delta="NA" if row.get("abs_delta") is None else f"{float(row['abs_delta']):.3e}",
                dpe="NA" if row.get("delta_per_epsilon") is None else f"{float(row['delta_per_epsilon']):.3e}",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config")
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--max_loops", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--adapter_dtype", default="float32")
    parser.add_argument("--attn_implementation", default="default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--fd_epsilon", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_scan_rows", type=int, default=2048)
    parser.add_argument("--row_id")
    parser.add_argument("--min_active_loop_labels", type=int)
    parser.add_argument("--grad_tol", type=float, default=1e-12)
    parser.add_argument("--fd_tol", type=float, default=1e-6)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    batch_info = select_audit_rows(
        Path(args.train_jsonl),
        out_dir / "audit_batch.jsonl",
        max_loops=args.max_loops,
        max_scan_rows=args.max_scan_rows,
        row_id=args.row_id,
        min_active_loop_labels=args.min_active_loop_labels,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = JsonlCausalDataset(
        batch_info["audit_batch_jsonl"],
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_train_loops=args.max_loops,
        train_on_prompt=bool(config.get("train_on_prompt", False)),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda rows: collate_causal_batch(rows, pad_token_id=tokenizer.pad_token_id),
    )
    batch = move_batch(next(iter(loader)), args.device)
    if "loop_labels" not in batch:
        raise RuntimeError("Selected audit batch has no loop_labels")

    wrapper_args = SimpleNamespace(
        model_name=args.model_name,
        split=args.split,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        device=args.device,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        adapter_dtype=args.adapter_dtype,
    )
    wrapper = load_recurrent_wrapper(wrapper_args, args.checkpoint)
    set_audit_requires_grad(wrapper)
    wrapper.eval()

    gradient_rows = per_loop_gradient_matrix(wrapper, batch, max_loops=args.max_loops)
    fd_rows = finite_difference_bridge_prelude(
        wrapper,
        batch,
        max_loops=args.max_loops,
        epsilon=args.fd_epsilon,
        seed=args.seed,
    )
    interpretation = interpret_gradient_signature(
        gradient_rows,
        fd_rows,
        thresholds=SignatureThresholds(grad_tol=args.grad_tol, fd_tol=args.fd_tol),
    )
    summary = {
        "kind": "gradient_path_audit",
        "status": interpretation["status"],
        "args": vars(args),
        "batch_selection": batch_info,
        "static_source_audit": static_source_audit(),
        "bridge_weight_stats": bridge_weight_stats(wrapper),
        "optimizer_bookkeeping": optimizer_bookkeeping(config, wrapper),
        "gradient_matrix": gradient_rows,
        "finite_difference": fd_rows,
        "interpretation": interpretation,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(out_dir / "summary.md", summary)
    print(json.dumps({"summary": str(out_dir / "summary.json"), "status": summary["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
