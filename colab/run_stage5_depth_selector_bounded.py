"""Run the bounded S1/S2 learned-depth selector assessment on frozen N24 states."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoTokenizer

ROOT = Path(os.environ.get("STAGE5_ROOT", Path(__file__).resolve().parents[1]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.run_stage5_natural_surface_transfer import restore_checkpoint
from colab.stage5_publish_utils import publishable_artifact_paths
from eval.eval_mcq import load_recurrent_wrapper
from eval.eval_synthetic_depth_active_labels import (
    LETTER_SYMBOLS,
    candidates_for_row,
    prompt_for_row,
)
from models.halting import expected_loop_count
from training.depth_selector_bounded import (
    SELECTOR_PARAMETER_NAMES,
    assert_active_selector_gradient,
    assert_frozen_gradients_zero,
    configure_selector_only,
    evaluate_s1_gate,
    evaluate_s2_gate,
    frozen_parameter_hash,
    halting_weights_from_features,
    ponder_outcome_loss,
    supervised_depth_loss,
    summarize_selector_rows,
    truncated_geometric_prior,
)


RUN_SOURCE = ROOT / "outputs/stage5/stage5_n24_support12_rung_20260707_140139"
TRAIN_ROWS = RUN_SOURCE / "data/train_chain_mcq.jsonl"
HELDOUT64 = RUN_SOURCE / "data/test_chain_mcq_heldout64.jsonl"
N24_KEEPER_DRIVE = Path(
    "/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/"
    "stage5_n24_support12_rung_20260707_140139/"
    "anneal_to_outcome_final/unfrozen_recurrent_step_6000.pt"
)
N24_KEEPER_SHA256 = "898a259db2ab344ece4545e2910b051840e8408dbe4927f799e7cdb3cdd8c7dc"
DRIVE_ARTIFACT_ROOT = Path(
    "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/"
    "stage5_depth_selector_bounded_n24_step6000"
)
DRIVE_CHECKPOINT_ROOT = Path(
    "/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/"
    "stage5_depth_selector_bounded_n24_step6000"
)


def locked_spec() -> dict[str, Any]:
    return {
        "kind": "bounded_depth_selector_preregistration",
        "substrate": "N24_step6000",
        "source_checkpoint_sha256": N24_KEEPER_SHA256,
        "max_loops": 12,
        "train_depths": list(range(1, 13)),
        "heldout_rows_per_depth": 64,
        "steps_per_arm": 2000,
        "batch_size": 8,
        "optimizer": "adamw",
        "trainable_parameter_names": sorted(SELECTOR_PARAMETER_NAMES),
        "oracle_target_controls_frozen": True,
        "s1_name": "S1_supervised_depth_reading",
        "s1_min_correct_per_depth": 46,
        "s1_answer_delta_floor": -0.03,
        "s2_name": "S2_ponder_outcome",
        "s2_geometric_prior_mean": 6.0,
        "s2_beta": 0.02,
        "s2_strong_spearman_floor": 0.8,
        "s2_partial_spearman_floor": 0.3,
        "s2_answer_delta_from_s1_floor": -0.05,
        "canary_exemption": (
            "The frozen mechanism and all forced-T logits are cached before selector training. "
            "Only the halt projection, loop embedding, and loop bias are optimized; the selector "
            "chooses among immutable forced-T outputs and cannot alter computation at any fixed T."
        ),
        "do_not_claim": [
            "S1 is depth reading from an explicitly stated prompt field, not intelligent difficulty allocation.",
            "Learned halting on held-out hard reasoning is not established.",
        ],
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_for_cli(path: str | Path) -> str:
    value = Path(path)
    try:
        return value.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(value)


def publish(run_dir: Path, message: str) -> None:
    subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", "main"], cwd=ROOT, check=False)
    for path in publishable_artifact_paths(run_dir):
        if path.suffix == ".pt":
            continue
        subprocess.run(["git", "add", "-f", path_for_cli(path)], cwd=ROOT, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        return
    subprocess.run(["git", "commit", "-m", message], cwd=ROOT, check=True)
    pushed = subprocess.run(["git", "push", "origin", "main"], cwd=ROOT)
    if pushed.returncode:
        subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", "main"], cwd=ROOT, check=True)
        subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=True)


def _prompt_and_completion(row: dict[str, Any], *, split: str) -> tuple[str, str]:
    prompt = prompt_for_row(row, prediction_space="full_symbols", prompt_style="question_only")
    return prompt, f" {row['target']}"


def _single_suffix_token(tokenizer: Any, prompt: str, completion: str) -> int:
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    full_ids = tokenizer(prompt + completion, add_special_tokens=True)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids or len(full_ids) != len(prompt_ids) + 1:
        raise RuntimeError(
            "Depth-selector cache requires a one-token target completion with an unchanged prompt prefix. "
            f"prompt_tail={prompt[-30:]!r}, completion={completion!r}"
        )
    return int(full_ids[-1])


def _cache_signature(*, split: str, data_path: Path) -> dict[str, Any]:
    return {
        "kind": "depth_selector_frozen_feature_cache",
        "split": split,
        "source_checkpoint_sha256": N24_KEEPER_SHA256,
        "data_path": path_for_cli(data_path),
        "data_sha256": sha256_file(data_path),
        "model_name": os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
        "max_loops": 12,
        "prediction_space": "full_symbols",
        "prompt_style": "question_only",
        "value_prefix": "letter:",
        "logits_to_keep": 1,
    }


def _cache_is_valid(path: Path, signature: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception:
        return False
    return payload.get("signature") == signature


def extract_frozen_cache(
    wrapper: Any,
    tokenizer: Any,
    *,
    data_path: Path,
    split: str,
    cache_path: Path,
    device: str,
    extraction_batch_size: int,
) -> dict[str, Any]:
    signature = _cache_signature(split=split, data_path=data_path)
    if _cache_is_valid(cache_path, signature):
        print(f"reusing_frozen_cache={cache_path}", flush=True)
        return torch.load(cache_path, map_location="cpu")

    rows = read_jsonl(data_path)
    expected = 3072 if split == "train" else 768
    if len(rows) != expected:
        raise RuntimeError(f"Unexpected {split} row count: {len(rows)} != {expected}")
    depth_counts = {
        str(depth): sum(int(row["depth"]) == depth for row in rows)
        for depth in range(1, 13)
    }
    expected_per_depth = 256 if split == "train" else 64
    if depth_counts != {str(depth): expected_per_depth for depth in range(1, 13)}:
        raise RuntimeError(f"Unbalanced {split} depth rows: {depth_counts}")

    prompts: list[str] = []
    target_token_ids: list[int] = []
    target_indices: list[int] = []
    candidate_ids_by_row: list[list[int]] = []
    for row in rows:
        prompt, completion = _prompt_and_completion(row, split=split)
        prompts.append(prompt)
        target_token_ids.append(_single_suffix_token(tokenizer, prompt, completion))
        target = str(row["target"])
        target_indices.append(LETTER_SYMBOLS.index(target))
        if split == "heldout":
            candidates = candidates_for_row(
                row,
                prediction_space="full_symbols",
                value_prefix="letter:",
            )
            candidate_ids_by_row.append(
                [_single_suffix_token(tokenizer, prompt, candidates[symbol]) for symbol in LETTER_SYMBOLS[:24]]
            )

    feature_chunks: list[torch.Tensor] = []
    nll_chunks: list[torch.Tensor] = []
    prediction_chunks: list[torch.Tensor] = []
    wrapper.eval()
    with torch.no_grad():
        for start in range(0, len(rows), extraction_batch_size):
            end = min(start + extraction_batch_size, len(rows))
            encoded = tokenizer(
                prompts[start:end],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                add_special_tokens=True,
            ).to(device)
            output = wrapper(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                labels=None,
                max_loops=12,
                num_trajectories=1,
                particle_update_mode="none",
                use_cache=False,
                return_dict=True,
                return_loop_logits=True,
                return_loop_recurrent_states=True,
                logits_to_keep=1,
            )
            states = output.loop_recurrent_states[:, 0]
            mask = encoded["attention_mask"][:, None, :, None].to(dtype=states.dtype)
            pooled = (states * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
            logits = output.loop_logits[:, 0, :, -1, :].float()
            target_ids = torch.tensor(
                target_token_ids[start:end],
                device=logits.device,
                dtype=torch.long,
            ).view(-1, 1, 1).expand(-1, logits.shape[1], 1)
            target_nll = -torch.log_softmax(logits, dim=-1).gather(-1, target_ids).squeeze(-1)
            feature_chunks.append(pooled.detach().to(device="cpu", dtype=torch.bfloat16))
            nll_chunks.append(target_nll.detach().to(device="cpu", dtype=torch.float32))
            if split == "heldout":
                candidate_ids = torch.tensor(
                    candidate_ids_by_row[start:end],
                    device=logits.device,
                    dtype=torch.long,
                )
                candidate_logits = logits.gather(
                    -1,
                    candidate_ids[:, None, :].expand(-1, logits.shape[1], -1),
                )
                prediction_chunks.append(candidate_logits.argmax(dim=-1).detach().cpu())
            if start == 0 or end % 128 == 0 or end == len(rows):
                print(f"selector_cache_progress split={split} rows={end}/{len(rows)}", flush=True)

    payload = {
        "signature": signature,
        "row_ids": [str(row.get("id") or row.get("instance_id")) for row in rows],
        "depths": torch.tensor([int(row["depth"]) for row in rows], dtype=torch.long),
        "target_indices": torch.tensor(target_indices, dtype=torch.long),
        "pooled_features": torch.cat(feature_chunks, dim=0),
        "per_loop_target_nll": torch.cat(nll_chunks, dim=0),
        "forced_predictions": (
            torch.cat(prediction_chunks, dim=0)
            if prediction_chunks
            else None
        ),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    print(f"saved_frozen_cache={cache_path}", flush=True)
    return payload


def _copy_cache_to_drive(local_path: Path, drive_path: Path) -> None:
    drive_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_path, drive_path)
    print(f"backed_up_frozen_cache={drive_path}", flush=True)


def _selector_state(wrapper: Any) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in wrapper.named_parameters()
        if name in SELECTOR_PARAMETER_NAMES
    }


def _restore_selector_state(wrapper: Any, state: dict[str, torch.Tensor]) -> None:
    named = dict(wrapper.named_parameters())
    if set(state) != SELECTOR_PARAMETER_NAMES:
        raise RuntimeError("Selector reset state does not match the locked trainable set")
    with torch.no_grad():
        for name, tensor in state.items():
            named[name].copy_(tensor.to(device=named[name].device, dtype=named[name].dtype))


def _save_selector_checkpoint(
    wrapper: Any,
    path: Path,
    *,
    arm: str,
    steps: int,
    frozen_hash: str,
    source_checkpoint_sha256: str,
) -> dict[str, Any]:
    payload = {
        "kind": "bounded_depth_selector_checkpoint",
        "arm": arm,
        "steps": steps,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "frozen_parameter_hash": frozen_hash,
        "selector_state_dict": _selector_state(wrapper),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    drive = DRIVE_CHECKPOINT_ROOT / arm / path.name
    drive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, drive)
    return {
        "local_path": path_for_cli(path),
        "drive_path": str(drive),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _batch_indices(
    count: int,
    batch_size: int,
    *,
    steps: int,
    seed: int,
) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    output: list[torch.Tensor] = []
    while len(output) < steps:
        permutation = torch.randperm(count, generator=generator)
        for start in range(0, count, batch_size):
            chunk = permutation[start : start + batch_size]
            if chunk.numel() != batch_size:
                continue
            output.append(chunk)
            if len(output) == steps:
                break
    return output


def train_selector_arm(
    wrapper: Any,
    train_cache: dict[str, Any],
    *,
    arm: str,
    steps: int,
    batch_size: int,
    learning_rate: float,
    beta: float,
    prior_mean: float,
    device: str,
    trace_path: Path,
) -> list[dict[str, Any]]:
    configure_selector_only(wrapper)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in wrapper.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=0.0,
    )
    prior = truncated_geometric_prior(max_loops=12, target_mean=prior_mean)
    batches = _batch_indices(
        len(train_cache["row_ids"]),
        batch_size,
        steps=steps,
        seed=20260717 if arm == "S1_supervised_depth_reading" else 20260718,
    )
    trace: list[dict[str, Any]] = []
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as trace_file:
        for step, indices in enumerate(batches, start=1):
            features = train_cache["pooled_features"][indices].to(device=device, dtype=torch.float32)
            depths = train_cache["depths"][indices].to(device)
            if arm == "S1_supervised_depth_reading":
                loss, weights = supervised_depth_loss(wrapper.halt_predictor, features, depths)
                metrics = {
                    "loss": loss.detach(),
                    "target_nll": loss.detach(),
                    "kl": torch.zeros((), device=loss.device),
                }
            elif arm == "S2_ponder_outcome":
                outcome_nll = train_cache["per_loop_target_nll"][indices].to(device)
                loss, metrics, weights = ponder_outcome_loss(
                    wrapper.halt_predictor,
                    features,
                    outcome_nll,
                    prior=prior,
                    beta=beta,
                )
            else:
                raise ValueError(f"Unknown selector arm: {arm}")
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Nonfinite selector loss at step {step}: {loss}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            frozen_gradient_tensors = assert_frozen_gradients_zero(wrapper)
            grad_norms = assert_active_selector_gradient(wrapper)
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in wrapper.parameters() if parameter.requires_grad],
                1.0,
                error_if_nonfinite=True,
            )
            optimizer.step()
            row = {
                "step": step,
                **{key: float(value.detach().float().item()) for key, value in metrics.items()},
                "mean_selected_depth": float(weights.argmax(dim=-1).float().add(1).mean().item()),
                "mean_expected_depth": float(expected_loop_count(weights).mean().item()),
                "frozen_gradient_tensors": frozen_gradient_tensors,
                "selector_gradient_norm": sum(grad_norms.values()),
            }
            trace.append(row)
            trace_file.write(json.dumps(row, ensure_ascii=True) + "\n")
            if step == 1 or step % 100 == 0 or step == steps:
                print(f"selector_train arm={arm} {row}", flush=True)
    return trace


def evaluate_selector(
    wrapper: Any,
    heldout_cache: dict[str, Any],
    *,
    device: str,
    batch_size: int = 32,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    forced = heldout_cache["forced_predictions"]
    if forced is None:
        raise RuntimeError("Held-out cache is missing forced predictions")
    rows: list[dict[str, Any]] = []
    wrapper.halt_predictor.eval()
    with torch.no_grad():
        for start in range(0, len(heldout_cache["row_ids"]), batch_size):
            end = min(start + batch_size, len(heldout_cache["row_ids"]))
            features = heldout_cache["pooled_features"][start:end].to(device=device, dtype=torch.float32)
            weights = halting_weights_from_features(wrapper.halt_predictor, features).float().cpu()
            selected = weights.argmax(dim=-1) + 1
            expected = expected_loop_count(weights)
            for offset, selected_depth in enumerate(selected.tolist()):
                index = start + offset
                depth = int(heldout_cache["depths"][index])
                target = int(heldout_cache["target_indices"][index])
                selected_prediction = int(forced[index, selected_depth - 1])
                forced_prediction = int(forced[index, depth - 1])
                rows.append(
                    {
                        "id": heldout_cache["row_ids"][index],
                        "depth": depth,
                        "selected_loop": selected_depth,
                        "expected_loops": float(expected[offset]),
                        "halting_weights": [float(value) for value in weights[offset].tolist()],
                        "target": LETTER_SYMBOLS[target],
                        "forced_prediction": LETTER_SYMBOLS[forced_prediction],
                        "selected_prediction": LETTER_SYMBOLS[selected_prediction],
                        "forced_hit": forced_prediction == target,
                        "selected_hit": selected_prediction == target,
                    }
                )
    fixed_t: dict[str, Any] = {}
    targets = heldout_cache["target_indices"]
    depths = heldout_cache["depths"]
    for loop in range(1, 13):
        hits = forced[:, loop - 1].eq(targets)
        fixed_t[str(loop)] = {
            "overall_accuracy": float(hits.float().mean().item()),
            "by_true_depth": {
                str(depth): float(hits[depths.eq(depth)].float().mean().item())
                for depth in range(1, 13)
            },
        }
    return rows, {
        **summarize_selector_rows(rows),
        "fixed_t_structural_comparison": fixed_t,
        "fixed_t_scope": (
            "Completeness control only. Fixed T is not an intelligent allocation baseline "
            "because the row states the required depth."
        ),
    }


def write_summary(run_dir: Path, payload: dict[str, Any]) -> None:
    write_json(run_dir / "summary.json", payload)
    lines = [
        f"# Bounded Depth Selector Assessment - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Source SHA: `{payload.get('source_checkpoint_sha256')}`",
        f"- Frozen mechanism hash: `{payload.get('frozen_parameter_hash')}`",
        f"- Canary: `{payload['locked_spec']['canary_exemption']}`",
        "",
        "## Arms",
        "",
        "| Arm | Status | Selection accuracy | Answer accuracy | Mean depth | Spearman |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for arm_name in ("S1", "S2"):
        arm = payload.get(arm_name) or {}
        gate = arm.get("gate") or {}
        summary = arm.get("evaluation") or {}
        lines.append(
            f"| {arm_name} | {gate.get('status', 'pending')} | "
            f"{summary.get('selection_accuracy', 0.0):.4f} | "
            f"{summary.get('selected_answer_accuracy', 0.0):.4f} | "
            f"{summary.get('mean_selected_depth', 0.0):.3f} | "
            f"{gate.get('spearman_selected_vs_true', 0.0):.3f} |"
        )
    lines.extend(
        [
            "",
            "S1 reads an explicitly stated depth. It is not evidence of difficulty inference.",
            "Learned halting on held-out hard reasoning remains unestablished.",
        ]
    )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((run_dir / "summary.md").read_text(encoding="utf-8"), flush=True)


def main() -> int:
    run_id = os.environ.get("STAGE5_DEPTH_SELECTOR_RUN_ID") or time.strftime(
        "stage5_depth_selector_bounded_%Y%m%d_%H%M%S"
    )
    run_dir = ROOT / "outputs/stage5" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    spec = locked_spec()
    payload: dict[str, Any] = {
        "kind": "stage5_depth_selector_bounded_assessment",
        "run_id": run_id,
        "status": "started",
        "locked_spec": spec,
        "S1": None,
        "S2": None,
    }
    write_summary(run_dir, payload)
    publish(run_dir, f"Record bounded depth-selector preregistration {run_id} [skip ci]")

    source_checkpoint, restore_receipt = restore_checkpoint(
        [N24_KEEPER_DRIVE],
        run_dir / "restored/n24_step6000.pt",
        label="depth_selector_n24_step6000",
    )
    source_sha = sha256_file(source_checkpoint)
    if source_sha != N24_KEEPER_SHA256:
        raise RuntimeError(f"N24 selector source SHA mismatch: {source_sha} != {N24_KEEPER_SHA256}")
    payload["source_checkpoint"] = restore_receipt
    payload["source_checkpoint_sha256"] = source_sha

    device = os.environ.get("DEVICE", "cuda")
    load_args = SimpleNamespace(
        model_name=os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
        dtype=os.environ.get("STAGE5_DEPTH_SELECTOR_DTYPE", "bfloat16"),
        attn_implementation=os.environ.get("ATTN_IMPLEMENTATION", "default"),
        device=device,
        split="6,18",
        bridge_projection_mode="split",
        adapter_dtype="float32",
        lora_rank=0,
        lora_alpha=1,
        trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(load_args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    wrapper = load_recurrent_wrapper(load_args, str(source_checkpoint))
    trainable_names = configure_selector_only(wrapper)
    payload["trainable_parameter_names"] = sorted(trainable_names)
    payload["frozen_parameter_hash"] = frozen_parameter_hash(wrapper)
    initial_selector = _selector_state(wrapper)

    extraction_batch_size = int(os.environ.get("STAGE5_DEPTH_SELECTOR_EXTRACTION_BATCH", "8"))
    train_cache_local = run_dir / "cache/train_cache.pt"
    heldout_cache_local = run_dir / "cache/heldout_cache.pt"
    train_cache_drive = DRIVE_ARTIFACT_ROOT / "train_cache.pt"
    heldout_cache_drive = DRIVE_ARTIFACT_ROOT / "heldout_cache.pt"
    for local, drive_cache, split, data_path in (
        (train_cache_local, train_cache_drive, "train", TRAIN_ROWS),
        (heldout_cache_local, heldout_cache_drive, "heldout", HELDOUT64),
    ):
        if _cache_is_valid(drive_cache, _cache_signature(split=split, data_path=data_path)):
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(drive_cache, local)
            print(f"restored_frozen_cache={drive_cache} -> {local}", flush=True)
        cache = extract_frozen_cache(
            wrapper,
            tokenizer,
            data_path=data_path,
            split=split,
            cache_path=local,
            device=device,
            extraction_batch_size=extraction_batch_size,
        )
        if cache["signature"] != _cache_signature(split=split, data_path=data_path):
            raise RuntimeError(f"Frozen cache signature mismatch after extraction: {split}")
        _copy_cache_to_drive(local, drive_cache)

    train_cache = torch.load(train_cache_local, map_location="cpu")
    heldout_cache = torch.load(heldout_cache_local, map_location="cpu")
    baseline_rows, baseline_eval = evaluate_selector(wrapper, heldout_cache, device=device)
    payload["untrained_selector_baseline"] = baseline_eval
    payload["status"] = "cache_ready"
    write_summary(run_dir, payload)
    publish(run_dir, f"Record depth-selector frozen cache receipts {run_id} [skip ci]")

    steps = int(os.environ.get("STAGE5_DEPTH_SELECTOR_STEPS", str(spec["steps_per_arm"])))
    batch_size = int(os.environ.get("STAGE5_DEPTH_SELECTOR_BATCH_SIZE", str(spec["batch_size"])))
    _restore_selector_state(wrapper, initial_selector)
    s1_trace = train_selector_arm(
        wrapper,
        train_cache,
        arm=spec["s1_name"],
        steps=steps,
        batch_size=batch_size,
        learning_rate=float(os.environ.get("STAGE5_DEPTH_SELECTOR_S1_LR", "1e-3")),
        beta=0.0,
        prior_mean=spec["s2_geometric_prior_mean"],
        device=device,
        trace_path=run_dir / "S1/training_trace.jsonl",
    )
    if frozen_parameter_hash(wrapper) != payload["frozen_parameter_hash"]:
        raise RuntimeError("Frozen mechanism hash changed during S1")
    if sha256_file(source_checkpoint) != source_sha:
        raise RuntimeError("Source checkpoint file changed during S1")
    s1_rows, s1_eval = evaluate_selector(wrapper, heldout_cache, device=device)
    s1_gate = evaluate_s1_gate(
        s1_rows,
        min_correct_per_depth=spec["s1_min_correct_per_depth"],
        answer_delta_floor=spec["s1_answer_delta_floor"],
    )
    write_jsonl(run_dir / "S1/eval_rows.jsonl", s1_rows)
    write_json(run_dir / "S1/eval_summary.json", s1_eval)
    write_json(run_dir / "S1/gate.json", s1_gate)
    s1_checkpoint = _save_selector_checkpoint(
        wrapper,
        run_dir / f"S1/{spec['s1_name']}_step_{steps}.pt",
        arm=spec["s1_name"],
        steps=steps,
        frozen_hash=payload["frozen_parameter_hash"],
        source_checkpoint_sha256=source_sha,
    )
    payload["S1"] = {
        "arm": spec["s1_name"],
        "training_trace": path_for_cli(run_dir / "S1/training_trace.jsonl"),
        "evaluation": s1_eval,
        "gate": s1_gate,
        "checkpoint": s1_checkpoint,
    }
    payload["status"] = f"S1_{s1_gate['status']}"
    write_summary(run_dir, payload)
    publish(run_dir, f"Record bounded selector S1 {run_id} [skip ci]")

    _restore_selector_state(wrapper, initial_selector)
    s2_trace = train_selector_arm(
        wrapper,
        train_cache,
        arm=spec["s2_name"],
        steps=steps,
        batch_size=batch_size,
        learning_rate=float(os.environ.get("STAGE5_DEPTH_SELECTOR_S2_LR", "1e-3")),
        beta=spec["s2_beta"],
        prior_mean=spec["s2_geometric_prior_mean"],
        device=device,
        trace_path=run_dir / "S2/training_trace.jsonl",
    )
    if frozen_parameter_hash(wrapper) != payload["frozen_parameter_hash"]:
        raise RuntimeError("Frozen mechanism hash changed during S2")
    if sha256_file(source_checkpoint) != source_sha:
        raise RuntimeError("Source checkpoint file changed during S2")
    s2_rows, s2_eval = evaluate_selector(wrapper, heldout_cache, device=device)
    s2_gate = evaluate_s2_gate(
        s2_rows,
        training_trace=s2_trace,
        s1_gate=s1_gate,
        strong_spearman_floor=spec["s2_strong_spearman_floor"],
        partial_spearman_floor=spec["s2_partial_spearman_floor"],
        answer_delta_from_s1_floor=spec["s2_answer_delta_from_s1_floor"],
    )
    write_jsonl(run_dir / "S2/eval_rows.jsonl", s2_rows)
    write_json(run_dir / "S2/eval_summary.json", s2_eval)
    write_json(run_dir / "S2/gate.json", s2_gate)
    s2_checkpoint = _save_selector_checkpoint(
        wrapper,
        run_dir / f"S2/{spec['s2_name']}_step_{steps}.pt",
        arm=spec["s2_name"],
        steps=steps,
        frozen_hash=payload["frozen_parameter_hash"],
        source_checkpoint_sha256=source_sha,
    )
    payload["S2"] = {
        "arm": spec["s2_name"],
        "training_trace": path_for_cli(run_dir / "S2/training_trace.jsonl"),
        "evaluation": s2_eval,
        "gate": s2_gate,
        "checkpoint": s2_checkpoint,
    }
    payload["source_checkpoint_sha256_end"] = sha256_file(source_checkpoint)
    payload["frozen_parameter_hash_end"] = frozen_parameter_hash(wrapper)
    payload["frozen_contract_pass"] = bool(
        payload["source_checkpoint_sha256_end"] == source_sha
        and payload["frozen_parameter_hash_end"] == payload["frozen_parameter_hash"]
    )
    payload["status"] = (
        "finished"
        if s1_gate["status"] == "pass" and s2_gate["status"] in {"strong", "partial"}
        else "finished_blocked"
    )
    write_summary(run_dir, payload)
    publish(run_dir, f"Finish bounded depth-selector assessment {run_id} [skip ci]")
    return 0 if payload["status"] == "finished" else 2


if __name__ == "__main__":
    raise SystemExit(main())
