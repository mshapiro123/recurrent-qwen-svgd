"""Execute the five Paper Two Phase T0 contracts without training."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import transformers
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_mcq import load_recurrent_wrapper  # noqa: E402
from training.internal_think_token_runtime import (  # noqa: E402
    forced_loop_accounting,
    install_internal_control_tokens,
    mask_internal_control_logits,
    one_loop_identity_max_abs_diff,
)
from training.internal_think_token_spec import (  # noqa: E402
    INTERNAL_CONTROL_TOKENS,
    phase_t0_spec,
)


def loader_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model_name=args.model_name,
        checkpoint=None,
        split=args.split,
        bridge_projection_mode="split",
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        device=args.device,
        lora_rank=0,
        lora_alpha=16,
        adapter_dtype="float32",
        base_lora_layer_range="all",
    )


def forward(wrapper: Any, encoded: dict[str, torch.Tensor], max_loops: int) -> Any:
    wrapper.eval()
    with torch.no_grad():
        return wrapper(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            labels=None,
            max_loops=max_loops,
            num_trajectories=1,
            particle_update_mode="none",
            use_cache=False,
            return_dict=True,
            return_loop_logits=True,
            logits_to_keep=1,
        )


def write_receipt(output_dir: Path, summary: dict[str, Any]) -> None:
    contracts = summary["contracts"]
    lines = [
        "# Paper Two Phase T0 Preflight",
        "",
        f"- Status: `{summary['status']}`",
        f"- Training performed: `{summary['training_performed']}`",
        f"- Model: `{summary['model_name']}`",
        f"- Device: `{summary['environment']['device']}`",
        "",
        "| Contract | Result |",
        "|---|---|",
        f"| Tokenizer collision | `{contracts['tokenizer_collision']['passed']}` |",
        f"| Exactly three rows and tie policy | `{contracts['vocabulary_resize']['passed']}` |",
        f"| Visible-generation masking | `{contracts['visible_generation_masking']['passed']}` |",
        f"| One-loop identity below 1e-3 | `{contracts['one_loop_identity']['passed']}` |",
        f"| Forced loop accounting | `{contracts['loop_accounting']['passed']}` |",
        "",
        "No checkpoint was written and no training was authorized by this receipt.",
        "",
    ]
    (output_dir / "receipt.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt", default="Return the symbol A.\nAnswer:")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collision_tokens = sorted(set(INTERNAL_CONTROL_TOKENS) & set(tokenizer.get_vocab()))
    if collision_tokens:
        raise AssertionError(f"Control-token collision before resize: {collision_tokens}")
    wrapper = load_recurrent_wrapper(loader_args(args), None)
    encoded = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=True).to(args.device)

    before = forward(wrapper, encoded, max_loops=1)
    resize = install_internal_control_tokens(tokenizer, wrapper.base_model)
    after = forward(wrapper, encoded, max_loops=1)
    identity_max_abs_diff = one_loop_identity_max_abs_diff(
        before.loop_logits,
        after.loop_logits,
        original_model_vocab_size=resize.original_vocab_size,
    )
    if identity_max_abs_diff >= 1e-3:
        raise AssertionError(
            f"One-loop inactive-control identity failed: {identity_max_abs_diff}"
        )

    last_logits = after.loop_logits[0, 0, 0, -1].detach().clone()
    adversarial = last_logits.clone()
    adversarial[list(resize.control_token_ids)] = adversarial.max() + 1000
    masked = mask_internal_control_logits(adversarial, resize.control_token_ids)
    visible_token_id = int(masked.argmax().item())
    if visible_token_id in resize.control_token_ids:
        raise AssertionError("Visible generation selected an internal control token")
    visible_text = tokenizer.decode([visible_token_id], skip_special_tokens=False)
    if visible_text in INTERNAL_CONTROL_TOKENS:
        raise AssertionError("Visible decoded output contains an internal control token")

    forced_output = forward(wrapper, encoded, max_loops=4)
    executed_loops = int(forced_output.loop_logits.shape[2])
    accounting = forced_loop_accounting(
        requested_loops=4,
        executed_loops=executed_loops,
        selected_loop=4,
    )

    summary = {
        "kind": "paper2_internal_think_token_phase_t0_preflight",
        "status": "passed_all_five_contracts",
        "training_performed": False,
        "checkpoint_written": False,
        "phase_t1_authorized_by_this_receipt": False,
        "model_name": args.model_name,
        "split": args.split,
        "policy": phase_t0_spec(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": args.device,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "dtype": args.dtype,
        },
        "contracts": {
            "tokenizer_collision": {
                "passed": not collision_tokens,
                "collisions": collision_tokens,
                "planned_tokens": list(INTERNAL_CONTROL_TOKENS),
            },
            "vocabulary_resize": {"passed": True, **resize.to_dict()},
            "visible_generation_masking": {
                "passed": True,
                "adversarial_control_logits": True,
                "selected_visible_token_id": visible_token_id,
                "selected_visible_text": visible_text,
                "control_token_ids": list(resize.control_token_ids),
            },
            "one_loop_identity": {
                "passed": identity_max_abs_diff < 1e-3,
                "maximum": 1e-3,
                "observed_max_abs_diff": identity_max_abs_diff,
                "compared_original_vocab_rows": resize.original_vocab_size,
            },
            "loop_accounting": {"passed": True, **accounting},
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_receipt(output_dir, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
