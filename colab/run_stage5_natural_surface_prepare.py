"""Generate the Stage 5 natural-surface transfer datasets.

This is CPU-only prep.  It writes deterministic relay/pointer verbal surfaces
and the rung-zero mixed SFT curriculum, then publishes the artifact without
changing the current Stage 5 source pointer.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colab.stage5_chain_consolidation_utils import ROOT, path_for_cli, publish_run, write_json
from training.natural_surface_transfer import NaturalSurfaceConfig, verify_single_token_names, write_natural_surface_dataset


def write_markdown(run_dir: Path, payload: dict[str, Any]) -> None:
    data = payload.get("dataset", {})
    manifests = data.get("manifests", {})
    lines = [
        f"# Natural-Surface Transfer Prep - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Dataset summary: `{payload['data_summary']}`",
        f"- Symbols: `{payload['config']['n_symbols']}` names",
        f"- Relay train: `{manifests.get('train_relay_chain_symbol_sft', {}).get('rows')}` rows",
        f"- Rung-zero mix: `{manifests.get('rung0_train_mix_chain_symbol_sft', {}).get('rows')}` rows",
        f"- Relay eval: `{manifests.get('relay_test_chain_mcq', {}).get('rows')}` rows",
        f"- Pointer eval: `{manifests.get('pointer_test_chain_mcq', {}).get('rows')}` rows",
        f"- Tokenizer verification: `{payload.get('tokenizer_verification', {}).get('all_single_token')}`",
        "",
        "## Queue Role",
        "",
        "- Ungated CPU/data-prep artifact for the natural-surface transfer program.",
        "- Does not update `config/stage5_current_source_summary.txt`; the GPU battery remains the active line.",
        "- Rung zero trains relay verbal plus symbolic rehearsal; pointer is a held-out zero-shot template read.",
        "",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def maybe_verify_tokenizer(model_name: str, n_symbols: int) -> dict[str, Any]:
    if os.environ.get("STAGE5_NATURAL_VERIFY_TOKENIZER", "0").strip().lower() not in {"1", "true", "yes", "y"}:
        return {
            "enabled": False,
            "all_single_token": None,
            "note": "Set STAGE5_NATURAL_VERIFY_TOKENIZER=1 to verify Qwen tokenizer name tokens.",
        }
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    verdict = verify_single_token_names(tokenizer, n_symbols=n_symbols)
    verdict["enabled"] = True
    verdict["model_name"] = model_name
    if not verdict["all_single_token"]:
        raise RuntimeError(f"Natural-surface answer names are not all single-token under {model_name}: {verdict}")
    return verdict


def main() -> int:
    run_id = os.environ.get("STAGE5_NATURAL_SURFACE_RUN_ID") or time.strftime(
        "stage5_natural_surface_transfer_%Y%m%d_%H%M%S"
    )
    run_dir = ROOT / "outputs" / "stage5" / run_id
    data_dir = run_dir / "data"
    config = NaturalSurfaceConfig(
        n_symbols=int(os.environ.get("STAGE5_NATURAL_N_SYMBOLS", "20")),
        train_max_depth=int(os.environ.get("STAGE5_NATURAL_TRAIN_MAX_DEPTH", "8")),
        eval_max_depth=int(os.environ.get("STAGE5_NATURAL_EVAL_MAX_DEPTH", "12")),
        train_rows_per_depth=int(os.environ.get("STAGE5_NATURAL_TRAIN_ROWS_PER_DEPTH", "256")),
        val_rows_per_depth=int(os.environ.get("STAGE5_NATURAL_VAL_ROWS_PER_DEPTH", "64")),
        eval_rows_per_depth=int(os.environ.get("STAGE5_NATURAL_EVAL_ROWS_PER_DEPTH", "128")),
        seed=int(os.environ.get("STAGE5_NATURAL_SEED", "910031")),
        max_target_loops=int(os.environ.get("STAGE5_NATURAL_MAX_TARGET_LOOPS", "12")),
        value_prefix="name:",
    )
    dataset = write_natural_surface_dataset(output_dir=data_dir, config=config)
    tokenizer_verification = maybe_verify_tokenizer(
        os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
        n_symbols=config.n_symbols,
    )
    payload = {
        "kind": "stage5_natural_surface_transfer_prepare",
        "run_id": run_id,
        "status": "finished",
        "config": dataset["config"],
        "data_summary": path_for_cli(data_dir / "summary.json"),
        "dataset": dataset,
        "tokenizer_verification": tokenizer_verification,
        "queue_role": {
            "ungated_cpu_prep": True,
            "updates_current_source_pointer": False,
            "post_battery_training_slot": 1,
            "next_training_target": "verbal_rung_zero",
        },
    }
    write_json(run_dir / "summary.json", payload)
    write_markdown(run_dir, payload)
    publish_run(
        run_dir,
        message=f"Record Stage 5 natural-surface transfer prep {run_id} [skip ci]",
        update_pointer=False,
    )
    print(json.dumps({"run_id": run_id, "status": payload["status"], "summary": path_for_cli(run_dir / "summary.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
