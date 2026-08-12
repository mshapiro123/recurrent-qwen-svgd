"""Run the score-blind P3.4 v1 task-inference preflight on both i1 endpoints."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.cache_paper2_phase3_agreement_oracle import _load_phase3_module
from eval.eval_paper2_phase3_p31_references import (
    MODEL_SPECS,
    _chat_prompt,
    _generation_prompt,
    _mcq,
    _mcq_prompt,
)
from eval.eval_paper2_phase3_p34_task_inference import (
    P34TaskInferenceGraph,
    task_graph_preflight,
)


RUN_ID = "stage5_paper2_phase3_p34_prerequisites_20260812"
DRIVE_STAGE5 = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5")
DRIVE_RUN = DRIVE_STAGE5 / RUN_ID
P31_RUN = DRIVE_STAGE5 / "stage5_paper2_phase3_p31_completion_20260810"
I1_RUN = DRIVE_STAGE5 / "stage5_paper2_phase3_p33_i1_20260812"
MIGRATION_RUN = DRIVE_STAGE5 / "stage5_paper2_phase3_p31_p32_receipts_20260810"
P33_RUN = DRIVE_STAGE5 / "stage5_paper2_phase3_p33_20260811"
EXPECTED_CHECKPOINTS = {
    0: "01c804bc69d35a01730fff236cf5a8d974899d2e4de7e15b92a227b2a9ce5d88",
    1: "2ed3296f510a6c3a66c451051ecbe2284de03b35dde4052827174a66a10c1d4a",
}
EXPECTED_MIGRATED = {
    0: "d0f2b735825d29ab9801a5200493ca9aa65294778aea2fb7f728eb8e85dfc519",
    1: "3ca1cdf8dd16bf4f435e81a675d7514778144c5c881af52a70171659f7734b4f",
}
EXPECTED_P33 = {
    0: "84dc0fb2d1f69114b20888acd95101d6b31c810974a536dc36358b69fe13c70e",
    1: "e80ad205eb3c4712fdee5303a4887260488f67ff858a2b4b005d724675e52067",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def prompt_text(row: dict[str, object], tokenizer: object) -> str:
    if row["battery"] in {"arc_easy", "arc_challenge", "mmlu"}:
        question, choices, _answer = _mcq(row)
        return _mcq_prompt(question, choices)
    content, _cap = _generation_prompt(row)
    return _chat_prompt(tokenizer, content)


def load_composed_sidecar(*, seed: int, embedding_weight: torch.Tensor):
    migrated = MIGRATION_RUN / f"private/migrated_checkpoints/seed_{seed}_full_a2_phase3_migrated.pt"
    p33 = P33_RUN / f"private/seed_{seed}/checkpoint_step_1000.pt"
    i1 = I1_RUN / f"private/seed_{seed}/resume.pt"
    expected = ((migrated, EXPECTED_MIGRATED[seed]), (p33, EXPECTED_P33[seed]), (i1, EXPECTED_CHECKPOINTS[seed]))
    for path, digest in expected:
        if sha256_file(path) != digest:
            raise RuntimeError(f"P3.4 composed endpoint SHA mismatch: {path}")
    module, migrated_receipt = _load_phase3_module(
        checkpoint=migrated,
        embedding_weight=embedding_weight,
        device="cuda",
    )
    receipts = {"migrated": migrated_receipt}
    for label, path in (("p33", p33), ("i1", i1)):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("trainable_state")
        if not isinstance(state, dict):
            raise RuntimeError(f"P3.4 {label} endpoint lacks trainable_state")
        current = dict(module.named_parameters())
        unknown = sorted(set(state) - set(current))
        if unknown:
            raise RuntimeError(f"P3.4 {label} state contains unknown parameters: {unknown}")
        with torch.no_grad():
            for name, value in state.items():
                current[name].copy_(value.to(device=current[name].device, dtype=current[name].dtype))
        receipts[label] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "state_keys": sorted(state),
            "step": int(payload["step"]),
        }
    return module.eval(), receipts


def main() -> int:
    receipts = DRIVE_RUN / "receipts"
    status = receipts / "task_preflight_status.json"
    try:
        write_json(status, {"status": "loading", "updated_at_unix": time.time()})
        rows_path = P31_RUN / "private/p31_partitioned_rows.jsonl"
        rows = [
            json.loads(line)
            for line in rows_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        dev = [row for row in rows if row["partition"] == "dev"][:8]
        if len(dev) != 8 or any(row["partition"] != "dev" for row in dev):
            raise RuntimeError("P3.4 preflight could not select eight unsealed DEV prompts")
        spec = MODEL_SPECS["base"]
        tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"])
        model = AutoModelForCausalLM.from_pretrained(
            spec["model"],
            revision=spec["revision"],
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda").eval()
        embedding_weight = model.get_output_embeddings().weight.detach().cpu()
        seed_receipts = []
        for seed in (0, 1):
            checkpoint = I1_RUN / f"private/seed_{seed}/resume.pt"
            observed = sha256_file(checkpoint)
            if observed != EXPECTED_CHECKPOINTS[seed]:
                raise RuntimeError(f"P3.4 i1 endpoint SHA mismatch for seed {seed}")
            sidecar, checkpoint_receipt = load_composed_sidecar(
                seed=seed, embedding_weight=embedding_weight
            )
            sidecar.bridge.set_gate_ceiling(0.08)
            graph = P34TaskInferenceGraph(base_model=model, sidecar=sidecar)
            prompt_receipts = []
            for row in dev:
                encoded = tokenizer(prompt_text(row, tokenizer), return_tensors="pt").to("cuda")
                receipt = task_graph_preflight(
                    graph,
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                )
                failed = [name for name, passed in receipt["assertions"].items() if not passed]
                if failed:
                    raise RuntimeError(f"P3.4 task graph failed seed={seed}: {failed}")
                prompt_receipts.append(
                    {
                        "item_id": row["item_id"],
                        "battery": row["battery"],
                        "tokens": int(encoded["attention_mask"].sum()),
                        "selected_write_cells": receipt["selected_write_cells"],
                        "repeat_max_abs_difference": receipt["repeat_max_abs_difference"],
                    }
                )
            seed_receipts.append(
                {
                    "seed": seed,
                    "checkpoint": checkpoint_receipt,
                    "checkpoint_sha256": observed,
                    "gate_ceiling": 0.08,
                    "prompts": prompt_receipts,
                }
            )
            del graph, sidecar
            torch.cuda.empty_cache()
        result = {
            "kind": "paper2_phase3_p34_task_inference_preflight_v1",
            "status": "complete_score_blind",
            "base_model": spec,
            "serving_dtype": "bfloat16",
            "attn_implementation": "sdpa",
            "seeds": seed_receipts,
            "task_scores_computed": False,
            "correctness_computed": False,
            "gap_closed_computed": False,
            "sealed_partitions_touched": False,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "training_authorized": False,
        }
        output = receipts / "task_inference_preflight.json"
        write_json(output, result)
        write_json(
            status,
            {"status": "complete", "updated_at_unix": time.time(), "receipt": str(output)},
        )
        print(json.dumps({"status": "complete", "receipt": str(output)}, indent=2))
        return 0
    except Exception as error:
        write_json(
            status,
            {
                "status": "failed",
                "updated_at_unix": time.time(),
                "exception_type": type(error).__name__,
                "exception": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
