"""Cache prompt-only Bicameral states for the W2-prime D4 site screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.eval_paper2_stage2b_campaign import _forced_target
from models.bicameral import BicameralTaskInferenceGraph, SEQUENTIAL_EXECUTION_SCHEDULE
from models.recurrent_wrapper import LayerSplit, RecurrentQwenForCausalLM


KIND = "paper2_bicameral_w2p_d4_cache_v1"
AUTHORITY_SHA256 = "f89b45ef100fa46536dd93a3ef936aa8c9cfa1fc624b401b4bfc0d2b50bc2aa4"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SITES = (8, 12, 16, 18)
EXPECTED_TORCH = "2.11.0+cu128"
EXPECTED_CUDA = "12.8"
EXPECTED_PYTHON = "3.13.15"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_rows(manifest: Path, reference_rows: Path) -> list[dict[str, Any]]:
    selected = read_jsonl(manifest)
    reference = {str(row["item_id"]): row for row in read_jsonl(reference_rows)}
    rows = [reference[str(row["item_id"])] for row in selected]
    if len(rows) != 256 or len({str(row["item_id"]) for row in rows}) != 256:
        raise RuntimeError("W2-prime D4 requires the frozen 256-row Stage-0 manifest")
    return rows


def state_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def pool(value: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    weights = attention_mask.to(value.device).float().unsqueeze(-1)
    return ((value.float() * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)).cpu()


def runtime_receipt() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("W2-prime D4 requires an A100 runtime")
    properties = torch.cuda.get_device_properties(0)
    receipt = {
        "gpu": properties.name,
        "total_gib": properties.total_memory / 2**30,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
    }
    expected = {
        "torch": EXPECTED_TORCH,
        "cuda": EXPECTED_CUDA,
        "python": EXPECTED_PYTHON,
    }
    observed = {key: receipt[key] for key in expected}
    if "A100-SXM4" not in properties.name or observed != expected:
        raise RuntimeError(f"W2-prime D4 runtime mismatch: {receipt}")
    return receipt


def cache_seed(
    graph: BicameralTaskInferenceGraph,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    initializer: Path,
    *,
    seed: int,
    deadline: float,
) -> dict[str, Any]:
    graph.core.zero_gates()
    graph.core.load_branch_initializers(initializer)
    graph.core.bind_strategy_operating_gates(source_receipt_sha256=AUTHORITY_SHA256)
    versions_before = {name: parameter._version for name, parameter in graph.named_parameters()}
    before = state_digest(graph.core)
    collected = {
        site: {"base": [], "branch_a": [], "branch_b": []}
        for site in SITES
    }
    parity = None
    started = time.perf_counter()
    with torch.inference_mode():
        for index, row in enumerate(rows):
            if time.perf_counter() >= deadline:
                raise TimeoutError("W2-prime D4 stopped before its 0.25 A100-hour cap")
            prompt, _target = _forced_target(tokenizer, row)
            input_ids = torch.tensor(prompt, dtype=torch.long, device=graph.device).unsqueeze(0)
            attention = torch.ones_like(input_ids)
            states = graph.cache_site_states(
                input_ids=input_ids,
                attention_mask=attention,
                sites=SITES,
            )
            if index == 0:
                direct = graph.cache_branch_states(
                    input_ids=input_ids, attention_mask=attention
                )
                direct_base, _context = graph._encode_middle(
                    bicameral=False,
                    input_ids=input_ids,
                    attention_mask=attention,
                )
                parity = {
                    "branch_a_exact": bool(torch.equal(states.branch_a[18], direct.branch_a)),
                    "branch_b_exact": bool(torch.equal(states.branch_b[18], direct.branch_b)),
                    "base_exact": bool(torch.equal(states.base[18], direct_base.combined)),
                }
                if not all(parity.values()):
                    raise RuntimeError(f"W2-prime D4 chunked-site parity failed: {parity}")
            for site in SITES:
                collected[site]["base"].append(pool(states.base[site], attention)[0])
                collected[site]["branch_a"].append(pool(states.branch_a[site], attention)[0])
                collected[site]["branch_b"].append(pool(states.branch_b[site], attention)[0])
            if (index + 1) % 32 == 0 or index + 1 == len(rows):
                print(f"w2p_d4_progress seed={seed} rows={index + 1}/{len(rows)}", flush=True)
    after = state_digest(graph.core)
    versions_after = {name: parameter._version for name, parameter in graph.named_parameters()}
    if before != after or versions_before != versions_after:
        raise RuntimeError("W2-prime D4 mutated the frozen graph")
    return {
        "kind": KIND,
        "seed": seed,
        "item_ids": [str(row["item_id"]) for row in rows],
        "batteries": [str(row["battery"]) for row in rows],
        "sites": {
            site: {name: torch.stack(values) for name, values in payload.items()}
            for site, payload in collected.items()
        },
        "interface_site": 18,
        "feature_list": [
            "base_prompt_state_h",
            "bicameral_prompt_state_h_A",
            "bicameral_prompt_state_h_B",
            "m_equals_half_hA_plus_hB",
            "d_equals_half_hA_minus_hB",
        ],
        "input_provenance": "student_prompt_only",
        "gold_answer_used": False,
        "teacher_forward_used": False,
        "oracle_routing_used": False,
        "prompt_construction": "forced_target_formatter_prompt_component_only_target_discarded",
        "execution_schedule": SEQUENTIAL_EXECUTION_SCHEDULE,
        "sites_parity": parity,
        "initializer_sha256": sha256_file(initializer),
        "graph_state_digest_before": before,
        "graph_state_digest_after": after,
        "parameter_versions_unchanged": True,
        "seconds": time.perf_counter() - started,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference_rows", type=Path, required=True)
    parser.add_argument("--initializer_seed_0", type=Path, required=True)
    parser.add_argument("--initializer_seed_1", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_cache", type=Path, required=True)
    parser.add_argument("--wall_seconds_cap", type=float, default=840.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_all = time.perf_counter()
    deadline = started_all + float(args.wall_seconds_cap)
    rows = load_rows(args.manifest, args.reference_rows)
    runtime = runtime_receipt()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=args.model_cache
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=args.model_cache,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).eval().to("cuda")
    wrapper = RecurrentQwenForCausalLM(
        base, layer_split=LayerSplit(prelude_end=6, recurrent_end=18)
    ).eval()
    graph = BicameralTaskInferenceGraph(wrapper).eval()
    outputs = {}
    for seed, initializer in (
        (0, args.initializer_seed_0),
        (1, args.initializer_seed_1),
    ):
        payload = cache_seed(
            graph,
            tokenizer,
            rows,
            initializer,
            seed=seed,
            deadline=deadline,
        )
        path = args.output_dir / f"seed_{seed}_w2p_d4_cache.pt"
        torch.save(payload, path)
        outputs[str(seed)] = {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "seconds": payload["seconds"],
            "sites_parity": payload["sites_parity"],
        }
    summary = {
        "kind": "paper2_bicameral_w2p_d4_summary_v1",
        "status": "complete_forward_only",
        "authority_sha256": AUTHORITY_SHA256,
        "runtime": runtime,
        "manifest": {"bytes": args.manifest.stat().st_size, "sha256": sha256_file(args.manifest)},
        "reference_rows_sha256": sha256_file(args.reference_rows),
        "seeds": outputs,
        "gpu_seconds_total": time.perf_counter() - started_all,
        "wall_seconds_cap": float(args.wall_seconds_cap),
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    summary_path = args.output_dir / "w2p_d4_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
