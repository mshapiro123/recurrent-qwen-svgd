"""Run the single registered EVAL-C pass and immutable Stage A verdict."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.cache_speculative_depth_d0_teachers import load_drafter, read_jsonl  # noqa: E402
from eval.eval_paper2_dc0_depth_by_append import (  # noqa: E402
    anchor_registered_k0,
    evaluate_append_arm,
    evaluate_inplace,
    parameter_fingerprint,
    transition_counts,
)
from eval.eval_paper2_dc1_stage_a_verdict import aggregate  # noqa: E402
from eval.eval_speculative_depth_d0_floor import load_partition_cache  # noqa: E402
from models.coconut_composite import CoconutRecurrentQwen  # noqa: E402
from training.speculative_depth_d0_corpus import sha256_file  # noqa: E402


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)


def write_jsonl_once(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise RuntimeError("immutable EVAL-C scoring cache already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return sha256_file(path)


def transition_cell(
    before: torch.Tensor, after: torch.Tensor, teacher: torch.Tensor
) -> dict[str, int]:
    report = transition_counts(before, after, teacher)
    return {
        "helps": int(report["helps"]),
        "hurts": int(report["hurts"]),
        "neutral": int(report["neutral"]),
        "before_correct": int(report["before_correct"]),
        "after_correct": int(report["after_correct"]),
    }


def build_scoring_rows(
    *,
    rows: Sequence[dict[str, Any]],
    teacher_rows: dict[int, dict[str, Any]],
    inplace_rows: Sequence[torch.Tensor],
    trained_rows: Sequence[torch.Tensor],
    untrained_rows: Sequence[torch.Tensor],
) -> list[dict[str, Any]]:
    if not (
        len(rows)
        == len(teacher_rows)
        == len(inplace_rows)
        == len(trained_rows)
        == len(untrained_rows)
    ):
        raise ValueError("Stage A immutable cache inputs are not row-aligned")
    result = []
    for index, row in enumerate(rows):
        teacher = teacher_rows[index]["teacher_greedy_token_id"].long()
        inplace = inplace_rows[index].long()
        trained = trained_rows[index].long()
        untrained = untrained_rows[index].long()
        if inplace.ndim != 2 or inplace.shape[1] < 3:
            raise ValueError("Stage A in-place grid must contain depths 1 through 3")
        k0 = inplace[:, 0]
        result.append(
            {
                "row_id": str(row["row_id"]),
                "stratum": str(row["stratum"]),
                "scored_positions": int(len(teacher)),
                "arms": {
                    "trained_append_k1": transition_cell(k0, trained, teacher),
                    "untrained_append_k1": transition_cell(k0, untrained, teacher),
                    "inplace_depth2_descriptive": transition_cell(
                        k0, inplace[:, 1], teacher
                    ),
                    "inplace_depth3_descriptive": transition_cell(
                        k0, inplace[:, 2], teacher
                    ),
                },
            }
        )
    return result


def summarize_by_stratum(
    scoring_rows: list[dict[str, Any]], arms: Sequence[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stratum in ("code", "general", "pooled"):
        selected = (
            scoring_rows
            if stratum == "pooled"
            else [row for row in scoring_rows if row["stratum"] == stratum]
        )
        result[stratum] = {}
        for arm in arms:
            summary = aggregate(selected, arm)
            before_correct = sum(
                int(row["arms"][arm]["before_correct"]) for row in selected
            )
            after_correct = sum(
                int(row["arms"][arm]["after_correct"]) for row in selected
            )
            scored = int(summary["scored_positions"])
            result[stratum][arm] = {
                **summary,
                "before_correct": before_correct,
                "after_correct": after_correct,
                "before_accuracy": before_correct / scored,
                "after_accuracy": after_correct / scored,
            }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", required=True)
    parser.add_argument("--eval_freeze_summary", required=True)
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--document_manifest", required=True)
    parser.add_argument("--teacher_cache_summary", required=True)
    parser.add_argument("--init_checkpoint", required=True)
    parser.add_argument("--trained_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--private_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--append_batch_size", type=int, default=24)
    args = parser.parse_args()

    prereg_path = Path(args.prereg)
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    freeze = json.loads(Path(args.eval_freeze_summary).read_text(encoding="utf-8"))
    if prereg.get("locked_before_training") is not True:
        raise RuntimeError("Stage A EVAL-C requires the locked preregistration")
    if freeze.get("read_once_scoring_spent") is not False:
        raise RuntimeError("registered EVAL-C scoring pass was already spent")
    if freeze.get("scores_exposed") is not False:
        raise RuntimeError("EVAL-C freeze receipt exposed outcomes before scoring")
    eval_policy = prereg["eval_partition"]
    assertions = {
        "data_jsonl": (args.data_jsonl, eval_policy["jsonl_sha256"]),
        "document_manifest": (args.document_manifest, eval_policy["manifest_sha256"]),
        "teacher_cache": (args.teacher_cache_summary, eval_policy["teacher_cache_sha256"]),
        "init_checkpoint": (args.init_checkpoint, prereg["init_checkpoint_sha256"]),
    }
    for name, (path, expected) in assertions.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"Stage A EVAL-C {name} SHA-256 mismatch")

    trained_payload = torch.load(
        args.trained_checkpoint, map_location="cpu", weights_only=False
    )
    if trained_payload.get("prereg_sha256") != sha256_file(prereg_path):
        raise RuntimeError("trained bridge checkpoint preregistration mismatch")
    trained_checkpoint_sha = sha256_file(args.trained_checkpoint)

    private = Path(args.private_dir)
    private.mkdir(parents=True, exist_ok=True)
    immutable_cache = private / "immutable_scoring_cache.jsonl"
    immutable_receipt = private / "immutable_scoring_cache_receipt.json"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not immutable_cache.exists():
        rows = read_jsonl(args.data_jsonl)
        teacher_summary = json.loads(
            Path(args.teacher_cache_summary).read_text(encoding="utf-8")
        )
        teacher_rows = load_partition_cache(teacher_summary, "teacher_7b", "eval_c")
        _tokenizer, wrapper, resize, _original_vocab = load_drafter(
            checkpoint=Path(args.init_checkpoint),
            device=args.device,
            dtype="float32",
            attn_implementation="sdpa",
        )
        for parameter in wrapper.parameters():
            parameter.requires_grad_(False)
        wrapper.eval()
        composite = CoconutRecurrentQwen(
            wrapper, latent_token_id=int(resize.control_token_ids[2])
        ).to(device=args.device, dtype=torch.float32).eval()
        before = parameter_fingerprint(wrapper)

        inplace_rows = evaluate_inplace(
            wrapper,
            rows,
            vocab_size=resize.original_tokenizer_size,
            device=args.device,
            resume_dir=private / "arm_batches/inplace_locked_init",
        )
        composite.horizontal_bridge.load_state_dict(
            trained_payload["horizontal_bridge"]
        )
        trained_grid, trained_counters = evaluate_append_arm(
            composite,
            rows,
            arm="trained_append_k1",
            feedback_mode="raw",
            reference_rms=None,
            neutral_token_id=int(resize.control_token_ids[2]),
            vocab_size=resize.original_tokenizer_size,
            device=args.device,
            batch_size=args.append_batch_size,
            resume_dir=private / f"arm_batches/trained_{trained_checkpoint_sha[:16]}",
            append_steps=1,
        )
        trained_rows = [grid[:, 1] for grid in trained_grid]
        with torch.no_grad():
            composite.horizontal_bridge.delta.weight.zero_()
        untrained_grid, untrained_counters = evaluate_append_arm(
            composite,
            rows,
            arm="untrained_append_k1",
            feedback_mode="raw",
            reference_rms=None,
            neutral_token_id=int(resize.control_token_ids[2]),
            vocab_size=resize.original_tokenizer_size,
            device=args.device,
            batch_size=args.append_batch_size,
            resume_dir=private / "arm_batches/untrained_identity",
            append_steps=1,
        )
        untrained_rows = [grid[:, 1] for grid in untrained_grid]
        anchor_diagnostics = {
            "trained_append_cached_k0_vs_registered_disagreements": sum(
                int(grid[:, 0].ne(inplace[:, 0]).sum())
                for grid, inplace in zip(trained_grid, inplace_rows)
            ),
            "untrained_append_cached_k0_vs_registered_disagreements": sum(
                int(grid[:, 0].ne(inplace[:, 0]).sum())
                for grid, inplace in zip(untrained_grid, inplace_rows)
            ),
            "primary_k0_source": "registered_full_sequence_depth_1",
            "positive_k_source": "incremental_cache_append",
        }
        scoring_rows = build_scoring_rows(
            rows=rows,
            teacher_rows=teacher_rows,
            inplace_rows=inplace_rows,
            trained_rows=trained_rows,
            untrained_rows=untrained_rows,
        )
        cache_sha = write_jsonl_once(immutable_cache, scoring_rows)
        after = parameter_fingerprint(wrapper)
        if before != after:
            raise RuntimeError("Stage A EVAL-C mutated the frozen recurrent model")
        write_json(
            immutable_receipt,
            {
                "kind": "paper2_dc1_stage_a_immutable_scoring_cache",
                "status": "complete_write_once",
                "sha256": cache_sha,
                "rows": len(scoring_rows),
                "positions": sum(row["scored_positions"] for row in scoring_rows),
                "arms": prereg["evaluation"]["arms"],
                "trained_append_counters": trained_counters,
                "untrained_append_counters": untrained_counters,
                "execution_path_anchor_diagnostics": anchor_diagnostics,
                "read_once_scoring_spent": True,
            },
        )
    else:
        if not immutable_receipt.exists():
            raise RuntimeError("immutable cache exists without its atomic receipt")
        cache_receipt = json.loads(immutable_receipt.read_text(encoding="utf-8"))
        cache_sha = sha256_file(immutable_cache)
        if cache_sha != cache_receipt["sha256"]:
            raise RuntimeError("immutable scoring cache changed after creation")
        scoring_rows = [
            json.loads(line)
            for line in immutable_cache.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        anchor_diagnostics = cache_receipt["execution_path_anchor_diagnostics"]
        print(f"stage_a_eval_resume immutable_cache={immutable_cache}", flush=True)

    verdict_path = output_dir / "verdict.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / prereg["evaluation"]["verdict_script"]),
            "--immutable_cache",
            str(immutable_cache),
            "--expected_cache_sha256",
            cache_sha,
            "--prereg",
            str(prereg_path),
            "--output_summary",
            str(verdict_path),
        ],
        cwd=ROOT,
        check=True,
    )
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    arms = [
        "trained_append_k1",
        "untrained_append_k1",
        "inplace_depth2_descriptive",
        "inplace_depth3_descriptive",
    ]
    summary = {
        "kind": "paper2_dc1_stage_a_single_eval_c_pass",
        "status": "complete_registered_attempt_consumed",
        "verdict": verdict,
        "by_stratum": summarize_by_stratum(scoring_rows, arms),
        "immutable_scoring_cache_sha256": cache_sha,
        "eval_partition": {
            "data_jsonl_sha256": eval_policy["jsonl_sha256"],
            "manifest_sha256": eval_policy["manifest_sha256"],
            "teacher_cache_sha256": eval_policy["teacher_cache_sha256"],
        },
        "read_once_scoring_spent": True,
        "arm_specific_rescoring_performed": False,
        "execution_path_anchor_diagnostics": anchor_diagnostics,
        "trained_checkpoint_sha256": trained_checkpoint_sha,
        "training_performed_during_evaluation": False,
        "optimizer_steps_during_evaluation": 0,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
