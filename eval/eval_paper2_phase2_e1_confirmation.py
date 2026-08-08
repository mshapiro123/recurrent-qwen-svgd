"""Run the locked, read-once Phase-2 E1 confirmation on frozen EVAL-D."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from eval.eval_paper2_phase2_option_b_bootstrap_audit import (
    _bootstrap_document_means,
)
from training.paper2_phase2_e1_confirmation import (
    LOCKED_REGISTRATION,
    git_lf_sha256_file,
    sha256_file,
)
from training.run_paper2_phase2_a2 import _load_module, _tensor_digest, evaluate
from training.run_paper2_phase2_matched_alpha import (
    _decoder_for_alpha,
    _load_trainable_state,
)


ROOT = Path(__file__).resolve().parents[1]
LOCK_COMMIT = "ebe4ea4b27d0633ca5b5dcaabc1cb3cdacc4ca37"
LOCKED_REGISTRATION_SHA256 = (
    "436b75f06c9ff4859f0526f72ec428ff7e5f08691308b37edc5848e1f5b219a1"
)
RUN_KIND = "paper2_phase2_e1_confirmation_v1"
LEASE_KIND = "paper2_phase2_e1_read_once_lease_v1"
A1_SHA256 = {
    0: "823c1865878a86079a6423fabf432b6f1d36d431ec4381800846019882afb136",
    1: "a9c20510f6cf2561f6208fa8d1915626e2ec6e68a588228d3f0edd9cd0efde89",
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _assert_lock_ancestor() -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", LOCK_COMMIT, "HEAD"], cwd=ROOT
    )
    if result.returncode:
        raise RuntimeError("E1 runner commit does not descend from the preregistration lock")


def _finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if isinstance(value, float):
        return bool(np.isfinite(value))
    return True


def _measurement(point: float, samples: np.ndarray) -> dict[str, Any]:
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "estimate": float(point),
        "document_bootstrap_95_ci": [float(lower), float(upper)],
    }


def _weighted_eal(
    values: np.ndarray, strata: list[str], weights: Mapping[str, float]
) -> float:
    return float(
        sum(
            float(weights[name]) * float(values[np.asarray(strata) == name].mean())
            for name in ("general", "code")
        )
    )


def _arm_rows_path(private_dir: Path, *, seed: int, arm: str) -> Path:
    return private_dir / f"seed_{seed}_{arm}_rows.pt"


def _claim_lease(path: Path, registration_sha256: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": LEASE_KIND,
        "status": "claimed_unscored",
        "claimed_at_unix": time.time(),
        "registration_sha256": registration_sha256,
        "lock_commit": LOCK_COMMIT,
        "read_once_scoring_spent": False,
        "score_exposure_started": False,
    }
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        existing = json.loads(path.read_text(encoding="utf-8"))
        raise RuntimeError(
            "E1 read-once lease already exists; automatic rerun is prohibited: "
            f"{existing.get('status')}"
        ) from exc
    return payload


def _update_lease(path: Path, payload: dict[str, Any], status: str, **updates: Any) -> None:
    payload.update({"status": status, "updated_at_unix": time.time(), **updates})
    write_json(path, payload)


def validate_static_inputs(
    *,
    registration: Mapping[str, Any],
    freeze_receipt: Path,
    readiness_receipt: Path,
    sparse_qc_receipt: Path,
    endpoint_receipt: Path,
    option_b_summary: Path,
    cache_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _assert_lock_ancestor()
    if git_lf_sha256_file(LOCKED_REGISTRATION) != LOCKED_REGISTRATION_SHA256:
        raise RuntimeError("E1 locked preregistration SHA differs")
    if registration.get("status") != "locked_before_e1_scoring":
        raise RuntimeError("E1 preregistration status is not locked")
    if registration.get("locked_before_e1_scoring") is not True:
        raise RuntimeError("E1 lock flag is absent")
    if registration.get("e1_evaluation_authorized") is not True:
        raise RuntimeError("E1 evaluation authorization is absent")
    if registration.get("training_authorized") is not False:
        raise RuntimeError("E1 training must remain prohibited")
    if registration.get("lock_blockers") != []:
        raise RuntimeError("E1 lock contains unresolved blockers")
    if registration["resource_note"]["e1_scoring"] != (
        "A100_80GB_required_conservatively_no_score_blind_endpoint_memory_preflight_run"
    ):
        raise RuntimeError("E1 resource lock differs")

    artifact_paths = {
        "eval_d_freeze": freeze_receipt,
        "readiness": readiness_receipt,
        "sparse_support_qc": sparse_qc_receipt,
        "endpoint_lock_preparation": endpoint_receipt,
    }
    for name, path in artifact_paths.items():
        expected = registration["lock_artifacts"][name]["sha256"]
        if git_lf_sha256_file(path) != expected:
            raise RuntimeError(f"E1 lock artifact SHA differs: {name}")
    inventory = ROOT / registration["lock_artifacts"]["rule_inventory"]["path"]
    if git_lf_sha256_file(inventory) != registration["lock_artifacts"][
        "rule_inventory"
    ]["sha256"]:
        raise RuntimeError("E1 rule inventory SHA differs")
    scorer_path = ROOT / registration["evaluation"]["scorer_source"].split(":", 1)[0]
    if git_lf_sha256_file(scorer_path) != registration["evaluation"][
        "scorer_source_sha256"
    ]:
        raise RuntimeError("E1 inherited evaluator SHA differs")

    freeze = json.loads(freeze_receipt.read_text(encoding="utf-8"))
    readiness = json.loads(readiness_receipt.read_text(encoding="utf-8"))
    sparse = json.loads(sparse_qc_receipt.read_text(encoding="utf-8"))
    endpoint = json.loads(endpoint_receipt.read_text(encoding="utf-8"))
    option_b = json.loads(option_b_summary.read_text(encoding="utf-8"))
    if readiness.get("ready_to_lock") is not True or readiness.get("blockers") != []:
        raise RuntimeError("E1 readiness receipt is not clean")
    if readiness.get("read_once_scoring_spent") is not False:
        raise RuntimeError("E1 readiness reports prior score exposure")
    if freeze.get("status") != "complete_frozen_unscored":
        raise RuntimeError("E1 EVAL-D cache is not frozen unscored")
    if freeze.get("read_once_scoring_spent") is not False:
        raise RuntimeError("E1 freeze reports prior score exposure")
    if sparse.get("all_emitted_metrics_finite") is not True:
        raise RuntimeError("E1 sparse-support QC is not finite")
    if endpoint.get("ready_for_lock_transcription") is not True:
        raise RuntimeError("E1 endpoint receipt is incomplete")
    if sha256_file(cache_path) != registration["evaluation"]["private_cache_sha256"]:
        raise RuntimeError("E1 private cache SHA differs")
    if sha256_file(option_b_summary) != readiness["inputs"]["option_b_summary_sha256"]:
        raise RuntimeError("E1 Option B public summary SHA differs")
    return freeze, option_b


def load_final_module(
    *,
    seed: int,
    arm: str,
    a1_checkpoint: Path,
    endpoint: Path,
    registration: Mapping[str, Any],
    option_b_summary: Mapping[str, Any],
    embedding_weight: torch.Tensor,
    rms_cap: float,
    device: str,
) -> tuple[nn.Module, nn.Embedding, dict[str, Any]]:
    name = f"seed_{seed}_{arm}"
    locked = registration["checkpoints"][name]
    if sha256_file(endpoint) != locked["sha256"]:
        raise RuntimeError(f"E1 endpoint file SHA differs: {name}")
    embedding = nn.Embedding.from_pretrained(embedding_weight.float(), freeze=True).to(device)
    module, _source = _load_module(
        seed=seed,
        checkpoint_path=a1_checkpoint,
        expected_sha=A1_SHA256[seed],
        embedding=embedding,
        rms_cap=rms_cap,
        device=device,
        arm=arm,
    )
    saved = torch.load(endpoint, map_location="cpu", weights_only=False)
    if (
        saved.get("kind") != "paper2_phase2_option_b_arm_v1"
        or saved.get("name") != name
        or int(saved.get("seed", -1)) != seed
        or saved.get("arm") != arm
        or int(saved.get("step", -1)) != int(locked["expected_step"])
        or saved.get("abort_reason") is not None
    ):
        raise RuntimeError(f"E1 endpoint metadata differs: {name}")
    state = saved.get("trainable_state")
    if _tensor_digest(state) != locked["semantic_trainable_state_digest"]:
        raise RuntimeError(f"E1 endpoint semantic digest differs: {name}")
    _load_trainable_state(module, state)
    active = {key: value for key, value in module.named_parameters() if value.requires_grad}
    if _tensor_digest(active) != locked["semantic_trainable_state_digest"]:
        raise RuntimeError(f"E1 loaded trainable state differs: {name}")
    source_row = next(
        row
        for row in option_b_summary["arms"]
        if int(row["seed"]) == seed and row["arm"] == arm
    )
    frozen = {key: value for key, value in module.named_parameters() if not value.requires_grad}
    if _tensor_digest(frozen) != source_row["frozen_parameter_hash_after"]:
        raise RuntimeError(f"E1 reconstructed frozen substrate differs: {name}")
    all_before = _tensor_digest(dict(module.named_parameters()))
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in module.parameters()):
        raise RuntimeError("E1 module unexpectedly retains trainable parameters")
    return module, embedding, {
        "name": name,
        "file_sha256": locked["sha256"],
        "semantic_trainable_state_digest": locked["semantic_trainable_state_digest"],
        "frozen_parameter_digest": source_row["frozen_parameter_hash_after"],
        "all_parameter_digest_before": all_before,
        "a1_checkpoint_sha256": A1_SHA256[seed],
    }


def summarize(
    *,
    arm_summaries: dict[str, dict[str, Any]],
    arm_rows: dict[str, dict[str, torch.Tensor]],
    documents: list[str],
    strata: list[str],
    positions: torch.Tensor,
    registration: Mapping[str, Any],
) -> dict[str, Any]:
    trajectories = {
        name: rows["accepted_length"].double().numpy()[:, None]
        for name, rows in arm_rows.items()
    }
    bootstrap_spec = registration["evaluation"]["bootstrap"]
    bootstrap = _bootstrap_document_means(
        trajectories=trajectories,
        document_ids=documents,
        replicates=int(bootstrap_spec["replicates"]),
        seed=int(bootstrap_spec["seed"]),
    )
    seeds = []
    primary_passes = []
    quality_passes = []
    dev_weights = registration["dev_mixture_reweighted_secondary"]["weights"]
    strata_array = np.asarray(strata)
    position_array = positions.numpy()
    for seed in (0, 1):
        full_name = f"seed_{seed}_full_a2"
        control_name = f"seed_{seed}_draft_only_control"
        full = trajectories[full_name][:, 0]
        control = trajectories[control_name][:, 0]
        point_gap = float(full.mean() - control.mean())
        point_relative = point_gap / max(float(control.mean()), 1e-12)
        bootstrap_gap = bootstrap[full_name][:, 0] - bootstrap[control_name][:, 0]
        bootstrap_relative = bootstrap_gap / np.clip(
            bootstrap[control_name][:, 0], 1e-12, None
        )
        gap = _measurement(point_gap, bootstrap_gap)
        relative = _measurement(point_relative, bootstrap_relative)
        primary_pass = bool(relative["document_bootstrap_95_ci"][0] > 0.0)
        primary_passes.append(primary_pass)
        full_summary = arm_summaries[full_name]
        quality_pass = bool(
            float(full_summary["retention"])
            >= float(registration["quality"]["point_retention_minimum"])
            and float(full_summary["retention_wilson_95_lower"])
            >= float(registration["quality"]["wilson_95_lower_minimum"])
        )
        quality_passes.append(quality_pass)
        by_stratum = {}
        for stratum in ("general", "code"):
            mask = strata_array == stratum
            by_stratum[stratum] = {
                "anchors": int(mask.sum()),
                "full_mean_eal": float(full[mask].mean()),
                "control_mean_eal": float(control[mask].mean()),
                "relative_full_gain": float(
                    (full[mask].mean() - control[mask].mean())
                    / max(float(control[mask].mean()), 1e-12)
                ),
            }
        full_dev = _weighted_eal(full, strata, dev_weights)
        control_dev = _weighted_eal(control, strata, dev_weights)
        by_bucket = {}
        bucket_edges = {
            "position_0": position_array == 0,
            "position_1_3": (position_array >= 1) & (position_array <= 3),
            "position_4_31": (position_array >= 4) & (position_array <= 31),
            "position_32_127": (position_array >= 32) & (position_array <= 127),
            "position_128_plus": position_array >= 128,
        }
        for label, mask in bucket_edges.items():
            by_bucket[label] = {
                "anchors": int(mask.sum()),
                "full_minus_control_eal": (
                    float((full[mask] - control[mask]).mean()) if bool(mask.any()) else None
                ),
            }
        seeds.append(
            {
                "seed": seed,
                "primary_absolute_eal_gain": gap,
                "primary_relative_eal_gain": relative,
                "primary_pass": primary_pass,
                "quality": {
                    "baseline_correct": int(full_summary["baseline_correct"]),
                    "retained_correct": int(full_summary["retained_correct"]),
                    "retention": float(full_summary["retention"]),
                    "wilson_95_lower": float(full_summary["retention_wilson_95_lower"]),
                    "pass": quality_pass,
                    "diagnostic_0p997_met": bool(float(full_summary["retention"]) >= 0.997),
                },
                "by_stratum": by_stratum,
                "by_position_bucket": by_bucket,
                "dev_mixture_reweighted_secondary": {
                    "weights": dev_weights,
                    "full_mean_eal": full_dev,
                    "control_mean_eal": control_dev,
                    "relative_full_gain": (full_dev - control_dev)
                    / max(control_dev, 1e-12),
                },
                "one_percent_exploratory_target_met": bool(point_relative >= 0.01),
            }
        )
    if all(primary_passes) and all(quality_passes):
        verdict = "CONFIRMED_WITH_MEASURED_PARETO"
    elif all(primary_passes):
        verdict = "EFFECT_REPLICATES_QUALITY_BOUNDARY_FAILS"
    elif all(quality_passes):
        verdict = "MECHANISM_NOT_CONFIRMED"
    else:
        verdict = "BOTH_CONFIRMATION_REQUIREMENTS_FAIL"
    return {
        "bootstrap": {
            "unit": "document",
            "method": "paired percentile cluster bootstrap",
            "confidence_level": 0.95,
            "replicates": int(bootstrap_spec["replicates"]),
            "seed": int(bootstrap_spec["seed"]),
            "anchors": len(documents),
            "unique_documents": len(set(documents)),
            "seed_inference": False,
        },
        "seeds": seeds,
        "scripted_verdict": verdict,
        "primary_pass_both_seeds": all(primary_passes),
        "quality_pass_both_seeds": all(quality_passes),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("E1 requires CUDA")
    if torch.cuda.get_device_properties(0).total_memory < 70 * 2**30:
        raise RuntimeError("locked E1 resource note requires A100 80GB class VRAM")
    registration = json.loads(LOCKED_REGISTRATION.read_text(encoding="utf-8"))
    freeze, option_b = validate_static_inputs(
        registration=registration,
        freeze_receipt=args.freeze_receipt,
        readiness_receipt=args.readiness_receipt,
        sparse_qc_receipt=args.sparse_qc_receipt,
        endpoint_receipt=args.endpoint_receipt,
        option_b_summary=args.option_b_summary,
        cache_path=args.cache,
    )
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    if cache.get("kind") != registration["evaluation"]["option_b_cache_schema_required"]:
        raise RuntimeError("E1 cache kind differs")
    if len(cache["documents"]) != int(registration["evaluation"]["actual_anchor_count"]):
        raise RuntimeError("E1 cache anchor count differs")
    required = set(freeze["option_b_cache"]["fields"])
    if not required.issubset(cache):
        raise RuntimeError(f"E1 cache fields missing: {sorted(required - set(cache))}")

    student_head = torch.load(args.student_head, map_location="cpu", weights_only=False)
    teacher_head = torch.load(args.teacher_head, map_location="cpu", weights_only=False)
    expected_revisions = freeze["teacher_stack"]["model_revisions"]
    for key, payload in (("student_0p5b", student_head), ("teacher_14b", teacher_head)):
        if (
            payload.get("kind") != "paper2_phase2_stage0a_lm_head"
            or payload.get("model_key") != key
            or payload.get("revision") != expected_revisions[key]
        ):
            raise RuntimeError(f"E1 frozen LM-head identity differs: {key}")
    student_weight = student_head["weight_bfloat16"]
    teacher_weight = teacher_head["weight_bfloat16"]
    teacher_embedding = nn.Embedding.from_pretrained(
        teacher_weight.float(), freeze=True
    ).to(args.device)
    decoder, decoder_bias = _decoder_for_alpha(cache, alpha=0.5, device=args.device)
    indices = torch.arange(len(cache["documents"]), dtype=torch.long)

    lease = _claim_lease(args.lease, LOCKED_REGISTRATION_SHA256)
    scoring_started = False
    arm_summaries: dict[str, dict[str, Any]] = {}
    arm_rows: dict[str, dict[str, torch.Tensor]] = {}
    identities: dict[str, dict[str, Any]] = {}
    try:
        for seed in (0, 1):
            for arm in ("full_a2", "draft_only_control"):
                name = f"seed_{seed}_{arm}"
                module, embedding, identity = load_final_module(
                    seed=seed,
                    arm=arm,
                    a1_checkpoint=args.a1_checkpoints[seed],
                    endpoint=args.endpoints[name],
                    registration=registration,
                    option_b_summary=option_b,
                    embedding_weight=student_weight,
                    rms_cap=float(args.rms_cap),
                    device=args.device,
                )
                if not scoring_started:
                    scoring_started = True
                    _update_lease(
                        args.lease,
                        lease,
                        "scoring_started",
                        score_exposure_started=True,
                        read_once_scoring_spent=True,
                    )
                metrics, rows = evaluate(
                    module=module,
                    cache=cache,
                    indices=indices,
                    embedding=embedding,
                    teacher_embedding=teacher_embedding,
                    decoder=decoder,
                    decoder_bias=decoder_bias,
                    arm=arm,
                    device=args.device,
                )
                if not _finite(metrics) or any(
                    not bool(torch.isfinite(value).all())
                    for value in rows.values()
                    if value.is_floating_point()
                ):
                    raise RuntimeError(f"E1 non-finite arm output: {name}")
                identity["all_parameter_digest_after"] = _tensor_digest(
                    dict(module.named_parameters())
                )
                if identity["all_parameter_digest_after"] != identity[
                    "all_parameter_digest_before"
                ]:
                    raise RuntimeError(f"E1 model mutation detected: {name}")
                row_payload = {
                    **rows,
                    "documents": list(cache["documents"]),
                    "anchor_keys": list(cache["anchor_keys"]),
                    "strata": list(cache["strata"]),
                    "positions": cache["positions"].clone(),
                }
                row_path = _arm_rows_path(args.private_dir, seed=seed, arm=arm)
                row_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = row_path.with_suffix(".pt.tmp")
                torch.save(row_payload, temporary)
                os.replace(temporary, row_path)
                row_hash = sha256_file(row_path)
                if abs(float(rows["accepted_length"].mean()) - float(metrics[
                    "mean_accepted_length"
                ])) > 2e-6:
                    raise RuntimeError(f"E1 row/public aggregate mismatch: {name}")
                arm_summaries[name] = {**metrics, "row_receipt_sha256": row_hash}
                arm_rows[name] = rows
                identities[name] = identity
                print(
                    f"e1_arm_complete name={name} mean_eal={metrics['mean_accepted_length']:.9f} "
                    f"retention={metrics['retention']:.9f} rows_sha256={row_hash}",
                    flush=True,
                )
                del module, embedding
                torch.cuda.empty_cache()

        analysis = summarize(
            arm_summaries=arm_summaries,
            arm_rows=arm_rows,
            documents=list(cache["documents"]),
            strata=list(cache["strata"]),
            positions=cache["positions"],
            registration=registration,
        )
        result = {
            "kind": RUN_KIND,
            "status": "complete",
            "launcher_commit": _git_head(),
            "lock_commit": LOCK_COMMIT,
            "registration_sha256": LOCKED_REGISTRATION_SHA256,
            "read_once_scoring_spent": True,
            "eval_e_touched": False,
            "training_started": False,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "cache": {
                "sha256": sha256_file(args.cache),
                "anchors": len(cache["documents"]),
                "documents": len(set(cache["documents"])),
                "strata": {
                    name: list(cache["strata"]).count(name)
                    for name in ("general", "code")
                },
            },
            "endpoint_identities": identities,
            "arms": arm_summaries,
            **analysis,
            "do_not_claim": [
                "two seeds estimate a seed population",
                "alpha 0.5 is selected or optimal",
                "the effect is at least one percent unless measured here",
                "quality neutrality",
                "generalization beyond EVAL-D and this frozen protocol",
                "a descriptive secondary rescues a failed primary endpoint",
            ],
        }
        if not _finite(result):
            raise RuntimeError("E1 public summary contains a non-finite value")
        write_json(args.output, result)
        _update_lease(
            args.lease,
            lease,
            "spent_complete",
            read_once_scoring_spent=True,
            score_exposure_started=True,
            output_sha256=sha256_file(args.output),
            scripted_verdict=result["scripted_verdict"],
        )
        return result
    except BaseException as exc:
        _update_lease(
            args.lease,
            lease,
            "spent_failed" if scoring_started else "claimed_failed_unspent",
            read_once_scoring_spent=scoring_started,
            score_exposure_started=scoring_started,
            exception_type=type(exc).__name__,
            exception=str(exc),
            traceback=traceback.format_exc(),
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze_receipt", type=Path, required=True)
    parser.add_argument("--readiness_receipt", type=Path, required=True)
    parser.add_argument("--sparse_qc_receipt", type=Path, required=True)
    parser.add_argument("--endpoint_receipt", type=Path, required=True)
    parser.add_argument("--option_b_summary", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--student_head", type=Path, required=True)
    parser.add_argument("--teacher_head", type=Path, required=True)
    parser.add_argument("--a1_checkpoint_seed_0", type=Path, required=True)
    parser.add_argument("--a1_checkpoint_seed_1", type=Path, required=True)
    for seed in (0, 1):
        for arm in ("full_a2", "draft_only_control"):
            parser.add_argument(f"--endpoint_seed_{seed}_{arm}", type=Path, required=True)
    parser.add_argument("--rms_cap", type=float, required=True)
    parser.add_argument("--private_dir", type=Path, required=True)
    parser.add_argument("--lease", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.a1_checkpoints = {
        seed: getattr(args, f"a1_checkpoint_seed_{seed}") for seed in (0, 1)
    }
    args.endpoints = {
        f"seed_{seed}_{arm}": getattr(args, f"endpoint_seed_{seed}_{arm}")
        for seed in (0, 1)
        for arm in ("full_a2", "draft_only_control")
    }
    return args


def main() -> int:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
