"""E3b: matched verbal fine-tuning from installed and fresh R16 initializations."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.adapter_parity_common import (
    ARM_E_ALPHA,
    ARM_E_FINAL_SHA256,
    ARM_E_PRETRAINED_BASE_SHA256,
    ARM_E_RANK,
    MODEL_NAME,
    adapter_resume_config,
    assert_adapter_training_summary,
    lora_eval_args,
    path_for_cli,
    read_json,
    restore_arm_e_checkpoint,
    run,
    sha256_file,
    write_json,
)
from colab.stage5_chain_consolidation_utils import publish_run
from training.adapter_verbal_transference import (
    PAIRED_ALPHA,
    TRANSFER_THRESHOLD,
    classify_regression,
    first_threshold_crossing,
    score_transference,
    summarize_archived_active_diagonal,
)


RUN_ID = os.environ.get(
    "STAGE5_ADAPTER_VERBAL_RUN_ID", "stage5_adapter_verbal_transference_e3b_20260720"
)
RUN_DIR = ROOT / "outputs/stage5" / RUN_ID
DATA_ROOT = ROOT / "outputs/stage5/stage5_natural_surface_transfer_20260708_230229/data"
TRAIN_DATA = DATA_ROOT / "rung0_train_mix_chain_symbol_sft.jsonl"
RELAY_DATA = DATA_ROOT / "relay_test_chain_mcq.jsonl"
POINTER_DATA = DATA_ROOT / "pointer_test_chain_mcq.jsonl"
SYNTHETIC_DATA = DATA_ROOT / "synthetic_rehearsal_chain_symbol_sft.jsonl"
TIER1_DATA = (
    ROOT / "outputs/stage5/stage5_adapter_budget_arm_e_20260718/data/base_capability_canary_64.jsonl"
)
FULL_BLOCK_ZERO_SHOT = (
    ROOT / "outputs/stage5/stage5_natural_surface_transfer_rung0_fixed_prompt_20260709_133812/summary.json"
)
LORA_ZERO_SHOT = ROOT / "outputs/stage5/stage5_adapter_parity_e3a_20260719/summary.json"
LEDGER = ROOT / "docs/part1_claim_evidence_ledger.json"
DRIVE_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints") / RUN_ID
MAX_STEPS = int(os.environ.get("STAGE5_ADAPTER_VERBAL_STEPS", "6000"))
CHECKPOINT_STEPS = list(range(1000, MAX_STEPS + 1, 1000))
EXPECTED_DATA_SHA256 = {
    "train": "4e963fcc2d464f7bbf0746697068701e9e0361611c875563d5e47220db207f58",
    "relay": "bde97fdbfab213845726a56af7a6b65f96426694457d068049cbe0476598951e",
    "pointer": "a3c15806ed4c089fdae125d972908c744a51c5a3ecbcf6d7b9d9d1683a481da6",
    "synthetic": "6b054327db1ae71aef8091b49e1da96b5dc3755d057c87824977dfb17a22d308",
}


def canonical_sha256(path: Path) -> str:
    raw = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def assert_frozen_data() -> dict[str, str]:
    paths = {"train": TRAIN_DATA, "relay": RELAY_DATA, "pointer": POINTER_DATA, "synthetic": SYNTHETIC_DATA}
    observed = {name: canonical_sha256(path) for name, path in paths.items()}
    if observed != EXPECTED_DATA_SHA256:
        raise RuntimeError(f"E3b frozen data changed: {observed}")
    print(f"[assert-ok] e3b_data_sha256={observed}", flush=True)
    return observed


def zero_shot_receipt() -> dict[str, Any]:
    full = read_json(FULL_BLOCK_ZERO_SHOT)["frozen_baseline"]["n24"]
    lora = read_json(LORA_ZERO_SHOT)
    result: dict[str, Any] = {"full_block": {}, "lora": {}}
    for family in ("relay", "pointer"):
        result["full_block"][family] = summarize_archived_active_diagonal(
            full[family]["active_diagonal"], rows_per_depth=128
        )
        result["lora"][family] = dict(lora["results"][family]["same_reader_total"])
    result["sources"] = {
        "full_block": path_for_cli(FULL_BLOCK_ZERO_SHOT),
        "lora": path_for_cli(LORA_ZERO_SHOT),
    }
    result["reading"] = "minimal_zero_shot_transfer_both_budgets"
    return result


def fresh_arm_s_checkpoint() -> tuple[Path, dict[str, Any]]:
    out = RUN_DIR / "init" / "arm_s"
    checkpoint = out / "peft_identity_step_0.pt"
    summary = out / "summary.json"
    drive_checkpoint = DRIVE_ROOT / "arm_s_init" / checkpoint.name
    if not checkpoint.exists() and drive_checkpoint.exists():
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(drive_checkpoint, checkpoint)
    if not checkpoint.exists() or not summary.exists():
        run(
            [
                sys.executable,
                "eval/eval_peft_identity.py",
                "--model_name",
                MODEL_NAME,
                "--rank",
                str(ARM_E_RANK),
                "--alpha",
                str(ARM_E_ALPHA),
                "--seed",
                "0",
                "--threshold",
                "0.001",
                "--dtype",
                os.environ.get("STAGE5_ADAPTER_PARITY_DTYPE", "bfloat16"),
                "--adapter_dtype",
                "float32",
                "--device",
                os.environ.get("DEVICE", "cuda"),
                "--output_summary",
                path_for_cli(summary),
                "--output_checkpoint",
                path_for_cli(checkpoint),
            ]
        )
    payload = read_json(summary)
    if not payload.get("passed") or payload.get("pretrained_base_sha256") != ARM_E_PRETRAINED_BASE_SHA256:
        raise RuntimeError(f"Arm S identity or base-hash gate failed: {payload}")
    observed = sha256_file(checkpoint)
    drive_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, drive_checkpoint)
    payload.update({"checkpoint_sha256": observed, "drive_backup": str(drive_checkpoint)})
    write_json(summary, payload)
    return checkpoint, payload


def diagonal_eval(
    *, label: str, checkpoint: Path, data_jsonl: Path, max_depth: int, value_prefix: str
) -> dict[str, Any]:
    out = RUN_DIR / "guardrails" / label
    summary = out / "summary.json"
    if not summary.exists():
        run(
            [
                sys.executable,
                "eval/eval_synthetic_diagonal_guardrail.py",
                "--data_jsonl",
                path_for_cli(data_jsonl),
                "--checkpoint",
                path_for_cli(checkpoint),
                "--output_jsonl",
                path_for_cli(out / "rows.jsonl"),
                "--output_summary",
                path_for_cli(summary),
                "--max_depth",
                str(max_depth),
                "--value_prefix",
                value_prefix,
                "--progress_every",
                "64",
                *lora_eval_args(),
            ]
        )
    return read_json(summary)


def final_symbol_eval(*, arm: str, step: int, family: str, checkpoint: Path) -> dict[str, Any]:
    out = RUN_DIR / "eval" / f"step_{step}" / arm / family
    rows_path = out / "rows.jsonl"
    summary = out / "summary.json"
    if not summary.exists():
        data = RELAY_DATA if family == "relay" else POINTER_DATA
        run(
            [
                sys.executable,
                "eval/eval_synthetic_depth_final_symbol.py",
                "--data_jsonl",
                path_for_cli(data),
                "--checkpoint",
                path_for_cli(checkpoint),
                "--output_jsonl",
                path_for_cli(rows_path),
                "--output_summary",
                path_for_cli(summary),
                "--max_loops",
                "12",
                "--threshold",
                str(TRANSFER_THRESHOLD),
                "--prompt_style",
                "question_only",
                "--value_prefix",
                "name:",
                "--progress_every",
                "64",
                *lora_eval_args(),
            ]
        )
    rows = read_jsonl(rows_path)
    compact = [
        {
            "id": row["id"],
            "depth": int(row["depth"]),
            "same_reader_final_hit": bool(row["same_reader_final_hit"]),
        }
        for row in rows
    ]
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in compact),
        encoding="utf-8",
    )
    return read_json(summary)


def write_train_config(
    *, arm: str, checkpoint: Path, canary_baseline: float
) -> Path:
    output_dir = RUN_DIR / "train" / arm
    cfg: dict[str, Any] = {
        "model_name": MODEL_NAME,
        "dtype": os.environ.get("STAGE5_ADAPTER_PARITY_DTYPE", "bfloat16"),
        "adapter_dtype": "float32",
        "attn_implementation": os.environ.get("ATTN_IMPLEMENTATION", "default"),
        "layer_split": "6,18",
        "max_length": 640,
        "max_loops": 8,
        "loop_loss_mode": "per_loop_labels",
        "initial_halt_prob": 0.15,
        "beta": 0.0,
        "halt_target_nll_weight": 0.0,
        "loop_control_ce_weight": 0.0,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "minimum_effective_batch_size": 1,
        "seed": 0,
        "optimizer": "adamw",
        "reject_muon": True,
        "learning_rate": 1e-5,
        "adamw_lr": 1e-5,
        "weight_decay": 0.0,
        "max_grad_norm": 0.5,
        "max_steps": MAX_STEPS,
        "save_every": 1000,
        "log_every": 100,
        "bridge_projection_mode": "split",
        "bridge_prelude_lr_multiplier": 10.0,
        "bridge_prelude_weight_decay": 0.0,
        "bridge_prelude_lr_include_norm": False,
        "bridge_prelude_grad_multiplier": 1.0,
        "train_on_prompt": False,
        "gradient_checkpointing": True,
        "require_active_supervision": True,
        "require_nonzero_train_gradient": True,
        "require_frozen_base_hash": True,
        "output_dir": path_for_cli(output_dir),
        "resume_from": path_for_cli(checkpoint),
        "recurrence_curriculum": {
            "enabled": False,
            "start_loop": 8,
            "end_loop": 8,
            "schedule": "linear",
            "target_source": "row_capped",
            "ramp_compute": False,
        },
        "track_loop_dose": True,
        "dose_assert_every": 1000,
        "canary_every": 1000,
        "canary_jsonl": path_for_cli(TIER1_DATA),
        "canary_value_prefix": "name:",
        "canary_baseline_accuracy": float(canary_baseline),
        "canary_hard_stop_delta": -0.03,
        "checkpoint_backup_every": 1000,
        "checkpoint_backup_dir": str(DRIVE_ROOT / arm),
        "progress_backup_path": str(DRIVE_ROOT / arm / "progress.json"),
        "synthetic_phase": "adapter_verbal_transference_e3b",
        "synthetic_stage": arm,
        "disposable_measurement_mode": True,
        "synthetic_regression_floor_enforced": False,
        **adapter_resume_config(),
    }
    path = RUN_DIR / "config" / f"{arm}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def train_arm(*, arm: str, checkpoint: Path, canary_baseline: float) -> dict[str, Any]:
    train_dir = RUN_DIR / "train" / arm
    summary_path = train_dir / "train_unfrozen_recurrent_summary.json"
    final_checkpoint = train_dir / f"unfrozen_recurrent_step_{MAX_STEPS}.pt"
    drive_final = DRIVE_ROOT / arm / final_checkpoint.name
    if summary_path.exists() and not final_checkpoint.exists() and drive_final.exists():
        final_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(drive_final, final_checkpoint)
        print(f"restored_{arm}_final={final_checkpoint}", flush=True)
    if not summary_path.exists() or not final_checkpoint.exists():
        config = write_train_config(arm=arm, checkpoint=checkpoint, canary_baseline=canary_baseline)
        result = run(
            [
                sys.executable,
                "training/train_unfrozen_recurrent.py",
                "--config",
                path_for_cli(config),
                "--train_jsonl",
                path_for_cli(TRAIN_DATA),
                "--device",
                os.environ.get("DEVICE", "cuda"),
            ],
            accepted_returncodes={0, 2},
        )
        train_dir.mkdir(parents=True, exist_ok=True)
        (train_dir / "train.log").write_text(result.stdout or "", encoding="utf-8")
    if not summary_path.exists():
        raise RuntimeError(f"{arm} training ended without a summary")
    summary = read_json(summary_path)
    assert_adapter_training_summary(summary)
    if not final_checkpoint.exists():
        return {"status": "blocked", "summary": summary, "final_checkpoint": None}
    return {
        "status": "finished",
        "summary": summary,
        "final_checkpoint": path_for_cli(final_checkpoint),
        "final_checkpoint_sha256": sha256_file(final_checkpoint),
    }


def checkpoint_for(arm: str, step: int, init: Path) -> Path:
    if step == 0:
        return init
    local = RUN_DIR / "train" / arm / f"unfrozen_recurrent_step_{step}.pt"
    if local.exists():
        return local
    drive = DRIVE_ROOT / arm / local.name
    if not drive.exists():
        raise FileNotFoundError(f"Missing {arm} step {step} locally and on Drive")
    local.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(drive, local)
    return local


def paired_rows(arm: str, step: int, family: str) -> list[dict[str, Any]]:
    return read_jsonl(RUN_DIR / "eval" / f"step_{step}" / arm / family / "rows.jsonl")


def aligned_hits(arm: str, step: int, family: str) -> dict[str, bool]:
    rows = paired_rows(arm, step, family)
    result = {str(row["id"]): bool(row["same_reader_final_hit"]) for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"Duplicate row ids in {arm}/{step}/{family}")
    return result


def comparison_at(step: int) -> dict[str, Any]:
    result: dict[str, Any] = {"step": step, "families": {}, "by_depth": {}}
    pooled_t: list[bool] = []
    pooled_s: list[bool] = []
    for family in ("relay", "pointer"):
        t = aligned_hits("arm_t", step, family)
        s = aligned_hits("arm_s", step, family)
        if t.keys() != s.keys():
            raise RuntimeError(f"Frozen row mismatch at step {step}/{family}")
        ids = sorted(t)
        result["families"][family] = score_transference(
            t_hits=[t[row_id] for row_id in ids],
            s_hits=[s[row_id] for row_id in ids],
            alpha=PAIRED_ALPHA,
        )
        pooled_t.extend(t[row_id] for row_id in ids)
        pooled_s.extend(s[row_id] for row_id in ids)
        rows = {str(row["id"]): row for row in paired_rows("arm_t", step, family)}
        for depth in range(1, 13):
            depth_ids = [row_id for row_id in ids if int(rows[row_id]["depth"]) == depth]
            bucket = result["by_depth"].setdefault(str(depth), {"t": [], "s": []})
            bucket["t"].extend(t[row_id] for row_id in depth_ids)
            bucket["s"].extend(s[row_id] for row_id in depth_ids)
    result["pooled"] = score_transference(t_hits=pooled_t, s_hits=pooled_s, alpha=PAIRED_ALPHA)
    result["by_depth"] = {
        depth: score_transference(t_hits=value["t"], s_hits=value["s"], alpha=PAIRED_ALPHA)
        for depth, value in result["by_depth"].items()
    }
    return result


def update_ledger(summary_path: Path, decision: dict[str, Any]) -> None:
    ledger = read_json(LEDGER)
    claims = [claim for claim in ledger["claims"] if claim.get("id") != "adapter_verbal_transference"]
    claims.append(
        {
            "id": "adapter_verbal_transference",
            "claim": "Matched adapter-budget verbal fine-tuning measured initialization transfer and synthetic regression.",
            "status": "supported_bounded",
            "scope": "single seed, frozen base, relay-trained and pointer-held-out verbal families, disposable Arm T/S continuations",
            "metrics": {
                "transference_verdict": decision["transference"]["verdict"],
                "regression_verdict": decision["regression"]["verdict"],
            },
            "evidence": [{"path": path_for_cli(summary_path), "locator": "decision and checkpoint_curves"}],
        }
    )
    ledger["claims"] = claims
    LEDGER.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")


def publish_ledger() -> None:
    subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", "main"], cwd=ROOT, check=True)
    subprocess.run(["git", "add", path_for_cli(LEDGER)], cwd=ROOT, check=True)
    diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT)
    if diff.returncode == 0:
        return
    subprocess.run(
        ["git", "commit", "-m", f"Bank E3b adapter verbal transference {RUN_ID} [skip ci]"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=True)


def main() -> int:
    data_sha = assert_frozen_data()
    p1 = zero_shot_receipt()
    arm_t_init, t_restore = restore_arm_e_checkpoint(RUN_DIR / "init" / "arm_t_e1.pt")
    arm_s_init, s_identity = fresh_arm_s_checkpoint()
    init_checkpoints = {"arm_t": arm_t_init, "arm_s": arm_s_init}
    init_receipt = {
        "kind": "stage5_adapter_verbal_transference_e3b",
        "run_id": RUN_ID,
        "status": "initialized",
        "p1_zero_shot": p1,
        "data_sha256": data_sha,
        "arm_t": {"source": "installed_E1_R16", "checkpoint_sha256": ARM_E_FINAL_SHA256, "restore": t_restore},
        "arm_s": {"source": "fresh_base_surgery_R16", "identity": s_identity},
        "locked_protocol": {
            "only_variable": "initialization_symbolic_history",
            "training_mix": "2048 relay plus 2048 synthetic rehearsal; pointer held out",
            "steps": MAX_STEPS,
            "checkpoints": CHECKPOINT_STEPS,
            "optimizer": "adamw",
            "learning_rate": 1e-5,
            "seed": 0,
            "base_weights": "frozen",
            "synthetic_regression": "measurement_only_arm_t",
            "tier1": "hard_stop_each_arm_against_own_step0_baseline",
        },
    }
    write_json(RUN_DIR / "summary.json", init_receipt)
    publish_run(RUN_DIR, message=f"Record E3b initialization {RUN_ID} [skip ci]")

    canary_baselines: dict[str, dict[str, Any]] = {}
    training: dict[str, Any] = {}
    for arm in ("arm_t", "arm_s"):
        baseline = diagonal_eval(
            label=f"{arm}_step_0_tier1",
            checkpoint=init_checkpoints[arm],
            data_jsonl=TIER1_DATA,
            max_depth=1,
            value_prefix="name:",
        )
        canary_baselines[arm] = baseline
        training[arm] = train_arm(
            arm=arm,
            checkpoint=init_checkpoints[arm],
            canary_baseline=float(baseline["accuracy"]),
        )
        init_receipt.update({"status": f"{arm}_training_{training[arm]['status']}", "canary_baselines": canary_baselines, "training": training})
        write_json(RUN_DIR / "summary.json", init_receipt)
        publish_run(RUN_DIR, message=f"Record E3b {arm} training {RUN_ID} [skip ci]")
        if training[arm]["status"] != "finished":
            return 2

    comparisons: dict[str, Any] = {}
    synthetic_by_step: dict[str, dict[str, float]] = {}
    tier1_by_step: dict[str, Any] = {}
    for step in [0, *CHECKPOINT_STEPS]:
        for arm in ("arm_t", "arm_s"):
            checkpoint = checkpoint_for(arm, step, init_checkpoints[arm])
            for family in ("relay", "pointer"):
                final_symbol_eval(arm=arm, step=step, family=family, checkpoint=checkpoint)
        t_checkpoint = checkpoint_for("arm_t", step, arm_t_init)
        synthetic = diagonal_eval(
            label=f"arm_t_step_{step}_synthetic",
            checkpoint=t_checkpoint,
            data_jsonl=SYNTHETIC_DATA,
            max_depth=8,
            value_prefix="letter:",
        )
        tier1 = diagonal_eval(
            label=f"arm_t_step_{step}_tier1",
            checkpoint=t_checkpoint,
            data_jsonl=TIER1_DATA,
            max_depth=1,
            value_prefix="name:",
        )
        synthetic_by_step[str(step)] = {key: float(value) for key, value in synthetic["active_diagonal"].items()}
        tier1_by_step[str(step)] = {"correct": tier1["correct"], "rows": tier1["rows"], "accuracy": tier1["accuracy"]}
        comparisons[str(step)] = comparison_at(step)
        init_receipt.update(
            {
                "status": f"evaluated_step_{step}",
                "comparisons": comparisons,
                "arm_t_synthetic_by_step": synthetic_by_step,
                "arm_t_tier1_by_step": tier1_by_step,
            }
        )
        write_json(RUN_DIR / "summary.json", init_receipt)
        publish_run(RUN_DIR, message=f"Record E3b matched eval step {step} {RUN_ID} [skip ci]")

    pooled_curves = {
        arm: {
            step: float(comparisons[step]["pooled"][arm]["accuracy"])
            for step in comparisons
        }
        for arm in ("arm_t", "arm_s")
    }
    dose_crossing = {
        arm: first_threshold_crossing(curve, threshold=TRANSFER_THRESHOLD)
        for arm, curve in pooled_curves.items()
    }
    endpoint = comparisons[str(MAX_STEPS)]["pooled"]
    regression = classify_regression(synthetic_by_step)
    decision = {
        "transference": endpoint,
        "dose_to_0p71_pooled": dose_crossing,
        "regression": regression,
        "full_block_endpoint_reference": {
            "relay": {"correct": 1321, "total": 1536, "accuracy": 1321 / 1536},
            "pointer": {"correct": 1213, "total": 1536, "accuracy": 1213 / 1536},
            "comparison": "descriptive_only",
        },
    }
    init_receipt.update(
        {
            "status": "finished",
            "comparisons": comparisons,
            "pooled_curves": pooled_curves,
            "arm_t_synthetic_by_step": synthetic_by_step,
            "arm_t_tier1_by_step": tier1_by_step,
            "decision": decision,
            "limitations": ["single_seed", "relay_trained_pointer_held_out", "disposable_no_lineage_promotion"],
        }
    )
    summary_path = RUN_DIR / "summary.json"
    write_json(summary_path, init_receipt)
    (RUN_DIR / "summary.md").write_text(
        "\n".join(
            [
                f"# E3b Adapter Verbal Transference - {RUN_ID}",
                "",
                f"- P1 full-block zero-shot: `{p1['full_block']}`",
                f"- P1 LoRA zero-shot: `{p1['lora']}`",
                f"- Matched endpoint: `{endpoint}`",
                f"- Dose to 0.71 pooled: `{dose_crossing}`",
                f"- Synthetic regression: `{regression}`",
                f"- Tier-1 trajectory: `{tier1_by_step}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    update_ledger(summary_path, decision)
    publish_run(RUN_DIR, message=f"Record completed E3b verbal transference {RUN_ID} [skip ci]")
    publish_ledger()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
