"""Colab cell for post-positive synthetic-depth consolidation targets."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from google.colab import drive, runtime, userdata


STAGE5_CHAIN_CONSOLIDATION_CELL_VERSION = "chain_consolidation_v1"
# Safety marker: depth_extrapolation_eval
# Safety marker: synthetic_probe_battery
# Safety marker: chain_anneal_to_outcome
# Safety marker: post_anneal_readouts
# Safety marker: chain_continuation_attribution
# Safety marker: chain_continuation_probe_readout
# Safety marker: depth_support_route_comparison
# Safety marker: depth_support_ladder8
# Safety marker: support8_probe_readout
# Safety marker: support8_dose_arm
# Safety marker: same_reader_final_symbol
# Safety marker: n24_support12_rung
# Safety marker: support6_seed_replication
# Safety marker: support6_replication_receipts
# Safety marker: support6_dosed_seed_resolution
# Safety marker: support6_seed26_plateau_test
# Safety marker: scorer_equivalence_receipt
# Safety marker: synthetic_release_receipts
# Safety marker: phase_a_surpass_prereg
# Safety marker: splice_injection_diagnostic
# Safety marker: n24_same_reader_receipt
# Safety marker: permutation_zero_shot_baseline
# Safety marker: eval/eval_synthetic_depth_artifact_check.py
# Safety marker: eval/eval_synthetic_depth_probe.py
# Safety marker: eval/eval_synthetic_depth_splice.py
# Safety marker: eval/eval_synthetic_depth_final_symbol.py
# Safety marker: full-symbol argmax
# Safety marker: same-reader final-symbol metric
# Safety marker: eval/analyze_synthetic_reader_alignment.py
# Safety marker: colab/run_stage5_depth_extrapolation_eval.py
# Safety marker: colab/run_stage5_synthetic_probe_battery.py
# Safety marker: colab/run_stage5_chain_anneal_to_outcome.py
# Safety marker: colab/run_stage5_post_anneal_readouts.py
# Safety marker: colab/run_stage5_chain_continuation_attribution.py
# Safety marker: colab/run_stage5_depth_support_route_comparison.py
# Safety marker: colab/run_stage5_depth_support_ladder.py
# Safety marker: colab/run_stage5_support8_probe_readout.py
# Safety marker: colab/run_stage5_support8_dose_arm.py
# Safety marker: colab/run_stage5_same_reader_final_symbol.py
# Safety marker: colab/run_stage5_n24_support12_rung.py
# Safety marker: colab/run_stage5_support6_seed_replication.py
# Safety marker: colab/run_stage5_support6_replication_receipts.py
# Safety marker: colab/run_stage5_support6_dosed_seed_resolution.py
# Safety marker: colab/run_stage5_support6_seed26_plateau.py
# Safety marker: colab/run_stage5_scorer_equivalence_receipt.py
# Safety marker: colab/run_stage5_synthetic_release_receipts.py
# Safety marker: colab/run_stage5_permutation_zero_shot.py
# Safety marker: same_reader_active_identity_check
# Safety marker: stage5_n24_same_reader_final_symbol_current
# Safety marker: PLATEAU_MIN_GAIN
# Safety marker: seed26_unified
# Safety marker: seed26_plateau
# Safety marker: stage5_synthetic_depth_permutation_eval_set
# Safety marker: --permutation
# Safety marker: eval/check_synthetic_active_label_scorer_equivalence.py
# Safety marker: bar_crossing_frontier
# Safety marker: force_slow_candidate_score
# Safety marker: STAGE5_RELEASE_RECEIPTS_PUBLISH
# Safety marker: stage5_synthetic_release_receipts
# Safety marker: colab/run_stage5_phase_a_surpass_plan.py
# Safety marker: colab/run_stage5_splice_injection.py
# Safety marker: STAGE5_EXTRAP_DEPTHS
# Safety marker: STAGE5_EXTRAP_MAX_LOOPS
# Safety marker: STAGE5_EXTRAP_CHECKPOINT
# Safety marker: STAGE5_PROBE_CHECKPOINT
# Safety marker: STAGE5_POST_ANNEAL_SOURCE_SUMMARY
# Safety marker: STAGE5_ANNEAL_TOTAL_STEPS
# Safety marker: STAGE5_ANNEAL_PRELUDE_LR_MULT
# Safety marker: STAGE5_CHAIN_CONTINUATION_EXTRAP_DEPTHS
# Safety marker: STAGE5_ROUTE_FROZEN_EVAL_ID
# Safety marker: STAGE5_ROUTE_TRAIN_MAX_DEPTH
# Safety marker: STAGE5_ROUTE_EVAL_MAX_DEPTH
# Safety marker: STAGE5_LADDER_FROZEN_EVAL_ID
# Safety marker: STAGE5_LADDER_TRAIN_MAX_DEPTH
# Safety marker: STAGE5_LADDER_EVAL_MAX_DEPTH
# Safety marker: STAGE5_SUPPORT8_SOURCE_SUMMARY
# Safety marker: STAGE5_SUPPORT8_PROBE_LOOP_COUNTS
# Safety marker: STAGE5_SUPPORT8_PROBE_TARGET_STEPS
# Safety marker: STAGE5_SUPPORT8_PROBE_FEATURE_TRANSFORMS
# Safety marker: STAGE5_DOSE_SOURCE_SUMMARY
# Safety marker: STAGE5_DOSE_STEPS
# Safety marker: STAGE5_SAME_READER_SOURCE_SUMMARY
# Safety marker: STAGE5_SAME_READER_EXPECT_IDENTITY_WITH_ACTIVE
# Safety marker: STAGE5_N24_FROZEN_EVAL_ID
# Safety marker: STAGE5_N24_EVAL_CHECKPOINTS
# Safety marker: STAGE5_RUNG_CANARY_HARD_STOP
# Safety marker: STAGE5_SUPPORT6_REPLICATION_SEEDS
# Safety marker: STAGE5_SUPPORT6_DOSED_RECEIPT_SUMMARY
# Safety marker: STAGE5_SUPPORT6_DOSED_STEPS
# Safety marker: STAGE5_SEED26_PLATEAU_SOURCE_SUMMARY
# Safety marker: STAGE5_PERM_SOURCE_SUMMARY
# Safety marker: STAGE5_PERM_PARITY_TOLERANCE
# Safety marker: STAGE5_ROUTE_TRAIN_SEED
# Safety marker: STAGE5_PHASE_A_PLAN_RUN_ID
# Safety marker: soft_depth10_min_correct
# Safety marker: soft_depth11_min_correct
# Safety marker: N24_STRONG_SCALING_MIN_CORRECT
# Safety marker: N24_CHANCE_REJECTION_MIN_CORRECT
# Safety marker: STRONG_SCALING_MIN_CORRECT = 91
# Safety marker: ASYMPTOTE_REJECTION_MIN_CORRECT = 79
# Safety marker: CHANCE_REJECTION_MIN_CORRECT = 14
# Safety marker: NONREGRESSION_FLOORS = {"1": 0.93
# Safety marker: STAGE5_SPLICE_SOURCE_SUMMARY
# Safety marker: STAGE5_SPLICE_POINTS
# Safety marker: source_orbit_fraction_j1_to_j3
# Safety marker: source_state_continuation
# Safety marker: lawful_fraction_j1_to_j3
# Safety marker: prompt_position_shortcut
# Safety marker: SELECTION_MIN_CORRECT
# Safety marker: NONREGRESSION_FLOORS
# Safety marker: STAGE5_ANNEAL_LOOP_LOSS_MODE
# Safety marker: per_loop_labels
# Safety marker: STAGE5_PROBE_FEATURE_TRANSFORMS
# Safety marker: loop_index_probe
# Safety marker: router_leak_exclusion
# Safety marker: state_envelope
# Safety marker: loop_loss_mode='annealed_chain_to_outcome'
# Safety marker: tests/test_eval_synthetic_depth_probe.py
# Safety marker: tests/test_stage5_chain_consolidation.py
# Safety marker: tests/test_recurrent_wrapper_tiny.py::test_annealed_chain_to_outcome_loss_mixes_chain_and_target_ce_on_tiny_model

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
TARGET = os.environ.get("STAGE5_CURRENT_A100_TARGET", "depth_extrapolation_eval")

TARGETS = {
    "depth_extrapolation_eval": {
        "script": "colab/run_stage5_depth_extrapolation_eval.py",
        "tests": [
            "tests/test_eval_synthetic_depth_active_labels.py",
            "tests/test_stage5_chain_consolidation.py",
            "tests/test_stage5_notebooks.py::test_chain_consolidation_targets_are_wired_and_guarded",
        ],
        "disconnect_env": "STAGE5_EXTRAP_DISCONNECT",
    },
    "synthetic_probe_battery": {
        "script": "colab/run_stage5_synthetic_probe_battery.py",
        "tests": [
            "tests/test_eval_synthetic_depth_probe.py",
            "tests/test_stage5_chain_consolidation.py",
            "tests/test_stage5_notebooks.py::test_chain_consolidation_targets_are_wired_and_guarded",
        ],
        "disconnect_env": "STAGE5_PROBE_DISCONNECT",
    },
    "chain_continuation_probe_readout": {
        "script": "colab/run_stage5_synthetic_probe_battery.py",
        "tests": [
            "tests/test_eval_synthetic_depth_probe.py",
            "tests/test_stage5_chain_consolidation.py",
            "tests/test_stage5_notebooks.py::test_chain_consolidation_targets_are_wired_and_guarded",
        ],
        "disconnect_env": "STAGE5_PROBE_DISCONNECT",
    },
    "chain_anneal_to_outcome": {
        "script": "colab/run_stage5_chain_anneal_to_outcome.py",
        "tests": [
            "tests/test_recurrent_wrapper_tiny.py::test_annealed_chain_to_outcome_loss_mixes_chain_and_target_ce_on_tiny_model",
            "tests/test_train_unfrozen_recurrent.py::test_chain_label_weight_ramps_then_holds_zero",
            "tests/test_stage5_chain_consolidation.py",
            "tests/test_stage5_notebooks.py::test_chain_consolidation_targets_are_wired_and_guarded",
        ],
        "disconnect_env": "STAGE5_ANNEAL_DISCONNECT",
    },
    "post_anneal_readouts": {
        "script": "colab/run_stage5_post_anneal_readouts.py",
        "tests": [
            "tests/test_analyze_synthetic_reader_alignment.py",
            "tests/test_eval_synthetic_depth_active_labels.py",
            "tests/test_eval_synthetic_depth_probe.py",
            "tests/test_stage5_chain_consolidation.py",
            "tests/test_stage5_notebooks.py::test_chain_consolidation_targets_are_wired_and_guarded",
        ],
        "disconnect_env": "STAGE5_POST_ANNEAL_DISCONNECT",
    },
    "post_anneal_extended_readouts": {
        "script": "colab/run_stage5_post_anneal_readouts.py",
        "tests": [
            "tests/test_analyze_synthetic_reader_alignment.py",
            "tests/test_eval_synthetic_depth_active_labels.py",
            "tests/test_eval_synthetic_depth_probe.py",
            "tests/test_stage5_chain_consolidation.py",
            "tests/test_stage5_notebooks.py::test_chain_consolidation_targets_are_wired_and_guarded",
        ],
        "disconnect_env": "STAGE5_POST_ANNEAL_DISCONNECT",
    },
    "chain_continuation_attribution": {
        "script": "colab/run_stage5_chain_continuation_attribution.py",
        "tests": [
            "tests/test_recurrent_wrapper_tiny.py::test_per_loop_label_loss_mode_uses_active_intermediate_labels_on_tiny_model",
            "tests/test_eval_synthetic_depth_active_labels.py",
            "tests/test_stage5_chain_consolidation.py",
            "tests/test_stage5_notebooks.py::test_chain_consolidation_targets_are_wired_and_guarded",
        ],
        "disconnect_env": "STAGE5_CHAIN_CONTINUATION_DISCONNECT",
    },
    "depth_support_route_comparison": {
        "script": "colab/run_stage5_depth_support_route_comparison.py",
        "tests": [
            "tests/test_recurrent_wrapper_tiny.py::test_per_loop_label_loss_mode_uses_active_intermediate_labels_on_tiny_model",
            "tests/test_eval_synthetic_depth_active_labels.py",
            "tests/test_stage5_chain_consolidation.py",
            "tests/test_train_unfrozen_recurrent.py::test_trainable_parameter_norm_stats_groups_recurrent_and_bridge_params",
            "tests/test_stage5_notebooks.py::test_chain_consolidation_targets_are_wired_and_guarded",
        ],
        "disconnect_env": "STAGE5_ROUTE_DISCONNECT",
    },
    "depth_support_ladder8": {
        "script": "colab/run_stage5_depth_support_ladder.py",
        "tests": [
            "tests/test_recurrent_wrapper_tiny.py::test_per_loop_label_loss_mode_uses_active_intermediate_labels_on_tiny_model",
            "tests/test_eval_synthetic_depth_active_labels.py",
            "tests/test_stage5_chain_consolidation.py",
            "tests/test_train_unfrozen_recurrent.py::test_trainable_parameter_norm_stats_groups_recurrent_and_bridge_params",
            "tests/test_stage5_notebooks.py::test_chain_consolidation_targets_are_wired_and_guarded",
        ],
        "disconnect_env": "STAGE5_LADDER_DISCONNECT",
    },
    "support8_probe_readout": {
        "script": "colab/run_stage5_support8_probe_readout.py",
        "tests": [
            "tests/test_eval_synthetic_depth_probe.py",
            "tests/test_stage5_chain_consolidation.py",
            "tests/test_stage5_notebooks.py::test_chain_consolidation_targets_are_wired_and_guarded",
        ],
        "disconnect_env": "STAGE5_SUPPORT8_PROBE_DISCONNECT",
    },
    "support8_dose_arm": {
        "script": "colab/run_stage5_support8_dose_arm.py",
        "tests": [
            "tests/test_recurrent_wrapper_tiny.py::test_per_loop_label_loss_mode_uses_active_intermediate_labels_on_tiny_model",
            "tests/test_eval_synthetic_depth_active_labels.py",
            "tests/test_stage5_chain_consolidation.py",
            "tests/test_train_unfrozen_recurrent.py::test_trainable_parameter_norm_stats_groups_recurrent_and_bridge_params",
            "tests/test_stage5_notebooks.py::test_chain_consolidation_targets_are_wired_and_guarded",
        ],
        "disconnect_env": "STAGE5_DOSE_DISCONNECT",
    },
    "same_reader_final_symbol": {
        "script": "colab/run_stage5_same_reader_final_symbol.py",
        "tests": [
            "tests/test_eval_synthetic_depth_final_symbol.py",
            "tests/test_eval_synthetic_depth_active_labels.py",
            "tests/test_stage5_chain_consolidation.py",
            "tests/test_stage5_notebooks.py::test_chain_consolidation_targets_are_wired_and_guarded",
        ],
        "disconnect_env": "STAGE5_SAME_READER_DISCONNECT",
    },
    "n24_same_reader_receipt": {
        "script": "colab/run_stage5_same_reader_final_symbol.py",
        "tests": [
            "tests/test_eval_synthetic_depth_final_symbol.py",
            "tests/test_eval_synthetic_depth_active_labels.py",
            "tests/test_stage5_chain_consolidation.py",
            "tests/test_stage5_notebooks.py::test_chain_consolidation_targets_are_wired_and_guarded",
        ],
        "env": {
            "STAGE5_SAME_READER_RUN_ID": "stage5_n24_same_reader_final_symbol_current",
            "STAGE5_SAME_READER_SOURCE_SUMMARY": "outputs/stage5/stage5_n24_support12_rung_20260707_140139/summary.json",
            "STAGE5_SAME_READER_DATA_JSONL": "outputs/stage5/stage5_synthetic_depth_frozen_eval_v3_depth22_n24/data/test_chain_mcq.jsonl",
            "STAGE5_SAME_READER_MAX_LOOPS": "22",
            "STAGE5_SAME_READER_EXPECT_IDENTITY_WITH_ACTIVE": "1",
            "STAGE5_SAME_READER_IDENTITY_TOLERANCE": "0.000001",
            "STAGE5_SAME_READER_DTYPE": "bfloat16",
        },
        "disconnect_env": "STAGE5_SAME_READER_DISCONNECT",
    },
    "n24_support12_rung": {
        "script": "colab/run_stage5_n24_support12_rung.py",
        "tests": [
            "tests/test_stage5_n24_rung.py",
            "tests/test_eval_synthetic_depth_active_labels.py",
            "tests/test_stage5_chain_consolidation.py",
            "tests/test_stage5_notebooks.py::test_chain_consolidation_targets_are_wired_and_guarded",
        ],
        "disconnect_env": "STAGE5_N24_DISCONNECT",
    },
    "support6_seed_replication": {
        "script": "colab/run_stage5_support6_seed_replication.py",
        "tests": [
            "tests/test_stage5_support6_seed_replication.py",
            "tests/test_stage5_chain_consolidation.py",
            "tests/test_stage5_notebooks.py::test_chain_consolidation_targets_are_wired_and_guarded",
        ],
        "disconnect_env": "STAGE5_SUPPORT6_REPLICATION_DISCONNECT",
    },
    "support6_replication_receipts": {
        "script": "colab/run_stage5_support6_replication_receipts.py",
        "tests": [
            "tests/test_stage5_support6_seed_replication.py",
            "tests/test_stage5_chain_consolidation.py",
            "tests/test_stage5_notebooks.py::test_chain_consolidation_targets_are_wired_and_guarded",
        ],
        "disconnect_env": "STAGE5_SUPPORT6_RECEIPTS_DISCONNECT",
        "requires_gpu": False,
        "mount_drive": False,
    },
    "support6_dosed_seed_resolution": {
        "script": "colab/run_stage5_support6_dosed_seed_resolution.py",
        "tests": [
            "tests/test_stage5_support6_seed_replication.py",
            "tests/test_stage5_chain_consolidation.py",
            "tests/test_stage5_notebooks.py::test_chain_consolidation_targets_are_wired_and_guarded",
        ],
        "disconnect_env": "STAGE5_SUPPORT6_DOSED_DISCONNECT",
    },
    "support6_seed26_plateau_test": {
        "script": "colab/run_stage5_support6_seed26_plateau.py",
        "tests": [
            "tests/test_stage5_support6_seed_replication.py",
            "tests/test_stage5_chain_consolidation.py",
            "tests/test_stage5_notebooks.py::test_chain_consolidation_targets_are_wired_and_guarded",
        ],
        "env": {
            "STAGE5_SEED26_PLATEAU_SOURCE_SUMMARY": "outputs/stage5/stage5_support6_dosed_seed_resolution_20260708_004504_seed_20260726_dose2000/summary.json",
            "STAGE5_SEED26_PLATEAU_STEPS": "2000",
            "STAGE5_SEED26_PLATEAU_SEED": "20260726",
            "STAGE5_SEED26_PLATEAU_DTYPE": "bfloat16",
        },
        "disconnect_env": "STAGE5_SEED26_PLATEAU_DISCONNECT",
    },
    "scorer_equivalence_receipt": {
        "script": "colab/run_stage5_scorer_equivalence_receipt.py",
        "tests": [
            "tests/test_eval_synthetic_depth_active_labels.py",
            "tests/test_stage5_chain_consolidation.py",
            "tests/test_stage5_notebooks.py::test_chain_consolidation_targets_are_wired_and_guarded",
        ],
        "disconnect_env": "STAGE5_SCORER_EQUIV_DISCONNECT",
    },
    "synthetic_release_receipts": {
        "script": "colab/run_stage5_synthetic_release_receipts.py",
        "tests": [
            "tests/test_stage5_support6_seed_replication.py",
            "tests/test_stage5_chain_consolidation.py",
            "tests/test_stage5_notebooks.py::test_chain_consolidation_targets_are_wired_and_guarded",
        ],
        "disconnect_env": "STAGE5_RELEASE_RECEIPTS_DISCONNECT",
        "requires_gpu": False,
        "mount_drive": False,
    },
    "phase_a_surpass_prereg": {
        "script": "colab/run_stage5_phase_a_surpass_plan.py",
        "tests": [
            "tests/test_stage5_phase_a_surpass.py",
            "tests/test_stage5_chain_consolidation.py",
            "tests/test_stage5_notebooks.py::test_chain_consolidation_targets_are_wired_and_guarded",
        ],
        "disconnect_env": "STAGE5_PHASE_A_PLAN_DISCONNECT",
    },
    "permutation_zero_shot_baseline": {
        "script": "colab/run_stage5_permutation_zero_shot.py",
        "tests": [
            "tests/test_synthetic_depth_task.py",
            "tests/test_eval_synthetic_depth_active_labels.py",
            "tests/test_stage5_chain_consolidation.py",
            "tests/test_stage5_notebooks.py::test_chain_consolidation_targets_are_wired_and_guarded",
        ],
        "env": {
            "STAGE5_PERM_SOURCE_SUMMARY": "outputs/stage5/stage5_n24_support12_rung_20260707_140139/summary.json",
            "STAGE5_PERM_EVAL_ID": "stage5_synthetic_depth_permutation_eval_v1_n24_depth12",
            "STAGE5_PERM_N_SYMBOLS": "24",
            "STAGE5_PERM_MAX_DEPTH": "12",
            "STAGE5_PERM_ROWS_PER_DEPTH": "128",
            "STAGE5_PERM_PARITY_TOLERANCE": "0.05",
            "STAGE5_PERM_DTYPE": "bfloat16",
        },
        "disconnect_env": "STAGE5_PERM_DISCONNECT",
    },
    "splice_injection_diagnostic": {
        "script": "colab/run_stage5_splice_injection.py",
        "tests": [
            "tests/test_recurrent_wrapper_tiny.py::test_recurrent_state_override_applies_before_requested_next_loop_on_tiny_model",
            "tests/test_eval_synthetic_depth_splice.py",
            "tests/test_stage5_chain_consolidation.py",
            "tests/test_stage5_notebooks.py::test_chain_consolidation_targets_are_wired_and_guarded",
        ],
        "disconnect_env": "STAGE5_SPLICE_DISCONNECT",
    },
}


def secret(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
        try:
            value = userdata.get(name)
        except Exception:
            value = None
        if value:
            return value
    return None


GH_TOKEN = secret("GH_TOKEN", "GITHUB_TOKEN")
HF_TOKEN = secret("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
assert GH_TOKEN, "Missing GH_TOKEN or GITHUB_TOKEN in Colab secrets."
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
    print("HF token loaded", flush=True)
else:
    print("HF token missing; model downloads will use anonymous Hub access.", flush=True)


def redact(text: str) -> str:
    out = str(text)
    for token in (GH_TOKEN, HF_TOKEN):
        if token:
            out = out.replace(token, "****")
    return out


def run(cmd: list[str | os.PathLike[str]], *, cwd: Path = ROOT, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("$", redact(" ".join(map(str, cmd))), flush=True)
    process = subprocess.Popen(
        list(map(str, cmd)),
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    chunks: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        safe = redact(line)
        print(safe, end="", flush=True)
        chunks.append(safe)
    stdout = "".join(chunks)
    proc = subprocess.CompletedProcess(list(map(str, cmd)), process.wait(), stdout, None)
    if check and proc.returncode:
        print("FAILED_COMMAND_TAIL_START", flush=True)
        print("\n".join(stdout.splitlines()[-180:]), flush=True)
        print("FAILED_COMMAND_TAIL_END", flush=True)
        raise subprocess.CalledProcessError(proc.returncode, proc.args, output=stdout)
    return proc


def env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def sync_repo() -> None:
    clone_url = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
    if ROOT.exists():
        run(["git", "remote", "set-url", "origin", clone_url])
        run(["git", "fetch", "origin", "main"])
        run(["git", "checkout", "main"])
        run(["git", "reset", "--hard", "origin/main"])
    else:
        run(["git", "clone", clone_url, ROOT], cwd=Path("/content"))
    run(["git", "config", "user.email", "colab-runner@local"])
    run(["git", "config", "user.name", "Colab Runner"])
    run(["git", "log", "--oneline", "-5"], check=False)


def require_gpu_runtime() -> None:
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("Attach a GPU runtime first. L4/T4 is sufficient for these consolidation targets.")
    run(["nvidia-smi"], cwd=Path("/content"), check=False)


try:
    if TARGET not in TARGETS:
        raise ValueError(f"Unknown consolidation target: {TARGET}")
    target = TARGETS[TARGET]
    if target.get("requires_gpu", True):
        require_gpu_runtime()
    else:
        print(f"Skipping GPU requirement for CPU-safe target {TARGET}.", flush=True)
    if target.get("mount_drive", True):
        drive.mount("/content/drive", force_remount=False)
    else:
        print(f"Skipping Drive mount for target {TARGET}.", flush=True)
    sync_repo()
    os.chdir(ROOT)
    os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
    print(f"STAGE5_CHAIN_CONSOLIDATION_CELL_VERSION={STAGE5_CHAIN_CONSOLIDATION_CELL_VERSION}", flush=True)
    print(f"stage5_chain_consolidation_target={TARGET}", flush=True)
    run(["python", "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run([sys.executable, "-m", "pytest", "-q", *target["tests"]])
    run([sys.executable, target["script"]])
    if env_flag(str(target["disconnect_env"]), "0"):
        print(f"Disconnecting Colab runtime after {TARGET}.", flush=True)
        runtime.unassign()
except Exception:
    print(f"Stage 5 chain consolidation target errored: {TARGET}; leaving runtime connected.", flush=True)
    raise
