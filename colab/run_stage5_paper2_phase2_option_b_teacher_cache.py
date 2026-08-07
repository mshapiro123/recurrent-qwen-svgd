"""Run and publish the locked, resumable Option B fresh-anchor teacher cache."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    import resource
except ModuleNotFoundError:  # Windows unit-test collection; Colab provides resource.
    resource = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.cache_paper2_phase2_stage0a import read_jsonl, write_json, write_jsonl  # noqa: E402
from training.paper2_phase2_option_b import (  # noqa: E402
    build_anchor_admission_rows,
    build_cache_config,
    choose_full_anchor_count,
    fixed_anchor_subset,
    load_locked_registration,
)
from training.paper2_phase2_stage0a import sha256_file  # noqa: E402


# Safety marker: locked fresh documents all-admitted-anchor 14B states no optimizer no training
RUN_ID = "stage5_paper2_phase2_option_b_teacher_cache_20260806"
RUN_DIR = ROOT / "outputs/stage5" / RUN_ID
DRIVE_RUN = Path(
    os.environ.get(
        "STAGE5_PHASE2_OPTION_B_DRIVE_ROOT",
        f"/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/{RUN_ID}",
    )
)
REGISTRATION = ROOT / "training/paper2_phase2_option_b_preregistration.json"
PILOT_ANCHORS = 512
DATA_EXCLUSION_RUNS = (
    "stage5_paper2_d0_20260726",
    "stage5_paper2_dc0_20260728",
    "stage5_paper2_dc1_preflight_20260729",
    "stage5_paper2_dc1_stage_a_20260730",
)
DERIVED_EXCLUSION_RECEIPTS = {
    "stage5_paper2_phase2_prewindow_20260731": (
        ROOT / "outputs/stage5/stage5_paper2_phase2_prewindow_20260731"
    ),
    "stage5_paper2_phase2_stage0a_20260803": (
        ROOT / "outputs/stage5/stage5_paper2_phase2_stage0a_20260803"
    ),
}
SOURCE_DATA_HASH_KEYS = {"data_jsonl_sha256", "dev_data_sha256", "data_sha256"}


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    tail: list[str] = []
    for line in process.stdout:
        print(line, end="", flush=True)
        tail.append(line.rstrip())
        tail = tail[-300:]
    code = process.wait()
    if code:
        print("\nOption B child-process tail:\n" + "\n".join(tail), flush=True)
        raise subprocess.CalledProcessError(code, command)


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def collect_source_data_hashes(payload: Any) -> set[str]:
    hashes: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if (
                key in SOURCE_DATA_HASH_KEYS
                and isinstance(value, str)
                and len(value) == 64
                and all(character in "0123456789abcdef" for character in value.lower())
            ):
                hashes.add(value.lower())
            hashes.update(collect_source_data_hashes(value))
    elif isinstance(payload, list):
        for value in payload:
            hashes.update(collect_source_data_hashes(value))
    return hashes


def receipt_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def collect_exclusion_files(
    *,
    base: Path | None = None,
    data_runs: tuple[str, ...] = DATA_EXCLUSION_RUNS,
    derived_receipts: dict[str, Path] = DERIVED_EXCLUSION_RECEIPTS,
) -> tuple[list[Path], dict[str, Any]]:
    base = base or Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5")
    files: list[Path] = []
    files_by_sha256: dict[str, list[str]] = {}
    data_run_receipts: list[dict[str, Any]] = []
    for run_id in data_runs:
        root = base / run_id
        if not root.is_dir():
            raise FileNotFoundError(f"Missing locked Option B exclusion run: {root}")
        candidates = sorted(root.rglob("*.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"No JSONL exclusion records under: {root}")
        candidate_receipts = []
        for path in candidates:
            digest = sha256_file(path)
            files_by_sha256.setdefault(digest, []).append(str(path))
            candidate_receipts.append({"path": str(path), "sha256": digest})
        files.extend(candidates)
        data_run_receipts.append(
            {
                "run_id": run_id,
                "mode": "data_bearing",
                "jsonl_files": candidate_receipts,
            }
        )

    derived_run_receipts: list[dict[str, Any]] = []
    for run_id, receipt_root in derived_receipts.items():
        drive_root = base / run_id
        if not drive_root.is_dir():
            raise FileNotFoundError(f"Missing locked derived Option B exclusion run: {drive_root}")
        summaries = sorted(receipt_root.rglob("summary.json"))
        if not summaries:
            raise FileNotFoundError(
                f"No canonical derived receipts for Option B exclusion run: {receipt_root}"
            )
        source_hashes: set[str] = set()
        summary_receipts = []
        for path in summaries:
            payload = json.loads(path.read_text(encoding="utf-8"))
            observed = collect_source_data_hashes(payload)
            source_hashes.update(observed)
            summary_receipts.append(
                {
                    "path": receipt_path(path),
                    "sha256": sha256_file(path),
                    "source_data_sha256": sorted(observed),
                }
            )
        if not source_hashes:
            raise RuntimeError(
                f"Derived Option B exclusion run has no source-data hashes: {receipt_root}"
            )
        unresolved = sorted(source_hashes - files_by_sha256.keys())
        if unresolved:
            raise RuntimeError(
                "Derived Option B exclusion lineage is not closed over the actual "
                f"quarantined JSONL files for {run_id}: {unresolved}"
            )
        derived_run_receipts.append(
            {
                "run_id": run_id,
                "mode": "receipt_only_derived",
                "source_data_sha256": sorted(source_hashes),
                "resolved_source_files": {
                    digest: files_by_sha256[digest] for digest in sorted(source_hashes)
                },
                "canonical_receipts": summary_receipts,
            }
        )
    resolution = {
        "kind": "paper2_phase2_option_b_exclusion_lineage_closure",
        "status": "complete",
        "data_bearing_runs": data_run_receipts,
        "derived_receipt_only_runs": derived_run_receipts,
        "jsonl_file_count": len(files),
        "unique_jsonl_sha256_count": len(files_by_sha256),
        "all_derived_source_hashes_resolved": True,
        "training_started": False,
        "optimizer_steps": 0,
    }
    return files, resolution


def select_scratch() -> dict[str, Any]:
    minimum_total_bytes = int(
        os.environ.get("STAGE5_PHASE2_OPTION_B_MIN_SCRATCH_TOTAL_GIB", "300")
    ) * 1024**3
    minimum_free_bytes = int(
        os.environ.get("STAGE5_PHASE2_OPTION_B_MIN_SCRATCH_FREE_GIB", "250")
    ) * 1024**3
    listing = subprocess.check_output(
        ["df", "-B1", "--output=target,size,avail"], text=True
    )
    candidates: list[tuple[bool, int, int, Path]] = []
    for line in listing.splitlines()[1:]:
        fields = line.split()
        if len(fields) != 3:
            continue
        target, size_text, free_text = fields
        try:
            size, free = int(size_text), int(free_text)
        except ValueError:
            continue
        path = Path(target)
        if (
            size < minimum_total_bytes
            or free < minimum_free_bytes
            or target.startswith("/content/drive")
        ):
            continue
        if path.is_dir() and os.access(path, os.W_OK):
            candidates.append(("scratch" in target.lower(), free, size, path))
    if not candidates:
        raise RuntimeError(
            "Option B has no writable local disk meeting the active total/free "
            f"profile: total>={minimum_total_bytes / 1024**3:.0f} GiB, "
            f"free>={minimum_free_bytes / 1024**3:.0f} GiB"
        )
    candidates.sort(reverse=True)
    _named, free, size, mount = candidates[0]
    job = mount / RUN_ID
    staging = job / "staging"
    cache = job / "cache"
    for path in (staging, cache, job / "tmp"):
        path.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(cache / "huggingface" / "hub")
    os.environ["TMPDIR"] = str(job / "tmp")
    return {
        "mount": str(mount),
        "total_bytes": size,
        "free_bytes_at_start": free,
        "minimum_total_bytes": minimum_total_bytes,
        "minimum_free_bytes": minimum_free_bytes,
        "staging_dir": str(staging),
        "job_root": str(job),
    }


def cache_command(
    *, data: Path, config: Path, private: Path, output: Path, staging: Path
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "eval.cache_paper2_phase2_stage0a",
        "--data_jsonl",
        str(data),
        "--config_json",
        str(config),
        "--private_dir",
        str(private),
        "--output_summary",
        str(output),
        "--staging_dir",
        str(staging),
        "--device",
        "cuda",
        "--dtype",
        "bfloat16",
        "--attn_implementation",
        "sdpa",
    ]
    if os.environ.get("STAGE5_PHASE2_OPTION_B_OFFLOAD_32B", "0") == "1":
        command.extend(
            [
                "--offload_32b",
                "--offload_dir",
                str(staging / "offload_32b"),
            ]
        )
    return command


def publish(summary_path: Path) -> str:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "add", "-f", "--", summary_path.relative_to(ROOT).as_posix()])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        run(["git", "commit", "-m", "Record Option B teacher cache receipt [skip ci]"])
        run(["git", "push", "origin", "main"])
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> int:
    registration = load_locked_registration(REGISTRATION)
    DRIVE_RUN.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    scratch = select_scratch()
    private = DRIVE_RUN / "private"
    receipts = DRIVE_RUN / "receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    exclusions, exclusion_resolution = collect_exclusion_files()
    exclusion_resolution_path = receipts / "exclusion_lineage_closure.json"
    write_json(exclusion_resolution_path, exclusion_resolution)
    data = private / "new_documents_target.jsonl"
    target_config = private / "target_config.json"
    freeze_summary = receipts / "new_document_freeze_summary.json"
    target = int(registration["teacher_pass"]["new_training_anchor_target"])
    floor = int(registration["teacher_pass"]["new_training_anchor_minimum"])
    if not data.is_file() or not target_config.is_file() or not freeze_summary.is_file():
        command = [
            sys.executable,
            "-u",
            "-m",
            "eval.prepare_paper2_phase2_option_b_cache",
            "--registration",
            str(REGISTRATION),
            "--output_data",
            str(data),
            "--output_config",
            str(target_config),
            "--output_summary",
            str(freeze_summary),
            "--anchor_count",
            str(target),
            "--run_id",
            RUN_ID,
        ]
        for path in exclusions:
            command.extend(["--excluded_jsonl", str(path)])
        run(command)

    offload_32b = os.environ.get("STAGE5_PHASE2_OPTION_B_OFFLOAD_32B", "0") == "1"
    pilot_mode = "a10040_offload" if offload_32b else "a10080_resident"
    pilot_config = private / f"pilot_config_{pilot_mode}.json"
    write_json(
        pilot_config,
        build_cache_config(
            registration=registration,
            data_path=data,
            anchor_count=PILOT_ANCHORS,
            run_id=RUN_ID + f"_pilot_{pilot_mode}",
        ),
    )
    pilot_private = private / f"pilot_{pilot_mode}"
    pilot_output = receipts / f"pilot_cache_summary_{pilot_mode}.json"
    pilot_started = time.perf_counter()
    run(
        cache_command(
            data=data,
            config=pilot_config,
            private=pilot_private,
            output=pilot_output,
            staging=Path(scratch["staging_dir"]) / f"pilot_{pilot_mode}",
        )
    )
    pilot_elapsed = time.perf_counter() - pilot_started
    pilot_total = directory_bytes(pilot_private)
    pilot_fixed = sum(
        path.stat().st_size for path in pilot_private.rglob("lm_head.pt")
    )
    storage = choose_full_anchor_count(
        target=target,
        floor=floor,
        pilot_anchors=PILOT_ANCHORS,
        pilot_total_bytes=pilot_total,
        pilot_fixed_bytes=pilot_fixed,
        scratch_free_bytes=shutil.disk_usage(scratch["mount"]).free,
        drive_free_bytes=shutil.disk_usage(DRIVE_RUN).free,
    )
    storage["pilot_execution_mode"] = pilot_mode
    storage["pilot_elapsed_seconds"] = pilot_elapsed
    pilot_summary = json.loads(pilot_output.read_text(encoding="utf-8"))
    storage["model_throughput"] = {
        key: {
            "samples_per_second": value.get("samples_per_second_this_invocation"),
            "peak_gpu_memory_bytes": value.get("peak_gpu_memory_bytes_this_invocation"),
            "shard_seconds_per_sample": value.get(
                "shard_seconds_per_sample_this_invocation"
            ),
        }
        for key, value in pilot_summary["model_caches"].items()
    }
    storage["cascade_32b_fraction"] = pilot_summary["cascade_32b"]["selected_fraction"]
    storage["peak_child_system_ram_kib"] = (
        int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
        if resource is not None
        else None
    )
    storage["peak_local_scratch_bytes_observed"] = directory_bytes(
        Path(scratch["job_root"])
    )
    storage["drive_write_bytes_per_second_observed"] = pilot_total / max(
        pilot_elapsed, 1e-9
    )
    low_ratios: list[float] = []
    high_ratios: list[float] = []
    for model in pilot_summary["model_caches"].values():
        quantiles = model.get("shard_seconds_per_sample_this_invocation") or {}
        median = quantiles.get("median")
        low = quantiles.get("p10")
        high = quantiles.get("p90")
        if median and low is not None and high is not None:
            low_ratios.append(float(low) / float(median))
            high_ratios.append(float(high) / float(median))
    low_factor = min(low_ratios) if low_ratios else 0.8
    high_factor = max(high_ratios) if high_ratios else 1.25
    storage["projected_runtime_seconds"] = {
        "target_point": pilot_elapsed * target / PILOT_ANCHORS,
        "target_low_from_shard_variation": (
            pilot_elapsed * target / PILOT_ANCHORS * low_factor
        ),
        "target_high_from_shard_variation": (
            pilot_elapsed * target / PILOT_ANCHORS * high_factor
        ),
        "floor_point": pilot_elapsed * floor / PILOT_ANCHORS,
        "floor_low_from_shard_variation": (
            pilot_elapsed * floor / PILOT_ANCHORS * low_factor
        ),
        "floor_high_from_shard_variation": (
            pilot_elapsed * floor / PILOT_ANCHORS * high_factor
        ),
        "uncertainty_note": (
            "End-to-end pilot scaled by the minimum p10/median and maximum "
            "p90/median observed across per-model shard timings."
        ),
    }
    write_json(receipts / "throughput_storage_preflight.json", storage)

    if os.environ.get("STAGE5_PHASE2_OPTION_B_PREFLIGHT_ONLY", "0") == "1":
        summary = {
            "kind": "paper2_phase2_option_b_teacher_cache_preflight",
            "status": "complete_preflight_full_cache_not_started",
            "hardware_mode": "a100_40gb_32b_accelerate_offload",
            "pilot_anchor_count": PILOT_ANCHORS,
            "selected_full_anchor_count_if_authorized": storage[
                "selected_anchor_count"
            ],
            "exclusion_lineage_closure_sha256": sha256_file(
                exclusion_resolution_path
            ),
            "throughput_storage_preflight": storage,
            "training_started": False,
            "optimizer_steps": 0,
            "full_cache_started": False,
            "next_required_action": (
                "review measured runtime and storage before authorizing full cache"
            ),
        }
        public_summary = RUN_DIR / "preflight_summary.json"
        write_json(public_summary, summary)
        shutil.copy2(public_summary, receipts / "preflight_summary.json")
        commit = publish(public_summary)
        print(json.dumps({**summary, "publish_commit": commit}, indent=2, sort_keys=True))
        return 0

    selected = int(storage["selected_anchor_count"])
    full_config = private / "full_config.json"
    write_json(
        full_config,
        build_cache_config(
            registration=registration,
            data_path=data,
            anchor_count=selected,
            run_id=RUN_ID,
        ),
    )
    full_private = private / "full"
    full_output = receipts / "full_cache_summary.json"
    run(
        cache_command(
            data=data,
            config=full_config,
            private=full_private,
            output=full_output,
            staging=Path(scratch["staging_dir"]) / "full",
        )
    )
    samples = read_jsonl(full_private / "sample_manifest.jsonl")
    cascade = json.loads(
        (full_private / "teacher_32b_cascade_indices.json").read_text(encoding="utf-8")
    )
    admission_rows = build_anchor_admission_rows(
        samples, set(int(value) for value in cascade["sample_indices"])
    )
    admission_receipt = write_jsonl(private / "anchor_admission.jsonl", admission_rows)
    subset = fixed_anchor_subset(admission_rows, count=8031, seed=20260806)
    fixed_subset_path = private / "fixed_new_train_subset.json"
    write_json(fixed_subset_path, {"anchor_indices": subset})
    full_summary = json.loads(full_output.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_summary.read_text(encoding="utf-8"))
    summary = {
        "kind": "paper2_phase2_option_b_teacher_cache",
        "status": "complete_teacher_cache_training_still_prohibited",
        "selected_anchor_count": selected,
        "horizon_sample_count": selected * 4,
        "teacher_14b_state_samples": full_summary["teacher_states"]["samples"],
        "teacher_14b_state_coverage_policy": "all_admitted_anchors",
        "per_anchor_label_tier_admission": admission_receipt,
        "fixed_new_train_subset": {
            "count": len(subset),
            "sha256": sha256_file(fixed_subset_path),
        },
        "new_document_id_sha256": freeze["new_document_id_sha256"],
        "excluded_document_id_sha256": freeze["excluded_document_id_sha256"],
        "exclusion_lineage_closure_sha256": sha256_file(exclusion_resolution_path),
        "zero_overlap_with_excluded_documents": freeze[
            "zero_overlap_with_excluded_documents"
        ],
        "new_data_sha256": sha256_file(data),
        "sample_manifest_sha256": full_summary["manifest"]["sample_manifest_sha256"],
        "position_key_sha256": full_summary["manifest"]["position_key_sha256"],
        "full_logit_audit_samples": full_summary["manifest"]["full_logit_audit_samples"],
        "full_logit_audit_sample_keys_sha256": full_summary["manifest"][
            "full_logit_audit_sample_keys_sha256"
        ],
        "model_cache_ledger_hashes": {
            key: sha256_file(full_private / "model_cache" / key / "summary.json")
            for key in full_summary["model_caches"]
        },
        "lattice_summary_sha256": sha256_file(full_private / "lattice" / "summary.json"),
        "teacher_state_ledger_sha256": sha256_file(
            full_private / "model_cache" / "teacher_14b" / "summary.json"
        ),
        "model_execution_modes": {
            key: value.get("execution_mode")
            for key, value in full_summary["model_caches"].items()
        },
        "teacher_32b_hf_device_map": full_summary["model_caches"]["teacher_32b"].get(
            "hf_device_map", {}
        ),
        "throughput_storage_preflight": storage,
        "runtime_storage": scratch,
        "hardware_mode": (
            "a100_40gb_32b_accelerate_offload"
            if os.environ.get("STAGE5_PHASE2_OPTION_B_OFFLOAD_32B", "0") == "1"
            else "a100_80gb_fully_resident"
        ),
        "training_started": False,
        "optimizer_steps": 0,
        "evaluation_partition_touched": False,
        "training_authorized": False,
        "next_required_action": "hash_only_preregistration_amendment_before_training_launcher",
    }
    public_summary = RUN_DIR / "summary.json"
    write_json(public_summary, summary)
    shutil.copy2(public_summary, receipts / "summary.json")
    commit = publish(public_summary)
    print(json.dumps({**summary, "publish_commit": commit}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
