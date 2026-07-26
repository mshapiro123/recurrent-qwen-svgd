"""Freeze the D0 corpus and commit the preregistration lock; never label or train."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.speculative_depth_d0_corpus import (
    assert_document_disjoint,
    choose_stratum_mix,
    collect_partition_rows,
    collect_probe_rows,
    iter_fineweb_documents,
    iter_in_era_documents,
    iter_stack_documents,
    select_pilot_rows,
    sha256_file,
    stable_fraction,
    token_quotas,
    write_jsonl,
)
from training.speculative_depth_d0_spec import (
    D0ExecutionPolicy,
    DRAFTER_CHECKPOINT_SHA256,
    DRAFTER_MODEL,
    DRAFTER_MODEL_REVISION,
    FINEWEB_DATASET,
    FINEWEB_DUMP,
    FINEWEB_REVISION,
    GOVERNING_DOCUMENT,
    GOVERNING_DOCUMENT_HANDOFF_SHA256,
    GOVERNING_DOCUMENT_SHA256,
    STACK_DATASET,
    STACK_REVISION,
    TEACHER_14B,
    TEACHER_14B_REVISION,
    TEACHER_7B,
    TEACHER_7B_REVISION,
    locked_d0_from_manifest,
    prelock_contract,
    validate_locked_d0,
)


RUN_ID = os.environ.get("STAGE5_PAPER2_D0_PRELOCK_RUN_ID", "stage5_paper2_d0_preregistration_20260726")
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
DRIVE_ROOT = Path(
    os.environ.get(
        "STAGE5_PAPER2_D0_DRIVE_ROOT",
        f"/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/{RUN_ID}",
    )
)
CHECKPOINT_SOURCE = Path(
    os.environ.get(
        "STAGE5_PAPER2_D0_DRAFTER_CHECKPOINT",
        "/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/"
        "stage5_paper2_t1_lite_r_20260725/checkpoints/t1_lite_r_raw_step_10500.pt",
    )
)


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(command: list[str], *, allowed: tuple[int, ...] = (0,)) -> int:
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
    for line in process.stdout:
        print(line, end="", flush=True)
    code = process.wait()
    if code not in allowed:
        raise subprocess.CalledProcessError(code, command)
    return code


def assert_model_and_dataset_revisions() -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    observed = {
        "drafter": api.model_info(DRAFTER_MODEL, revision=DRAFTER_MODEL_REVISION).sha,
        "teacher_7b": api.model_info(TEACHER_7B, revision=TEACHER_7B_REVISION).sha,
        "teacher_14b": api.model_info(TEACHER_14B, revision=TEACHER_14B_REVISION).sha,
        "fineweb": api.dataset_info(FINEWEB_DATASET, revision=FINEWEB_REVISION).sha,
        "stack_smol": api.dataset_info(STACK_DATASET, revision=STACK_REVISION).sha,
    }
    expected = {
        "drafter": DRAFTER_MODEL_REVISION,
        "teacher_7b": TEACHER_7B_REVISION,
        "teacher_14b": TEACHER_14B_REVISION,
        "fineweb": FINEWEB_REVISION,
        "stack_smol": STACK_REVISION,
    }
    drift = {name: {"observed": observed[name], "expected": value} for name, value in expected.items() if observed[name] != value}
    if drift:
        raise RuntimeError(f"D0 pinned Hugging Face revisions drifted: {drift}")
    return {"status": "verified", "revisions": observed}


def restore_drafter() -> Path:
    if not CHECKPOINT_SOURCE.exists():
        raise FileNotFoundError(
            f"Locked D0 drafter checkpoint is missing: {CHECKPOINT_SOURCE}. "
            "Do not substitute a stage-state or EMA checkpoint."
        )
    observed = sha256_file(CHECKPOINT_SOURCE)
    if observed != DRAFTER_CHECKPOINT_SHA256:
        raise RuntimeError(f"Locked D0 drafter SHA mismatch: {observed}")
    destination = RUN_DIR / "restored" / CHECKPOINT_SOURCE.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or sha256_file(destination) != observed:
        shutil.copy2(CHECKPOINT_SOURCE, destination)
    return destination


def copy_to_drive(path: Path) -> Path:
    relative = path.relative_to(RUN_DIR)
    destination = DRIVE_ROOT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    if sha256_file(path) != sha256_file(destination):
        raise RuntimeError(f"D0 Drive backup hash mismatch for {relative}")
    return destination


def copy_governing_document_to_drive(path: Path) -> Path:
    destination = DRIVE_ROOT / "governing" / path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    if sha256_file(path) != sha256_file(destination):
        raise RuntimeError("D0 governing-document Drive backup hash mismatch")
    return destination


def publish_lock() -> str:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    paths = [
        Path(GOVERNING_DOCUMENT),
        RUN_DIR.relative_to(ROOT) / "preregistration.json",
        RUN_DIR.relative_to(ROOT) / "data_manifest.json",
        RUN_DIR.relative_to(ROOT) / "density" / "summary.json",
        RUN_DIR.relative_to(ROOT) / "source_access.json",
        RUN_DIR.relative_to(ROOT) / "lock_receipt.json",
        RUN_DIR.relative_to(ROOT) / "summary.json",
        RUN_DIR.relative_to(ROOT) / "summary.md",
    ]
    run(["git", "add", "--", *[path.as_posix() for path in paths]])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        raise RuntimeError("D0 lock produced no staged preregistration artifacts")
    run(["git", "commit", "-m", "Lock Paper Two D0 preregistration [skip ci]"])
    lock_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    run(["git", "push", "origin", "main"])
    return lock_commit


def _merge_and_order(rows_by_stratum: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, list[dict[str, Any]]]:
    merged: dict[str, list[dict[str, Any]]] = {}
    for partition in ("label_train", "calibration", "evaluation"):
        rows = rows_by_stratum["general"][partition] + rows_by_stratum["code"][partition]
        merged[partition] = sorted(rows, key=lambda row: stable_fraction(str(row["row_id"]), seed=20260725))
    return merged


def verify_code_corpus_access() -> dict[str, Any]:
    document = next(iter_stack_documents(), None)
    if document is None:
        raise RuntimeError(
            "Pinned bigcode/the-stack-smol produced no licensed direct-text rows. "
            "Confirm the HF account accepted the dataset terms; do not substitute a corpus."
        )
    return {
        "status": "direct_text_access_verified_before_model_load",
        "dataset": STACK_DATASET,
        "revision": STACK_REVISION,
        "document_id": document.document_id,
        "content_characters": len(document.text),
        "content_sha256": hashlib.sha256(document.text.encode("utf-8")).hexdigest(),
        "metadata": document.metadata,
        "aws_used": False,
        "software_heritage_api_used": False,
    }


def main() -> int:
    D0ExecutionPolicy(density_probe_authorized=True).assert_allowed(
        density_probe=True, labeling=False, training=False
    )
    contract = prelock_contract()
    governing_path = ROOT / GOVERNING_DOCUMENT
    if sha256_file(governing_path) != GOVERNING_DOCUMENT_SHA256:
        raise RuntimeError("D0 governing Draft 7 does not match its authenticated hash")
    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        raise RuntimeError("D0 pre-lock requires HF_TOKEN for gated bigcode/the-stack-smol")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    governing_drive_path = copy_governing_document_to_drive(governing_path)
    revisions = assert_model_and_dataset_revisions()
    source_access = verify_code_corpus_access()
    source_access_path = RUN_DIR / "source_access.json"
    write_json(source_access_path, source_access)
    copy_to_drive(source_access_path)
    checkpoint = restore_drafter()
    tokenizer = AutoTokenizer.from_pretrained(DRAFTER_MODEL, revision=DRAFTER_MODEL_REVISION)

    data_dir = RUN_DIR / "data"
    probe_general, excluded_general = collect_probe_rows(
        iter_fineweb_documents(), tokenizer, stratum="general", token_budget=100_000
    )
    probe_code, excluded_code = collect_probe_rows(
        iter_stack_documents(), tokenizer, stratum="code", token_budget=100_000
    )
    probe_paths = {
        "general": data_dir / "density_general_100k.jsonl",
        "code": data_dir / "density_code_100k.jsonl",
    }
    probe_receipts = {
        "general": write_jsonl(probe_paths["general"], probe_general),
        "code": write_jsonl(probe_paths["code"], probe_code),
    }
    for path in probe_paths.values():
        copy_to_drive(path)

    density_path = RUN_DIR / "density" / "summary.json"
    run(
        [
            sys.executable,
            "eval/eval_speculative_depth_d0_density.py",
            "--data_jsonl",
            str(probe_paths["general"]),
            "--data_jsonl",
            str(probe_paths["code"]),
            "--checkpoint",
            str(checkpoint),
            "--output_summary",
            str(density_path),
            "--device",
            os.environ.get("DEVICE", "cuda"),
            "--dtype",
            os.environ.get("STAGE5_PAPER2_D0_DTYPE", "bfloat16"),
            "--attn_implementation",
            os.environ.get("ATTN_IMPLEMENTATION", "default"),
        ]
    )
    density = json.loads(density_path.read_text(encoding="utf-8"))
    densities = {
        name: float(density["strata"][name]["rejection_density_per_1000_tokens"])
        for name in ("general", "code")
    }
    mix = choose_stratum_mix(densities)
    quotas = token_quotas(2_000_000, mix["mix"])
    rows_by_stratum = {
        "general": collect_partition_rows(
            iter_fineweb_documents(),
            tokenizer,
            stratum="general",
            quotas=quotas["general"],
            excluded_document_ids=excluded_general,
        ),
        "code": collect_partition_rows(
            iter_stack_documents(),
            tokenizer,
            stratum="code",
            quotas=quotas["code"],
            excluded_document_ids=excluded_code,
        ),
    }
    merged = _merge_and_order(rows_by_stratum)
    disjoint = assert_document_disjoint(merged)
    partition_paths = {
        partition: data_dir / f"{partition}.jsonl" for partition in merged
    }
    partition_receipts = {
        partition: write_jsonl(partition_paths[partition], rows)
        for partition, rows in merged.items()
    }
    stratum_partition_paths = {
        f"{stratum}_{partition}": data_dir / f"{stratum}_{partition}.jsonl"
        for stratum in ("general", "code")
        for partition in ("label_train", "calibration", "evaluation")
    }
    stratum_partition_receipts = {
        f"{stratum}_{partition}": write_jsonl(
            stratum_partition_paths[f"{stratum}_{partition}"],
            rows_by_stratum[stratum][partition],
        )
        for stratum in ("general", "code")
        for partition in ("label_train", "calibration", "evaluation")
    }
    pilot = select_pilot_rows(merged["label_train"], count=256)
    pilot_path = data_dir / "pilot_256.jsonl"
    pilot_receipt = write_jsonl(pilot_path, pilot)
    in_era_rows, _ = collect_probe_rows(
        iter_in_era_documents(), tokenizer, stratum="general_in_era", token_budget=100_000
    )
    in_era_path = data_dir / "in_era_contrast_100k.jsonl"
    in_era_receipt = write_jsonl(in_era_path, in_era_rows)
    for path in [
        *partition_paths.values(),
        *stratum_partition_paths.values(),
        pilot_path,
        in_era_path,
        density_path,
    ]:
        copy_to_drive(path)

    artifacts = {
        "label_train": partition_receipts["label_train"],
        "calibration": partition_receipts["calibration"],
        "evaluation": partition_receipts["evaluation"],
        "density_general": probe_receipts["general"],
        "density_code": probe_receipts["code"],
        "in_era_contrast": in_era_receipt,
        "pilot_256": pilot_receipt,
        **stratum_partition_receipts,
    }
    for receipt in artifacts.values():
        local = Path(receipt["path"])
        receipt["path"] = local.relative_to(ROOT).as_posix()
        receipt["drive_path"] = str(DRIVE_ROOT / local.relative_to(RUN_DIR))
    manifest = {
        "kind": "paper2_d0_frozen_corpus_manifest",
        "status": "frozen_before_labeling",
        "dataset_revisions": revisions["revisions"],
        "fineweb_dump": FINEWEB_DUMP,
        "code_corpus": {
            "dataset": STACK_DATASET,
            "revision": STACK_REVISION,
            "lineage": "Stack_v1",
            "provenance_period": "in_pretraining_era",
            "content_store": "huggingface_direct_text",
        },
        "mix_decision": mix,
        "token_quotas": quotas,
        **disjoint,
        "density_documents_excluded_from_partitions": True,
        "artifacts": artifacts,
    }
    prereg = locked_d0_from_manifest(manifest)
    validate_locked_d0(prereg)
    write_json(RUN_DIR / "data_manifest.json", manifest)
    write_json(RUN_DIR / "preregistration.json", prereg)
    lock_receipt = {
        "kind": "paper2_d0_preregistration_lock_receipt",
        "status": "ready_to_commit_locked_before_labeling",
        "governing_document": GOVERNING_DOCUMENT,
        "governing_document_sha256": GOVERNING_DOCUMENT_SHA256,
        "governing_document_handoff_sha256": GOVERNING_DOCUMENT_HANDOFF_SHA256,
        "governing_document_drive_path": str(governing_drive_path),
        "drafter_checkpoint_source": str(CHECKPOINT_SOURCE),
        "drafter_checkpoint_sha256": sha256_file(checkpoint),
        "hf_revisions": revisions,
        "source_access": source_access_path.relative_to(ROOT).as_posix(),
        "density_summary": density_path.relative_to(ROOT).as_posix(),
        "corpus_manifest": (RUN_DIR / "data_manifest.json").relative_to(ROOT).as_posix(),
        "teacher_labeling_proper_forwards": 0,
        "teacher_14b_forwards": 0,
        "optimizer_steps": 0,
        "training_checkpoints_written": 0,
    }
    write_json(RUN_DIR / "lock_receipt.json", lock_receipt)
    summary = {
        "kind": "stage5_paper2_d0_preregistration",
        "run_id": RUN_ID,
        "status": "locked_before_labeling",
        "preregistration": (RUN_DIR / "preregistration.json").relative_to(ROOT).as_posix(),
        "data_manifest": (RUN_DIR / "data_manifest.json").relative_to(ROOT).as_posix(),
        "mix_decision": mix,
        "drafter_checkpoint_sha256": DRAFTER_CHECKPOINT_SHA256,
        "post_lock_launcher_exists": False,
        "labeling_proper_started": False,
        "training_started": False,
    }
    write_json(RUN_DIR / "summary.json", summary)
    (RUN_DIR / "summary.md").write_text(
        "# Paper Two D0 Preregistration Lock\n\n"
        f"- Status: locked before labeling\n"
        f"- FineWeb-Edu dump: `{FINEWEB_DUMP}` at `{FINEWEB_REVISION}`\n"
        f"- Code corpus: `{STACK_DATASET}` (Stack v1 lineage) at `{STACK_REVISION}`\n"
        "- Code content access: direct Hugging Face text; no AWS or Software Heritage API\n"
        f"- Frozen mix: `{mix['mix']}`\n"
        f"- Drafter SHA-256: `{DRAFTER_CHECKPOINT_SHA256}`\n"
        "- Labeling proper: not started\n"
        "- Training: not started\n",
        encoding="utf-8",
    )
    for receipt_path in (
        RUN_DIR / "preregistration.json",
        RUN_DIR / "data_manifest.json",
        density_path,
        RUN_DIR / "lock_receipt.json",
        RUN_DIR / "summary.json",
        RUN_DIR / "summary.md",
        source_access_path,
    ):
        copy_to_drive(receipt_path)
    lock_commit = publish_lock()
    print(json.dumps({**summary, "lock_commit": lock_commit}, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
