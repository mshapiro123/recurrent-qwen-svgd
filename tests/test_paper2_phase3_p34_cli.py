from __future__ import annotations

import tarfile
from pathlib import Path

import pytest
import zstandard

from colab.run_stage5_paper2_phase3_p34_cli import (
    PrivateRelease,
    assert_training_amendment,
    package_receipts,
    restore_receipts,
    verify_sha256s,
)


ROOT = Path(__file__).resolve().parents[1]


def test_cli_launcher_is_mount_free_private_and_durable() -> None:
    source = (ROOT / "colab/run_stage5_paper2_phase3_p34_cli.py").read_text(
        encoding="utf-8"
    )
    assert "drive.mount" not in source
    assert 'PRIVATE_REPO = "mshapiro123/recurrent-qwen-svgd-runtime-private"' in source
    assert "transport_parts_manifest.json" in source
    assert "resume_name" in source
    assert "campaign_release.upload" in source
    assert '"status": "running"' in source
    assert 'checkpoint_step = int(checkpoint["step"])' in source
    assert 'campaign_release.upload(final_bundle, f"{label}-latest-receipts.tar.zst")' in source
    assert "--transport_only" in source
    assert "--preflight_only" in source
    assert "--guardrail_amendment_sha256" in source
    assert '"confirm_scored": False' not in source  # No evaluation code is embedded here.


def test_transport_sha_verifier_rejects_mutation(tmp_path: Path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"registered")
    (tmp_path / "SHA256SUMS").write_text(
        "6890f46ff2c48e5d45d18cc0a4c4385dfaa49b212e8838c17eaa10a31acc1916  payload.bin\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="SHA mismatch"):
        verify_sha256s(tmp_path)


def test_private_release_requires_a_nonempty_token(tmp_path: Path) -> None:
    token = tmp_path / "token"
    token.write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="token is empty"):
        PrivateRelease(token_file=token, tag="unused")


def test_training_amendment_gate_requires_ratified_nested_rule(tmp_path: Path) -> None:
    lock = tmp_path / "lock.json"
    lock.write_text(
        '{"guardrail_amendment":{"sha256":"approved"},'
        '"guardrails":{"tier_s_consecutive_looks":4,'
        '"tier_w_consecutive_looks":2},'
        '"loss_share_contract":{'
        '"estimator_cadence":"non-overlapping trailing 100-step windows read at steps divisible by 100",'
        '"rule":"two consecutive breaches demote one controller rung; four stop"}}\n',
        encoding="utf-8",
    )
    assert_training_amendment(lock_path=lock, expected_sha256="approved")

    with pytest.raises(RuntimeError, match="absent or mismatched"):
        assert_training_amendment(lock_path=lock, expected_sha256="different")

    lock.write_text(
        '{"guardrail_amendment":{"sha256":"approved"},'
        '"guardrails":{"tier_s_consecutive_looks":2,'
        '"tier_w_consecutive_looks":2},'
        '"loss_share_contract":{'
        '"estimator_cadence":"non-overlapping trailing 100-step windows read at steps divisible by 100",'
        '"rule":"two consecutive breaches demote one controller rung; four stop"}}\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Tier-S"):
        assert_training_amendment(lock_path=lock, expected_sha256="approved")


def test_receipt_bundle_does_not_archive_itself_or_prior_bundles(tmp_path: Path) -> None:
    output = tmp_path / "output"
    private = tmp_path / "private"
    output.mkdir()
    private.mkdir()
    (output / "summary.json").write_text("{}\n", encoding="utf-8")
    (private / "campaign.log").write_text("healthy\n", encoding="utf-8")
    (private / "resume.pt").write_bytes(b"checkpoint")
    (private / "old-receipts.tar.zst").write_bytes(b"old")
    destination = private / "latest-receipts.tar.zst"

    package_receipts(output_dir=output, private_dir=private, destination=destination)

    with destination.open("rb") as source:
        with zstandard.ZstdDecompressor().stream_reader(source) as decoded:
            with tarfile.open(fileobj=decoded, mode="r|") as archive:
                names = [member.name for member in archive]
    assert "outputs/summary.json" in names
    assert "private/campaign.log" in names
    assert all("resume.pt" not in name for name in names)
    assert all("receipts.tar.zst" not in name for name in names)


def test_receipt_bundle_restores_prior_scientific_artifacts(tmp_path: Path) -> None:
    old_output = tmp_path / "old-output"
    old_private = tmp_path / "old-private"
    old_output.mkdir()
    old_private.mkdir()
    (old_output / "task_summary_look_01.json").write_text("{}\n", encoding="utf-8")
    (old_private / "task_rows_look_01.jsonl").write_text("{}\n", encoding="utf-8")
    bundle = tmp_path / "latest-receipts.tar.zst"
    package_receipts(
        output_dir=old_output, private_dir=old_private, destination=bundle
    )
    new_output = tmp_path / "new-output"
    new_private = tmp_path / "new-private"
    restore_receipts(
        bundle=bundle, output_dir=new_output, private_dir=new_private
    )
    assert (new_output / "task_summary_look_01.json").is_file()
    assert (new_private / "task_rows_look_01.jsonl").is_file()
