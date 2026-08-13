from __future__ import annotations

from pathlib import Path

import pytest

from colab.run_stage5_paper2_phase3_p34_cli import (
    PrivateRelease,
    assert_training_amendment,
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
    assert "--transport_only" in source
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
        '"tier_w_consecutive_looks":2}}\n',
        encoding="utf-8",
    )
    assert_training_amendment(lock_path=lock, expected_sha256="approved")

    with pytest.raises(RuntimeError, match="absent or mismatched"):
        assert_training_amendment(lock_path=lock, expected_sha256="different")

    lock.write_text(
        '{"guardrail_amendment":{"sha256":"approved"},'
        '"guardrails":{"tier_s_consecutive_looks":2,'
        '"tier_w_consecutive_looks":2}}\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Tier-S"):
        assert_training_amendment(lock_path=lock, expected_sha256="approved")
