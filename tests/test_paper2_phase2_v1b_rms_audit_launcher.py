from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rms_audit_target_is_cpu_only_and_uses_existing_private_rows() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(
        encoding="utf-8"
    )
    runner = (ROOT / "colab/run_stage5_paper2_phase2_v1b_rms_audit.py").read_text(
        encoding="utf-8"
    )
    for marker in (
        "paper2_phase2_v1b_rms_audit",
        "STAGE5_PAPER2_PHASE2_V1B_RMS_AUDIT_CELL.py",
        "paper2_phase2_v1b_rms_audit_v1",
        "CPU only existing private V1b records no model inference no training",
        "p99 or fixed multiple median cap recommendation",
    ):
        assert marker in bootstrap
    assert "v1b_neutral_v2" in runner
    assert "audit_paper2_phase2_v1b_rms" in runner
    assert "nvidia-smi" not in runner
    assert "torch.optim" not in runner
