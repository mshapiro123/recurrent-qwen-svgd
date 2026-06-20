from __future__ import annotations

from colab.run_stage5_arc_agi_candidate_gate import build_variants


def test_candidate_gate_exposes_symbolic_program_variants(monkeypatch) -> None:
    monkeypatch.delenv("STAGE5_ARC_AGI_GATE_VARIANTS", raising=False)

    variants = {variant.name: variant for variant in build_variants()}

    assert variants["symbolic_only"].symbolic_candidate_format == "grid"
    assert variants["symbolic_program_only"].symbolic_candidate_format == "program"
    assert variants["base_hybrid_program_first"].symbolic_candidate_format == "program"
    assert variants["phase1_hybrid_program_first"].symbolic_candidate_format == "program"
