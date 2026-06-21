from __future__ import annotations

import json

from colab.build_stage5_claim_packet import build_claim_packet, main


def _write(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _release(path, *, passed: bool = True):
    _write(
        path,
        {
            "run_id": "release",
            "gate": "stage5_release_benchmark_readiness",
            "status": "ready_for_broader_benchmarks" if passed else "needs_benchmark_confirmation",
            "passed": passed,
            "next_step": "broader benchmarks",
        },
    )
    return path


def _broader(path, *, passed: bool = True):
    _write(
        path,
        {
            "run_id": "broader",
            "gate": "stage5_broader_benchmark_suite",
            "status": "passed" if passed else "needs_recurrent_recovery",
            "passed": passed,
            "next_step": "claim packet",
        },
    )
    return path


def _recipe(path, *, passed: bool = True):
    _write(
        path,
        {
            "run_id": "recipe",
            "gate": "stage5_same_recipe_architecture",
            "status": "passed" if passed else "failed",
            "passed": passed,
            "next_step": "replicate",
        },
    )
    return path


def _selector_conversion(path, *, passed: bool = True):
    _write(
        path,
        {
            "run_id": "selector_conversion",
            "gate": "stage5_same_recipe_selector_conversion",
            "kind": "recipe_selector_conversion",
            "status": "passed" if passed else "failed",
            "passed": passed,
            "next_step": "broader benchmark gate",
            "passing_selectors": [{"label": "recovered", "selection_strategy": "reliability_vote"}] if passed else [],
            "best_selector": {"label": "recovered", "selection_strategy": "reliability_vote"},
        },
    )
    return path


def _selector_replication(path, *, passed: bool = True):
    _write(
        path,
        {
            "run_id": "selector_replication",
            "gate": "stage5_selector_replication",
            "kind": "selector_replication",
            "status": "passed" if passed else "needs_confirmation",
            "passed": passed,
            "next_step": "particle gate",
            "replicated_comparisons": ["recovered__selector_reliability_vote_vs_source"] if passed else [],
        },
    )
    return path


def _particle_gate(path, *, passed: bool = True):
    _write(
        path,
        {
            "run_id": "particle_gate",
            "gate": "stage5_gate2_particle_mechanism",
            "status": "passed" if passed else "failed",
            "passed": passed,
            "next_step": "release gate",
            "best_variant": {"variant": "svgd", "selected_delta": 1, "best_of_k_delta": 1} if passed else None,
        },
    )
    return path


def _export(
    path,
    *,
    with_hash: bool = True,
    checkpoint: str = "outputs/stage5/run/phase1.pt",
    source_summary: str | None = "outputs/stage5/run/summary.json",
):
    metadata = {
        "checkpoint_source_path": checkpoint,
        "source": {"summary_path": source_summary},
    }
    if with_hash:
        metadata["checkpoint_sha256"] = "abc123"
    _write(
        path,
        {
            "run_id": "export",
            "export_dir": "outputs/hf_exports/export",
            "checkpoint": checkpoint,
            "hf_repo_id": "mshapiro123/recurrent-qwen-test",
            "metadata": metadata,
        },
    )
    return path


def _arc_agi(
    path,
    *,
    passed: bool = True,
    checkpoint: str | None = "outputs/stage5/run/phase1.pt",
    source_summary: str | None = "outputs/stage5/run/summary.json",
):
    candidate_metadata = {}
    if checkpoint:
        candidate_metadata["recovered_checkpoint"] = checkpoint
    candidate = {"metadata": candidate_metadata}
    if source_summary:
        candidate["path"] = source_summary
    _write(
        path,
        {
            "run_id": "arc_agi",
            "gate": "stage5_arc_agi_sota_comparison",
            "status": "passed" if passed else "failed",
            "passed": passed,
            "candidate": candidate,
        },
    )
    return path


def test_claim_packet_distinguishes_release_candidate_from_sota(tmp_path) -> None:
    payload = build_claim_packet(
        release_gate_summary=_release(tmp_path / "release" / "summary.json"),
        broader_benchmark_summary=_broader(tmp_path / "broader" / "summary.json"),
        recipe_control_summary=_recipe(tmp_path / "recipe" / "summary.json"),
        selector_replication_summary=_selector_replication(tmp_path / "selector_replication" / "summary.json"),
        particle_mechanism_summary=_particle_gate(tmp_path / "particle_gate" / "summary.json"),
        hf_export_summary=_export(tmp_path / "export" / "summary.json"),
        arc_agi_comparison_summary=None,
    )

    assert payload["status"] == "ready_for_release_candidate_not_sota"
    assert payload["claim_level"] == "release_candidate"
    assert payload["passed"] is True
    assert payload["criteria"][-1]["passed"] is False


def test_claim_packet_accepts_selector_conversion_as_architecture_evidence(tmp_path) -> None:
    payload = build_claim_packet(
        release_gate_summary=_release(tmp_path / "release" / "summary.json"),
        broader_benchmark_summary=_broader(tmp_path / "broader" / "summary.json"),
        recipe_control_summary=_recipe(tmp_path / "recipe" / "summary.json", passed=False),
        selector_conversion_summary=_selector_conversion(tmp_path / "selector_conversion" / "summary.json"),
        selector_replication_summary=_selector_replication(tmp_path / "selector_replication" / "summary.json"),
        particle_mechanism_summary=_particle_gate(tmp_path / "particle_gate" / "summary.json"),
        hf_export_summary=_export(tmp_path / "export" / "summary.json"),
        arc_agi_comparison_summary=None,
    )

    assert payload["status"] == "ready_for_release_candidate_not_sota"
    assert payload["claim_level"] == "release_candidate"
    assert payload["criteria"][2]["passed"] is True
    assert payload["artifacts"]["same_recipe_selector_conversion"]["passed"] is True


def test_claim_packet_requires_selector_replication_before_release_candidate(tmp_path) -> None:
    payload = build_claim_packet(
        release_gate_summary=_release(tmp_path / "release" / "summary.json"),
        broader_benchmark_summary=_broader(tmp_path / "broader" / "summary.json"),
        recipe_control_summary=_recipe(tmp_path / "recipe" / "summary.json"),
        selector_replication_summary=None,
        particle_mechanism_summary=_particle_gate(tmp_path / "particle_gate" / "summary.json"),
        hf_export_summary=_export(tmp_path / "export" / "summary.json"),
        arc_agi_comparison_summary=None,
    )

    assert payload["status"] == "needs_selector_replication"
    assert payload["claim_level"] == "not_ready"
    assert payload["passed"] is False


def test_claim_packet_requires_particle_mechanism_gate_before_release_candidate(tmp_path) -> None:
    payload = build_claim_packet(
        release_gate_summary=_release(tmp_path / "release" / "summary.json"),
        broader_benchmark_summary=_broader(tmp_path / "broader" / "summary.json"),
        recipe_control_summary=_recipe(tmp_path / "recipe" / "summary.json"),
        selector_replication_summary=_selector_replication(tmp_path / "selector_replication" / "summary.json"),
        particle_mechanism_summary=None,
        hf_export_summary=_export(tmp_path / "export" / "summary.json"),
        arc_agi_comparison_summary=None,
    )

    assert payload["status"] == "needs_particle_mechanism_gate"
    assert payload["claim_level"] == "not_ready"
    assert payload["passed"] is False


def test_claim_packet_can_mark_sota_candidate_when_authoritative_comparison_exists(tmp_path) -> None:
    payload = build_claim_packet(
        release_gate_summary=_release(tmp_path / "release" / "summary.json"),
        broader_benchmark_summary=_broader(tmp_path / "broader" / "summary.json"),
        recipe_control_summary=_recipe(tmp_path / "recipe" / "summary.json"),
        selector_replication_summary=_selector_replication(tmp_path / "selector_replication" / "summary.json"),
        particle_mechanism_summary=_particle_gate(tmp_path / "particle_gate" / "summary.json"),
        hf_export_summary=_export(tmp_path / "export" / "summary.json"),
        arc_agi_comparison_summary=_arc_agi(tmp_path / "arc_agi" / "summary.json"),
    )

    assert payload["status"] == "sota_claim_ready"
    assert payload["claim_level"] == "sota_candidate"
    assert all(row["passed"] for row in payload["criteria"])


def test_claim_packet_requires_sota_export_checkpoint_linkage(tmp_path) -> None:
    payload = build_claim_packet(
        release_gate_summary=_release(tmp_path / "release" / "summary.json"),
        broader_benchmark_summary=_broader(tmp_path / "broader" / "summary.json"),
        recipe_control_summary=_recipe(tmp_path / "recipe" / "summary.json"),
        selector_replication_summary=_selector_replication(tmp_path / "selector_replication" / "summary.json"),
        particle_mechanism_summary=_particle_gate(tmp_path / "particle_gate" / "summary.json"),
        hf_export_summary=_export(tmp_path / "export" / "summary.json", checkpoint="outputs/stage5/run/phase1.pt"),
        arc_agi_comparison_summary=_arc_agi(
            tmp_path / "arc_agi" / "summary.json",
            checkpoint="outputs/stage5/other/phase1.pt",
        ),
    )

    assert payload["status"] == "ready_for_release_candidate_needs_sota_export_linkage"
    assert payload["claim_level"] == "release_candidate"
    assert payload["passed"] is True
    assert payload["artifacts"]["sota_export_linkage"]["matched_on"] == "checkpoint"
    assert payload["artifacts"]["sota_export_linkage"]["passed"] is False
    assert payload["criteria"][-1]["passed"] is False


def test_claim_packet_accepts_source_summary_linkage_when_checkpoint_missing(tmp_path) -> None:
    payload = build_claim_packet(
        release_gate_summary=_release(tmp_path / "release" / "summary.json"),
        broader_benchmark_summary=_broader(tmp_path / "broader" / "summary.json"),
        recipe_control_summary=_recipe(tmp_path / "recipe" / "summary.json"),
        selector_replication_summary=_selector_replication(tmp_path / "selector_replication" / "summary.json"),
        particle_mechanism_summary=_particle_gate(tmp_path / "particle_gate" / "summary.json"),
        hf_export_summary=_export(
            tmp_path / "export" / "summary.json",
            checkpoint="outputs/stage5/run/phase1.pt",
            source_summary="outputs/stage5/run/summary.json",
        ),
        arc_agi_comparison_summary=_arc_agi(
            tmp_path / "arc_agi" / "summary.json",
            checkpoint=None,
            source_summary="outputs/stage5/run/summary.json",
        ),
    )

    assert payload["status"] == "sota_claim_ready"
    assert payload["artifacts"]["sota_export_linkage"]["matched_on"] == "source_summary"


def test_claim_packet_blocks_sota_when_export_linkage_is_unverified(tmp_path) -> None:
    payload = build_claim_packet(
        release_gate_summary=_release(tmp_path / "release" / "summary.json"),
        broader_benchmark_summary=_broader(tmp_path / "broader" / "summary.json"),
        recipe_control_summary=_recipe(tmp_path / "recipe" / "summary.json"),
        selector_replication_summary=_selector_replication(tmp_path / "selector_replication" / "summary.json"),
        particle_mechanism_summary=_particle_gate(tmp_path / "particle_gate" / "summary.json"),
        hf_export_summary=_export(
            tmp_path / "export" / "summary.json",
            checkpoint="outputs/stage5/run/phase1.pt",
            source_summary=None,
        ),
        arc_agi_comparison_summary=_arc_agi(
            tmp_path / "arc_agi" / "summary.json",
            checkpoint=None,
            source_summary=None,
        ),
    )

    assert payload["status"] == "ready_for_release_candidate_needs_sota_export_linkage"
    assert payload["artifacts"]["sota_export_linkage"]["verified"] is False
    assert payload["criteria"][-1]["passed"] is False


def test_claim_packet_requires_hf_export_hash(tmp_path) -> None:
    payload = build_claim_packet(
        release_gate_summary=_release(tmp_path / "release" / "summary.json"),
        broader_benchmark_summary=_broader(tmp_path / "broader" / "summary.json"),
        recipe_control_summary=_recipe(tmp_path / "recipe" / "summary.json"),
        selector_replication_summary=_selector_replication(tmp_path / "selector_replication" / "summary.json"),
        particle_mechanism_summary=_particle_gate(tmp_path / "particle_gate" / "summary.json"),
        hf_export_summary=_export(tmp_path / "export" / "summary.json", with_hash=False),
        arc_agi_comparison_summary=None,
    )

    assert payload["status"] == "needs_hf_export"
    assert payload["passed"] is False


def test_claim_packet_cli_writes_outputs(tmp_path, monkeypatch) -> None:
    release = _release(tmp_path / "release" / "summary.json")
    broader = _broader(tmp_path / "broader" / "summary.json")
    recipe = _recipe(tmp_path / "recipe" / "summary.json")
    selector_replication = _selector_replication(tmp_path / "selector_replication" / "summary.json")
    particle_gate = _particle_gate(tmp_path / "particle_gate" / "summary.json")
    export = _export(tmp_path / "export" / "summary.json")
    output_json = tmp_path / "claim.json"
    output_md = tmp_path / "claim.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_stage5_claim_packet.py",
            "--release_gate_summary",
            str(release),
            "--broader_benchmark_summary",
            str(broader),
            "--recipe_control_summary",
            str(recipe),
            "--selector_replication_summary",
            str(selector_replication),
            "--particle_mechanism_summary",
            str(particle_gate),
            "--hf_export_summary",
            str(export),
            "--output_json",
            str(output_json),
            "--output_md",
            str(output_md),
        ],
    )

    assert main() == 0
    assert json.loads(output_json.read_text(encoding="utf-8"))["status"] == "ready_for_release_candidate_not_sota"
    assert "Claim Guardrail" in output_md.read_text(encoding="utf-8")
