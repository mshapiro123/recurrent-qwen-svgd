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


def _export(path, *, with_hash: bool = True):
    _write(
        path,
        {
            "run_id": "export",
            "export_dir": "outputs/hf_exports/export",
            "checkpoint": "outputs/stage5/run/phase1.pt",
            "hf_repo_id": "mshapiro123/recurrent-qwen-test",
            "metadata": {"checkpoint_sha256": "abc123"} if with_hash else {},
        },
    )
    return path


def _arc_agi(path, *, passed: bool = True):
    _write(
        path,
        {
            "run_id": "arc_agi",
            "gate": "stage5_arc_agi_sota_comparison",
            "status": "passed" if passed else "failed",
            "passed": passed,
        },
    )
    return path


def test_claim_packet_distinguishes_release_candidate_from_sota(tmp_path) -> None:
    payload = build_claim_packet(
        release_gate_summary=_release(tmp_path / "release" / "summary.json"),
        broader_benchmark_summary=_broader(tmp_path / "broader" / "summary.json"),
        recipe_control_summary=_recipe(tmp_path / "recipe" / "summary.json"),
        hf_export_summary=_export(tmp_path / "export" / "summary.json"),
        arc_agi_comparison_summary=None,
    )

    assert payload["status"] == "ready_for_release_candidate_not_sota"
    assert payload["claim_level"] == "release_candidate"
    assert payload["passed"] is True
    assert payload["criteria"][-1]["passed"] is False


def test_claim_packet_can_mark_sota_candidate_when_authoritative_comparison_exists(tmp_path) -> None:
    payload = build_claim_packet(
        release_gate_summary=_release(tmp_path / "release" / "summary.json"),
        broader_benchmark_summary=_broader(tmp_path / "broader" / "summary.json"),
        recipe_control_summary=_recipe(tmp_path / "recipe" / "summary.json"),
        hf_export_summary=_export(tmp_path / "export" / "summary.json"),
        arc_agi_comparison_summary=_arc_agi(tmp_path / "arc_agi" / "summary.json"),
    )

    assert payload["status"] == "sota_claim_ready"
    assert payload["claim_level"] == "sota_candidate"
    assert all(row["passed"] for row in payload["criteria"])


def test_claim_packet_requires_hf_export_hash(tmp_path) -> None:
    payload = build_claim_packet(
        release_gate_summary=_release(tmp_path / "release" / "summary.json"),
        broader_benchmark_summary=_broader(tmp_path / "broader" / "summary.json"),
        recipe_control_summary=_recipe(tmp_path / "recipe" / "summary.json"),
        hf_export_summary=_export(tmp_path / "export" / "summary.json", with_hash=False),
        arc_agi_comparison_summary=None,
    )

    assert payload["status"] == "needs_hf_export"
    assert payload["passed"] is False


def test_claim_packet_cli_writes_outputs(tmp_path, monkeypatch) -> None:
    release = _release(tmp_path / "release" / "summary.json")
    broader = _broader(tmp_path / "broader" / "summary.json")
    recipe = _recipe(tmp_path / "recipe" / "summary.json")
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
