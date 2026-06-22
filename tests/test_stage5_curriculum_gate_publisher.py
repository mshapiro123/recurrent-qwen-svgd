from __future__ import annotations

import json
import subprocess

import pytest


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def green_gate(work_dir: str = "data/curriculum/programmatic_direct_deep_001") -> dict:
    return {
        "kind": "curriculum_sft_gate",
        "go": True,
        "status": "go_train_recurrent_sft",
        "work_dir": work_dir,
        "summary_json": f"{work_dir}/summary.json",
        "checks": {
            "positive_sft": {
                "rows": 128,
                "mode_requirements": {
                    "direct": {"required": 64, "observed": 64, "passed": True},
                    "deep_narrow": {"required": 64, "observed": 64, "passed": True},
                },
            }
        },
        "artifacts": {"positive_sft": f"{work_dir}/positive_sft.jsonl"},
    }


def test_publish_gate_updates_current_source_pointer(monkeypatch, tmp_path) -> None:
    import colab.publish_stage5_curriculum_gate as module

    root = tmp_path
    gate = root / "data" / "curriculum" / "programmatic" / "curriculum_sft_gate.json"
    gate_md = gate.with_suffix(".md")
    write_json(gate, green_gate())
    gate_md.write_text("# Gate\n", encoding="utf-8")
    monkeypatch.setattr(module, "ROOT", root)

    result = module.publish_gate(
        gate_json=gate,
        gate_md=gate_md,
        published_dir=root / "outputs" / "stage5" / "published_gate",
    )

    assert result["published_gate"] == "outputs/stage5/published_gate/curriculum_sft_gate.json"
    assert result["published_md"] == "outputs/stage5/published_gate/curriculum_sft_gate.md"
    assert result["pointer"] == "config/stage5_current_source_summary.txt"
    assert (root / result["published_gate"]).exists()
    assert (root / "config" / "stage5_current_source_summary.txt").read_text(
        encoding="utf-8"
    ) == "outputs/stage5/published_gate/curriculum_sft_gate.json\n"
    published_payload = json.loads((root / result["published_gate"]).read_text(encoding="utf-8"))
    assert published_payload["work_dir"] == "data/curriculum/programmatic_direct_deep_001"
    assert published_payload["summary_json"] == "data/curriculum/programmatic_direct_deep_001/summary.json"


def test_publish_gate_refuses_no_go_payload(monkeypatch, tmp_path) -> None:
    import colab.publish_stage5_curriculum_gate as module

    root = tmp_path
    gate = root / "gate.json"
    payload = green_gate()
    payload["go"] = False
    payload["status"] = "no_go"
    write_json(gate, payload)
    monkeypatch.setattr(module, "ROOT", root)

    with pytest.raises(ValueError, match="not green"):
        module.publish_gate(
            gate_json=gate,
            gate_md=None,
            published_dir=root / "outputs" / "stage5" / "published_gate",
        )


def test_publish_gate_refuses_missing_handoff_fields(monkeypatch, tmp_path) -> None:
    import colab.publish_stage5_curriculum_gate as module

    root = tmp_path
    gate = root / "gate.json"
    payload = green_gate()
    payload.pop("work_dir")
    write_json(gate, payload)
    monkeypatch.setattr(module, "ROOT", root)

    with pytest.raises(ValueError, match="missing work_dir"):
        module.publish_gate(
            gate_json=gate,
            gate_md=None,
            published_dir=root / "outputs" / "stage5" / "published_gate",
        )


def test_publish_gate_refuses_missing_mode_requirements(monkeypatch, tmp_path) -> None:
    import colab.publish_stage5_curriculum_gate as module

    root = tmp_path
    gate = root / "gate.json"
    payload = green_gate()
    payload["checks"]["positive_sft"].pop("mode_requirements")
    write_json(gate, payload)
    monkeypatch.setattr(module, "ROOT", root)

    with pytest.raises(ValueError, match="mode_requirements"):
        module.publish_gate(
            gate_json=gate,
            gate_md=None,
            published_dir=root / "outputs" / "stage5" / "published_gate",
        )


def test_publish_gate_refuses_failed_mode_requirement(monkeypatch, tmp_path) -> None:
    import colab.publish_stage5_curriculum_gate as module

    root = tmp_path
    gate = root / "gate.json"
    payload = green_gate()
    payload["checks"]["positive_sft"]["mode_requirements"]["direct"]["observed"] = 63
    payload["checks"]["positive_sft"]["mode_requirements"]["direct"]["passed"] = False
    write_json(gate, payload)
    monkeypatch.setattr(module, "ROOT", root)

    with pytest.raises(ValueError, match="mode requirement"):
        module.publish_gate(
            gate_json=gate,
            gate_md=None,
            published_dir=root / "outputs" / "stage5" / "published_gate",
        )


def test_gate_publisher_git_commit_stages_gate_and_pointer(monkeypatch, tmp_path) -> None:
    import colab.publish_stage5_curriculum_gate as module

    root = tmp_path
    published_gate = root / "outputs" / "stage5" / "published_gate" / "curriculum_sft_gate.json"
    pointer = root / "config" / "stage5_current_source_summary.txt"
    published_gate.parent.mkdir(parents=True)
    pointer.parent.mkdir(parents=True)
    published_gate.write_text("{}", encoding="utf-8")
    pointer.write_text("outputs/stage5/published_gate/curriculum_sft_gate.json\n", encoding="utf-8")
    commands: list[list[str]] = []

    def fake_run(cmd, *, check=True):
        commands.append([str(item) for item in cmd])
        if cmd[:3] == ["git", "diff", "--cached"]:
            return subprocess.CompletedProcess(cmd, 1, "", None)
        return subprocess.CompletedProcess(cmd, 0, "", None)

    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "run", fake_run)

    assert module.git_commit_and_push(
        [
            "outputs/stage5/published_gate/curriculum_sft_gate.json",
            "config/stage5_current_source_summary.txt",
        ],
        commit_message="Publish gate",
        push=True,
    ) is True

    assert [
        "git",
        "add",
        "-f",
        "outputs/stage5/published_gate/curriculum_sft_gate.json",
        "config/stage5_current_source_summary.txt",
    ] in commands
    assert ["git", "commit", "-m", "Publish gate"] in commands
    assert ["git", "push", "origin", "main"] in commands
