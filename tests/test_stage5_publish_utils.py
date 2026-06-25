from __future__ import annotations

from colab.stage5_publish_utils import publishable_artifact_paths


def test_publishable_artifact_paths_excludes_checkpoints(tmp_path) -> None:
    run_dir = tmp_path / "outputs" / "stage5" / "run"
    (run_dir / "summary.json").parent.mkdir(parents=True)
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    (run_dir / "summary.md").write_text("# Summary\n", encoding="utf-8")
    (run_dir / "phase1" / "phase1_step_25.pt").parent.mkdir(parents=True)
    (run_dir / "phase1" / "phase1_step_25.pt").write_bytes(b"checkpoint")
    (run_dir / "adapter.safetensors").write_bytes(b"weights")
    (run_dir / "debug.tmp").write_text("ignore me", encoding="utf-8")

    rels = [path.relative_to(run_dir).as_posix() for path in publishable_artifact_paths(run_dir)]

    assert rels == ["summary.json", "summary.md"]
