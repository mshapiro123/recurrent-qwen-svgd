from __future__ import annotations

import yaml


def test_phase2_config_contains_projected_svgd_knobs(monkeypatch, tmp_path) -> None:
    import colab.run_stage5_recovered_phase2_smoke as module

    monkeypatch.setattr(module, "RUN_DIR", tmp_path)
    monkeypatch.setattr(module, "RECOVERED_CHECKPOINT", tmp_path / "parent.pt")
    monkeypatch.setattr(module, "PROJECTION_DIM", "8")
    monkeypatch.setattr(module, "REPULSION_SCALE", "2")
    monkeypatch.setattr(module, "PARTICLE_INIT_NOISE", "0.05")

    cfg_path = module.phase2_config(tmp_path / "projection.pt")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    assert cfg["resume_from"].endswith("parent.pt")
    assert cfg["sample_latents"] is False
    assert cfg["particle_update_mode"] == "svgd"
    assert cfg["particle_init_noise"] == 0.05
    assert cfg["svgd_repulsion_scale"] == 2.0
    assert cfg["svgd_kernel_projection_path"].endswith("projection.pt")
    assert cfg["svgd_kernel_projection_dim"] == 8
    assert cfg["svgd_kernel_geometry"] == "euclidean"
