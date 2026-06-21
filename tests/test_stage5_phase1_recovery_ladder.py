from __future__ import annotations


def test_phase1_distillation_config_reflects_globals(monkeypatch) -> None:
    import colab.run_stage5_phase1_recovery_ladder as module

    monkeypatch.setattr(module, "DISTILL_ENABLED", True)
    monkeypatch.setattr(module, "DISTILL_WEIGHT", 0.25)
    monkeypatch.setattr(module, "DISTILL_TEMPERATURE", 2.5)
    monkeypatch.setattr(module, "DISTILL_ON", "all")
    monkeypatch.setattr(module, "DISTILL_TEACHER_MODEL_NAME", "teacher/model")
    monkeypatch.setattr(module, "DISTILL_DTYPE", "bfloat16")

    assert module.phase1_distillation_config() == {
        "enabled": True,
        "weight": 0.25,
        "temperature": 2.5,
        "on": "all",
        "teacher_model_name": "teacher/model",
        "dtype": "bfloat16",
    }
