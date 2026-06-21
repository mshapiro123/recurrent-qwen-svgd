from __future__ import annotations

from colab.stage5_model_metadata import configured_params_b, finite_float, infer_params_b, model_metadata


def test_infer_params_b_from_common_model_names() -> None:
    assert infer_params_b("Qwen/Qwen2.5-0.5B-Instruct") == 0.5
    assert infer_params_b("Qwen/Qwen2.5-1.5B-Instruct") == 1.5
    assert infer_params_b("frontier-reasoner-3B") == 3.0


def test_infer_params_b_uses_last_parameter_token() -> None:
    assert infer_params_b("family-0.5B/distilled-1.5B") == 1.5


def test_configured_params_b_prefers_stage5_environment_override() -> None:
    environ = {"STAGE5_MODEL_PARAMS_B": "0.75", "MODEL_PARAMS_B": "1.5"}

    assert configured_params_b(model_name="unknown-model", environ=environ) == 0.75


def test_configured_params_b_falls_back_to_model_params_override() -> None:
    environ = {"MODEL_PARAMS_B": "1.5"}

    assert configured_params_b(model_name="unknown-model", environ=environ) == 1.5


def test_configured_params_b_ignores_invalid_overrides() -> None:
    environ = {"STAGE5_MODEL_PARAMS_B": "nan", "MODEL_PARAMS_B": "not-a-number"}

    assert configured_params_b(model_name="Qwen/Qwen2.5-0.5B-Instruct", environ=environ) == 0.5


def test_model_metadata_includes_name_and_parameter_count() -> None:
    assert model_metadata("Qwen/Qwen2.5-0.5B-Instruct") == {
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "params_b": 0.5,
    }


def test_finite_float_rejects_bool_none_and_nonfinite_values() -> None:
    assert finite_float(True) is None
    assert finite_float(None) is None
    assert finite_float("inf") is None
    assert finite_float("-inf") is None
    assert finite_float("nan") is None
    assert finite_float("0.5") == 0.5
