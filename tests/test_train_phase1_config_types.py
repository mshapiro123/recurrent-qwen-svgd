from training.train_phase1_ponder import cfg_float, cfg_int


def test_cfg_float_accepts_yaml_scientific_notation_strings() -> None:
    assert cfg_float({"learning_rate": "1e-05"}, "learning_rate", 1e-4) == 1e-5
    assert cfg_float({"learning_rate": "2e-5"}, "learning_rate", 1e-4) == 2e-5


def test_cfg_numeric_helpers_use_defaults_for_none() -> None:
    assert cfg_float({"learning_rate": None}, "learning_rate", 1e-4) == 1e-4
    assert cfg_int({"max_steps": None}, "max_steps", 25) == 25


def test_cfg_int_accepts_string_values() -> None:
    assert cfg_int({"max_steps": "50"}, "max_steps", 25) == 50
