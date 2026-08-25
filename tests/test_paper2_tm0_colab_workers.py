from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_tm0_cache_worker_is_phase_isolated_and_score_blind() -> None:
    text = (ROOT / "colab" / "run_tm0_cache_model_worker.py").read_text()
    assert 'MODEL_KEYS = {"student", "teacher_7b", "teacher_14b"}' in text
    assert '"--model_key"' in text
    assert 'confirm_scored=False' in text
    assert 'eval_e_scored=False' in text
    assert "eval.eval_paper2_phase3_p31_references" not in text
    assert "except BaseException" not in text


def test_tm0_score_worker_is_resumable_and_seal_guarded() -> None:
    text = (ROOT / "colab" / "run_tm0_score_worker.py").read_text()
    assert 'OUTPUT / "teacher_7b_scores.jsonl"' in text
    assert '"sealed_before_model_scoring"' in text
    assert '"--confirm_seal_ledger"' in text
    assert 'if rows != 6144' in text
    assert 'confirm_scored=False' in text
    assert 'eval_e_scored=False' in text
    assert "except BaseException" not in text
