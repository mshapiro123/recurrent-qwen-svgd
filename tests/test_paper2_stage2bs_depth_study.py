from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

from colab import run_stage5_paper2_stage2bs_depth_study as depth_runner
from eval.eval_paper2_stage2bs_depth_study import Stage2BScheduleGraph
from models.paper2_dc2_student import Phase3StudentModules
from models.paper2_stage2b_depth import Stage2BDepthAttachment
from models.recurrent_wrapper import LayerSplit, RecurrentQwenForCausalLM
from tests.test_recurrent_wrapper_tiny import TinyCausalLM
from training.paper2_stage2bs_depth_study import (
    EXPECTED_INITIALIZATION_STATE_DIGESTS,
    EXPECTED_NATIVE_COUNTS,
    INITIALIZATION_SEED_BASE,
    load_lock,
    resolve_direct_branch,
    resolve_keys,
    schedule_amplitudes,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return depth_runner.sha256_file(path)


def _tiny_wrapper() -> tuple[RecurrentQwenForCausalLM, torch.Tensor, torch.Tensor]:
    torch.manual_seed(20260822)
    base = TinyCausalLM().eval()
    wrapper = RecurrentQwenForCausalLM(
        base, layer_split=LayerSplit(prelude_end=1, recurrent_end=3)
    ).eval()
    sidecar = Phase3StudentModules(
        tied_embedding=base.model.embed_tokens,
        hidden_size=8,
        latent_dim=8,
        n_slots=8,
        control_dim=4,
        draft_rank=4,
        max_steps=4,
        rms_cap=0.5,
    ).eval()
    wrapper.install_stage2b_depth_attachment(Stage2BDepthAttachment.from_phase3(sidecar))
    tokens = torch.tensor([[1, 2, 3, 4]])
    return wrapper, tokens, torch.ones_like(tokens)


def test_locked_contract_is_machine_readable() -> None:
    lock = load_lock(ROOT / "training/paper2_stage2bs_depth_study_lock.json")
    assert lock["expected_native_counts"] == {
        str(seed): values for seed, values in EXPECTED_NATIVE_COUNTS.items()
    }
    assert lock["optimizer_steps_allowed"] == 0
    assert lock["confirm_scored"] is False
    assert lock["eval_e_scored"] is False
    assert lock["runtime"]["generation_batch_size"] == 8
    assert lock["runtime"]["margin_batch_size"] == 2
    assert lock["initialization"]["seed_base"] == INITIALIZATION_SEED_BASE
    assert lock["initialization"]["state_digest_by_seed"] == {
        str(seed): digest
        for seed, digest in EXPECTED_INITIALIZATION_STATE_DIGESTS.items()
    }
    assert lock["expected_native_counts"]["1"] == [162, 9, 5, 1]
    assert lock["cascade"]["direct_discriminator"]["stop_after_both_seeds"] is True
    assert lock["cascade"]["all_three_fail_action"] == (
        "bank_SUBTRACTIVE_and_close_implementation_line"
    )


def test_schedule_amplitude_matrix_matches_lock() -> None:
    assert schedule_amplitudes("native_interleaved") == (0.0, 0.02, 0.05)
    assert schedule_amplitudes("deferred_terminal_write_no_reentry") == (0.0, 0.02, 0.05)
    assert schedule_amplitudes("per_loop_write_no_reentry") == (0.05,)
    assert schedule_amplitudes("partial_interleave_pairs") == (0.05,)


def test_schedule_dependent_key_requires_both_seeds() -> None:
    cells = []
    for seed in (0, 1):
        cells.extend(
            [
                {"seed": seed, "schedule": "native_interleaved", "k": 4, "correct": 2},
                {
                    "seed": seed,
                    "schedule": "deferred_terminal_write_no_reentry",
                    "k": 4,
                    "correct": 190,
                },
            ]
        )
    result = resolve_keys(cells, native_k1_by_seed={0: 162, 1: 162})
    assert result["ADDITIVE"] is True
    assert result["SCHEDULE_DEPENDENT"] is True
    assert result["SUBTRACTIVE"] is False
    assert result["seed_disagreement"] is False


def test_key_resolution_escalates_seed_disagreement() -> None:
    cells = [
        {
            "seed": 0,
            "schedule": "deferred_terminal_write_no_reentry",
            "k": 4,
            "correct": 190,
        },
        {
            "seed": 1,
            "schedule": "deferred_terminal_write_no_reentry",
            "k": 4,
            "correct": 170,
        },
    ]
    result = resolve_keys(cells, native_k1_by_seed={0: 162, 1: 162})
    assert result["seed_disagreement"] is True
    assert result["requires_strategy_escalation"] is True


def test_direct_cascade_stops_for_relay_before_recovery_branch() -> None:
    rows = []
    for seed in (0, 1):
        rows.extend(
            {
                "seed": seed,
                "endpoint": "initialization",
                "schedule": "deferred_terminal_write_no_reentry",
                "amplitude": 0.05,
                "k": k,
                "correct": 190 if k == 4 else 160,
            }
            for k in range(1, 5)
        )
    result = resolve_direct_branch(rows, native_k1_by_seed={0: 162, 1: 162})
    assert result["branch"] == "RECOVERY_BRANCH_AUTHORIZED_AWAITING_RELAY"
    assert result["requires_relay_before_branch"] is True


def test_direct_cascade_seed_split_enters_neither_branch() -> None:
    rows = []
    for seed, k4 in ((0, 190), (1, 170)):
        rows.extend(
            {
                "seed": seed,
                "endpoint": "initialization",
                "schedule": "deferred_terminal_write_no_reentry",
                "amplitude": 0.05,
                "k": k,
                "correct": k4 if k == 4 else 160,
            }
            for k in range(1, 5)
        )
    result = resolve_direct_branch(rows, native_k1_by_seed={0: 162, 1: 162})
    assert result["branch"] == "STOP_SEED_SPLIT_REQUIRED_RELAY"


def test_lock_json_has_no_optimizer_or_sealed_partition_escape_hatch() -> None:
    raw = json.loads(
        (ROOT / "training/paper2_stage2bs_depth_study_lock.json").read_text(
            encoding="utf-8"
        )
    )
    assert raw["training_authorized"] is False
    assert raw["optimizer_steps_allowed"] == 0
    assert raw["panels"]["confirm_scored"] is False
    assert raw["panels"]["eval_e_scored"] is False


def test_deferred_zero_write_is_exact_native_k1_for_every_update_count() -> None:
    wrapper, tokens, mask = _tiny_wrapper()
    native_k1 = Stage2BScheduleGraph(
        wrapper=wrapper,
        schedule="native_interleaved",
        k=1,
        amplitude=0.05,
    ).next_token(input_ids=tokens, attention_mask=mask).augmented_logits
    for k in range(1, 5):
        graph = Stage2BScheduleGraph(
            wrapper=wrapper,
            schedule="deferred_terminal_write_no_reentry",
            k=k,
            amplitude=0.0,
        )
        observed = graph.next_token(
            input_ids=tokens, attention_mask=mask
        ).augmented_logits
        assert torch.equal(observed, native_k1)
        assert graph.provenance.sidecar_updates == k
        assert graph.provenance.bridge_writes == 0


def test_score_only_schedule_patches_are_restored() -> None:
    wrapper, tokens, mask = _tiny_wrapper()
    observe = wrapper.stage2b_depth_attachment.observe
    reenter = wrapper.stage2b_depth_attachment.reenter
    run_layers = wrapper._run_layer_range
    for schedule in (
        "deferred_terminal_write_no_reentry",
        "per_loop_write_no_reentry",
        "partial_interleave_pairs",
    ):
        graph = Stage2BScheduleGraph(
            wrapper=wrapper, schedule=schedule, k=4, amplitude=0.05
        )
        result = graph.next_token(input_ids=tokens, attention_mask=mask)
        assert torch.isfinite(result.augmented_logits).all()
        assert wrapper.stage2b_depth_attachment.observe == observe
        assert wrapper.stage2b_depth_attachment.reenter == reenter
        assert wrapper._run_layer_range == run_layers


def test_partial_interleave_provenance_counts_ordered_pairs() -> None:
    wrapper, _tokens, _mask = _tiny_wrapper()
    expected = {1: 1, 2: 1, 3: 2, 4: 2}
    for k, reentries in expected.items():
        provenance = Stage2BScheduleGraph(
            wrapper=wrapper,
            schedule="partial_interleave_pairs",
            k=k,
            amplitude=0.05,
        ).provenance
        assert provenance.sidecar_updates == k
        assert provenance.bridge_writes == reentries
        assert provenance.recurrent_reentries == reentries


def test_every_schedule_emits_finite_full_sequence_logits() -> None:
    wrapper, tokens, mask = _tiny_wrapper()
    for schedule in (
        "native_interleaved",
        "deferred_terminal_write_no_reentry",
        "per_loop_write_no_reentry",
        "partial_interleave_pairs",
    ):
        logits = Stage2BScheduleGraph(
            wrapper=wrapper,
            schedule=schedule,
            k=3,
            amplitude=0.05,
        ).sequence_logits(input_ids=tokens, attention_mask=mask)
        assert logits.shape[:2] == tokens.shape
        assert torch.isfinite(logits).all()


def test_seed1_p34_sha_exact_fallback_is_provenance_receipted(
    tmp_path: Path, monkeypatch
) -> None:
    drive = tmp_path / "drive"
    inputs = {
        "migrated": drive
        / depth_runner.MIGRATION_ID
        / "private/migrated_checkpoints/seed_1_full_a2_phase3_migrated.pt",
        "p33": drive / depth_runner.P33_ID / "private/seed_1/checkpoint_step_1000.pt",
        "i1": drive / depth_runner.I1_ID / "private/seed_1/resume.pt",
    }
    for name, path in inputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"locked-{name}".encode())
    fallback = tmp_path / "seed1-p34.pt"
    fallback.write_bytes(b"locked-p34")
    monkeypatch.setattr(depth_runner, "DRIVE_STAGE5", drive)
    monkeypatch.setitem(depth_runner.MIGRATED_SHA, 1, _sha(inputs["migrated"]))
    monkeypatch.setitem(depth_runner.P33_SHA, 1, _sha(inputs["p33"]))
    monkeypatch.setitem(depth_runner.I1_SHA, 1, _sha(inputs["i1"]))
    monkeypatch.setitem(depth_runner.P34_SHA, 1, _sha(fallback))
    monkeypatch.setenv("STAGE2BS_SEED1_P34_FALLBACK", str(fallback))

    def copy(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    monkeypatch.setattr(depth_runner, "rsync", copy)
    chain, provenance = depth_runner.stage_registered_chain(tmp_path / "scratch", 1)
    assert _sha(chain["p34"]) == _sha(fallback)
    assert provenance["p34_source"] == "local_durable_sha_exact_fallback"
    assert provenance["observed_sha256"] == provenance["expected_sha256"]


def test_seed1_p34_fallback_rejects_wrong_hash(tmp_path: Path, monkeypatch) -> None:
    fallback = tmp_path / "wrong.pt"
    fallback.write_bytes(b"wrong")
    monkeypatch.setattr(depth_runner, "DRIVE_STAGE5", tmp_path / "missing-drive")
    monkeypatch.setenv("STAGE2BS_SEED1_P34_FALLBACK", str(fallback))
    try:
        depth_runner.stage_registered_chain(tmp_path / "scratch", 1)
    except RuntimeError as error:
        assert "retained-mirror SHA mismatch" in str(error)
    else:
        raise AssertionError("Wrong-hash seed-1 fallback was accepted")
