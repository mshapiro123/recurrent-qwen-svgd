from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path

import pytest
import torch

from models.ablation_lm import AblationLM, AblationLMConfig, estimate_dense_unique_parameters
from training.weft1_gtok_a1_contract import (
    A1ExecutionBlocked,
    A1_DEDUP_BINDING,
    A1_EXECUTION_DEFECTS,
    A1_OPTIMIZER_RECIPE,
    A1_OPTIMIZER_RECIPE_SHA256,
    A1_TOKENIZER_BINDING,
    SOURCE_FAMILIES,
    ScreenCorpusReceiptV2,
    SourceRouteBindingV2,
    StratumFloorReceiptV2,
    a1_contract_snapshot,
    a1_contract_snapshot_sha256,
    load_source_route_manifest,
    require_a1_execution_ready,
)
from training.weft1_gtok_contract import (
    GTOK_AMENDMENT_A1_SHA256,
    GTOK_EXECUTION_AUTHORITY_CHAIN,
    GTOK_EXECUTION_AUTHORITY_CHAIN_V2,
    GTOK_HELDOUT_BYTE_TARGET,
    GTOK_PIPELINE_RNG_NAMES,
    GTOK_PRETOKENIZER_REGEX,
    GTOK_PROXY_TOPOLOGY,
    GTOK_SCREEN_HELDOUT_STRATUM_TARGETS,
    GTOK_SCREEN_TRAIN_STRATUM_TARGETS,
    GTOK_STRATA,
    GTOK_TRAINING_BYTE_BUDGET,
    execution_authority_bound_sha256,
    execution_authority_v2_bound_sha256,
    sha256_bytes,
)


def _hash(label: str) -> str:
    return sha256_bytes(label.encode("utf-8"))


def _floors(stream: str, *, shortfall: int) -> tuple[StratumFloorReceiptV2, ...]:
    targets = dict(
        GTOK_SCREEN_TRAIN_STRATUM_TARGETS
        if stream == "T"
        else GTOK_SCREEN_HELDOUT_STRATUM_TARGETS
    )
    return tuple(
        StratumFloorReceiptV2(
            stream=stream,
            stratum=name,
            target_bytes=targets[name],
            realized_bytes=targets[name] - shortfall,
            ordered_document_ids_sha256=_hash(f"{stream}-{name}-order"),
            boundary_document_id_sha256=_hash(f"{stream}-{name}-boundary"),
            next_document_byte_count=None if shortfall == 0 else shortfall + 1,
        )
        for name in GTOK_STRATA
    )


def test_a1_appends_a_v2_chain_without_rekeying_bank_v1() -> None:
    assert GTOK_EXECUTION_AUTHORITY_CHAIN_V2 == (
        *GTOK_EXECUTION_AUTHORITY_CHAIN,
        GTOK_AMENDMENT_A1_SHA256,
    )
    payload = {"example": 1}
    v1 = execution_authority_bound_sha256("weft1_example_v1", payload)
    v2 = execution_authority_v2_bound_sha256("weft1_example_v2", payload)
    assert v1 != v2
    with pytest.raises(ValueError, match="explicit v2"):
        execution_authority_v2_bound_sha256("weft1_example_v1", payload)


def test_proxy_executes_ten_blocks_when_weft_is_structurally_off() -> None:
    assert GTOK_PROXY_TOPOLOGY.executing_block_count == 10
    assert (
        GTOK_PROXY_TOPOLOGY.n_prelude_layers,
        GTOK_PROXY_TOPOLOGY.n_core_blocks,
        GTOK_PROXY_TOPOLOGY.n_coda_layers,
    ) == (4, 2, 4)
    assert GTOK_PROXY_TOPOLOGY.use_recurrence is False


def test_all_four_proxy_parameter_counts_match_a1() -> None:
    expected = {
        16_384: 37_891_840,
        24_576: 42_086_144,
        32_768: 46_280_448,
        49_152: 54_669_056,
    }
    observed = {
        vocab_size: estimate_dense_unique_parameters(
            AblationLMConfig(
                vocab_size=vocab_size,
                n_prelude_layers=4,
                n_core_blocks=2,
                n_coda_layers=4,
            )
        )
        for vocab_size in expected
    }
    assert observed == expected


def test_structural_off_forward_visits_four_two_four_blocks_once() -> None:
    config = AblationLMConfig(
        vocab_size=64,
        d_model=16,
        n_heads=4,
        n_kv_heads=2,
        d_ff=32,
        n_prelude_layers=4,
        n_core_blocks=2,
        n_coda_layers=4,
        max_sequence_length=16,
        recurrent_steps=1,
        max_recurrent_steps=8,
    )
    model = AblationLM(config)
    counts = {"prelude": 0, "core": 0, "coda": 0}
    handles = []
    for name, blocks in (
        ("prelude", model.prelude_blocks),
        ("core", model.core_blocks),
        ("coda", model.coda_blocks),
    ):
        for block in blocks:
            handles.append(
                block.register_forward_hook(
                    lambda unused_module, unused_inputs, unused_output, key=name: counts.__setitem__(
                        key, counts[key] + 1
                    )
                )
            )
    try:
        with torch.no_grad():
            output = model(
                torch.tensor([[1, 2, 3]], dtype=torch.long),
                return_diagnostics=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    assert counts == {"prelude": 4, "core": 2, "coda": 4}
    assert output.diagnostics["executed_core_visits"] == 1
    assert output.diagnostics["executed_core_block_passes"] == 2
    assert model.scratch is None
    assert model.engram is None
    assert model.long_term_memory is None


def test_route_ledger_binds_every_family_and_preserves_discrepancies() -> None:
    manifest = load_source_route_manifest()
    assert tuple(route.source_family for route in manifest.routes) == SOURCE_FAMILIES
    assert all(route.declared_license == "odc-by" for route in manifest.routes)
    assert all(route.available_bytes > route.required_bytes for route in manifest.routes)
    assert len(manifest.manifest_sha256) == 64

    by_family = {route.source_family: route for route in manifest.routes}
    assert by_family["dolma_web"].asset_selector == (
        "data/common_crawl-*-0019/*.jsonl.zst"
    )
    assert by_family["olmocr"].repository == "allenai/dolma3_mix-6T"
    assert "zero such assets" in by_family["olmocr"].lineage_evidence
    assert by_family["wikipedia_wikibooks"].external_locator_manifest_sha256
    assert any("preflight receipt" in item for item in manifest.known_route_findings)


def test_route_ledger_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate-route-key.json"
    path.write_text(
        '{"schema":"weft1_gtok_source_route_manifest_v2",'
        '"routes":[],"routes":[]}\n',
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ValueError, match="repeats key: routes"):
        load_source_route_manifest(path)


def test_route_binding_rejects_branch_license_capacity_and_card_drift() -> None:
    route = load_source_route_manifest().routes[0]
    with pytest.raises(ValueError, match="commit SHA"):
        replace(route, revision="main")
    with pytest.raises(ValueError, match="ODC-By"):
        replace(route, declared_license="other")
    with pytest.raises(ValueError, match="no positive byte margin"):
        replace(route, available_bytes=route.required_bytes)
    with pytest.raises(ValueError, match="exact repository and revision"):
        replace(route, card_url=route.card_url.replace(route.revision, "0" * 40))


def test_a1_train_and_holdout_targets_are_independent_document_floors() -> None:
    receipt = ScreenCorpusReceiptV2(
        training=_floors("T", shortfall=1),
        heldout=_floors("H", shortfall=1),
        training_stream_sha256=_hash("training-stream"),
        heldout_stream_sha256=_hash("heldout-stream"),
        document_overlap_count=0,
        cluster_overlap_count=0,
    )
    assert receipt.training_target_bytes == GTOK_TRAINING_BYTE_BUDGET
    assert receipt.heldout_target_bytes == GTOK_HELDOUT_BYTE_TARGET
    assert receipt.training_realized_bytes == GTOK_TRAINING_BYTE_BUDGET - 4
    assert receipt.heldout_realized_bytes == GTOK_HELDOUT_BYTE_TARGET - 4
    assert receipt.heldout_target_bytes * 50 == receipt.training_target_bytes
    assert receipt.heldout_realized_bytes * 50 != receipt.training_realized_bytes
    assert len(receipt.receipt_sha256) == 64


def test_floor_never_overshoots_exceeds_tolerance_or_stops_early() -> None:
    target = dict(GTOK_SCREEN_TRAIN_STRATUM_TARGETS)["general"]
    base = StratumFloorReceiptV2(
        stream="T",
        stratum="general",
        target_bytes=target,
        realized_bytes=target - 10,
        ordered_document_ids_sha256=_hash("order"),
        boundary_document_id_sha256=_hash("boundary"),
        next_document_byte_count=11,
    )
    assert base.shortfall_bytes == 10
    with pytest.raises(ValueError, match="positive document floor"):
        replace(base, realized_bytes=target + 1)
    with pytest.raises(ValueError, match="exceeds 0.5 percent"):
        replace(base, realized_bytes=target - target // 100)
    with pytest.raises(ValueError, match="not maximal"):
        replace(base, next_document_byte_count=10)


def test_a1_headline_tokenizer_dedup_and_optimizer_values_are_bound() -> None:
    assert A1_TOKENIZER_BINDING.pretokenizer_regex == GTOK_PRETOKENIZER_REGEX
    assert "\\p{N}" in A1_TOKENIZER_BINDING.pretokenizer_regex
    assert A1_TOKENIZER_BINDING.min_frequency == 2
    assert A1_TOKENIZER_BINDING.byte_atom_count == 256
    assert A1_DEDUP_BINDING.minhash_components == 128
    assert (A1_DEDUP_BINDING.lsh_bands, A1_DEDUP_BINDING.lsh_rows_per_band) == (16, 8)
    assert dict(A1_OPTIMIZER_RECIPE.hyperparameters) == {
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "gradient_clip_norm": 1.0,
        "learning_rate": 3e-4,
        "weight_decay": 0.1,
    }
    assert len(A1_OPTIMIZER_RECIPE_SHA256) == 64
    assert A1_OPTIMIZER_RECIPE_SHA256 != A1_OPTIMIZER_RECIPE.recipe_sha256


def test_a1_rng_names_are_disjoint_but_root_seeds_remain_blocked() -> None:
    assert GTOK_PIPELINE_RNG_NAMES == tuple(sorted(GTOK_PIPELINE_RNG_NAMES))
    assert len(GTOK_PIPELINE_RNG_NAMES) == len(set(GTOK_PIPELINE_RNG_NAMES))
    assert any("campaign/corpus root seed" in item for item in A1_EXECUTION_DEFECTS)


def test_local_bpe_trainer_has_no_seed_hook_named_by_a1() -> None:
    tokenizers = pytest.importorskip("tokenizers")
    signature = inspect.signature(tokenizers.trainers.BpeTrainer)
    assert "seed" not in signature.parameters


def test_a1_contract_is_hashable_but_every_execution_action_fails_closed() -> None:
    snapshot = a1_contract_snapshot()
    assert snapshot["authority_chain"] == GTOK_EXECUTION_AUTHORITY_CHAIN_V2
    assert len(a1_contract_snapshot_sha256()) == 64
    with pytest.raises(A1ExecutionBlocked) as raised:
        require_a1_execution_ready("materialize corpus")
    message = str(raised.value)
    assert "run termination" in message
    assert "language-ID" in message
    assert "BpeTrainer" in message
    assert "tripwire" in message


def test_source_route_dataclass_requires_typed_mapping() -> None:
    with pytest.raises(TypeError, match="mapping"):
        SourceRouteBindingV2.from_mapping([])  # type: ignore[arg-type]
