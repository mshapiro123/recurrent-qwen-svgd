import copy

import torch
from torch import nn

from models.recurrent_wrapper import LayerSplit, RecurrentQwenForCausalLM
from training.phase_g_training import (
    backward_phase_g_trajectories,
)
from training.train_phase1_ponder import optimizer_parameters


class TinyDecoderLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
    ):
        return (torch.tanh(self.proj(hidden_states)),)


class TinyCore(nn.Module):
    def __init__(self, vocab_size=19, hidden_size=8, num_layers=4):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([TinyDecoderLayer(hidden_size) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_size)
        self.gradient_checkpointing = False

    def _update_causal_mask(self, attention_mask, input_tensor, cache_position, past_key_values, output_attentions):
        return attention_mask

    def rotary_emb(self, hidden_states, position_ids):
        return None


class TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type(
            "Config",
            (),
            {
                "hidden_size": 8,
                "use_return_dict": True,
                "output_attentions": False,
                "output_hidden_states": False,
                "use_cache": False,
            },
        )()
        self.model = TinyCore()
        self.lm_head = nn.Linear(8, 19, bias=False)

    def forward(self, input_ids, attention_mask=None, use_cache=False, return_dict=True):
        hidden = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden, attention_mask=attention_mask)[0]
        logits = self.lm_head(self.model.norm(hidden))
        return type("Output", (), {"logits": logits})()


class RecordingBridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, hidden_states, prelude_hidden=None):
        self.calls.append(
            {
                "hidden_shape": tuple(hidden_states.shape),
                "prelude_shape": None if prelude_hidden is None else tuple(prelude_hidden.shape),
                "prelude_is_none": prelude_hidden is None,
            }
        )
        return hidden_states


def test_wrapper_identity_matches_tiny_base_model():
    torch.manual_seed(0)
    model = TinyCausalLM().eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=LayerSplit(1, 3)).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        original = model(input_ids=input_ids, attention_mask=attention_mask).logits
        wrapped = wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_loops=1,
            use_cache=False,
            return_dict=True,
        ).logits

    assert torch.allclose(original, wrapped)


def test_recurrent_reentry_bridge_receives_prelude_hidden_after_first_loop():
    torch.manual_seed(0)
    model = TinyCausalLM().eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=LayerSplit(1, 3)).eval()
    bridge = RecordingBridge()
    wrapper.bridge = bridge
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_loops=3,
            use_cache=False,
            return_dict=True,
        )

    assert len(bridge.calls) == 2
    assert all(call["prelude_shape"] == call["hidden_shape"] for call in bridge.calls)
    assert not any(call["prelude_is_none"] for call in bridge.calls)


def test_recurrent_loop1_does_not_call_reentry_bridge():
    torch.manual_seed(0)
    model = TinyCausalLM().eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=LayerSplit(1, 3)).eval()
    bridge = RecordingBridge()
    wrapper.bridge = bridge
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_loops=1,
            use_cache=False,
            return_dict=True,
        )

    assert bridge.calls == []


def test_internal_control_flag_off_is_exact():
    torch.manual_seed(17)
    wrapper = RecurrentQwenForCausalLM(TinyCausalLM().eval(), layer_split=LayerSplit(1, 3)).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_loops=4,
        use_cache=False,
        return_dict=True,
        return_loop_logits=True,
    )

    with torch.no_grad():
        baseline = wrapper(**kwargs)
        explicit_off = wrapper(
            **kwargs,
            internal_control_enabled=False,
            internal_control_token_ids=(17, 18),
            internal_control_readout_positions=torch.tensor([2]),
        )

    assert torch.equal(baseline.logits, explicit_off.logits)
    assert torch.equal(baseline.loop_logits, explicit_off.loop_logits)
    assert explicit_off.executed_loops is None


def test_internal_control_forced_stop_and_continue_cause_execution():
    torch.manual_seed(19)
    wrapper = RecurrentQwenForCausalLM(TinyCausalLM().eval(), layer_split=LayerSplit(1, 3)).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_loops=5,
        use_cache=False,
        return_dict=True,
        return_loop_logits=True,
        internal_control_enabled=True,
        internal_control_token_ids=(17, 18),
        internal_control_readout_positions=torch.tensor([2]),
    )

    with torch.no_grad():
        stopped = wrapper(
            **kwargs,
            internal_control_overrides={1: "continue", 2: "continue", 3: "stop"},
        )
        continued = wrapper(
            **kwargs,
            internal_control_overrides={
                1: "continue",
                2: "continue",
                3: "continue",
                4: "continue",
                5: "continue",
            },
        )

    assert stopped.executed_loops.tolist() == [3]
    assert stopped.selected_loop_counts.tolist() == [3]
    assert stopped.internal_control_decisions.tolist() == [[0, 0, 1]]
    assert stopped.loop_logits.shape[2] == 3
    assert torch.equal(stopped.logits, stopped.loop_logits[:, 0, -1])

    assert continued.executed_loops.tolist() == [5]
    assert continued.selected_loop_counts.tolist() == [5]
    assert continued.internal_control_decisions.tolist() == [[0, 0, 0, 0, 0]]
    assert continued.loop_logits.shape[2] == 5


def test_internal_control_requires_batch_one_and_valid_token_ids():
    wrapper = RecurrentQwenForCausalLM(TinyCausalLM().eval(), layer_split=LayerSplit(1, 3)).eval()
    with torch.no_grad():
        try:
            wrapper(
                input_ids=torch.tensor([[1, 2], [3, 4]]),
                attention_mask=torch.ones((2, 2), dtype=torch.long),
                max_loops=2,
                internal_control_enabled=True,
                internal_control_token_ids=(17, 18),
                internal_control_readout_positions=torch.tensor([1, 1]),
                return_dict=True,
            )
        except ValueError as exc:
            assert "batch size 1" in str(exc)
        else:
            raise AssertionError("internal control accepted batch size 2")

        try:
            wrapper(
                input_ids=torch.tensor([[1, 2]]),
                attention_mask=torch.ones((1, 2), dtype=torch.long),
                max_loops=2,
                internal_control_enabled=True,
                internal_control_token_ids=(18, 18),
                internal_control_readout_positions=torch.tensor([1]),
                return_dict=True,
            )
        except ValueError as exc:
            assert "distinct" in str(exc)
        else:
            raise AssertionError("internal control accepted duplicate token IDs")


def test_bridge_prelude_ablation_flag_off_is_exact_through_wrapper():
    torch.manual_seed(0)
    wrapper = RecurrentQwenForCausalLM(TinyCausalLM().eval(), layer_split=LayerSplit(1, 3)).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_loops=3,
        use_cache=False,
        return_dict=True,
        return_loop_logits=True,
    )

    with torch.no_grad():
        baseline = wrapper(**kwargs).loop_logits
        explicit_off = wrapper(**kwargs, bridge_prelude_ablation_basis=None).loop_logits

    assert baseline is not None
    assert torch.equal(baseline, explicit_off)


def test_bridge_prelude_ablation_changes_only_reentry_loops_through_wrapper():
    torch.manual_seed(0)
    wrapper = RecurrentQwenForCausalLM(TinyCausalLM().eval(), layer_split=LayerSplit(1, 3)).eval()
    wrapper.bridge.convert_to_split_projection()
    with torch.no_grad():
        wrapper.bridge.prelude_proj.weight.copy_(torch.eye(8))
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    basis = torch.eye(8)[:, :1]
    kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_loops=3,
        use_cache=False,
        return_dict=True,
        return_loop_logits=True,
    )

    with torch.no_grad():
        baseline = wrapper(**kwargs).loop_logits
        ablated = wrapper(**kwargs, bridge_prelude_ablation_basis=basis).loop_logits

    assert baseline is not None and ablated is not None
    assert torch.equal(baseline[:, :, 0], ablated[:, :, 0])
    assert not torch.equal(baseline[:, :, 1:], ablated[:, :, 1:])


def test_reentry_rescale_mode_preserves_loop1_identity_on_tiny_model():
    torch.manual_seed(0)
    model = TinyCausalLM().eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=LayerSplit(1, 3)).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        original = model(input_ids=input_ids, attention_mask=attention_mask).logits
        wrapped = wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_loops=1,
            reentry_rescale_mode="entry_rms",
            use_cache=False,
            return_dict=True,
        ).logits

    assert torch.allclose(original, wrapped)


def test_reentry_rescale_mode_runs_on_recurrent_tiny_model():
    torch.manual_seed(0)
    model = TinyCausalLM().eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=LayerSplit(1, 3)).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output = wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_loops=4,
            reentry_rescale_mode="entry_rms",
            use_cache=False,
            return_dict=True,
        )

    assert output.logits.shape[:2] == input_ids.shape
    assert torch.isfinite(output.metrics["mean_expected_loops"])


def test_reentry_adapter_spectral_mode_runs_on_recurrent_tiny_model():
    torch.manual_seed(0)
    model = TinyCausalLM().eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=LayerSplit(1, 3)).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output = wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_loops=4,
            use_reentry_adapter=True,
            reentry_adapter_mode="spectral",
            use_cache=False,
            return_dict=True,
        )

    assert output.logits.shape[:2] == input_ids.shape
    assert torch.isfinite(output.metrics["mean_expected_loops"])


def test_target_loop_loss_mode_uses_requested_loop_on_tiny_model():
    torch.manual_seed(0)
    model = TinyCausalLM().eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=LayerSplit(1, 3)).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    output = wrapper(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        target_loop_counts=torch.tensor([2]),
        max_loops=3,
        loop_loss_mode="target",
        use_cache=False,
        return_dict=True,
    )

    assert output.loss is not None
    assert torch.isfinite(output.loss)
    assert torch.allclose(output.loss.detach(), output.metrics["target_loop_ce"])
    assert torch.allclose(output.metrics["expected_ce"], output.metrics["target_loop_ce"])


def test_per_loop_label_loss_mode_uses_active_intermediate_labels_on_tiny_model():
    torch.manual_seed(0)
    model = TinyCausalLM().eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=LayerSplit(1, 3)).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    labels[:, -1] = input_ids[:, -1]
    loop_labels = torch.full((1, 3, input_ids.shape[1]), -100, dtype=torch.long)
    loop_labels[:, 0, -1] = 5
    loop_labels[:, 1, -1] = 6

    output = wrapper(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        loop_labels=loop_labels,
        max_loops=3,
        loop_loss_mode="per_loop_labels",
        use_cache=False,
        return_dict=True,
    )

    assert output.loss is not None
    assert torch.isfinite(output.loss)
    assert torch.allclose(output.loss.detach(), output.metrics["per_loop_label_ce"])
    assert output.metrics["per_loop_label_active"].item() == 2


def test_weighted_per_loop_loss_uses_fixed_batch_normalization() -> None:
    torch.manual_seed(0)
    model = TinyCausalLM().eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=LayerSplit(1, 3)).eval()
    input_ids = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    labels[:, -1] = input_ids[:, -1]
    loop_labels = torch.full((2, 3, input_ids.shape[1]), -100, dtype=torch.long)
    loop_labels[:, 0, -1] = 5
    loop_labels[0, 1, -1] = 6

    output = wrapper(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        loop_labels=loop_labels,
        max_loops=3,
        loop_loss_mode="weighted_per_loop_labels",
        loop_label_weights=torch.tensor([0.5, 1.5, 0.0]),
        use_cache=False,
        return_dict=True,
    )

    expected = sum(output.metrics[f"per_loop_label_weighted_ce_{loop}"] for loop in range(1, 4))
    assert output.loss is not None
    assert torch.allclose(output.loss.detach(), expected)
    assert output.metrics["per_loop_label_weighted_active"].item() == 2.5
    assert output.metrics["per_loop_label_weight_1"].item() == 0.5
    assert output.metrics["per_loop_label_weight_2"].item() == 1.5


def test_weighted_per_loop_loss_accepts_row_specific_weight_profiles() -> None:
    torch.manual_seed(0)
    model = TinyCausalLM().eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=LayerSplit(1, 3)).eval()
    input_ids = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    labels[:, -1] = input_ids[:, -1]
    loop_labels = torch.full((2, 3, input_ids.shape[1]), -100, dtype=torch.long)
    loop_labels[:, :, -1] = torch.tensor([[5, 6, 7], [5, 6, 7]])
    weights = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.5, 2.0]])

    output = wrapper(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        loop_labels=loop_labels,
        max_loops=3,
        loop_loss_mode="weighted_per_loop_labels",
        loop_label_weights=weights,
        use_cache=False,
        return_dict=True,
    )

    expected = sum(output.metrics[f"per_loop_label_weighted_ce_{loop}"] for loop in range(1, 4))
    assert output.loss is not None
    assert torch.allclose(output.loss.detach(), expected)
    assert output.metrics["per_loop_label_weighted_active"].item() == 3.5
    assert output.metrics["per_loop_label_weight_1"].item() == 0.5


def test_annealed_chain_to_outcome_loss_mixes_chain_and_target_ce_on_tiny_model():
    torch.manual_seed(0)
    model = TinyCausalLM().eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=LayerSplit(1, 3)).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    labels[:, -1] = input_ids[:, -1]
    loop_labels = torch.full((1, 3, input_ids.shape[1]), -100, dtype=torch.long)
    loop_labels[:, 0, -1] = 5
    loop_labels[:, 1, -1] = 6

    output = wrapper(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        loop_labels=loop_labels,
        target_loop_counts=torch.tensor([2]),
        max_loops=3,
        loop_loss_mode="annealed_chain_to_outcome",
        loop_label_loss_weight=0.25,
        outcome_label_loss_weight=1.0,
        use_cache=False,
        return_dict=True,
    )

    expected = output.metrics["outcome_target_ce"] + 0.25 * output.metrics["per_loop_label_ce"]
    assert output.loss is not None
    assert torch.isfinite(output.loss)
    assert torch.allclose(output.loss.detach(), expected)
    assert output.metrics["loop_label_loss_weight"].item() == 0.25
    assert output.metrics["outcome_label_loss_weight"].item() == 1.0


def test_latent_injection_modes_run_on_tiny_model():
    torch.manual_seed(0)
    model = TinyCausalLM().eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=LayerSplit(1, 3)).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        for mode in ("pre", "post", "both"):
            output = wrapper(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_loops=2,
                num_trajectories=2,
                sample_latents=True,
                latent_injection_mode=mode,
                use_cache=False,
                return_dict=True,
            )
            assert output.logits.shape[:2] == input_ids.shape
            assert torch.isfinite(output.metrics["trajectory_diversity"])


def test_logits_to_keep_returns_last_token_logits_on_tiny_model():
    torch.manual_seed(0)
    model = TinyCausalLM().eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=LayerSplit(1, 3)).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        full = wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_loops=2,
            use_cache=False,
            return_dict=True,
        )
        last_only = wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_loops=2,
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
        )

    assert last_only.logits.shape == (1, 1, 19)
    assert torch.allclose(full.logits[:, -1:, :], last_only.logits)


def test_logits_to_keep_preserves_trajectory_last_token_logits_on_tiny_model():
    torch.manual_seed(0)
    model = TinyCausalLM().eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=LayerSplit(1, 3)).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        full = wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_loops=2,
            num_trajectories=2,
            sample_latents=False,
            use_cache=False,
            return_dict=True,
        )
        last_only = wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_loops=2,
            num_trajectories=2,
            sample_latents=False,
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
        )

    assert last_only.logits.shape == (1, 1, 19)
    assert last_only.trajectory_logits.shape == (1, 2, 1, 19)
    assert torch.allclose(full.logits[:, -1:, :], last_only.logits)
    assert torch.allclose(full.trajectory_logits[:, :, -1:, :], last_only.trajectory_logits)


def test_phase_g_preserves_unpooled_logits_and_seeded_trajectory_differences():
    torch.manual_seed(0)
    model = TinyCausalLM().eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=LayerSplit(1, 3)).eval()
    wrapper.enable_phase_g_guidance(
        latent_dim=4,
        projection_seed=7,
        injection_scale_init=1.0,
    )
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output = wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_loops=2,
            num_trajectories=2,
            phase_g_enabled=True,
            phase_g_trajectory_seeds=[101, 202],
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
        )

    assert output.logits.shape == (2, 1, 19)
    assert output.trajectory_logits.shape == (1, 2, 1, 19)
    assert not torch.equal(
        output.trajectory_logits[:, 0],
        output.trajectory_logits[:, 1],
    )
    assert torch.isfinite(output.metrics["phase_g_prior_variance"])


def test_phase_g_k1_exposes_explicit_trajectory_axis():
    torch.manual_seed(0)
    wrapper = RecurrentQwenForCausalLM(
        TinyCausalLM().eval(),
        layer_split=LayerSplit(1, 3),
    ).eval()
    wrapper.enable_phase_g_guidance(latent_dim=4, projection_seed=7)
    input_ids = torch.tensor([[1, 2, 3, 4]])

    with torch.no_grad():
        output = wrapper(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_loops=2,
            num_trajectories=1,
            phase_g_enabled=True,
            phase_g_trajectory_seeds=[101],
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
        )

    assert output.trajectory_logits is not None
    assert output.trajectory_logits.shape == (1, 1, 1, 19)


def test_phase_g_wrapper_factor_one_preserves_default_output():
    torch.manual_seed(0)
    wrapper = RecurrentQwenForCausalLM(
        TinyCausalLM().eval(),
        layer_split=LayerSplit(1, 3),
    ).eval()
    wrapper.enable_phase_g_guidance(
        latent_dim=4,
        projection_seed=7,
        injection_scale_init=0.5,
    )
    input_ids = torch.tensor([[1, 2, 3, 4]])
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "max_loops": 2,
        "num_trajectories": 1,
        "phase_g_enabled": True,
        "phase_g_trajectory_seeds": [101],
        "use_cache": False,
        "logits_to_keep": 1,
        "return_dict": True,
    }

    with torch.no_grad():
        default = wrapper(**kwargs)
        factor_one = wrapper(**kwargs, phase_g_injection_multiplier=1.0)
        factor_three = wrapper(**kwargs, phase_g_injection_multiplier=3.0)

    assert torch.equal(default.logits, factor_one.logits)
    assert torch.equal(default.trajectory_logits, factor_one.trajectory_logits)
    assert torch.allclose(
        factor_three.metrics["phase_g_injection_scale"],
        3.0 * default.metrics["phase_g_injection_scale"],
    )
    assert not torch.equal(default.logits, factor_three.logits)


def test_oracle_conditioner_installation_is_exact_through_tiny_wrapper():
    torch.manual_seed(0)
    wrapper = RecurrentQwenForCausalLM(
        TinyCausalLM().eval(),
        layer_split=LayerSplit(1, 3),
    ).eval()
    wrapper.enable_oracle_reentry_conditioner(bottleneck_dim=4)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    commands = wrapper.base_model.model.embed_tokens(
        torch.tensor([[5, 6]])
    ).detach()
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "max_loops": 2,
        "use_cache": False,
        "return_loop_logits": True,
        "return_dict": True,
    }

    with torch.no_grad():
        baseline = wrapper(**kwargs)
        for mode in ("additive", "film"):
            conditioned = wrapper(
                **kwargs,
                oracle_reentry_enabled=True,
                oracle_reentry_mode=mode,
                oracle_reentry_targets=commands,
            )
            assert torch.equal(baseline.loop_logits, conditioned.loop_logits)


def test_oracle_force_identity_bypasses_trained_route_through_tiny_wrapper():
    torch.manual_seed(0)
    wrapper = RecurrentQwenForCausalLM(
        TinyCausalLM().eval(),
        layer_split=LayerSplit(1, 3),
    ).eval()
    wrapper.enable_oracle_reentry_conditioner(bottleneck_dim=4)
    with torch.no_grad():
        wrapper.oracle_reentry_conditioner.branch_a.net[-1].weight.normal_()
        wrapper.oracle_reentry_conditioner.branch_b.net[-1].weight.normal_()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    commands = wrapper.base_model.model.embed_tokens(
        torch.tensor([[5, 6]])
    ).detach()
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "max_loops": 2,
        "use_cache": False,
        "return_loop_logits": True,
        "return_dict": True,
    }

    with torch.no_grad():
        baseline = wrapper(**kwargs)
        active = wrapper(
            **kwargs,
            oracle_reentry_enabled=True,
            oracle_reentry_mode="film",
            oracle_reentry_targets=commands,
        )
        bypassed = wrapper(
            **kwargs,
            oracle_reentry_enabled=True,
            oracle_reentry_mode="film",
            oracle_reentry_targets=commands,
            oracle_reentry_force_identity=True,
        )

    assert not torch.equal(baseline.loop_logits, active.loop_logits)
    assert torch.equal(baseline.loop_logits, bypassed.loop_logits)


def test_oracle_trainable_contract_freezes_tiny_keeper():
    wrapper = RecurrentQwenForCausalLM(
        TinyCausalLM(),
        layer_split=LayerSplit(1, 3),
    )
    wrapper.enable_oracle_reentry_conditioner(bottleneck_dim=4)

    names = wrapper.configure_oracle_reentry_trainable()

    assert names
    assert all(name.startswith("oracle_reentry_conditioner.") for name in names)
    assert all(
        parameter.requires_grad == name.startswith("oracle_reentry_conditioner.")
        for name, parameter in wrapper.named_parameters()
    )


def test_oracle_intrablock_installation_is_exact_through_tiny_wrapper():
    torch.manual_seed(0)
    wrapper = RecurrentQwenForCausalLM(
        TinyCausalLM().eval(),
        layer_split=LayerSplit(1, 3),
    ).eval()
    wrapper.enable_oracle_intrablock_conditioner(bottleneck_dim=4)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    commands = wrapper.base_model.model.embed_tokens(
        torch.tensor([[5, 6]])
    ).detach()
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "max_loops": 2,
        "use_cache": False,
        "return_loop_logits": True,
        "return_dict": True,
    }

    with torch.no_grad():
        baseline = wrapper(**kwargs)
        conditioned = wrapper(
            **kwargs,
            oracle_intrablock_enabled=True,
            oracle_intrablock_targets=commands,
        )

    assert torch.equal(baseline.loop_logits, conditioned.loop_logits)


def test_oracle_intrablock_reuses_one_conditioner_before_every_recurrent_layer():
    torch.manual_seed(0)
    wrapper = RecurrentQwenForCausalLM(
        TinyCausalLM().eval(),
        layer_split=LayerSplit(1, 3),
    ).eval()
    wrapper.enable_oracle_intrablock_conditioner(bottleneck_dim=4)
    with torch.no_grad():
        wrapper.oracle_intrablock_conditioner.branch_a.net[-1].weight.normal_()
        wrapper.oracle_intrablock_conditioner.branch_b.net[-1].weight.normal_()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    commands = wrapper.base_model.model.embed_tokens(torch.tensor([[5, 6]])).detach()

    with torch.no_grad():
        output = wrapper(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_loops=2,
            use_cache=False,
            return_loop_logits=True,
            return_dict=True,
            oracle_intrablock_enabled=True,
            oracle_intrablock_targets=commands,
        )

    assert output.metrics["oracle_intrablock_applications"].item() == 4
    assert output.metrics["oracle_intrablock_residual_rms_ratio"].item() > 0


def test_oracle_intrablock_trainable_contract_freezes_tiny_keeper():
    wrapper = RecurrentQwenForCausalLM(
        TinyCausalLM(),
        layer_split=LayerSplit(1, 3),
    )
    wrapper.enable_oracle_intrablock_conditioner(bottleneck_dim=4)

    names = wrapper.configure_oracle_intrablock_trainable()

    assert names
    assert all(name.startswith("oracle_intrablock_conditioner.") for name in names)
    assert all(
        parameter.requires_grad == name.startswith("oracle_intrablock_conditioner.")
        for name, parameter in wrapper.named_parameters()
    )


def test_phase_g_tiny_wrapper_microbatch_matches_vectorized_gradients():
    torch.manual_seed(0)
    vectorized = RecurrentQwenForCausalLM(
        TinyCausalLM(),
        layer_split=LayerSplit(1, 3),
    )
    vectorized.enable_phase_g_guidance(
        latent_dim=4,
        projection_seed=7,
        injection_scale_init=0.1,
    )
    microbatched = copy.deepcopy(vectorized)
    vectorized.configure_phase_g_trainable()
    microbatched.configure_phase_g_trainable()
    vectorized.train()
    microbatched.train()

    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    labels[:, -1] = 7
    loop_labels = torch.full((1, 2, 4), -100, dtype=torch.long)
    loop_labels[:, 0, -1] = 5
    loop_labels[:, 1, -1] = 6
    with torch.no_grad():
        posterior_targets = vectorized.base_model.model.embed_tokens(
            torch.tensor([[5, 6]])
        ).detach()
    forward_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "loop_labels": loop_labels,
        "target_loop_counts": torch.tensor([2]),
        "max_loops": 2,
        "loop_loss_mode": "per_loop_labels",
        "particle_update_mode": "none",
        "use_cache": False,
        "return_dict": True,
        "phase_g_enabled": True,
        "phase_g_use_posterior": True,
        "phase_g_posterior_targets": posterior_targets,
        "phase_g_kl_balance": 0.8,
        "phase_g_kl_coefficient": 1e-3,
    }

    vectorized_result = backward_phase_g_trajectories(
        vectorized,
        forward_kwargs=forward_kwargs,
        trajectory_seeds=[101, 202],
        microbatch_size=2,
    )
    microbatched_result = backward_phase_g_trajectories(
        microbatched,
        forward_kwargs=forward_kwargs,
        trajectory_seeds=[101, 202],
        microbatch_size=1,
    )

    assert abs(vectorized_result.loss - microbatched_result.loss) < 1e-6
    vectorized_parameters = dict(vectorized.named_parameters())
    microbatched_parameters = dict(microbatched.named_parameters())
    for name, parameter in vectorized_parameters.items():
        if not parameter.requires_grad:
            continue
        assert parameter.grad is not None
        assert microbatched_parameters[name].grad is not None
        assert torch.allclose(
            parameter.grad,
            microbatched_parameters[name].grad,
            atol=2e-6,
            rtol=2e-5,
        ), name


def test_return_loop_recurrent_states_exposes_one_state_per_loop_on_tiny_model():
    torch.manual_seed(0)
    model = TinyCausalLM().eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=LayerSplit(1, 3)).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output = wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_loops=3,
            use_cache=False,
            return_dict=True,
            return_loop_recurrent_states=True,
        )

    assert output.loop_recurrent_states.shape == (1, 1, 3, 4, 8)
    assert torch.allclose(output.final_recurrent_hidden, output.loop_recurrent_states[:, :, -1])


def test_recurrent_state_override_applies_before_requested_next_loop_on_tiny_model():
    torch.manual_seed(0)
    model = TinyCausalLM().eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=LayerSplit(1, 3)).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        baseline = wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_loops=3,
            use_cache=False,
            return_dict=True,
            return_loop_logits=True,
            return_loop_recurrent_states=True,
        )
        override_state = torch.zeros_like(baseline.loop_recurrent_states[:, :, 0]).reshape(1, 4, 8)
        spliced = wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_loops=3,
            recurrent_state_overrides={1: override_state},
            use_cache=False,
            return_dict=True,
            return_loop_logits=True,
            return_loop_recurrent_states=True,
        )

    assert torch.allclose(baseline.loop_logits[:, :, 0], spliced.loop_logits[:, :, 0])
    assert not torch.allclose(baseline.loop_logits[:, :, 1], spliced.loop_logits[:, :, 1])
    assert torch.allclose(spliced.loop_recurrent_states[:, :, 0], baseline.loop_recurrent_states[:, :, 0])
    assert not torch.allclose(spliced.loop_recurrent_states[:, :, 1], baseline.loop_recurrent_states[:, :, 1])


def test_recurrent_state_override_rejects_wrong_shape_on_tiny_model():
    torch.manual_seed(0)
    model = TinyCausalLM().eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=LayerSplit(1, 3)).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    bad_state = torch.zeros(1, 3, 8)

    try:
        wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_loops=2,
            recurrent_state_overrides={1: bad_state},
            use_cache=False,
            return_dict=True,
        )
    except ValueError as exc:
        assert "recurrent_state_overrides" in str(exc)
    else:
        raise AssertionError("Expected bad recurrent_state_overrides shape to raise")


def test_halting_target_nll_weight_adds_supervised_loop_loss_on_tiny_model():
    torch.manual_seed(0)
    model = TinyCausalLM().eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=LayerSplit(1, 3)).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    target_loop_counts = torch.tensor([3])

    with torch.no_grad():
        baseline = wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            target_loop_counts=target_loop_counts,
            max_loops=4,
            beta=0.0,
            halt_target_nll_weight=0.0,
            use_cache=False,
            return_dict=True,
        )
        supervised = wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            target_loop_counts=target_loop_counts,
            max_loops=4,
            beta=0.0,
            halt_target_nll_weight=0.5,
            use_cache=False,
            return_dict=True,
        )

    assert "target_mean_loops" in supervised.metrics
    assert "target_loop_abs_error" in supervised.metrics
    assert "halting_target_nll" in supervised.metrics
    assert torch.isfinite(supervised.metrics["halting_target_nll"])
    assert supervised.loss > baseline.loss


def test_phase1_optimizer_parameters_can_select_halt_only_on_tiny_model():
    model = TinyCausalLM().eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=LayerSplit(1, 3)).eval()
    wrapper.freeze_base_model()
    all_params = optimizer_parameters(wrapper, {"optimizer_modules": "all"})
    halt_params = optimizer_parameters(wrapper, {"optimizer_modules": "halt"})

    expected_halt = [param for param in wrapper.halt_predictor.parameters() if param.requires_grad]
    assert [id(param) for param in halt_params] == [id(param) for param in expected_halt]
    assert len(halt_params) < len(all_params)


def test_phase1_optimizer_parameters_can_select_reentry_only_on_tiny_model():
    model = TinyCausalLM().eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=LayerSplit(1, 3)).eval()
    wrapper.freeze_base_model()
    reentry_params = optimizer_parameters(wrapper, {"optimizer_modules": "reentry"})

    expected = [param for param in wrapper.reentry_adapter.parameters() if param.requires_grad]
    assert [id(param) for param in reentry_params] == [id(param) for param in expected]


def test_phase1_optimizer_parameters_match_stage3_reentry_repair_modules():
    model = TinyCausalLM().eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=LayerSplit(1, 3)).eval()
    wrapper.freeze_base_model()
    selected = optimizer_parameters(wrapper, {"optimizer_modules": "bridge,reentry,halt"})

    expected = [
        *[param for param in wrapper.bridge.parameters() if param.requires_grad],
        *[param for param in wrapper.reentry_adapter.parameters() if param.requires_grad],
        *[param for param in wrapper.halt_predictor.parameters() if param.requires_grad],
    ]
    assert [id(param) for param in selected] == [id(param) for param in expected]


def test_phase1_optimizer_parameters_can_select_bridge_projection_without_gate():
    model = TinyCausalLM().eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=LayerSplit(1, 3)).eval()
    wrapper.freeze_base_model()
    selected = optimizer_parameters(wrapper, {"optimizer_modules": "bridge_proj,reentry,halt"})
    selected_ids = {id(param) for param in selected}

    assert id(wrapper.bridge.bridge_gate) not in selected_ids
    for param in wrapper.bridge.proj.parameters():
        assert id(param) in selected_ids
    for param in wrapper.reentry_adapter.parameters():
        assert id(param) in selected_ids
    for param in wrapper.halt_predictor.parameters():
        assert id(param) in selected_ids


def test_phase1_optimizer_parameters_can_select_bridge_gate_only():
    model = TinyCausalLM().eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=LayerSplit(1, 3)).eval()
    wrapper.freeze_base_model()
    selected = optimizer_parameters(wrapper, {"optimizer_modules": "bridge_gate"})

    assert selected == [wrapper.bridge.bridge_gate]


def test_phase1_optimizer_parameters_deduplicate_overlapping_modules():
    model = TinyCausalLM().eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=LayerSplit(1, 3)).eval()
    wrapper.freeze_base_model()
    selected = optimizer_parameters(wrapper, {"optimizer_modules": "bridge,bridge_proj"})

    assert len({id(param) for param in selected}) == len(selected)


def test_invalid_latent_injection_mode_raises():
    model = TinyCausalLM().eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=LayerSplit(1, 3)).eval()
    input_ids = torch.tensor([[1, 2]])

    try:
        wrapper(
            input_ids=input_ids,
            max_loops=1,
            sample_latents=True,
            latent_injection_mode="sideways",
            use_cache=False,
            return_dict=True,
        )
    except ValueError as exc:
        assert "latent_injection_mode" in str(exc)
    else:
        raise AssertionError("Expected invalid latent_injection_mode to raise")


def test_invalid_reentry_rescale_mode_raises():
    model = TinyCausalLM().eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=LayerSplit(1, 3)).eval()
    input_ids = torch.tensor([[1, 2]])

    try:
        wrapper(
            input_ids=input_ids,
            max_loops=2,
            reentry_rescale_mode="sideways",
            use_cache=False,
            return_dict=True,
        )
    except ValueError as exc:
        assert "reentry_rescale_mode" in str(exc)
    else:
        raise AssertionError("Expected invalid reentry_rescale_mode to raise")


def test_invalid_reentry_adapter_mode_raises():
    model = TinyCausalLM().eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=LayerSplit(1, 3)).eval()
    input_ids = torch.tensor([[1, 2]])

    try:
        wrapper(
            input_ids=input_ids,
            max_loops=2,
            use_reentry_adapter=True,
            reentry_adapter_mode="sideways",
            use_cache=False,
            return_dict=True,
        )
    except ValueError as exc:
        assert "reentry_adapter_mode" in str(exc)
    else:
        raise AssertionError("Expected invalid reentry_adapter_mode to raise")


def test_svgd_particle_update_runs_on_tiny_model():
    torch.manual_seed(0)
    model = TinyCausalLM().eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=LayerSplit(1, 3)).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output = wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_loops=2,
            num_trajectories=2,
            sample_latents=False,
            particle_update_mode="svgd",
            particle_init_noise=0.01,
            svgd_repulsion_scale=0.5,
            use_cache=False,
            return_dict=True,
        )

    assert output.logits.shape[:2] == input_ids.shape
    assert torch.isfinite(output.metrics["trajectory_diversity"])
    assert torch.isfinite(output.metrics["svgd_pairwise_distance"])


def test_svgd_projected_kernel_runs_on_tiny_model():
    torch.manual_seed(0)
    model = TinyCausalLM().eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=LayerSplit(1, 3)).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output = wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_loops=2,
            num_trajectories=2,
            sample_latents=False,
            particle_update_mode="svgd",
            particle_init_noise=0.01,
            svgd_repulsion_scale=0.5,
            svgd_kernel_projection_dim=4,
            use_cache=False,
            return_dict=True,
        )

    assert output.logits.shape[:2] == input_ids.shape
    assert torch.isfinite(output.metrics["svgd_pairwise_distance"])
    assert torch.isfinite(output.metrics["svgd_repulsion_rms"])


def test_svgd_spherical_projected_kernel_runs_on_tiny_model():
    torch.manual_seed(0)
    model = TinyCausalLM().eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=LayerSplit(1, 3)).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output = wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_loops=2,
            num_trajectories=2,
            sample_latents=False,
            particle_update_mode="svgd",
            particle_init_noise=0.01,
            svgd_repulsion_scale=0.5,
            svgd_kernel_projection_dim=4,
            svgd_kernel_geometry="spherical",
            use_cache=False,
            return_dict=True,
        )

    assert output.logits.shape[:2] == input_ids.shape
    assert torch.isfinite(output.metrics["svgd_pairwise_distance"])
    assert torch.isfinite(output.metrics["svgd_repulsion_rms"])


def test_svgd_k1_matches_standard_recurrent_path_on_tiny_model():
    torch.manual_seed(0)
    model = TinyCausalLM().eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=LayerSplit(1, 3)).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        standard = wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_loops=2,
            num_trajectories=1,
            sample_latents=False,
            particle_update_mode="none",
            use_cache=False,
            return_dict=True,
        ).logits
        svgd = wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_loops=2,
            num_trajectories=1,
            sample_latents=False,
            particle_update_mode="svgd",
            svgd_eps=1.0,
            use_cache=False,
            return_dict=True,
        ).logits

    assert torch.allclose(standard, svgd)


def test_svgd_accepts_grouped_trajectory_inputs_on_tiny_model():
    torch.manual_seed(0)
    model = TinyCausalLM().eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=LayerSplit(1, 3)).eval()
    input_ids = torch.tensor([[[1, 2, 3], [1, 2, 4]]])
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output = wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_loops=2,
            num_trajectories=2,
            sample_latents=False,
            particle_update_mode="svgd",
            svgd_repulsion_scale=0.5,
            use_cache=False,
            return_dict=True,
        )

    assert output.logits.shape == (1, 3, 19)
    assert output.trajectory_logits.shape == (1, 2, 3, 19)
    assert output.final_recurrent_hidden.shape[:3] == (1, 2, 3)
    assert torch.isfinite(output.metrics["svgd_pairwise_distance"])


def test_svgd_grouped_trajectory_noise_breaks_identical_inputs_on_tiny_model():
    torch.manual_seed(0)
    model = TinyCausalLM().eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=LayerSplit(1, 3)).eval()
    input_ids = torch.tensor([[[1, 2, 3], [1, 2, 3]]])
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        no_noise = wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_loops=2,
            num_trajectories=2,
            sample_latents=False,
            particle_update_mode="svgd",
            particle_init_noise=0.0,
            svgd_repulsion_scale=0.5,
            use_cache=False,
            return_dict=True,
        )
        with_noise = wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_loops=2,
            num_trajectories=2,
            sample_latents=False,
            particle_update_mode="svgd",
            particle_init_noise=0.05,
            svgd_repulsion_scale=0.5,
            use_cache=False,
            return_dict=True,
        )

    no_noise_delta = (no_noise.final_recurrent_hidden[0, 0] - no_noise.final_recurrent_hidden[0, 1]).abs().max()
    with_noise_delta = (with_noise.final_recurrent_hidden[0, 0] - with_noise.final_recurrent_hidden[0, 1]).abs().max()
    assert no_noise_delta == 0
    assert with_noise_delta > 0


def test_particle_init_noise_works_without_svgd_on_tiny_model():
    torch.manual_seed(0)
    model = TinyCausalLM().eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=LayerSplit(1, 3)).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        no_noise = wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_loops=2,
            num_trajectories=2,
            sample_latents=False,
            particle_update_mode="none",
            particle_init_noise=0.0,
            use_cache=False,
            return_dict=True,
        )
        with_noise = wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_loops=2,
            num_trajectories=2,
            sample_latents=False,
            particle_update_mode="none",
            particle_init_noise=0.05,
            use_cache=False,
            return_dict=True,
        )

    no_noise_delta = (no_noise.final_recurrent_hidden[0, 0] - no_noise.final_recurrent_hidden[0, 1]).abs().max()
    with_noise_delta = (with_noise.final_recurrent_hidden[0, 0] - with_noise.final_recurrent_hidden[0, 1]).abs().max()
    assert no_noise_delta == 0
    assert with_noise_delta > 0
