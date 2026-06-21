import torch
from torch import nn

from models.recurrent_wrapper import LayerSplit, RecurrentQwenForCausalLM


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
