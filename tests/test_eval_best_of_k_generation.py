import torch

from eval.eval_best_of_k_jsonl import generate_candidates, pathway_split_diagnostics


class Encoded(dict):
    def to(self, device):
        return self


class DummyTokenizer:
    eos_token_id = 99

    def __call__(self, prompt, return_tensors="pt"):
        return Encoded(
            {
                "input_ids": torch.tensor([[1]]),
                "attention_mask": torch.tensor([[1]]),
            }
        )

    def batch_decode(self, completions, skip_special_tokens=True):
        return ["ok" for _ in range(len(completions))]


class DummyWrapper:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        logits = torch.zeros(1, 1, 100)
        logits[:, :, 2] = 1.0
        return type("Output", (), {"logits": logits, "metrics": {}})()


def test_generate_candidates_requests_last_token_logits_only():
    wrapper = DummyWrapper()

    result = generate_candidates(
        wrapper,
        DummyTokenizer(),
        "prompt",
        max_new_tokens=1,
        max_loops=4,
        num_trajectories=1,
        sample_latents=False,
        latent_injection_mode="post",
        particle_update_mode="none",
        particle_init_noise=0.0,
        svgd_eps=1.0,
        svgd_repulsion_scale=0.5,
        svgd_bandwidth="median",
        svgd_bandwidth_floor=1e-6,
        svgd_repulsion_max_norm=None,
        svgd_kernel_projection_dim=None,
        svgd_kernel_projection_path=None,
        svgd_kernel_geometry="euclidean",
        svgd_projection_seed=0,
        reentry_rescale_mode="entry_rms",
        particle_noise_every_step=False,
        particle_noise_steps=0,
        stop_on_final_answer=False,
        temperature=0.0,
        device="cpu",
    )

    assert result.generation_steps == 1
    assert wrapper.calls[0]["logits_to_keep"] == 1
    assert wrapper.calls[0]["reentry_rescale_mode"] == "entry_rms"


def test_pathway_split_diagnostics_separates_correct_and_wrong_candidates():
    states = torch.tensor(
        [
            [0.0, 0.0],
            [0.05, 0.0],
            [5.0, 0.0],
            [10.0, 0.0],
        ]
    )
    hits = [True, True, False, False]

    out = pathway_split_diagnostics(states, hits)

    assert out["all"]["count"] == 4
    assert out["correct"]["count"] == 2
    assert out["wrong"]["count"] == 2
    assert out["all"]["effective_pathways"]["2"] >= out["correct"]["effective_pathways"]["2"]
