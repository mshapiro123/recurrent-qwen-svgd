"""Telemetry eval for stochastic latent trajectory sampling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_identity import model_load_kwargs, parse_split
from models.recurrent_wrapper import RecurrentQwenForCausalLM


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", default="Find one valid 4-queens placement.")
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--max_loops", type=int, default=4)
    parser.add_argument("--num_trajectories", type=int, default=2)
    parser.add_argument("--latent_injection_mode", default="pre", choices=("pre", "post", "both"))
    parser.add_argument("--particle_update_mode", default="none", choices=("none", "svgd"))
    parser.add_argument("--particle_init_noise", type=float, default=0.0)
    parser.add_argument("--svgd_eps", type=float, default=1.0)
    parser.add_argument("--svgd_repulsion_scale", type=float, default=1.0)
    parser.add_argument("--svgd_bandwidth", default="median")
    parser.add_argument("--svgd_bandwidth_floor", type=float, default=1e-6)
    parser.add_argument("--svgd_repulsion_max_norm", type=float)
    parser.add_argument(
        "--diagnostic_latent_scale",
        type=float,
        default=None,
        help="Override latent scale only for eval diagnostics.",
    )
    parser.add_argument(
        "--diagnostic_adapter_std",
        type=float,
        default=None,
        help="Reinitialize latent adapter weights only for eval diagnostics.",
    )
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attn_implementation", default="default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_load_kwargs(args.dtype, args.attn_implementation),
    ).to(args.device)
    model.eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=parse_split(args.split)).to(args.device)
    wrapper.eval()

    if args.diagnostic_latent_scale is not None:
        with torch.no_grad():
            wrapper.latent_trajectory.adapter.latent_scale.fill_(args.diagnostic_latent_scale)
    if args.diagnostic_adapter_std is not None:
        with torch.no_grad():
            wrapper.latent_trajectory.adapter.proj.weight.normal_(0.0, args.diagnostic_adapter_std)
            wrapper.latent_trajectory.adapter.proj.bias.zero_()

    encoded = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    with torch.no_grad():
        output = wrapper(
            **encoded,
            max_loops=args.max_loops,
            num_trajectories=args.num_trajectories,
            sample_latents=args.particle_update_mode == "none",
            latent_injection_mode=args.latent_injection_mode,
            particle_update_mode=args.particle_update_mode,
            particle_init_noise=args.particle_init_noise,
            svgd_eps=args.svgd_eps,
            svgd_repulsion_scale=args.svgd_repulsion_scale,
            svgd_bandwidth=args.svgd_bandwidth,
            svgd_bandwidth_floor=args.svgd_bandwidth_floor,
            svgd_repulsion_max_norm=args.svgd_repulsion_max_norm,
            use_cache=False,
            return_dict=True,
        )

    print(f"expected_loops={output.expected_loops.squeeze(0).tolist()}")
    for key, value in output.metrics.items():
        print(f"{key}={float(value):.6f}")
    if output.final_recurrent_hidden is not None and output.final_recurrent_hidden.shape[1] > 1:
        hidden = output.final_recurrent_hidden[0]
        delta = (hidden[0] - hidden[1]).abs()
        print(f"trajectory_hidden_max_abs_delta={float(delta.max()):.8f}")
        print(f"trajectory_hidden_mean_abs_delta={float(delta.mean()):.8f}")
    if output.trajectory_logits is not None:
        last_token_ids = output.trajectory_logits[0, :, -1].argmax(dim=-1).tolist()
        print(f"trajectory_next_token_argmax={last_token_ids}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
