"""Calibrate a low-dimensional SVGD kernel projection from recurrent particle states."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_best_of_k_jsonl import load_wrapper, parse_optional_float, parse_seeds, read_tasks, set_seed
from models.halting import masked_mean


def eta_squared(values: torch.Tensor, labels: torch.Tensor) -> float:
    values = values.float()
    labels = labels.cpu()
    total = (values - values.mean()).pow(2).sum()
    if total <= 0:
        return 0.0
    between = values.new_zeros(())
    for label in labels.unique():
        group = values[labels == label]
        if group.numel():
            between = between + group.numel() * (group.mean() - values.mean()).pow(2)
    return float((between / total).item())


def covariance_eigh(samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    centered = samples.float() - samples.float().mean(dim=0, keepdim=True)
    return covariance_eigh_centered(centered)


def covariance_eigh_centered(centered: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    centered = centered.float()
    denom = max(centered.shape[0] - 1, 1)
    cov = centered.t().matmul(centered) / float(denom)
    evals, evecs = torch.linalg.eigh(cov)
    return evals.flip(0).contiguous(), evecs.flip(1).contiguous()


def center_by_group(samples: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    centered = samples.float().clone()
    labels = labels.cpu()
    for label in labels.unique():
        mask = labels == label
        centered[mask] -= centered[mask].mean(dim=0, keepdim=True)
    return centered


def capture_fraction(centered_samples: torch.Tensor, projection: torch.Tensor, dim: int) -> float:
    if centered_samples.numel() == 0:
        return 0.0
    dim = max(1, min(int(dim), projection.shape[1]))
    total = centered_samples.float().pow(2).sum()
    if total <= 0:
        return 0.0
    projected = centered_samples.float().matmul(projection[:, :dim].float())
    return float((projected.pow(2).sum() / total).item())


def choose_knee(evals: torch.Tensor, *, floor: int, cap: int, search_max: int) -> int:
    if evals.numel() < 2:
        return max(1, floor)
    limit = min(search_max, evals.numel() - 1, cap)
    eps = max(float(evals[0].item()) * 1e-8, 1e-12)
    ratios = evals[:limit] / evals[1 : limit + 1].clamp_min(eps)
    knee = int(torch.argmax(ratios).item()) + 1
    return max(floor, min(knee, cap))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--tasks_jsonl", default="eval/smoke_exact_tasks_v2.jsonl")
    parser.add_argument("--phase2_checkpoint", required=True)
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--max_loops", type=int, default=4)
    parser.add_argument("--num_trajectories", type=int, default=4)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--adapter_dtype", default="float32")
    parser.add_argument("--phase2_sample_latents", action="store_true")
    parser.add_argument("--phase2_latent_injection_mode", default="post", choices=("pre", "post", "both"))
    parser.add_argument("--phase2_particle_update_mode", default="svgd", choices=("none", "svgd"))
    parser.add_argument("--particle_init_noise", type=float, default=0.05)
    parser.add_argument("--svgd_eps", type=float, default=1.0)
    parser.add_argument("--svgd_repulsion_scale", type=float, default=1.0)
    parser.add_argument("--svgd_bandwidth", default="median")
    parser.add_argument("--svgd_bandwidth_floor", type=float, default=1e-6)
    parser.add_argument("--svgd_repulsion_max_norm", type=parse_optional_float)
    parser.add_argument("--svgd_projection_seed", type=int, default=123)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--mode_floor", type=int, default=8)
    parser.add_argument("--knee_search_max", type=int, default=64)
    parser.add_argument(
        "--calibration_centering",
        default="within_group",
        choices=("global", "within_task", "within_group"),
        help=(
            "Covariance used for the saved projection. 'within_group' centers each task/seed particle group, "
            "matching the particles the kernel actually compares."
        ),
    )
    parser.add_argument("--output", default="outputs/calibration/svgd_hidden_pca_projection.pt")
    args = parser.parse_args()

    tasks = read_tasks(args.tasks_jsonl)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    wrapper = load_wrapper(args, args.phase2_checkpoint)

    pooled_rows: list[torch.Tensor] = []
    task_labels: list[int] = []
    seed_labels: list[int] = []
    group_labels: list[int] = []
    group_idx = 0

    with torch.no_grad():
        for seed in parse_seeds(args.seeds):
            set_seed(seed)
            for task_idx, task in enumerate(tasks):
                encoded = tokenizer(task.prompt, return_tensors="pt").to(args.device)
                output = wrapper(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded.get("attention_mask"),
                    max_loops=args.max_loops,
                    num_trajectories=args.num_trajectories,
                    sample_latents=args.phase2_sample_latents,
                    latent_injection_mode=args.phase2_latent_injection_mode,
                    particle_update_mode=args.phase2_particle_update_mode,
                    particle_init_noise=args.particle_init_noise,
                    svgd_eps=args.svgd_eps,
                    svgd_repulsion_scale=args.svgd_repulsion_scale,
                    svgd_bandwidth=args.svgd_bandwidth,
                    svgd_bandwidth_floor=args.svgd_bandwidth_floor,
                    svgd_repulsion_max_norm=args.svgd_repulsion_max_norm,
                    svgd_projection_seed=args.svgd_projection_seed,
                    use_cache=False,
                    return_dict=True,
                )
                hidden = output.final_recurrent_hidden.reshape(
                    args.num_trajectories,
                    output.final_recurrent_hidden.shape[-2],
                    output.final_recurrent_hidden.shape[-1],
                )
                mask = encoded.get("attention_mask")
                if mask is not None:
                    mask = mask.repeat_interleave(args.num_trajectories, dim=0)
                pooled = masked_mean(hidden, mask).detach().float().cpu()
                pooled_rows.append(pooled)
                task_labels.extend([task_idx] * pooled.shape[0])
                seed_labels.extend([seed] * pooled.shape[0])
                group_labels.extend([group_idx] * pooled.shape[0])
                group_idx += 1

    z = torch.cat(pooled_rows, dim=0)
    norms = z.norm(dim=-1)
    task_tensor = torch.tensor(task_labels)
    seed_tensor = torch.tensor(seed_labels)
    group_tensor = torch.tensor(group_labels)

    zn = torch.nn.functional.normalize(z, p=2, dim=-1, eps=1e-8)
    z_within_task = center_by_group(z, task_tensor)
    z_within_group = center_by_group(z, group_tensor)
    zn_within_task = center_by_group(zn, task_tensor)
    zn_within_group = center_by_group(zn, group_tensor)

    global_direction_evals, _ = covariance_eigh(zn)
    global_hidden_evals, global_hidden_evecs = covariance_eigh(z)
    within_task_direction_evals, _ = covariance_eigh_centered(zn_within_task)
    within_task_hidden_evals, within_task_hidden_evecs = covariance_eigh_centered(z_within_task)
    within_group_direction_evals, _ = covariance_eigh_centered(zn_within_group)
    within_group_hidden_evals, within_group_hidden_evecs = covariance_eigh_centered(z_within_group)

    if args.calibration_centering == "global":
        projection_source = global_hidden_evecs
        dimension_source_evals = global_direction_evals
    elif args.calibration_centering == "within_task":
        projection_source = within_task_hidden_evecs
        dimension_source_evals = within_task_direction_evals
    else:
        projection_source = within_group_hidden_evecs
        dimension_source_evals = within_group_direction_evals

    hidden_evals = {
        "global": global_hidden_evals,
        "within_task": within_task_hidden_evals,
        "within_group": within_group_hidden_evals,
    }
    direction_evals = {
        "global": global_direction_evals,
        "within_task": within_task_direction_evals,
        "within_group": within_group_direction_evals,
    }
    available_dim = min(args.projection_dim, projection_source.shape[1])
    recommended_dim = choose_knee(
        dimension_source_evals,
        floor=args.mode_floor,
        cap=available_dim,
        search_max=args.knee_search_max,
    )
    capture_dims = sorted({args.mode_floor, recommended_dim, min(2 * recommended_dim, available_dim), 16, 32, available_dim})
    capture_dims = [dim for dim in capture_dims if 0 < dim <= available_dim]
    within_group_capture = {
        "global_pca": {
            str(dim): capture_fraction(z_within_group, global_hidden_evecs, dim)
            for dim in capture_dims
        },
        "within_task_pca": {
            str(dim): capture_fraction(z_within_group, within_task_hidden_evecs, dim)
            for dim in capture_dims
        },
        "within_group_pca": {
            str(dim): capture_fraction(z_within_group, within_group_hidden_evecs, dim)
            for dim in capture_dims
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "projection": projection_source[:, :available_dim].contiguous(),
        "calibration_centering": args.calibration_centering,
        "recommended_dim": recommended_dim,
        "direction_eigenvalues": dimension_source_evals[: min(128, dimension_source_evals.numel())].contiguous(),
        "hidden_eigenvalues": hidden_evals[args.calibration_centering][
            : min(128, hidden_evals[args.calibration_centering].numel())
        ].contiguous(),
        "global_direction_eigenvalues": global_direction_evals[: min(128, global_direction_evals.numel())].contiguous(),
        "within_task_direction_eigenvalues": within_task_direction_evals[
            : min(128, within_task_direction_evals.numel())
        ].contiguous(),
        "within_group_direction_eigenvalues": within_group_direction_evals[
            : min(128, within_group_direction_evals.numel())
        ].contiguous(),
        "global_hidden_eigenvalues": global_hidden_evals[: min(128, global_hidden_evals.numel())].contiguous(),
        "within_task_hidden_eigenvalues": within_task_hidden_evals[
            : min(128, within_task_hidden_evals.numel())
        ].contiguous(),
        "within_group_hidden_eigenvalues": within_group_hidden_evals[
            : min(128, within_group_hidden_evals.numel())
        ].contiguous(),
        "radial_eta2_task": eta_squared(norms, task_tensor),
        "radial_eta2_seed": eta_squared(norms, seed_tensor),
        "radial_eta2_group": eta_squared(norms, group_tensor),
        "within_group_capture": within_group_capture,
        "num_samples": int(z.shape[0]),
        "num_groups": int(group_tensor.unique().numel()),
        "hidden_dim": int(z.shape[1]),
        "projection_dim": int(available_dim),
        "mode_floor": int(args.mode_floor),
        "knee_search_max": int(args.knee_search_max),
        "source": {
            "tasks_jsonl": args.tasks_jsonl,
            "seeds": args.seeds,
            "num_trajectories": args.num_trajectories,
            "checkpoint": args.phase2_checkpoint,
            "particle_update_mode": args.phase2_particle_update_mode,
            "particle_init_noise": args.particle_init_noise,
            "svgd_repulsion_scale": args.svgd_repulsion_scale,
            "svgd_repulsion_max_norm": args.svgd_repulsion_max_norm,
            "calibration_centering": args.calibration_centering,
        },
    }
    torch.save(payload, output_path)

    meta_path = output_path.with_suffix(".json")
    meta = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "projection",
            "direction_eigenvalues",
            "hidden_eigenvalues",
            "global_direction_eigenvalues",
            "within_task_direction_eigenvalues",
            "within_group_direction_eigenvalues",
            "global_hidden_eigenvalues",
            "within_task_hidden_eigenvalues",
            "within_group_hidden_eigenvalues",
        }
    }
    meta["direction_eigenvalues_top32"] = [float(x) for x in dimension_source_evals[:32]]
    meta["hidden_eigenvalues_top32"] = [float(x) for x in hidden_evals[args.calibration_centering][:32]]
    meta["global_direction_eigenvalues_top32"] = [float(x) for x in global_direction_evals[:32]]
    meta["within_task_direction_eigenvalues_top32"] = [float(x) for x in within_task_direction_evals[:32]]
    meta["within_group_direction_eigenvalues_top32"] = [float(x) for x in within_group_direction_evals[:32]]
    meta["global_hidden_eigenvalues_top32"] = [float(x) for x in global_hidden_evals[:32]]
    meta["within_task_hidden_eigenvalues_top32"] = [float(x) for x in within_task_hidden_evals[:32]]
    meta["within_group_hidden_eigenvalues_top32"] = [float(x) for x in within_group_hidden_evals[:32]]
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"saved_projection={output_path}")
    print(f"saved_metadata={meta_path}")
    print(f"num_samples={z.shape[0]}")
    print(f"num_groups={group_tensor.unique().numel()}")
    print(f"hidden_dim={z.shape[1]}")
    print(f"projection_dim={available_dim}")
    print(f"calibration_centering={args.calibration_centering}")
    print(f"recommended_dim={recommended_dim}")
    print(f"radial_eta2_task={payload['radial_eta2_task']:.6f}")
    print(f"radial_eta2_seed={payload['radial_eta2_seed']:.6f}")
    print(f"radial_eta2_group={payload['radial_eta2_group']:.6f}")
    print("within_group_capture=", within_group_capture)
    print("direction_eigenvalues_top16=", [round(float(x), 8) for x in dimension_source_evals[:16]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
