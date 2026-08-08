"""Complete the locked Option B document-bootstrap estimators from saved rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from training.paper2_phase2_matched_alpha import document_partition


KIND = "paper2_phase2_option_b_document_bootstrap_audit_v1"
SOURCE_KIND = "paper2_phase2_option_b_matrix_v1"
DEFAULT_REPLICATES = 10_000
DEFAULT_SEED = 20260808
REQUIRED_STEPS = (0, 2_000, 4_000, 6_000, *range(10_000, 20_001, 1_000))
ROOT = Path(__file__).resolve().parents[1]
METHOD_DOCUMENT = ROOT / "docs/PAPER2_PHASE2_OPTION_B_BOOTSTRAP_AUDIT_METHOD_20260808.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile_interval(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    tail = (1.0 - float(level)) / 2.0
    lower, upper = np.quantile(np.asarray(values, dtype=np.float64), [tail, 1.0 - tail])
    return float(lower), float(upper)


def ols_slope(trajectories: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """Return EAL slope per 1,000 updates for one or many trajectories."""

    values = np.asarray(trajectories, dtype=np.float64)
    x = np.asarray(steps, dtype=np.float64) / 1000.0
    centered = x - x.mean()
    denominator = float(np.square(centered).sum())
    if denominator <= 0.0:
        raise ValueError("OLS requires at least two distinct steps")
    return ((values - values.mean(axis=-1, keepdims=True)) * centered).sum(axis=-1) / denominator


def registered_e1_support(*, endpoint_relative_gain: float, late_slope_ci: tuple[float, float]) -> bool:
    """Apply the governing markdown's CI-qualified Option B support rule."""

    return bool(float(endpoint_relative_gain) >= 0.01 or float(late_slope_ci[0]) > 0.0)


def separated_positive_intervals(
    *, dose_ci: tuple[float, float], fresh_ci: tuple[float, float]
) -> bool:
    return bool(float(fresh_ci[0]) > float(dose_ci[1]))


def _measurement(point: float, bootstrap: np.ndarray) -> dict[str, Any]:
    lower, upper = percentile_interval(bootstrap)
    return {
        "estimate": float(point),
        "document_bootstrap_95_ci": [lower, upper],
        "ci_excludes_zero_positive": bool(lower > 0.0),
    }


def _read_manifest_documents(
    *, manifest_path: Path, stage0a_summary: Mapping[str, Any]
) -> list[str]:
    expected = str(stage0a_summary["manifest"]["sample_manifest_sha256"])
    if sha256_file(manifest_path) != expected:
        raise RuntimeError("Stage 0A sample-manifest SHA mismatch")
    documents: dict[int, str] = {}
    horizons: dict[int, set[int]] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            anchor = int(row["anchor_index"])
            document = str(row["document_id"])
            prior = documents.setdefault(anchor, document)
            if prior != document:
                raise RuntimeError(f"anchor {anchor} crosses documents")
            horizons.setdefault(anchor, set()).add(int(row["horizon"]))
    if not documents or sorted(documents) != list(range(max(documents) + 1)):
        raise RuntimeError("Stage 0A anchor indices are not contiguous")
    if any(values != {1, 2, 3, 4} for values in horizons.values()):
        raise RuntimeError("Stage 0A manifest does not contain four horizons per anchor")
    return [documents[index] for index in range(len(documents))]


def _load_accepted_length(path: Path, expected_rows: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"missing Option B row receipt: {path}")
    try:
        rows = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (RuntimeError, TypeError, ValueError):
        rows = torch.load(path, map_location="cpu", weights_only=True)
    if "accepted_length" not in rows:
        raise RuntimeError(f"row receipt lacks accepted_length: {path}")
    accepted = rows["accepted_length"].detach().float().numpy().astype(np.float64, copy=True)
    if accepted.shape != (expected_rows,) or not np.isfinite(accepted).all():
        raise RuntimeError(
            f"invalid accepted-length shape or values: path={path} shape={accepted.shape}"
        )
    return accepted


def _public_arm(summary: Mapping[str, Any], *, seed: int, arm: str) -> Mapping[str, Any]:
    matches = [
        value for value in summary["arms"]
        if int(value["seed"]) == int(seed) and str(value["arm"]) == arm
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one public arm for seed={seed} arm={arm}")
    return matches[0]


def _history_eal(arm: Mapping[str, Any], step: int) -> float:
    matches = [row for row in arm["history"] if int(row["step"]) == int(step)]
    if len(matches) != 1:
        raise RuntimeError(f"missing public fixed-evaluation step {step}")
    return float(matches[0]["evaluations"]["fixed_evaluation"]["mean_accepted_length"])


def _group_sums(values: np.ndarray, inverse: np.ndarray, documents: int) -> np.ndarray:
    result = np.zeros((documents, values.shape[1]), dtype=np.float64)
    np.add.at(result, inverse, values)
    return result


def _bootstrap_document_means(
    *,
    trajectories: Mapping[str, np.ndarray],
    document_ids: list[str],
    replicates: int,
    seed: int,
    chunk_size: int = 128,
) -> dict[str, np.ndarray]:
    if replicates < 1000:
        raise ValueError("document bootstrap requires at least 1,000 replicates")
    unique, inverse = np.unique(np.asarray(document_ids, dtype=object), return_inverse=True)
    document_count = len(unique)
    counts = np.bincount(inverse, minlength=document_count).astype(np.float64)
    sums = {
        name: _group_sums(np.asarray(values, dtype=np.float64), inverse, document_count)
        for name, values in trajectories.items()
    }
    widths = {values.shape[1] for values in trajectories.values()}
    if len(widths) != 1:
        raise ValueError("all bootstrap trajectories must share a checkpoint axis")
    width = widths.pop()
    outputs = {
        name: np.empty((replicates, width), dtype=np.float64) for name in trajectories
    }
    rng = np.random.default_rng(int(seed))
    probabilities = np.full(document_count, 1.0 / document_count, dtype=np.float64)
    for start in range(0, replicates, chunk_size):
        stop = min(replicates, start + chunk_size)
        weights = rng.multinomial(document_count, probabilities, size=stop - start)
        denominators = weights @ counts
        for name, document_sums in sums.items():
            outputs[name][start:stop] = (weights @ document_sums) / denominators[:, None]
        if start == 0 or stop == replicates or stop % 1_024 == 0:
            print(
                f"option_b_document_bootstrap_progress replicates={stop}/{replicates}",
                flush=True,
            )
    return outputs


def _slope_bundle(
    *, point: np.ndarray, bootstrap: np.ndarray, step_to_column: Mapping[int, int]
) -> dict[str, Any]:
    def segment(values: np.ndarray, start: int, end: int) -> np.ndarray:
        return (values[..., step_to_column[end]] - values[..., step_to_column[start]]) / (
            (end - start) / 1000.0
        )

    dose_point = float(segment(point, 2_000, 4_000))
    fresh_point = float(segment(point, 4_000, 6_000))
    dose_bootstrap = segment(bootstrap, 2_000, 4_000)
    fresh_bootstrap = segment(bootstrap, 4_000, 6_000)
    contrast_point = fresh_point - dose_point
    contrast_bootstrap = fresh_bootstrap - dose_bootstrap
    dose = _measurement(dose_point, dose_bootstrap)
    fresh = _measurement(fresh_point, fresh_bootstrap)
    contrast = _measurement(contrast_point, contrast_bootstrap)
    return {
        "dose_pre_splice_eal_per_1000": dose,
        "fresh_post_splice_eal_per_1000": fresh,
        "fresh_minus_dose_eal_per_1000": contrast,
        "fresh_and_dose_intervals_separated": separated_positive_intervals(
            dose_ci=tuple(dose["document_bootstrap_95_ci"]),
            fresh_ci=tuple(fresh["document_bootstrap_95_ci"]),
        ),
    }


def audit(
    *,
    source_summary_path: Path,
    stage0a_summary_path: Path,
    manifest_path: Path,
    private_root: Path,
    replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    source = json.loads(source_summary_path.read_text(encoding="utf-8"))
    if source.get("kind") != SOURCE_KIND or source.get("status") != "complete":
        raise RuntimeError("Option B source matrix is not a complete canonical receipt")
    stage0a = json.loads(stage0a_summary_path.read_text(encoding="utf-8"))
    anchor_documents = _read_manifest_documents(
        manifest_path=manifest_path, stage0a_summary=stage0a
    )
    evaluation_mask = document_partition(
        anchor_documents, evaluation_fraction=0.2, seed=20260804
    )
    evaluation_documents = [
        document for document, selected in zip(anchor_documents, evaluation_mask.tolist())
        if selected
    ]
    if len(evaluation_documents) != int(source["population"]["existing_evaluation_anchors"]):
        raise RuntimeError("reconstructed fixed-evaluation population count differs")
    unique_documents = len(set(evaluation_documents))
    step_to_column = {step: index for index, step in enumerate(REQUIRED_STEPS)}
    trajectories: dict[str, np.ndarray] = {}
    row_hashes: dict[str, dict[str, str]] = {}
    for seed in (0, 1):
        for arm in ("full_a2", "draft_only_control"):
            name = f"seed_{seed}_{arm}"
            public_arm = _public_arm(source, seed=seed, arm=arm)
            columns = []
            row_hashes[name] = {}
            for step in REQUIRED_STEPS:
                path = private_root / name / f"rows_fixed_evaluation_step_{step:05d}.pt"
                accepted = _load_accepted_length(path, len(evaluation_documents))
                public_mean = _history_eal(public_arm, step)
                if abs(float(accepted.mean()) - public_mean) > 2e-6:
                    raise RuntimeError(
                        f"row/public EAL mismatch: arm={name} step={step} "
                        f"rows={accepted.mean()} public={public_mean}"
                    )
                columns.append(accepted)
                row_hashes[name][str(step)] = sha256_file(path)
                print(
                    f"option_b_bootstrap_rows_verified arm={name} step={step} "
                    f"rows={accepted.size} mean={accepted.mean():.9f}",
                    flush=True,
                )
            trajectories[name] = np.stack(columns, axis=1)
    bootstrap = _bootstrap_document_means(
        trajectories=trajectories,
        document_ids=evaluation_documents,
        replicates=replicates,
        seed=bootstrap_seed,
    )
    point = {name: values.mean(axis=0) for name, values in trajectories.items()}
    seeds: list[dict[str, Any]] = []
    seed_bootstrap_metrics: dict[int, dict[str, np.ndarray]] = {}
    late_steps = np.asarray(tuple(range(10_000, 20_001, 1_000)), dtype=np.float64)
    late_columns = [step_to_column[int(step)] for step in late_steps]
    for seed in (0, 1):
        full_name = f"seed_{seed}_full_a2"
        control_name = f"seed_{seed}_draft_only_control"
        full_slopes = _slope_bundle(
            point=point[full_name], bootstrap=bootstrap[full_name], step_to_column=step_to_column
        )
        control_slopes = _slope_bundle(
            point=point[control_name],
            bootstrap=bootstrap[control_name],
            step_to_column=step_to_column,
        )
        full_late_point = float(ols_slope(point[full_name][late_columns], late_steps))
        full_late_bootstrap = ols_slope(bootstrap[full_name][:, late_columns], late_steps)
        full_late = _measurement(full_late_point, full_late_bootstrap)
        gap_point = point[full_name] - point[control_name]
        gap_bootstrap = bootstrap[full_name] - bootstrap[control_name]
        gap_growth_point = float(
            gap_point[step_to_column[20_000]] - gap_point[step_to_column[0]]
        )
        gap_growth_bootstrap = (
            gap_bootstrap[:, step_to_column[20_000]] - gap_bootstrap[:, step_to_column[0]]
        )
        late_gap_point = float(ols_slope(gap_point[late_columns], late_steps))
        late_gap_bootstrap = ols_slope(gap_bootstrap[:, late_columns], late_steps)
        control_endpoint = float(point[control_name][step_to_column[20_000]])
        endpoint_gap = float(gap_point[step_to_column[20_000]])
        endpoint_relative = endpoint_gap / max(control_endpoint, 1e-12)
        endpoint_relative_bootstrap = (
            gap_bootstrap[:, step_to_column[20_000]]
            / bootstrap[control_name][:, step_to_column[20_000]].clip(min=1e-12)
        )
        corrected_support = registered_e1_support(
            endpoint_relative_gain=endpoint_relative,
            late_slope_ci=tuple(full_late["document_bootstrap_95_ci"]),
        )
        seed_bootstrap_metrics[seed] = {
            "full_dose": (
                bootstrap[full_name][:, step_to_column[4_000]]
                - bootstrap[full_name][:, step_to_column[2_000]]
            ) / 2.0,
            "full_fresh": (
                bootstrap[full_name][:, step_to_column[6_000]]
                - bootstrap[full_name][:, step_to_column[4_000]]
            ) / 2.0,
            "control_dose": (
                bootstrap[control_name][:, step_to_column[4_000]]
                - bootstrap[control_name][:, step_to_column[2_000]]
            ) / 2.0,
            "control_fresh": (
                bootstrap[control_name][:, step_to_column[6_000]]
                - bootstrap[control_name][:, step_to_column[4_000]]
            ) / 2.0,
            "full_late": full_late_bootstrap,
            "gap_growth": gap_growth_bootstrap,
            "late_gap": late_gap_bootstrap,
            "endpoint_relative": endpoint_relative_bootstrap,
        }
        seeds.append(
            {
                "seed": seed,
                "full_system_segment_slopes": full_slopes,
                "control_segment_slopes": control_slopes,
                "full_second_half_exposure_slope_eal_per_1000": full_late,
                "writeback_gap_growth_step_0_to_20000": _measurement(
                    gap_growth_point, gap_growth_bootstrap
                ),
                "writeback_gap_second_half_slope_eal_per_1000": _measurement(
                    late_gap_point, late_gap_bootstrap
                ),
                "endpoint_relative_full_gain": _measurement(
                    endpoint_relative, endpoint_relative_bootstrap
                ),
                "one_percent_endpoint_gain": bool(endpoint_relative >= 0.01),
                "corrected_e1_support": corrected_support,
            }
        )
    combined: dict[str, Any] = {}
    for key in seed_bootstrap_metrics[0]:
        samples = (seed_bootstrap_metrics[0][key] + seed_bootstrap_metrics[1][key]) / 2.0
        if key == "endpoint_relative":
            points = [row["endpoint_relative_full_gain"]["estimate"] for row in seeds]
        elif key == "full_late":
            points = [
                row["full_second_half_exposure_slope_eal_per_1000"]["estimate"]
                for row in seeds
            ]
        elif key == "gap_growth":
            points = [row["writeback_gap_growth_step_0_to_20000"]["estimate"] for row in seeds]
        elif key == "late_gap":
            points = [
                row["writeback_gap_second_half_slope_eal_per_1000"]["estimate"]
                for row in seeds
            ]
        else:
            arm, segment = key.split("_", 1)
            source_key = f"{arm}_system_segment_slopes" if arm == "full" else "control_segment_slopes"
            metric_key = (
                "dose_pre_splice_eal_per_1000" if segment == "dose"
                else "fresh_post_splice_eal_per_1000"
            )
            points = [row[source_key][metric_key]["estimate"] for row in seeds]
        combined[key] = _measurement(float(np.mean(points)), samples)
    corrected_both = all(row["corrected_e1_support"] for row in seeds)
    growing_writeback = all(
        row["writeback_gap_growth_step_0_to_20000"]["estimate"] > 0.0 for row in seeds
    )
    return {
        "kind": KIND,
        "status": "complete",
        "mode": "read_only_cpu_post_processing",
        "optimizer_updates": 0,
        "model_loaded": False,
        "source": {
            "summary": str(source_summary_path),
            "summary_sha256": sha256_file(source_summary_path),
            "source_scripted_reading": source["scripted_reading"],
            "source_receipt_kind": source["kind"],
        },
        "method": {
            "document": str(METHOD_DOCUMENT.relative_to(ROOT)),
            "document_sha256": sha256_file(METHOD_DOCUMENT),
            "fixed_before_audit": True,
        },
        "audit_reason": {
            "governing_requirement": (
                "positive second-half exposure slope with a document-bootstrap "
                "95 percent interval excluding zero"
            ),
            "landed_implementation": "positive point estimate only",
            "disposition": "preserve source receipt and issue corrected CI-qualified reading",
        },
        "bootstrap": {
            "unit": "document",
            "method": "paired percentile cluster bootstrap",
            "confidence_level": 0.95,
            "replicates": int(replicates),
            "seed": int(bootstrap_seed),
            "evaluation_anchors": len(evaluation_documents),
            "unique_documents": unique_documents,
            "seed_inference": False,
            "combined_scope": "descriptive mean conditional on the two registered seeds",
        },
        "row_receipt_sha256": row_hashes,
        "seeds": seeds,
        "combined_descriptive": combined,
        "corrected_scripted_reading": {
            "e1_support_in_any_seed": any(row["corrected_e1_support"] for row in seeds),
            "e1_support_in_both_seeds": corrected_both,
            "writeback_retained_for_e1": growing_writeback,
            "interpretation": (
                "curve_supports_E1_recipe_transfer"
                if corrected_both else "ci_qualified_E1_support_not_established"
            ),
        },
        "do_not_claim": [
            "document bootstrap supplies inference over seeds",
            "DEV accepted length is serving throughput",
            "this single splice is a general unique-data scaling law",
            "a flat curve proves architectural impossibility",
        ],
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    def metric(value: Mapping[str, Any]) -> str:
        lower, upper = value["document_bootstrap_95_ci"]
        return f"{value['estimate']:.6f} [{lower:.6f}, {upper:.6f}]"

    lines = [
        "# Option B Document-Bootstrap Audit",
        "",
        "Read-only completion of the confidence intervals required by the locked Option B protocol.",
        "No model was loaded and no optimizer update occurred.",
        "",
        "## Corrected reading",
        "",
        f"- Interpretation: `{summary['corrected_scripted_reading']['interpretation']}`",
        f"- E1 support in both seeds: `{str(summary['corrected_scripted_reading']['e1_support_in_both_seeds']).lower()}`",
        f"- Retain writeback for E1: `{str(summary['corrected_scripted_reading']['writeback_retained_for_e1']).lower()}`",
        "",
        "## Seed-level estimates",
        "",
        "All cells are estimate [document-bootstrap 95% CI].",
        "",
        "| Seed | Full dose slope | Full fresh slope | Full late slope | Gap growth 0-20k | Endpoint relative gain | Corrected E1 support |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in summary["seeds"]:
        lines.append(
            "| {seed} | {dose} | {fresh} | {late} | {gap} | {relative} | {support} |".format(
                seed=row["seed"],
                dose=metric(row["full_system_segment_slopes"]["dose_pre_splice_eal_per_1000"]),
                fresh=metric(row["full_system_segment_slopes"]["fresh_post_splice_eal_per_1000"]),
                late=metric(row["full_second_half_exposure_slope_eal_per_1000"]),
                gap=metric(row["writeback_gap_growth_step_0_to_20000"]),
                relative=metric(row["endpoint_relative_full_gain"]),
                support=str(row["corrected_e1_support"]).lower(),
            )
        )
    lines.extend(
        [
            "",
            "## Estimator correction",
            "",
            "The landed matrix labeled a positive late-slope point estimate as E1 support.",
            "The governing protocol additionally requires the document-bootstrap 95% interval",
            "to exclude zero. This receipt applies that requirement without changing the source",
            "matrix, thresholds, rows, or training lineage.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_summary", type=Path, required=True)
    parser.add_argument("--stage0a_summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--private_root", type=Path, required=True)
    parser.add_argument("--output_summary", type=Path, required=True)
    parser.add_argument("--output_markdown", type=Path, required=True)
    parser.add_argument("--bootstrap_replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--bootstrap_seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = audit(
        source_summary_path=args.source_summary,
        stage0a_summary_path=args.stage0a_summary,
        manifest_path=args.manifest,
        private_root=args.private_root,
        replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary["corrected_scripted_reading"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
