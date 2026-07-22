from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
PHASE_A_SUMMARY = ROOT / "outputs/stage5/stage5_phase_a_surpass_receipt_20260714/summary.json"
PHASE_A_DENSE_AUDIT = ROOT / "outputs/stage5/stage5_phase_a_dense_reader_audit_20260722/summary.json"
ARM_E_SUMMARY = ROOT / "outputs/stage5/stage5_adapter_budget_arm_e_20260718/summary.json"
DEFAULT_OUTPUT = ROOT / "docs/figures/figure4_phase_a_depth_profile.svg"

SERIES = (
    ("A", "Arm A: full block, 180.6M trainable", "#0B6E4F"),
    ("B_step4000", "Arm B: dense direct 0.5B", "#C44536"),
    ("C_step4000", "Arm C: dense scratchpad 0.5B", "#2667FF"),
    ("D_step4000", "Arm D: dense direct 1.5B", "#6F4E7C"),
    ("E", "Arm E: R16 + bridge, 6.0M trainable", "#D48C00"),
)


def load_counts() -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    phase_a = json.loads(PHASE_A_SUMMARY.read_text(encoding="utf-8"))
    dense_audit = json.loads(PHASE_A_DENSE_AUDIT.read_text(encoding="utf-8"))
    arm_e = json.loads(ARM_E_SUMMARY.read_text(encoding="utf-8"))
    counts = dict(phase_a["scoring"]["counts"])
    for label in ("B_step4000", "C_step4000", "D_step4000"):
        counts[label] = {
            depth: int(values["corrected_correct"])
            for depth, values in dense_audit["arms"][label]["by_depth"].items()
        }
    counts["E"] = arm_e["adapter_budget_depth_profile"]["counts_by_depth"]["E"]
    totals = phase_a["scoring"]["depth_totals"]
    return counts, totals


def arm_crossing(counts: dict[str, dict[str, int]]) -> float:
    delta_11 = counts["E"]["11"] - counts["A"]["11"]
    delta_12 = counts["E"]["12"] - counts["A"]["12"]
    if not (delta_11 > 0 and delta_12 < 0):
        raise ValueError("Expected the Arm A/E crossover between depths 11 and 12")
    return 11 + delta_11 / (delta_11 - delta_12)


def build_svg(counts: dict[str, dict[str, int]], totals: dict[str, int]) -> str:
    width, height = 1040, 650
    left, right, top, bottom = 76, 1015, 108, 565
    depths = list(range(1, 15))
    dx = (right - left) / (len(depths) - 1)

    def x(depth: float) -> float:
        return left + (depth - 1) * dx

    def y(accuracy: float) -> float:
        return bottom - accuracy * (bottom - top)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<rect x="{x(1) - dx / 2:.1f}" y="{top}" '
            f'width="{x(8) - x(1) + dx:.1f}" height="{bottom - top}" '
            'fill="#E8F2ED"/>'
        ),
        (
            f'<text x="{(x(1) + x(8)) / 2:.1f}" y="{top + 19}" text-anchor="middle" '
            'font-family="Arial" font-size="12" fill="#315E4D">'
            'trained support (depths 1-8)</text>'
        ),
        (
            '<text x="76" y="30" font-family="Arial" font-size="19" '
            'font-weight="bold">Depth profile on the frozen Phase A family</text>'
        ),
        (
            '<text x="76" y="52" font-family="Arial" font-size="12" fill="#444">'
            'First-completed-response accuracy; 128 identical rows per depth</text>'
        ),
    ]

    for tick in range(0, 101, 20):
        yy = y(tick / 100)
        lines.extend(
            [
                f'<line x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}" stroke="#D8D8D8"/>',
                (
                    f'<text x="{left - 12}" y="{yy + 4:.1f}" text-anchor="end" '
                    f'font-family="Arial" font-size="12">{tick}%</text>'
                ),
            ]
        )

    for depth in depths:
        lines.append(
            f'<text x="{x(depth):.1f}" y="{bottom + 28}" text-anchor="middle" '
            f'font-family="Arial" font-size="12">{depth}</text>'
        )

    for key, label, color in SERIES:
        points = " ".join(
            f"{x(depth):.1f},{y(counts[key][str(depth)] / totals[str(depth)]):.1f}"
            for depth in depths
        )
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
        for depth in depths:
            lines.append(
                f'<circle cx="{x(depth):.1f}" '
                f'cy="{y(counts[key][str(depth)] / totals[str(depth)]):.1f}" '
                f'r="2.8" fill="{color}"/>'
            )

    crossing = arm_crossing(counts)
    crossing_x = x(crossing)
    lines.extend(
        [
            (
                f'<line x1="{crossing_x:.1f}" y1="{top}" x2="{crossing_x:.1f}" '
                f'y2="{bottom}" stroke="#333" stroke-width="1.5" stroke-dasharray="6 5"/>'
            ),
            (
                f'<text x="{crossing_x - 7:.1f}" y="{top + 42}" text-anchor="end" '
                'font-family="Arial" font-size="11" fill="#222">'
                f'Arm A/E crossover: d{crossing:.2f}</text>'
            ),
        ]
    )

    legend_x = 535
    for index, (_, label, color) in enumerate(SERIES):
        row = index % 3
        column = index // 3
        lx = legend_x + column * 265
        ly = 25 + row * 23
        lines.extend(
            [
                f'<line x1="{lx}" y1="{ly}" x2="{lx + 25}" y2="{ly}" '
                f'stroke="{color}" stroke-width="3"/>',
                f'<text x="{lx + 32}" y="{ly + 4}" font-family="Arial" '
                f'font-size="11">{escape(label)}</text>',
            ]
        )

    lines.extend(
        [
            (
                f'<text x="{(left + right) / 2:.1f}" y="{height - 19}" '
                'text-anchor="middle" font-family="Arial" font-size="14">'
                'Composition depth</text>'
            ),
            (
                f'<text x="18" y="{(top + bottom) / 2:.1f}" text-anchor="middle" '
                f'transform="rotate(-90 18 {(top + bottom) / 2:.1f})" '
                'font-family="Arial" font-size="14">Accuracy</text>'
            ),
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    counts, totals = load_counts()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_svg(counts, totals), encoding="utf-8", newline="\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
