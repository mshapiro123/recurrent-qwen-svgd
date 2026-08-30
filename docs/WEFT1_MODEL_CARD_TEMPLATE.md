---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
tags:
  - weft-1
  - causal-lm
---

# WEFT-1

<!-- Fill only receipt-backed fields. Preserve every release guardrail below verbatim. -->

## Model description

<!-- Add frozen architecture, parameter count, context length, and checkpoint identity. -->

## Provenance

from-scratch in weights, not in data provenance — trained from random initialization on an open corpus including model-generated reasoning traces in a declared final phase.

## Training data and attribution

Dolma 3 — This training corpus includes Dolma 3 Pool and Dolma 3 Mix material from the Allen Institute for AI, licensed under the Open Data Commons Attribution License v1.0 (ODC-By); cite Team Olmo et al., “Olmo 3” (2025), arXiv:2512.13961.

FineWeb-Edu — This training corpus includes FineWeb-Edu by Anton Lozhkov, Loubna Ben Allal, Leandro von Werra, and Thomas Wolf, licensed under ODC-By v1.0 and subject to Common Crawl’s Terms of Use; cite “FineWeb-Edu: the Finest Collection of Educational Content” (2024), DOI 10.57967/hf/2497.

Stack-Edu — The code stratum includes Stack-Edu material routed through the pinned Dolma 3 Mix; Stack-Edu is a 125B-token educational-code corpus filtered from The Stack v2. Cite Loubna Ben Allal et al., “SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model” (2025), arXiv:2502.02737. The Stack v2’s original per-file licenses, attribution requirements, and removal/opt-out process remain applicable.

Stack-Edu route — Executed from allenai/dolma3_mix-6T at revision 689a3ea2d8217e64d73a5058913fa43ad15e81aa (2026-01-15T05:36:27Z). Stack-Edu is derived from The Stack v2/StarCoder2Data: the underlying source snapshot uses the Software Heritage graph dated 2023-09-06 and GitHub Archive metadata through 2023-09-14. The upstream card records that StarCoder2 used v2.0.1, which incorporated validated opt-outs through 2023-10-20; WEFT-1 inherits The Stack v2’s continuing removal/takedown posture.

Language identification retains the pinned fastText lid.176.bin classifier; no substitute model is authorized.

<!-- Add the remaining manifest-generated attribution rows without changing the text above. -->

## Corpus reproducibility and distribution

The public corpus artifact consists of the pipeline code, the manifest (pins, SHAs, seeds, and deduplication rates), and D1 replay instructions. The manifest SHA-256 is the public corpus identity, and replay is the reproducibility claim. Raw text shards are never published.

## Evaluation and claims

Comparative claims ride on the matched-compute control only.

no sentence of the form "WEFT-1 outperforms [named public model]" is ever written.

Named public-model comparisons: none.

<!-- Report only receipt-backed within-program and matched-control results. -->

## Limitations

<!-- Add evaluated limitations, known data limitations, and bounded failure cases. -->

## License

WEFT-1 weights are licensed under Apache License 2.0. Dataset attribution and upstream source-code license obligations remain as stated above and in the public corpus manifest.
