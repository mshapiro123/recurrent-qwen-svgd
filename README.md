# Recurrent-Depth Qwen

This repository studies whether a pretrained dense Qwen model can be converted into a recurrent-depth latent-computation system, and whether guided stochastic trajectories can later add useful reasoning width.

**Current status, July 15, 2026:** Part 1, the deterministic mechanism program, is closed. Part 2, guided stochastic width, is gated on a multi-valued forward-relation substrate screen.

- [Part 1 paper draft](docs/PAPER_ONE_DETERMINISTIC_RECURRENT_QWEN_20260715.md)
- [Part 1 decision record](docs/PART1_DETERMINISTIC_PROGRAM_CLOSEOUT_20260715.md)
- [Claim-to-evidence ledger](docs/part1_claim_evidence_ledger.json)
- [Current status index](docs/PROJECT_STATUS_PAPER.md)
- [Phase G-alpha specification](docs/PHASE_G_ALPHA_GUIDED_STOCHASTIC_TRANSITION_SPEC.md)

## Research Program

The project began with a deterministic recurrent-depth architecture:

```text
input -> recurrent loop -> learned halting depth -> one latent trajectory -> one answer
```

The longer-term GRAM-inspired target adds learned stochastic width:

```text
input -> target-conditioned latent prior -> K recurrent trajectories
      -> exact candidate coverage -> optional learned selection
```

The implementation is not a reproduction of GRAM. It combines a pretrained Qwen backbone, a Prelude/Recurrent-Block/Coda split, corrected input re-injection, probabilistic halting, and a proposed prior/posterior latent head. Early Gaussian and SVGD experiments predated the repaired loop closure and lacked target-conditioned variational guidance, so they do not answer the current width hypothesis.

## Deterministic Architecture

`Qwen/Qwen2.5-0.5B-Instruct` is partitioned into:

1. **Prelude:** computes an input-grounded representation.
2. **Recurrent Block:** a shared middle transformer region applied repeatedly.
3. **Coda:** maps the final recurrent state through the remaining pretrained layers and LM head.

The one-loop route preserves the base computation when recurrent additions are inactive. On later loops, a split bridge combines the persistent state with the re-injected Prelude representation. Exact per-loop labels can supervise the latent transition, while a sequence-level PonderNet-style controller can model loop depth.

The final deterministic capacity experiments unfroze the 12-layer recurrent block. The resulting trainable budget is 182,163,457 parameters; Part 1 is therefore an architecture-conversion and mechanism study, not a parameter-efficient recovery claim.

## Main Part 1 Result

On one frozen synthetic composition family with 128 rows at each depth from 1 through 14:

| Arm | Correct / 1,792 | Accuracy | Depths 11-14 |
|---|---:|---:|---:|
| Recurrent Qwen2.5-0.5B | `1,506` | `84.04%` | `272/512` |
| Dense 0.5B direct | `470` | `26.23%` | `60/512` |
| Dense 0.5B scratchpad | `952` | `53.13%` | `56/512` |
| Dense 1.5B direct | `322` | `17.97%` | `58/512` |

Against the strongest dense scratchpad control, the paired comparison was 607 helped, 53 hurt, and 1,132 tied, with two-sided exact `p=3.42e-120`.

This is a bounded synthetic-family system comparison. Training lineage, tokens, FLOPs, latency, and inference compute were not matched. It is not a broad natural-reasoning or external-benchmark claim.

## Other Established Findings

- The corrected architecture passes the tested one-loop identity contract.
- Intermediate-state supervision installed a persistent loop-indexed transition: after outcome-only annealing, the active diagonal remained `625/640` and `357/384` above-diagonal states continued the update.
- Support-depth scaling moved the N24 frozen frontier to `91.4%` at depth 14 and `70.3%` at depth 18, with a measured ceiling by depth 22.
- Controlled verbal relay and pointer surfaces reached `86.0%` and `79.0%` at the final checkpoint, with clear tail degradation.
- Inverse operations were learnable in isolation, but the tested full-block continuations did not preserve them together with the consolidated mechanism.
- The preregistered F9 multi-channel bridge precursor gate did not authorize an architecture change.

## Current Claim Boundary

Supported:

- identity-preserving recurrent model surgery on the tested Qwen split;
- trainable and persistent loop-indexed latent computation;
- a recurrent-system advantage over registered dense recipes on the frozen synthetic family;
- a bounded acquisition-retention failure under the tested inverse-task continuations.

Open:

- broad natural-benchmark superiority;
- reliable learned depth selection;
- matched-compute architectural causality;
- guided stochastic width and exact multi-solution coverage;
- any beneficial role for SVGD beyond a future ablation.

## Next GPU Gate

Use an L4 for the shared target `part1_closeout_pivot_session`. It runs:

1. a non-promotable loop-position transfer micro-test;
2. the N20 verbal branching-relations screen;
3. the N24 symbolic branching-relations screen.

The branching gate is pooled exact validity `>=0.70` with every depth `>=0.55` on at least one keeper. No block unfreeze is allowed. Phase G-alpha launches only after a green screen and a powered coverage margin is locked.

```python
import base64
import os
import requests

REPO = "mshapiro123/recurrent-qwen-svgd"
REF = "29a8b57dcac6f1c3e273c6c126305a528e791afa"
PATH = "colab/CURRENT_A100_BOOTSTRAP_CELL.py"

token = None
try:
    from google.colab import userdata
    token = userdata.get("GH_TOKEN")
except Exception:
    pass

headers = {"Accept": "application/vnd.github+json"}
if token:
    headers["Authorization"] = f"Bearer {token}"

url = f"https://api.github.com/repos/{REPO}/contents/{PATH}?ref={REF}"
response = requests.get(url, headers=headers, timeout=30)
response.raise_for_status()
code = base64.b64decode(response.json()["content"]).decode("utf-8")

os.environ["STAGE5_CURRENT_A100_TARGET"] = "part1_closeout_pivot_session"
os.environ["STAGE5_BOOTSTRAP_REF"] = REF
exec(compile(code, PATH, "exec"))
```

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

The training/evaluation entry points are under `training/`, `eval/`, and `colab/`. Durable experiment receipts are under `outputs/stage5/`. The current dependency order is maintained in [PROGRAM_TRACK_MASTER_SEQUENCE.md](docs/PROGRAM_TRACK_MASTER_SEQUENCE.md), and the chronological record is [EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md).

## Phase G-alpha Contract

If the branching substrate passes, the deterministic keeper remains frozen. Only these groups may train:

```text
phase_g_prior_head.*
phase_g_posterior_head.*
phase_g_injection_scale
```

The primary test is paired exact oracle coverage at K against:

1. entropy-matched answer-head sampling at the same K;
2. one deterministic trajectory at matched `K*T` transition compute.

Learned selection, per-trajectory halting, and SVGD remain closed until guided latent width beats both comparators.
