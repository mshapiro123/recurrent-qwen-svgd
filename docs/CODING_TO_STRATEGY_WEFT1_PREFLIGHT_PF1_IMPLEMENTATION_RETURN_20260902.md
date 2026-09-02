# CODING → STRATEGY — WEFT-1 PF-1 implementation return

**Date:** 2026-09-02  
**Status:** FAIL-CLOSED CPU RECEIPT · PF-1.1–1.5 implemented where executable · no PRE-FLIGHT GPU spend  
**Authority:** `STRATEGY_PREFLIGHT_AMENDMENT_PF1_20260902.md`, 12,285 bytes, SHA-256 `4e3186c432b57f71b9f32a444a269eec08557ca5181a6896b477078dbbb40861`, Drive `14czHXHXYMSx4zJKay7lCr0QmNCuWE1q4`
**Implementation commit:** `4ca0236c6c31c1da3abf1cdf20a66ef859e452b9`

## 0. Outcome

PF-1 corrected the Jacobian estimator and the gradient-liveness criterion successfully. The physical B1 plant now recovers `p=1.0` and `p=1.5` in both its zero-measurement-noise and noisy phases, the paired leave-one-probe-out jackknife is on the slope scale, and the K=1/K=4 eligibility-aware liveness gate passes the complete materialized bring-up graph.

Three items correctly prevent a PRE-FLIGHT pass receipt from minting:

1. **PF-1.2's registered branch fired:** with `Sxx=2.4022650695910066`, the literal published frontiers require `n=519` and `n=514`, so `n=512` does not stand without an explicit rounding/tolerance ruling.
2. **Catch #26:** PF-1.4 still describes the live scratch/re-entry/loop path as a product of linear or gated-linear factors. The actual graph contains RMSNorm, SiLU and a coupled `(hidden, lanes)` state update; using weight norms alone would not certify the implemented visit.
3. **Catch #27:** A7's phrase "each module OFF alone ≡ dense baseline, per K" has two incompatible literal readings, and both fail on the actual graph. Exact matched-background cells are executable, but strategy must bind their comparison graph before the matrix can be registered.

No unsafe production certificate was wired, no A7 pass was claimed, and no GPU meter was started.

## 1. Implemented CPU surface

| item | implementation | measured result |
|---|---|---|
| PF-1.1 natural-log coordinate | `analysis/weft1_jacobian_panel.py` | `Sxx=2.4022650695910066`; physical `lambda_T=-c*T**(-p)` recovers both registered plants |
| PF-1.2 exact power rerun | `analysis/jacobian_power3.py` | measurement SE term grows `1.4426950408889636x`; registered-`n` branch returns to strategy |
| PF-1.3 paired LOO jackknife | `paired_probe_jackknife` plus report/pilot schema | per-example leave-one-probe-out slopes; jackknife variance in slope coordinates; raw variance minus mean measurement variance, clipped with flag |
| B1 zero-noise then noisy coverage | `analysis/weft1_preflight_b1.py` | coverage counts `20/20`, `18/20`, `19/20`, `18/20`; all four phase means recover the plant within two Monte Carlo SE for `(p=1.0 zero/noisy, p=1.5 zero/noisy)` |
| B1 negative control | seeded scalar white-noise/no-power-law plant | derived `lambda_T=(+0.0818506,-0.225273,+0.411421,-0.812010)`; `sign_inconsistency` fires |
| n0 pilot semantics | same B1 runner | `n0=32`, `admissible_panel_result=false`, `reporting_status=pilot_diagnostic_only` |
| PF-1.4 standalone factor utility | `models/ablation_lm/certificates.py` | single-gate sidecar counterexample fixed (`1.50`, never `1.25`); certified and empirical quantities separated |
| A1/A2/A4 standing tests | standalone callosum, `Cl(2,0)` and mu-R product tests | green; production integration explicitly absent |
| PF-1.5 liveness | `models/ablation_lm/liveness.py` | K=1 and K=4 pass; K=1 re-entry ineligible by construction; every such tensor becomes eligible and live at K=4 |
| A7 typed plumbing | `models/ablation_lm/observational_invariance.py` | exact comparison, deferred cells and complete-coverage guard implemented; no matrix minted while catch #27 is open |
| Build-status disclosure | `docs/CODING_TO_STRATEGY_WEFT1_BUILD_STATUS_MATRIX_20260902.md` | every ratified surface classified as absent, standalone, integrated, or integrated + OBS-INV-tested with commit evidence |

The B1 receipt reports **zero A100-hours**. P-A files, workers and durability artifacts were untouched.

The deterministic 20-replicate B1 summary is:

| plant / phase | mean `p_hat` | Monte Carlo SE | 95% CI coverage |
|---|---:|---:|---:|
| `p=1.0`, zero measurement noise | `0.9979669927047136` | `0.0019364225831234572` | `20/20` |
| `p=1.0`, noisy | `1.003563140546964` | `0.0031566539903947056` | `18/20` |
| `p=1.5`, zero measurement noise | `1.496525522242151` | `0.002857298012182621` | `19/20` |
| `p=1.5`, noisy | `1.49638196486561` | `0.0027108882834993146` | `18/20` |

Every phase mean is within two Monte Carlo SE of its planted exponent. The zero-noise phases report `sigma_w=0`; the noisy-phase mean fitted `sigma_w` values are approximately `0.02016` and `0.01997`.

## 2. PF-1.2 branch — `n=512` does not meet the literal frontiers

The exact rerun uses the amendment's formula:

```text
SE(p_hat) = sqrt((sigma_slope^2 + sigma_w^2/Sxx) / n)
Sxx       = 2.4022650695910066
sigma_w   = 0.25
```

| frontier | literal target | realized at `n=512` | minimum integer `n` | covered `sigma_slope` at 512 |
|---|---:|---:|---:|---:|
| primary | `SE <= 0.051`, `sigma_slope=1.15` | `0.051320780244339934` | **519** | `1.1426700695027545` |
| secondary | `SE <= 0.036`, `sigma_slope=0.80` | `0.03606680845857909` | **514** | `0.7984578183833069` |

This is a small difference, but PF-1 explicitly says that if `n=512` does not stand, the new `n` returns through the amendment path. Coding therefore did not invent a rounding band. **Recommended literal:** register `n=520`, the smallest clean even panel size above both exact minima. The code remains at `MAIN_PANEL_EXAMPLES=512` until strategy rules.

## 3. Catch #26 — the corrected certificate still does not describe the implemented visit

PF-1.4 says re-entry, scratch update/injection and loop embedding are linear or gated-linear and asks for exact weight spectral norms times their gates. That characterization is false for the live graph:

```text
re-entry:       h' = h + alpha * scale * W * RMSNorm(prelude)
scratch step:   l' = l + alpha * W_out * SiLU(W_in [RMSNorm(l), W_ctx RMSNorm(h)+e_k])
scratch inject: h' = h + alpha * scale * W_read [mu(l'), delta(l')]
loop embedding: h' = h + alpha * e_k
```

Consequences:

- Re-entry and loop embedding are translations with respect to the recurrent state and therefore have state-Lipschitz factor exactly one; pricing their weight matrices as recurrent-state linear maps is the wrong derivative.
- Scratch is a nonlinear, coupled map on `(hidden, lanes)`. A product of `||W||_2` terms omitting RMSNorm, SiLU, the residual block structure, and the joint-state metric is not an upper bound on that map.
- The unresolved joint-state metric already identified in C-JAC-1 is load-bearing here. Without it, even a correct block-operator bound has no scalar norm in which to live.

The new utility therefore certifies only the formulas it actually receives, keeps absent rotor/callosum/sidecar placeholders out of `Lambda_adapters`, labels finite power iteration as an empirical local lower estimate, and prohibits a cL alarm or production-integration claim.

**Recommended binding:** retain exact structured certificates for the rotor, callosum and sidecar; treat re-entry and loop embedding as recurrent-state translations; estimate the complete nonlinear scratch+core visit locally as one explicitly empirical `Lambda_hat_visit` under a ratified joint-state metric. If strategy still wants a global upper bound for scratch, it must bind the joint norm and the RMSNorm/SiLU/block inequalities explicitly; coding should not infer them.

## 4. Catch #27 — A7's comparison graph is ambiguous

Two direct counterexamples establish that the literal wording cannot mint the requested Cartesian matrix:

| literal reading | measured counterexample |
|---|---|
| "module M OFF alone equals the dense baseline" while another optional module stays ON | engram OFF + Hadamard ON differs from dense in **160/160 logits**, max absolute difference **0.0059753283858299255** |
| "all optional modules OFF equals dense baseline at every K" | recurrence-only K=2 differs from ordinary K=1 dense in **160/160 logits**, max absolute difference **0.00009090080857276917** |

Both differences are expected behavior, not implementation defects: another active module changes the graph in the first case, and a shared dense core executed twice is not a dense core executed once in the second.

**Recommended minimal binding:** 

1. A per-module cell compares a freshly constructed structural-OFF model for `M` with an independently constructed matched-background model from which `M` is physically deleted before forward; all other flags, shared weights, inputs, K, dtype and backend are identical.
2. Dense-anchor identity is a separate all-optional-modules-OFF, K=1 cell.
3. K>1 cells use a matched recurrent-dense-core background, not the K=1 dense anchor.
4. Recurrence-OFF is K=1-only / N/A at K>1. If strategy instead wants requested `K>1` to collapse to executed K=1, that behavior must be bound explicitly because the current API rejects it.
5. Module/K compatibility is an eligibility matrix, not an unconditional Cartesian product.

Under that candidate semantics, independently constructed engram-OFF cells are already bit-identical in CPU FP32 and BF16. They remain evidence for executability, not a registered A7 pass.

## 5. Verification

Focused independent reruns on Windows Python 3.11:

```text
Combined PF-1/model regression             130 passed, 18 warnings
Full raw repository suite                  3,986 passed, 1 failed, 19 warnings
Strict exact-node quarantine gate          PASS
```

The one raw-suite failure is unchanged and governed:
`tests/test_paper2_claim_evidence_ledger.py::test_paper2_claim_evidence_paths_exist`.
No failure was added, removed or renamed. The immutable successor is
`training/ablation_lm_engineering_quarantine_20260901_pf1.json`, SHA-256
`f54bc252c560da3122fb02c2b6ccb6005eff3cca96e09b16e001547a3fbd656f`;
it retains the 2026-09-04 review date and forbids a repository-wide green claim.
Warnings are the existing PyTorch `torch.jit.script` deprecation warning plus
the existing anomaly-detection warning. `ruff` is not installed, so no lint
result is claimed.

## 6. Requested rulings

| ID | exact question | recommendation |
|---|---|---|
| **PF1-R1** | new main-panel `n`, or an explicit tolerance that preserves 512 | **520** |
| **PF1-R2 / catch #26** | actual nonlinear visit certificate topology and joint-state metric | exact structured factors + empirical complete-visit estimate; no false global certificate |
| **PF1-R3 / catch #27** | A7 comparison graph and K/module eligibility | matched-background semantics in §4; recurrence-OFF K>1 is N/A |

The B1 instrument calibration itself is green. These three rulings gate only the affected sample-size constant and the production certificate/A7 receipts. They do not invalidate PF-1.5, the standalone A1/A2/A4 tests, or the build-status matrix.

## 7. Preserved execution posture

```text
PF1_AUTHORITY_VERIFIED_BYTE_EXACT
B1_PHYSICAL_PLANT_AND_JACKKNIFE_GREEN
PF1_LIVENESS_K1_K4_GREEN
POWER_N_BRANCH_RETURNED_TO_STRATEGY
PRODUCTION_CERTIFICATE_NOT_WIRED
A7_MATRIX_NOT_MINTED
ZERO_A100_SPEND
P_A_UNTOUCHED
NO_TRAINING_OR_SEALED_DATA_CONTACT
```
