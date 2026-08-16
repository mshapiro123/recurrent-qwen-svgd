# Handoff: P3.5 No-Training Amplitude and Depth Diagnostics

**Date:** 2026-08-16  
**Program:** Paper Two, Phase 3 closeout and Stage 2A discrimination  
**Status:** complete; amplitude and K+ banked; T1 archive-blocked; D1 folded into K+  
**Data boundary:** reused DEV only; CONFIRM and EVAL-E remain sealed  
**Training boundary:** no optimizer constructed, zero optimizer steps

## 0. Bottom line

The amplitude study establishes a wider safe operating surface but does not identify one replicated performance optimum. All four predeclared ceilings, `0.02`, `0.05`, `0.08`, and `0.11`, passed the replicated safety rule: collateral `chi` was zero in both seeds and the protected floor half never lost more than two net rows. The preregistered structural rule therefore selects `0.11`, the largest tested safe ceiling. Task performance is non-monotonic and seed-dependent, however. Seed 0 peaked at `+15/1,024` rows at `0.08`; seed 1 peaked at `+8/1,024` at `0.05`. The safe amplitude range is larger than the P3.5 operating point, but amplitude alone does not explain or reliably repair the small effect.

The requested T1 late-window state extraction could not run under its registered contract. The P3.4 step-4,000 checkpoints exist and match their recorded hashes, but steps 3,400, 3,600, and 3,800 are absent for both seeds. The P3.5 late-window states belong to a different lineage and were correctly not substituted. This is an archive/design blocker, not a negative scientific result.

D1 found no landed per-K P3.4/P3.5 coda rows in the archives. It was therefore folded into the authorized K+ score-only sweep. Future-prediction KL is unavailable because the registered task graph keeps the draft head inactive.

K+ does not show a replicated benefit from additional recurrent depth. Seed 0 is exactly flat at `506/1,024` for every native depth from K=1 through K=4. Seed 1 scores `501`, `500`, `500`, then `508`, so its fourth loop adds eight net rows, primarily on GSM8K. The registered K=4 marginal is therefore zero in seed 0 and plus eight in seed 1; the predeclared both-seeds criterion fails. Depth changes many individual row outcomes, but its net task value is not stable across seeds. The exploratory clamped extension is harmful overall: K=5/K=6 score `504/505` in seed 0 and `503/502` in seed 1.

Taken together, E2 is the strongest measured contributor: wider writes are safe over the tested range and improve the mean DEV net, but the performance response is non-monotonic and heterogeneous. E3 is a real seed-specific interaction rather than a replicated explanation. E1, the knowledge-deficit hypothesis, remains unresolved; these diagnostics neither test nor eliminate it.

## 1. Questions and rationale

The wave was designed to discriminate among three explanations for P3.5's small task effect:

- **E1, knowledge deficit:** the sidecar cannot supply information the small model does not possess.
- **E2, amplitude limit:** the learned direction is useful, but the deployed write magnitude is too small.
- **E3, depth underuse:** additional recurrent computation still adds value and the fixed K=4 path is not saturated.

Four no-training jobs were authorized together:

1. A replicated amplitude surface on fixed Arm S EMA endpoints.
2. T1 state extraction from the registered P3.4 late window.
3. D1 archive reconstruction of per-loop value.
4. K+ scoring at native K=1 through 4 and exploratory K=5 through 6.

This sequencing isolates inference-time levers before another training campaign. It also keeps the recipe itself visible as a method contribution: matched estimators, sealed partitions, explicit operating ceilings, causal instrumentation, and resumable receipts are part of the result-production method, not incidental logistics.

## 2. Fixed experimental substrate

The amplitude and K+ studies use the two P3.5 Arm S EMA step-4,400 endpoints:

| Seed | Endpoint SHA-256 |
|---:|---|
| 0 | `a047e2e7b35320376a736492c79d913b8690937da785efa2af002c8f54d26ca6` |
| 1 | `e36cddb76407c8f853ccb43824c77cf01d15f144780726dd9aec23215467fccb` |

Both studies use the same 1,024-row DEV task panel. Checkpoint selection is barred. No CONFIRM or EVAL-E score is computed. The amplitude surface changes only the evaluation-time gate ceiling. K+ fixes the ceiling at `0.02` and changes only the number of recurrent flow applications.

Native K=1 through 4 use the learned per-step parameters directly. K=5 and 6 are explicitly exploratory: the flow step embedding index and bridge gate/rho index are clamped to the learned step-4 parameters. Those cells can show trajectory shape but cannot determine the registered depth conclusion.

## 3. Amplitude design and locked rule

The four ceilings were `0.02`, `0.05`, `0.08`, and `0.11`, evaluated in both seeds. The preregistered selection rule was structural rather than accuracy-maximizing:

> Select the largest predeclared ceiling for which both seeds have collateral `chi = 0` and protected floor-half net degradation no worse than minus two rows.

This rule was intentionally chosen to map a safe operating range without selecting a ceiling post hoc on DEV accuracy.

## 4. Amplitude results

The fixed base score was `502/1,024` in every cell.

| Ceiling | Seed 0 net | Seed 1 net | Mean net | Seed 0 fixes/regressions | Seed 1 fixes/regressions | Floor net S0/S1 | `chi` S0/S1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.02 | +4 | +6 | +5.0 | 49 / 45 | 50 / 44 | +1 / +1 | 0 / 0 |
| 0.05 | +8 | +8 | +8.0 | 50 / 42 | 47 / 39 | +3 / +1 | 0 / 0 |
| 0.08 | **+15** | +6 | +10.5 | 56 / 41 | 47 / 41 | +4 / +1 | 0 / 0 |
| 0.11 | +11 | +7 | +9.0 | 53 / 42 | 50 / 43 | +3 / +1 | 0 / 0 |

All four ceilings are safety-eligible under the locked rule, so the selected structural ceiling is `0.11`. That is not a claim that `0.11` is the performance optimum. Seed 0's task optimum on this surface is `0.08`; seed 1's is `0.05`; both decline afterward. Selecting either accuracy peak would be post-hoc and is prohibited.

The deployed-direction capture statistic also falls as the ceiling rises:

| Ceiling | Seed 0 `pi_dep` | Seed 1 `pi_dep` |
|---:|---:|---:|
| 0.02 | 28.30% | 27.59% |
| 0.05 | 19.36% | 19.80% |
| 0.08 | 17.28% | 17.00% |
| 0.11 | 16.66% | 16.77% |

`pi_dir` remains fixed within each seed because it is the matched forced-direction audit, not a function of the deployed ceiling. The falling `pi_dep` percentage should therefore not be read as fewer absolute task corrections without consulting the paired fixes/regressions and task net. The wider writes produce more task benefit in some cells while also changing the denominator and eligibility geometry of the deployed-capture statistic.

## 5. T1 state-extraction receipt

The T1 manifest was score-blind and loaded no model. It verified the 1,024-row panel and the intended cell schema: four loops, layer taps 6/12/18/24, eight prelude slots, eight recurrent slots per loop, 44 cells per row, and 128 dimensions per cell.

Available:

| Seed | Step | SHA-256 |
|---:|---:|---|
| 0 | 4,000 | `381955ec5b78d0a00883c29e9f940feac8cfc8665f7a3a4446c79734532f4ed7` |
| 1 | 4,000 | `97ad532a5bffd72b2563799047b517e531e00115793bf4808f060148dfffc1ec` |

Missing for each seed: steps 3,400, 3,600, and 3,800. Because the lock forbids substitution and the P3.5 landing is a different lineage, T1 exits as `blocked_missing_registered_late_window_checkpoints`. No fingerprint or temporal state claim follows from this receipt.

## 6. D1 archive read

The P3.4 and P3.5 coda archives do not contain landed task rows at multiple K values. D1 therefore cannot reconstruct an accuracy-versus-K curve solely from existing receipts and is folded into K+. The future-prediction KL requested where available is not estimable from the registered task graph because its draft head is inactive during scoring. This limitation is explicit rather than replaced with a different KL estimator.

## 7. K+ design and results

The registered K+ read fixed the P3.5 EMA endpoints and the `0.02` ceiling, then scored K=1 through 4 on the same 1,024 DEV rows. K=4 was transported from the canonical P3.5 receipt. K=1 through 3 were scored through the same registered task path. The K=5 and K=6 cells repeat the learned step-4 flow embedding and bridge gate/rho index and are labeled exploratory throughout.

| K | Seed 0 correct | Seed 0 vs base | Seed 1 correct | Seed 1 vs base | Scope |
|---:|---:|---:|---:|---:|---|
| 1 | 506 | +4 | 501 | -1 | registered |
| 2 | 506 | +4 | 500 | -2 | registered |
| 3 | 506 | +4 | 500 | -2 | registered |
| 4 | 506 | +4 | 508 | +6 | registered |
| 5 | 504 | +2 | 503 | +1 | exploratory, step-4 clamp |
| 6 | 505 | +3 | 502 | 0 | exploratory, step-4 clamp |

Equal pooled counts do not mean equal predictions. The paired row marginal shows substantial churn:

| Transition | Seed 0 fixes / regressions / net | Seed 1 fixes / regressions / net | Scope |
|---|---:|---:|---|
| K1 to K2 | 6 / 6 / 0 | 5 / 6 / -1 | registered |
| K2 to K3 | 4 / 4 / 0 | 4 / 4 / 0 | registered |
| K3 to K4 | 18 / 18 / 0 | 23 / 15 / **+8** | registered |
| K4 to K5 | 19 / 21 / -2 | 16 / 21 / -5 | exploratory |
| K5 to K6 | 3 / 2 / +1 | 1 / 2 / -1 | exploratory |

The seed-1 K3-to-K4 gain is concentrated on GSM8K: `19` fixes and `11` regressions for `+8` net rows. Its other battery marginals are ARC-Challenge `0`, ARC-Easy `-1`, MBPP `0`, MMLU `+1`, and Tier-1 `0`. Seed 0's K3-to-K4 changes are balanced overall: `18` fixes and `18` regressions, with ARC-Challenge `+1`, ARC-Easy `-2`, GSM8K `-1`, MBPP `0`, MMLU `+2`, and Tier-1 `0`.

The registered indicator `registered_k4_marginal_positive_both_seeds` is therefore `false`. The result does not support a general statement that K=4 is underexploited. It also does not support a universal saturation statement, because seed 1 has a material late K=4 gain. The accurate reading is a seed-by-depth interaction, with depth-sensitive row churn in both seeds and net benefit only in seed 1.

The K=5/K=6 extension falls outside the learned support and is not a fair test of trained deeper recurrence. Its decline is nevertheless useful engineering evidence: simply repeating the final learned step parameters is not a free path to more performance.

## 8. Discrimination among E1, E2, and E3

### E2: amplitude limit

**Partially supported and the strongest measured lever.** Every tested ceiling through `0.11` is safe under the registered rule. Mean net rows rise from `+5.0` at `0.02` to `+8.0` at `0.05`, `+10.5` at `0.08`, and `+9.0` at `0.11`. The response is not monotonic within either seed and the apparent optimum differs by seed. Amplitude therefore explains some unused headroom, but a single wider ceiling does not reliably solve the task conversion problem.

The cleanest replicated task point is `0.05`, where both seeds score `+8`; that observation was not the registered selection rule and must not be retroactively promoted as the chosen ceiling. The registered structural selection remains `0.11` because it is the largest safe tested value. A future lock may name `0.05` as a new operating candidate if it explicitly prioritizes replicated task consistency, but that is a prospective decision.

### E3: depth underuse

**Not supported as a replicated global explanation.** Seed 0 receives zero net benefit from K=2, K=3, or K=4 relative to K=1. Seed 1 receives a late eight-row benefit at K=4, mostly on GSM8K. The both-seed K=4 criterion fails. Depth remains a possible workload- or seed-dependent lever, but it should not lead Stage 2A without a design that explains and controls this interaction.

### E1: knowledge deficit

**Still open.** Neither inference-time amplitude nor native depth produces a stable replicated effect large enough to close the gap. That leaves knowledge availability as a plausible limiting factor, but this wave did not measure whether the required answer information exists in the substrate state. E1 should stay framed as a hypothesis until a teacher-supported knowledge/access diagnostic directly tests it.

### Program implication

The most defensible Stage 2A opening is: the sidecar recipe installs a selective, safe causal write channel, but task conversion is limited by a heterogeneous combination of write amplitude, seed-specific use of late recurrent depth, and possibly missing or inaccessible knowledge. The data favor an amplitude-aware repair before a depth expansion. They do not justify treating more loops alone as the next main experiment.

## 9. Limitations and do-not-claim boundaries

- All scored results are on a reused 1,024-row DEV panel.
- The amplitude rule maps safety; it does not nominate an accuracy-optimal ceiling.
- Only two endpoint seeds are available.
- K=5 and 6 reuse step-4 parameters and are exploratory only.
- D1 future-prediction KL is unavailable under the pinned task graph.
- T1 did not run; missing registered checkpoints cannot be interpreted as failed state structure.
- Zero collateral is scoped to the registered audit population and tested amplitudes.
- No capability claim is made on CONFIRM or EVAL-E.

## 10. Questions for strategy

1. Should Stage 2A use `0.05` as a prospectively locked replicated-consistency operating point, retain the structurally selected `0.11`, or train across a declared amplitude distribution so the mechanism is not tied to one inference ceiling?
2. Should the seed-1 GSM8K-specific K=4 gain trigger a stratified depth study, or is the failed both-seed criterion sufficient to keep depth secondary until the amplitude/knowledge questions are resolved?
3. Should T1 be amended to a cross-seed step-4,000 endpoint fingerprint using the two available checkpoints, or deferred until a future run deliberately saves a late-window trajectory? Recreating the historical P3.4 window would require new training and is not justified by the current receipt alone.
4. What direct diagnostic should arbitrate E1 next: teacher-token support in the frozen substrate state, retrieval from the canonical state, or a bounded teacher-state side channel? The diagnostic should distinguish absent knowledge from failed read/write use without another broad training campaign.
5. Should the next registered training recipe explicitly optimize robustness across ceilings and seeds, rather than evaluating a single ceiling after training?

## 11. Plain-language summary

We tested two simple explanations for why the mechanism helps only a little. First, we let it write more strongly. Stronger writes were safe at every tested level and usually improved the average result, but the best strength differed between the two runs. One run gained fifteen answers at a ceiling of `0.08`; the other peaked at eight answers at `0.05`. This says write strength matters, but there is no single proven best setting yet.

Second, we changed how many recurrent thinking steps the fixed model used. In one run, accuracy was exactly unchanged from one through four steps, even though different questions were fixed and broken. In the other run, the fourth step added eight answers, mostly on GSM8K. Because that benefit did not repeat across both runs, we cannot say that more recurrent depth is the general solution. Extending to five and six steps by reusing the last learned step made both runs worse, although those two points are exploratory because the model was not trained for them.

The practical conclusion is that amplitude is the strongest next lever, but it needs a recipe that is robust across seeds rather than a hand-picked ceiling. Extra depth is secondary and may matter for particular workloads. The possibility that the small model simply lacks some needed knowledge remains open and now needs a direct test.

## 12. Canonical artifacts

- Amplitude summary: `stage5_paper2_phase3_p35_amplitude_t1_20260816/receipts/summary.json`, SHA-256 `829c5f36b3ecd9e3e16eb90301bbf5ad28e571ae3a9fdfc54e880bd6bdfbac49`.
- T1 manifest: `stage5_paper2_phase3_p35_amplitude_t1_20260816/receipts/t1_extraction_manifest_preflight.json`, SHA-256 `336d5609589ac042888b5d944786e12a5e060bcf3754eaafa6440f488fb4c576`.
- K+ summary: `stage5_paper2_phase3_p35_depth_discrimination_20260816/receipts/summary.json`, SHA-256 `0a5106adc489663f8a94b3d4ebd31c8495907808eca760f691f58ec9951b0c77`.
- K+ status: `stage5_paper2_phase3_p35_depth_discrimination_20260816/receipts/status.json`, SHA-256 `47f094c76b0cfecb87e2026cc4b56d4721e53deba8fc40cf8b151afc2f2e6d11`.
- Diagnostic figure: `docs/figures/paper2_p35_diagnostic_wave_20260816.png`, SHA-256 `2b732b3256691dba160432ffdb07c893c493d0e70309bfc0f1cbd7f577f4f58f`.
- Diagnostic figure: `docs/figures/paper2_p35_diagnostic_wave_20260816.svg`, SHA-256 `1287882ded053b5193f4547de20de0e52ebfdd6be35d8d63f683cfa28203a22d`.
- This handoff: `docs/PAPER2_P35_DEPTH_DISCRIMINATION_HANDOFF_20260816.md`; its final SHA-256 and Drive ID are recorded in the companion upload receipt.
