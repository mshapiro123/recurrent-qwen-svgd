# Coding to Strategy — WEFT-1 D-MC-1 Allocation Correction

**Date:** 2026-09-04
**Status:** fail-closed planning correction; no run removed, no compute consumed,
and no existing governed receipt modified
**Scope:** re-derive the D-MC-1 final-plus-one-sampled-coda cost against the
ratified D-CUR-4 Scenario C allocation

## 1. Authorities checked

| authority | identity | relevant binding |
|---|---|---|
| `STRATEGY_CURRICULUM_DECISIONS_20260828.md` | 8,354 bytes; SHA-256 `61fc7727e456d822f43613db602c0251344b64ea92c7b256af5f1fe560cd8b6d`; Drive `123Ar8LQNFVKUxFCCZYWvMT8dtk9ilC8T` | D-CUR-4 ratifies Scenario C as rung A 59 + rung B 70 + dense control 59 + S2 about 15 + G-TOK 6.2 + observatory 25 = about 234 A100-hours, **against a programme allowance of 196–306 A100-hours**. The same section binds rung-B-first de-scope only if tripwires fire or the effective budget proves nearer the floor than the ceiling. |
| `STRATEGY_MATH_CHECK_RATIFICATION_20260903.md` | 2,868 bytes; SHA-256 `9c5822daef5dbb0609bc3e46019cc4b1e332991c30e8a42c1b4432800a747ab1` | D-MC-1 adds one sampled earlier-state coda decode whenever `K_exec > 1`; at `K_exec = 1`, there is no earlier decode. Its planning table gives multipliers 1.323077 at K=2, 1.238636 at K=4, and 1.189189 at K=6 for the stated 9/4/9 approximation. |
| `STRATEGY_TO_CODING_AGENT_LOOM1_HANDOFF_20260826.md` | 61,329 bytes; SHA-256 `498f34b5966f0879c7f0a15ca8be02a603558781c35f59f03fb29cc9edd3eb02` | Target rungs are A=9/4/9 and B=8/6/8; rounded active-equivalent counts and per-block dense/core costs are separately reported. |
| `STRATEGY_TO_CODING_AGENT_CORPUS_GTOK_20260828.md` | 16,431 bytes; SHA-256 `2aecb64711a2bf2776c8d1940350bc5d42b335f60eb774ac1e941f470b9cf74c` | Repeats D-CUR-4's conditional de-scope order and preserves rung A plus the dense control. |

The full D-CUR source hashes are also encoded at
`training/weft1_gtok_contract.py:52-62`. The local evidence is
`STRATEGY_MATH_CHECK_RATIFICATION_20260903.md:7-9`,
`STRATEGY_TO_CODING_AGENT_LOOM1_HANDOFF_20260826.md:147-173`, and
`STRATEGY_TO_CODING_AGENT_CORPUS_GTOK_20260828.md:126-130`.

## 2. The 1.238636 multiplier is correct but local to one composition

The math-check approximation uses, in millions of active-equivalent parameters,

```text
N_prelude = 105
N_recurrent = 57.5
N_coda = 105
K = 4
```

Therefore

```text
old_AE_A = 105 + 4*57.5 + 105 = 440
new_AE_A = 105 + 4*57.5 + 2*105 = 545
multiplier_A = 545 / 440 = 1.238636363636...
```

and the ratified 59-hour rung-A row becomes

```text
59 * 545 / 440 = 73.079545 A100-hours.
```

This verifies the stated multiplier as a **planning approximation for rung A at
K=4**. It is not a universal multiplier for Scenario C. The 440 M and 545 M
inputs are rounded: using the handoff's separately rounded dense-block estimate
(`9 * 11.796 M = 106.164 M` of coda) would give 1.241282 instead. Exact reminting
must therefore wait for the integrated graph's exact counts, as the existing
Step-6 spec already requires.

## 3. Why `234 * 545/440 = 289.84` is not the all-in re-derivation

D-CUR-4 defines the baseline as a sum of unlike rows:

```text
rung A, K=4                 59.0
rung B, K=4                 70.0
dense control, K=1          59.0
S2 proxy calibration       ~15.0
G-TOK                        6.2
observatory                 25.0
                              ----
baseline Scenario C        234.2 ~= 234 A100-hours
```

Multiplying all 234 hours by 545/440 incorrectly does three things:

1. it prices the 8/6/8 rung B with the 9/4/9 rung A's coda fraction;
2. it adds a second coda decode to the K=1 dense control even though D-MC-1
   explicitly forbids that decode at K=1;
3. it uplifts G-TOK and observatory compute even though neither row is a
   D-MC-1 sampled-decode target-training row.

It also calls 234 the “allowance.” D-CUR-4 instead calls 234 the Scenario C
**commitment/estimate** and 196–306 the **allowance**. The arithmetic value

```text
234 * 545 / 440 = 289.840909...
```

is numerically reproducible but does not estimate the ratified mixture of runs.

## 4. Component-wise planning estimate

### Rung A

As above: **73.079545 hours**.

### Rung B

Rung B has eight coda blocks, not nine. Using the handoff's rounded dense-block
cost gives

```text
N_coda_B ~= 8 * 11.796 M = 94.368 M
multiplier_B ~= (523 + 94.368) / 523 = 1.180435946...
hours_B ~= 70 * 1.180435946 = 82.630516.
```

As a cross-check using only the rounded active-equivalent columns,
`N_recurrent=(689-523)/(6-4)=83 M` and `N_fixed=523-4*83=191 M`.
Splitting that symmetric 8/6/8 outer stack equally gives `N_coda~=95.5 M`,
`multiplier_B~=1.182600`, and 82.782027 hours. The 0.1515-hour spread is the
expected consequence of rounded inputs and unisolated fixed overhead. It is not
receipt-grade uncertainty; the integrated count will replace it.

### Dense control, G-TOK, and observatory

- Dense control: **59 hours unchanged**, because `K_exec=1` has one coda decode.
- G-TOK: **6.2 hours unchanged**, because the registered screen is the
  structural-OFF S0 graph rather than the Step-6 sampled objective.
- Observatory: **25 hours unchanged** in this planning correction. D-MC-1 says
  inference decodes once, and no authority classifies the observatory allocation
  as sampled-coda training.

### S2

The D-CUR-4 record supplies only the aggregate `S2 ~= 15` hours. It does not
supply the hours by proxy topology, K, or training/evaluation cell. Therefore an
exact D-MC-1 uplift for S2 is **not mintable from the present ledger**.

Before S2 is scheduled, the receipt needs this literal table:

```text
cell_id | proxy topology | K_exec distribution | baseline A100-hours |
sampled objective active? | exact N_coda | exact old/new active-train count
```

Until that exists, two useful bounds are available:

- if S2's 15-hour line is unaffected, Scenario C is approximately
  `260.91–261.06` hours (the interval is the rung-B rounded-count spread);
- conservatively applying D-MC-1's largest stated multiplier, `430/325 =
  1.323077`, to **all** 15 S2 hours and also using the rung-A multiplier as an
  intentionally loose upper bound for rung B yields:

```text
73.079545 + (70*545/440) + 59 + (15*430/325) + 6.2 + 25
= 269.830245 A100-hours.
```

That deliberately pessimistic bound still leaves

```text
306 - 269.830245 = 36.169755 A100-hours
```

of headroom to the ratified ceiling. Using the rung-B composition rather than
the loose bound gives approximately 265.76–265.91 hours, about 40.1 hours below
the ceiling.

## 5. Disposition

1. **The old 234-hour estimate is exceeded.** The currently supportable
   component-wise planning range is about 261–266 hours, with a conservative
   fail-closed upper bound of 269.83 hours.
2. **The 306-hour programme ceiling is not exceeded.** The current evidence
   proves at least 36.16 hours of ceiling headroom even under the conservative
   bound.
3. **Rung-B de-scope does not fire from this arithmetic alone.** It fires if a
   tripwire fires or the effective available budget resolves near the 196-hour
   floor. No source inspected resolves the 196–306 allowance to one effective
   number. No run is removed locally.
4. **The existing Step-6 allocation paragraph needs supersession, not silent
   editing.** `WEFT1_STEP6_OBJECTIVE_AND_SAMPLED_DECODE_SPEC_20260903.md:89-96`
   should remain provenance for the caught calculation; a strategy amendment
   can adopt this component-wise correction.
5. **Exact remint remains gated on two inputs:** the integrated Step-3/4/5/6
   graph's exact `N_coda` and active-train counts, plus the S2 cell allocation
   table above.

## 6. Accounting implementation consequence

The current composition receipt records `coda_decodes_per_step`, but
`models/ablation_lm/accounting.py:569-572` still computes only
`N_fixed + executed_visits*N_recurrent`. That remains a valid inference-style
`N_active_eval`, but it cannot price D-MC-1 training.

Before Step 6 can mint an exact allocation receipt, add a distinct training
quantity rather than changing the meaning of `N_active_eval`, for example:

```text
N_active_train = N_fixed
               + K_exec * N_recurrent
               + (coda_decodes_per_step - 1) * N_coda
```

with exact tied-parameter partition checks for `N_coda`. This is a build-axis
accounting requirement, not authorization to start S2 or target training.
