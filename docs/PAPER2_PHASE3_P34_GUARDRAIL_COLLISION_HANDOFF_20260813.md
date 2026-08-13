# Handoff: P3.4 Tier-S and Tier-W Event Collision

Date: 2026-08-13. Audience: strategy and research review. Status: pre-training implementation audit; zero P3.4 optimizer steps.

## 0. Executive finding

P3.4 cannot launch under the executed lock as written because Tier-S and Tier-W are defined by the same operational event:

> one-sided 90% paired upper confidence bound below -3 points on two consecutive looks.

Tier-W is supposed to demote the controller and continue. Tier-S is supposed to stop the campaign. With identical predicates, every Tier-W demotion event is simultaneously a Tier-S stop. The registered three-tier behavior is therefore not executable.

This is a specification collision, not a modeling result and not an implementation discretion. Training remains disabled until strategy binds distinct operational predicates or explicitly retires one consequence.

## 1. How the collision arose

The calibration correctly searched for the smallest sustained true drop detected with at least 99% probability while controlling familywise false stops. At the conservative autocorrelation edge, that detectable effect was 5.5 points. However, `delta_cat = 0.055` describes the operating characteristic of the -3-point decision rule; it is not itself the rule's decision threshold.

The executed lock records both facts:

- Tier-S decision margin: `-0.03`;
- Tier-S `delta_cat`: `0.055`;
- Tier-W drop class and decision margin: `-0.03`;
- both use one-sided alpha `0.10` and two consecutive looks.

The runner initially implemented those fields literally. A direct source and unit-test audit then showed `tier_s_condition == tier_w_condition` for every possible panel result.

## 2. Quantified repair sensitivity

The existing simulator was rerun without touching model data or checkpoints: 1,024 rows, 20 looks, paired discordance `0.093099`, conservative adjacent-look correlation `0.887550`, one-sided alpha `0.10`, and 100,000 campaigns per condition.

| Operational paired-UCB margin | True sustained drop | Action probability | Upper-95 probability |
|---:|---:|---:|---:|
| -3.0 points | 0.0 points | 0.001% | 0.0047% |
| -3.0 points | 3.0 points | 30.659% | 30.900% |
| -3.0 points | 5.0 points | 98.607% | 98.667% |
| -3.0 points | 5.5 points | 99.811% | 99.833% |
| -5.5 points | 0.0 points | 0.000% | 0.0030% |
| -5.5 points | 3.0 points | 0.014% | 0.0219% |
| -5.5 points | 5.0 points | 12.613% | 12.787% |
| -5.5 points | 5.5 points | 30.658% | 30.899% |
| -5.5 points | 7.0 points | 93.862% | 93.986% |
| -5.5 points | 8.0 points | 99.858% | 99.877% |

Moving Tier-S's operational margin to -5.5 points creates a clean hierarchy and an even lower false-stop rate, but it changes the detectable catastrophe class: at least 99% power moves from about 5.5 points to about 8 points. It would be inaccurate to retain the old 5.5-point power statement after this repair.

## 3. Decision options

### Option A: distinct paired-UCB margins

- Tier-W: upper bound below -3 points twice -> demote and flag.
- Tier-S: upper bound below -5.5 points twice -> stop.
- Amend Tier-S's registered sensitivity: 30.7% at a true 5.5-point drop, 93.9% at 7 points, and 99.9% at 8 points under the conservative calibration.

This is the smallest implementation change and gives the intended consequence hierarchy. Its cost is reduced catastrophe-stop sensitivity.

### Option B: retain the -3-point predicate as Tier-S only

- Tier-S remains the calibrated stop with 99.8% power at a true 5.5-point drop.
- Tier-W is retired for this campaign because it has no distinct event.

This preserves the calibrated hard floor but removes automatic controller demotion. Ordinary controller transitions and all telemetry remain available.

### Option C: design and calibrate a new nested Tier-S estimator

For example, retain Tier-W's -3-point paired-UCB event and add a separate severity condition for Tier-S. This could preserve more power than Option A, but it is a new sequential estimator and must be simulated on the exact 20-look schedule before launch. It should not be improvised in the runner.

## 4. Coding recommendation

Option A is the cleanest three-tier implementation if strategy accepts an 8-point high-power catastrophe class. Option B is preferable if preserving the already-certified 5.5-point stop sensitivity matters more than automatic demotion. The coding lane does not recommend inventing Option C under launch pressure.

Whichever option is selected, the lock JSON, executed-lock prose, runner predicate, unit tests, and handoff language must change together. The final receipt must state the operational threshold separately from the true-drop class at which power was measured.

## 5. Separate implementation correction completed during the audit

The charter permits one sampled flow depth per batch with probability proportional to depth. The first runner draft applied that distribution to the main arms but forced the slot arm to depth 4, which introduced an unregistered second variable into the paired seed-0 comparison. The corrected implementation uses the same dedicated RNG seed and the same depth distribution `[0.10, 0.20, 0.30, 0.40]` for main seed 0 and slot seed 0. Slot deep supervision covers the executed states only, retaining weights `j/4` and normalization over the executed set.

This correction changes no guardrail and runs no training. It restores the registered comparison: the slot loss and its lift are the only arm-level differences.

## 6. Required ruling

Strategy should bind one of Options A-C, including exact consequence language and any revised power claim. P3.4 remains ready otherwise, but no Colab campaign should launch until that ruling is mirrored into the machine lock and tests.

