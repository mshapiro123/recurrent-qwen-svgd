# Addendum: A Nested P3.4 Guardrail Repair Preserves the Registered Power Class

Date: 2026-08-13. Amends the options analysis in `PAPER2_PHASE3_P34_GUARDRAIL_COLLISION_HANDOFF_20260813.md`. Status: CPU-only calibration; zero optimizer steps and no sealed-partition contact.

## 0. Result

The collision can be repaired without weakening Tier-S from the registered 5.5-point catastrophe class. Use the same paired upper-confidence-bound breach at both tiers but distinguish them by persistence:

- **Tier-W:** upper bound below -3 points for two consecutive looks -> demote one controller rung and flag review.
- **Tier-S:** upper bound below -3 points for four consecutive looks -> stop.

The Tier-S event is a strict subset of Tier-W. A persistent degradation first triggers the reversible response. It stops only if it survives the demotion and remains present for two additional evaluations. This matches the intended warning-versus-catastrophe semantics more closely than two unrelated magnitude thresholds.

At the executed lock's conservative edge, the four-look Tier-S rule has:

- familywise null upper-95 probability: `2.996e-5`, below the registered `1e-4` ceiling;
- power at a sustained 5.5-point true drop: `0.99036`, above the registered `0.99` floor;
- power at a sustained 5-point drop: `0.95138`;
- action probability at a sustained 3-point drop: `0.16141`.

Four looks is data-selected only within a pre-run engineering repair search: three and four both meet the 5.5-point power floor, while five does not (`0.98339` in the preliminary sensitivity and below `0.99` in the canonical receipt). Four is therefore the longest tested persistence rule that retains the registered catastrophe sensitivity, maximizing separation from Tier-W without changing the power class.

## 1. Why this is preferable to the earlier options

The initial Option A moved Tier-S's operational UCB margin to -5.5 points. That created a hierarchy but moved at least 99% power to an approximately 8-point true drop. The nested four-look rule avoids that loss.

The initial Option B retired Tier-W. That preserved Tier-S but discarded the controller's reversible response. The nested rule keeps both.

The addendum is the concrete version of initial Option C. It uses no new task estimator, no new panel, no new confidence level, and no new magnitude threshold. Only the registered persistence count differs by consequence.

## 2. Interpretation boundary

The simulation assumes a sustained degradation process. In the live controller, the Tier-W demotion after the second breach may improve the model before a fourth breach occurs. That is desirable: a recovered trajectory should not stop. The `0.99036` power number therefore describes a sustained drop that remains after the reversible intervention, not a transient pre-demotion drop.

The rule still requires explicit strategy ratification because persistence is part of the strict stop estimator. The coding lane has not silently changed the machine lock or runner.

## 3. Reproducibility

Command:

```text
python -m eval.eval_paper2_phase3_p34_guardrail_collision --lock training/paper2_phase3_p34_preregistration.json --output outputs/stage5/stage5_paper2_phase3_p34_guardrail_collision_20260813/summary.json --campaigns 100000 --seed 20260813
```

Receipt:

- path: `outputs/stage5/stage5_paper2_phase3_p34_guardrail_collision_20260813/summary.json`;
- SHA-256: `344426ffd0fbba57cdbe58a6c6c976e543ee9dddac106b1b5aa0109331159249`;
- Drive: `1MHiEB_apaWaC9r8-_Rwlp5OQwtnMBaK7`;
- campaigns: 100,000 per condition;
- rows: 1,024;
- looks: 20;
- paired discordance: `0.0930989583`;
- conservative autocorrelation: `0.8875501005`;
- one-sided alpha: `0.10`;
- sealed CONFIRM/EVAL-E scored: false;
- optimizer steps: zero.

Figure:

- `docs/figures/p34_guardrail_nested_rule_sensitivity_20260813.svg`, SHA-256 `fc79eb2c1e15894a53c703adc34fa729592a81ada8ca705e2113ff5faabaa846`;
- SVG Drive: `1z9BHoCvxNXZLF-v87whJjSOmQgc-oRw2`;
- PNG companion, SHA-256 `0d9a79e544808a68072dfde573c4705d0fdcda12f096c66c336b76dee1dd6e3f`, Drive `1EJivoLASjjkG2lt0jfYnf7RXqOJh-Bgv`.

## 4. Requested ruling

Ratify the nested two-look/four-look rule, or reject it with a specific alternative. On ratification, the coding lane will update together:

1. `tier_s_consecutive_looks` from 2 to 4 in the lock JSON;
2. the executed-lock prose and power statement;
3. the runner's independent Tier-S and Tier-W streak counters;
4. resume-state serialization for both counters;
5. unit tests proving Tier-S is nested within Tier-W;
6. the launch commit SHA.

No other campaign constant changes.
