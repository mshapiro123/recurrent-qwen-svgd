# Guardrail Doctrine — How and Why Rules Are Written in This Program

Date: 2026-08-06. Origin: Mark's review of the step-237 resolution — "we've created rules where thoughtfulness is required. By trying to avoid wasting GPU time, we're wasting GPU and LLM time repeatedly crashing into guardrails that are well intentioned but not well implemented. Guards exist to keep us from falling off a cliff, not to keep us from gathering data that will inform our research." This document is the consolidated answer: one purpose test that generates the design rules, replacing the accumulation of case-by-case principles. It binds on every future protocol, amendment, and runner; existing rules are grandfathered only after passing the section-3 sweep.

## 1. The purpose test (the keystone — everything else follows from it)

**A guardrail exists to prevent a cliff: an irrecoverable loss.** Cliffs in this program are exactly four: corrupted frozen lineage (the pretrained model or a frozen artifact mutates), invalidated science (frozen-slice contact, silently changed objective, broken preregistration), garbage training (non-finite state propagating through a budget), and irreversible quality damage escaping into a deliverable claim.

**A guardrail must never exist to prevent gathering data.** If a rule fires and the run it stopped would have produced informative, attributable measurements — even measurements of something going wrong — the rule destroyed data; it did not protect anything. A failing trajectory that is measurable is not a cliff: it is the experiment working.

The test, applied at rule-writing time: *name the cliff.* State, in one sentence, the irrecoverable loss this rule prevents. If the sentence cannot be written, the rule is telemetry. If the sentence describes a recoverable cost (some wasted steps, an ugly curve, a metric below aspiration), the rule is telemetry with an alarm, or at most a warning. Stop authority is earned by a named cliff, never by good intentions.

## 2. The design rules (all derivable from the purpose test; kept explicit for use)

1. **Default disposition is observe-and-log.** A proposed rule enters as telemetry and is promoted to warning, then stop, only with its cliff named and its constants measured. Demotion never requires ceremony; promotion always does.
2. **Tripwires, not shapers, in exploration.** Continuous penalties and caps that redirect what the optimizer can learn may shape only after their constants are empirically grounded on the population they police. Exploration runs measure; confirmation runs constrain.
3. **Name the estimator with the threshold.** Population, sample size, and cadence live in the same clause as the number. A verdict rendered by a different estimator than the one that defined the contract is void.
4. **Ground in-flight rules to the trajectory, endpoint rules to the endpoint.** An in-flight stop asks "is this run degrading from where it started"; a qualification bar asks "did it arrive" — and only at arrival. Arming an endpoint bar mid-flight prosecutes healing.
5. **Reference the population the rule polices.** Rolling statistics for nonstationary quantities; step-zero snapshots ground initializations only and are stale by construction as persistent references.
6. **Weigh the false-stop tax.** A rule's expected cost includes every healthy run it will halt — GPU, LLM cycles, amendment rounds, attention. A rule that fires on healthy runs more often than on cliffs is net-negative regardless of intent, and the four stops of the A1/A2 campaign are the measured evidence: every one halted a healthy run; none caught a cliff.
7. **Every stop must be resumable.** Checkpoint, optimizer state, and generator state preserved at the stop, so a false stop costs a pause, not a restart. This is the one property the four stops got right, and it is why their tax was days rather than weeks.

## 3. The audit obligation

Any runner that arms rules emits its **rule inventory** into the receipt: each rule's threshold, estimator, reference point, cadence, disposition (log / warn / stop), and named cliff. A rule without a named cliff in the inventory is automatically demoted to telemetry at launch. Protocol reviews check the inventory against this doctrine as a standing agenda item — one sweep per protocol, not one stop per rule.

## 4. What this doctrine is not

It is not a loosening. The four cliffs keep absolute, immediate, non-negotiable stops — non-finite state, lineage mutation, frozen-slice contact, and the registered quality floor are exactly as hard as they ever were. The doctrine moves everything *else* out of the stopping business and into the measuring business, which is where a research program's rules belong. The experiments exist to inform the research; the guards exist to make sure the experiments survive to do so.
