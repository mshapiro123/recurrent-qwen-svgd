# Strategy → Coding Agent — P3.5 Execution Handoff: Ratified, Locked, Go

Date: 2026-08-15. **Mark has ratified.** This handoff consolidates everything the coding agent needs to set the lock fields, launch P3.5, and run the parallel desk track. Governing chain: P3.5 charter response (Drive `1ZzWO3MzkFW5Ph0wAuF5r-ZpdYCzAEi6F`, SHA `3bf476f1…7c82`) → prerequisites response (`1RNbtzUx78urgXrsIpPp1I-rQUfbtOjru`, SHA `bc694c08…13e`) → Sidecar v2 pivot (`1hiL4Tv1s0NK2Ixb9BhC_1ZfjeDQ7llYK`) → thread tracker (`1OUgZg7mtAa8IXhtlodkekWAdMXA42DDQ`). Mark's ratification message of this date is the signature of record.

## 1. Immediate action: set the four lock fields

In the assembled P3.5 machine lock: `mark_ratified=true`, `locked_before_training=true`, `training_authorized=true`, `status=approved_for_training`. Record this handoff's Drive ID and SHA in the lock's authority block. Any material mismatch between the lock, the charter response, and the implementation stops before optimizer construction, per standing rule.

## 2. Training campaign (GPU)

**Arm S — stabilized landing, seeds 0 and 1 (the spine).** P3.4 A2 configuration with the eight ratified contracts: terminal LR decay over the final ~10 percent; controller parameters frozen and rung pinned during the landing window with counterfactual decisions logged; raw + EMA checkpoints through the landing window, **EMA primary** (declared, irreversible); every registered score read at the **pinned 0.02 ceiling**, 0.08 as secondary telemetry at the four audit looks only; per-token and per-row-minimum margin telemetry, battery- and class-stratified; exact row identities persisted at every look; causal audits on **cache v2 only** (`294358a7…2294`). Initialization per the lock's bound choice (re-landing from the post-audit resumable checkpoints or fresh retrain — whichever you priced and bound).

**Arm R — probe-pool reader A/B, seed 0.** Identical to Arm S seed 0 in every contract; the sole change is the control/gate reader: detached four-probe pooling (A = 4, stopgrad cells) replacing mean pooling, per the pivot document's `ProbePool` spec. Registered comparison against Arm S seed 0 on paired rows: gate precision at matched recall, π_dep under cache v2, per-row minimum margin lift, net rows.

**Carried forward unchanged:** dynamic log-weight share controller with its rung targets, share floors and stop rules, catastrophe tripwires, task inference graph (fresh scratchpad, K = 4, greedy, position zero closed, draft head inactive), sealed partitions untouched, sessions released promptly. No persistence anywhere in this campaign.

**Report-backs:** wave rule. Per look: net rows per battery per arm, margin summaries, churn vs prior look, controller state, π under cache v2 at audit looks. Endpoint reads land in the pre-committed branch tree (charter response §4): ≥ +10 stabilized mean → Branch A (P3.6 drafting); +8–10 → Branch B (margin tiebreaker, lever queue); < +8 → Branch C (effect size to Stage 2A).

## 3. Parallel desk track (no new authorization needed)

- **T1** — probe-pool vs mean-pool on cached P3.4 states: retrieval AUC, stability under the late-window checkpoint jitter set. Receipt due with the Stage 2A lock draft.
- **Stage 2A data spine** — teacher-functional fingerprints and expert clusters, prerequisite for T2/T3.
- **T3b addition** — the Stage 2A lock draft gains the literal n-gram Engram control: hash n-grams of existing prefix token IDs inside the sidecar, inject via the bridge, zero substrate modification. Runs beside T3 (concept-keyed) at matched budget.
- **`fast_wht` equivalence tests** — numerical identity against dense H matmul, both orientations, before any T2 work.

## 4. Standing boundaries

CONFIRM and EVAL-E sealed. No checkpoint or threshold selection from diagnostic reads. No claim language beyond exploratory DEV until the branch tree says otherwise. Code-review maintenance stays off the checkpoint-defining branch. The thread tracker is updated by strategy at each wave — flag anything that should move on it.

## 5. Plain-language summary

Everything is approved and the switches can be flipped. Two training runs launch: both seeds with the new gentle-landing procedure and the averaged official checkpoint, and a third run testing the smarter multi-probe reader against the current one on identical terms. All measurements use one fixed write-strength setting and the fully repaired measuring instrument. Meanwhile the desk work for the memory sidecar continues — including the newly added literal-lookup control experiment, which needs no changes to the base model. Results land in a decision tree written before anyone saw them.

---

**Go line:** set the four fields, run the preflights the lock requires, launch Arm S (seeds 0, 1) and Arm R (seed 0), report per wave. Good hunting.
