# Strategy Ruling — P3.4 Share-Contract Enforcement Cadence, Confirmed (r2)

Date: 2026-08-13. Supersedes Drive `1Evlf9999rXmZLw17EHX08McgfRxKm8Sa` (r1 wrote "warn" at 2 windows; this r2 matches the implementation's exact proposed consequence). Ratified with Mark's authority under the approved lock (Drive `1XNXZdovLLbxGUeZOJZ5yApLBH3s12-Cb`) and the governance-tiering record (Drive `1sTUmvJl9-zNRq6P0K2B7JfpRK592DG6y`):

**Per-loss share contract, strict trailing-window training estimator: 2 consecutive breach windows → controller demotes one rung and flags for strategy review; 4 consecutive breach windows → stop.**

The cadence (2/4) confirms the B6 binding verbatim (Drive `1mLkbVZYhKyoiOuiOwdbjT9_nwA3Z8eYI`). The 2-window consequence is accepted as *demote* rather than B6's *warn* — a strengthening entirely inside the reversible tier, consistent with Tier-W semantics and the non-monotonicity clause (rungs are re-earned, transients are ridden through, nothing stops). The resume-defect fixes (gate-ceiling restoration, full checkpoint and evaluator-progress recovery, commit `9ca919bf`) are banked as reported — exactly the resumability the doctrine mandates.

**Update the lock and launch all three sessions.**