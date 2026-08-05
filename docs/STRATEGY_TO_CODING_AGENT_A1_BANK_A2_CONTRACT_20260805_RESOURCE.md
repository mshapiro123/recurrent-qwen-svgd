# Strategy Resource: A1 Bank and A2 Calibration Authorization

Date: 2026-08-05

- Title: `STRATEGY_TO_CODING_AGENT_A1_BANK_A2_CONTRACT_20260805.md`
- Drive ID: `1CCIZqKgIvaveFit8IEOzcXfEcf-4YYWZ`
- Drive size: `9,181` bytes
- Decision: `BANK_A1_AND_AUTHORIZE_A2_CALIBRATION`

The strategy decision banks the two seed-specific A1 candidate passes as one
replicated `a1_pass` at alpha `0.5`, forbids an A1 extension, and authorizes
only a zero-update A2 calibration on both banked checkpoints. A2 training and
the two matched draft-head-only controls remain closed until the calibration
receipts are folded into a committed amendment lock.

The prior `35/35/10/20` shares are initialization targets only. The future A2
contract is directional: cumulative KL plus local CE jointly retain at least
50% of trainable-path gradient share; no individual non-primary loss exceeds
25%; preserve KL is descriptive and guarded by endpoint quality; and clipping
is a catastrophe tripwire derived from the A2 calibration rather than an
active shaper. No threshold in this resource authorizes optimizer updates.
