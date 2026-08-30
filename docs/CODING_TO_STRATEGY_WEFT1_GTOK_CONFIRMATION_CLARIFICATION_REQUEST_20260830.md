# WEFT-1 G-TOK confirmation clarification request

**Date:** 2026-08-30
**Status:** strategy ruling required before any A100 base-screen run
**Data posture:** no G-TOK base or confirmation result exists; no production A100 run has launched

The P-C implementation reaches two ambiguities that cannot be bound as replay
literals without choosing scientific semantics. Both were found before observing
any arm result. The implementation remains fail-closed.

## C-1. A common exact whole-step FLOP budget is not generally reachable

The binding states:

> both seeds; common FLOP budget is the minimum measured base-run FLOPs across
> the top two arms and two seeds

The ratified optimizer and packing contract also requires fixed global batches,
ordinary whole optimizer steps, and only the natural terminal short batch. A
vocabulary-dependent output surface gives each arm a different physical FLOP
step lattice. Therefore the minimum complete base-run total from one arm is not
generally an exact prefix sum for the other arm. Exact equality cannot be
guaranteed without adding an unregistered operation such as an artificial
partial optimizer step, dummy compute, another corpus pass, or an arbitrarily
large common multiple.

Current safe behavior: reconstruct every physical base step from the profiler
and unsupported-operator ledger, write an exact reachability receipt, and STOP
before confirmation calibration if any of the four arm-seed rows cannot reach
the common budget exactly.

Recommended ruling: define `F*` as the registered minimum measured base-run
FLOPs; for every confirmation row use the largest whole-step physical prefix
`F_i <= F*`; require both `(F* - F_i) / F*` to be below a registered small cap
and the slack to be strictly smaller than that row's next physical step FLOPs.
Record `F*`, `F_i`, relative slack, and next-step quantum. Prohibit partial
optimizer steps, dummy padding, extra T passes, and checkpoints. A row outside
the cap stops the line. A suggested cap is `5e-4` (0.05%), subject to strategy's
pricing judgment.

## C-2. “Top two” is undefined after asymmetric-band selection

The governing handoff says that a larger vocabulary displaces a smaller one
only by winning by more than `3 s_hat`, followed by compute-matched confirmation
on “the top two arms.” It does not define whether “top” means raw terminal-BPB
order or the decision-rule outcome.

The distinction can change the experiment. The 3-sigma traversal can retain
16K or 24K while the two lowest raw BPBs belong to 48K and 32K. Under the current
literal implementation, confirmation would then omit the selected vocabulary,
yet the later V receipt could freeze it.

Current safe behavior requested for the stable scaffold: if the selected
vocabulary is absent from the confirmation pair, STOP before confirmation and
return to strategy. No V receipt may mint for an arm that the registered rule
required but the confirmation did not test.

Recommended ruling: confirmation uses the selected vocabulary plus the best
distinct raw-BPB alternative, with the selected vocabulary named first. The
ruling must also define “reversal” for the case where the asymmetric band chose
the smaller arm despite a slightly worse raw BPB. Alternatives are acceptable,
but raw-top-two must explicitly acknowledge that the selected arm can be
omitted.

## Requested response

Please bind, before any A100 base run:

1. the physically realizable compute-matching rule and tolerance, if any;
2. the exact construction and ordering of the two confirmation arms; and
3. the reversal rule when the asymmetric-band winner is not the raw-BPB winner.

P-A materialization, P-B mechanical gates, tokenizer fitting, and build-axis
work can continue because none consumes an A100 base result or depends on either
ruling. The expensive CPU precompute should wait: its receipt binds the complete
G-TOK code closure, so a strategy-driven confirmation edit would invalidate it
and force the corpus scans to repeat. The A100 base screen remains unlaunched
until these semantics are fixed.
