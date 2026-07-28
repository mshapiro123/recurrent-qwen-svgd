# Strategy Addendum: Composite Depth-by-Append Markup Round 1

Date: 2026-07-28. This amends
`STRATEGY_TO_CODING_AGENT_COMPOSITE_DEPTH_BY_APPEND_20260728.md`. It does not
authorize training.

## Adopted Amendments

1. Add a fifth diagnostic arm. After each raw feedback slot, issue a transient
   readout query carrying the original token embedding and original rotary
   position. This is the causal implementation of the requested read-at-t
   diagnostic: the query is computed after the feedback slot so it can attend
   to that slot. Receipts must call it `read_at_t_query`; they must not describe
   it as literal backward attention from an earlier cached query.
2. Scope the harm-asymmetry claim to the post-D0 checkpoint. DC0 holds the
   checkpoint fixed and cannot determine whether utility-trained D1 labels
   would shrink the asymmetry. Add the free pre-D0 floor `1 -> 2` decomposition
   beside the post-D0 audit result.
3. Report layer-application costs. At `L=1`, an extra in-place loop costs 12
   recurrent-layer applications, while one append slot costs 24 full-stack
   layer applications plus attention overhead. Therefore append `k=1` is
   matched in layer applications to in-place depth 3, not depth 2.
4. Extend M7 eviction checks. After every eviction, later real-token position
   IDs must match the source sequence, cache length must equal real tokens
   processed, and an append-then-evict probe must reproduce all later real-token
   logits within the RG-1 tolerance.

## Unchanged Boundaries

EVAL-B remains read-once. DC0 remains forward-only. The persistent scratchpad,
bridge adaptation, RG-12, L greater than one, and all training remain
unauthorized. A red M7 precondition stops scoring.
