# Multi-Channel Bridge Precursor - backward_recovery

- Status: `finished`
- Checkpoint: `/content/recurrent-qwen-svgd/outputs/stage5/stage5_multichannel_bridge_precursor_replication_20260714/restored/backward_recovery.pt`
- Checkpoint SHA256: `fc98feb5d5bd450f7ecc4f6d43ce36fd436418d7ad2cd69df38a089d5ec453d1`
- Rows: `8`
- Max loops: `8`
- Random controls: `20`
- Query-head write basis: `final recurrent layer o_proj query-head input-column blocks, independently orthonormalized`

## M1
- Confirmed: `False`
- Reading: `smeared`
- Classification: `{'confirmed': False, 'reading': 'smeared', 'eligible_loops': 3, 'qualifying_loops': 0, 'required_qualifying_loops': 3, 'advantage_ratio_by_loop': {'6': 1.374470748669372, '7': 1.3735167992273312, '8': 1.3692546301472577}, 'minimum_loop': 6, 'locked_advantage_ratio': 2.0, 'locked_consistency_fraction': 0.75, 'outside_random_p95_loops': 3, 'required_outside_random_p95_loops': 3}`

## M2
- Confirmed: `False`
- Reading: `retrieval_heads_not_established`
- Classification: `{'confirmed': False, 'reading': 'retrieval_heads_not_established', 'qualifying_heads': [], 'qualifying_head_count': 0, 'actual_concentration': 0.2721971273422241, 'random_concentration_p95': 0.6039222061634064, 'random_null_win': False, 'locked_minimum_ratio': 3.0, 'locked_minimum_stable_fraction': 0.5}`

## Decision
- Measurements confirmed: `0`
- Battery specialization criterion: `False`
- Staircase reading one: `False`
- Architecture activation eligible: `False`

The battery does not change the experiment queue. Activation requires both at least two positive measurements and a staircase reading of per-position installation cost.
