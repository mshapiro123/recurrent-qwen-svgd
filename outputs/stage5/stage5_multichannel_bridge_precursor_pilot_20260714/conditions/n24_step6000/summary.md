# Multi-Channel Bridge Precursor - n24_step6000

- Status: `finished`
- Checkpoint: `/content/recurrent-qwen-svgd/outputs/stage5/stage5_multichannel_bridge_precursor_pilot_20260714/restored/n24_step6000.pt`
- Checkpoint SHA256: `898a259db2ab344ece4545e2910b051840e8408dbe4927f799e7cdb3cdd8c7dc`
- Rows: `14`
- Max loops: `14`
- Random controls: `20`
- Query-head write basis: `final recurrent layer o_proj query-head input-column blocks, independently orthonormalized`

## M1
- Confirmed: `False`
- Reading: `smeared`
- Classification: `{'confirmed': False, 'reading': 'smeared', 'eligible_loops': 9, 'qualifying_loops': 0, 'required_qualifying_loops': 7, 'advantage_ratio_by_loop': {'6': 1.370408699857283, '7': 1.3837667533368188, '8': 1.3932222046533962, '9': 1.3995508473702754, '10': 1.404983225254535, '11': 1.409804546414693, '12': 1.4126455641860387, '13': 1.41439730443112, '14': 1.4189167207790645}, 'minimum_loop': 6, 'locked_advantage_ratio': 2.0, 'locked_consistency_fraction': 0.75, 'outside_random_p95_loops': 9, 'required_outside_random_p95_loops': 7}`

## M2
- Confirmed: `True`
- Reading: `retrieval_heads_exist`
- Classification: `{'confirmed': True, 'reading': 'retrieval_heads_exist', 'qualifying_heads': ['L0H3', 'L11H0', 'L11H8', 'L1H13', 'L1H7', 'L2H0', 'L2H1', 'L2H3', 'L3H1', 'L3H12', 'L3H3', 'L3H4', 'L3H6', 'L4H1', 'L4H13', 'L4H9', 'L5H1', 'L5H2', 'L5H3', 'L5H4', 'L5H6', 'L6H1', 'L6H4', 'L6H5', 'L7H13', 'L7H8', 'L7H9', 'L8H12', 'L8H13', 'L8H3', 'L8H5', 'L8H9', 'L9H12', 'L9H13', 'L9H6', 'L9H7', 'L9H9'], 'qualifying_head_count': 37, 'actual_concentration': 0.7404117584228516, 'random_concentration_p95': 0.5228730320930481, 'random_null_win': True, 'locked_minimum_ratio': 3.0, 'locked_minimum_stable_fraction': 0.5}`

## Decision
- Measurements confirmed: `1`
- Battery specialization criterion: `False`
- Staircase reading one: `False`
- Architecture activation eligible: `False`

The battery does not change the experiment queue. Activation requires both at least two positive measurements and a staircase reading of per-position installation cost.
