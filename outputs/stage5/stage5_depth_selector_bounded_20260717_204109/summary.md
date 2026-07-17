# Bounded Depth Selector Assessment - stage5_depth_selector_bounded_20260717_204109

- Status: `S1_blocked`
- Source SHA: `898a259db2ab344ece4545e2910b051840e8408dbe4927f799e7cdb3cdd8c7dc`
- Frozen mechanism hash: `eb4c19ca7871ca1563b5bc2f3229fee8ee664907048aedeb2028fa48670f2d9b`
- Canary: `The frozen mechanism and all forced-T logits are cached before selector training. Only the halt projection, loop embedding, and loop bias are optimized; the selector chooses among immutable forced-T outputs and cannot alter computation at any fixed T.`

## Arms

| Arm | Status | Selection accuracy | Answer accuracy | Mean depth | Spearman |
|---|---|---:|---:|---:|---:|
| S1 | blocked | 0.0911 | 0.0951 | 2.620 | 0.000 |
| S2 | pending | 0.0000 | 0.0000 | 0.000 | 0.000 |

S1 reads an explicitly stated depth. It is not evidence of difficulty inference.
Learned halting on held-out hard reasoning remains unestablished.
