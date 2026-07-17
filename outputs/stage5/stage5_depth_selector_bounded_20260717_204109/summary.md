# Bounded Depth Selector Assessment - stage5_depth_selector_bounded_20260717_204109

- Status: `finished_blocked`
- Source SHA: `898a259db2ab344ece4545e2910b051840e8408dbe4927f799e7cdb3cdd8c7dc`
- Frozen mechanism hash: `839955196679b919820ef3e97f13d88589b63c4c634f174ac48a11295e6e7a71`
- Canary: `The frozen mechanism and all forced-T logits are cached before selector training. Only the halt projection, loop embedding, and loop bias are optimized; the selector chooses among immutable forced-T outputs and cannot alter computation at any fixed T.`

## Arms

| Arm | Status | Selection accuracy | Answer accuracy | Mean depth | Spearman |
|---|---|---:|---:|---:|---:|
| S1 | blocked | 0.0911 | 0.0951 | 2.620 | 0.000 |
| S2 | collapse | 0.0833 | 0.2253 | 12.000 | 0.000 |

S1 reads an explicitly stated depth. It is not evidence of difficulty inference.
Learned halting on held-out hard reasoning remains unestablished.
