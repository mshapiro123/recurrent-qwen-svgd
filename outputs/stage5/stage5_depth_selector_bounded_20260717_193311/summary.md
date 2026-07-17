# Bounded Depth Selector Assessment - stage5_depth_selector_bounded_20260717_193311

- Status: `started`
- Source SHA: `None`
- Frozen mechanism hash: `None`
- Canary: `The frozen mechanism and all forced-T logits are cached before selector training. Only the halt projection, loop embedding, and loop bias are optimized; the selector chooses among immutable forced-T outputs and cannot alter computation at any fixed T.`

## Arms

| Arm | Status | Selection accuracy | Answer accuracy | Mean depth | Spearman |
|---|---|---:|---:|---:|---:|
| S1 | pending | 0.0000 | 0.0000 | 0.000 | 0.000 |
| S2 | pending | 0.0000 | 0.0000 | 0.000 | 0.000 |

S1 reads an explicitly stated depth. It is not evidence of difficulty inference.
Learned halting on held-out hard reasoning remains unestablished.
