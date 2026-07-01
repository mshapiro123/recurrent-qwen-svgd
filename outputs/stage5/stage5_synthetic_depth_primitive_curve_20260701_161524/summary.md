# Synthetic Depth Primitive Curve

- Primitive bar: `0.71`
- Strong bar: `0.9`
- Largest N clearing primitive bar: `16`
- Largest N clearing strong bar: `16`
- Recommended Phase 2 N: `16`

| N | Base acc | Recurrent acc | Clears 0.71 | Clears 0.90 |
|---:|---:|---:|:---:|:---:|
| 8 | 0.203 | 0.984 | yes | yes |
| 12 | 0.277 | 0.973 | yes | yes |
| 16 | 0.227 | 0.961 | yes | yes |

Proceed to staged-depth forced-loop staircase at the largest N whose depth-1 primitive accuracy clears the primitive bar; prefer the largest N clearing the strong bar if available.