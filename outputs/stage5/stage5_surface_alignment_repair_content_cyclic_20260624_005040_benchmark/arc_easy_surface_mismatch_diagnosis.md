# MCQ Surface-Mismatch Diagnosis - arc_easy_stage5_surface_alignment_repair_content_cyclic_20260624_005040_benchmark

- Content delta: `-10` (138/256 vs 148/256)
- Candidate cyclic vs candidate content delta: `+67`
- Content losses: `15`
- Content losses rescued by cyclic: `13` (0.867)
- Stable cyclic rescues: `8` (0.533)
- Unrescued content losses: `2` (0.133)
- Order-sensitive content losses: `5` (0.333)
- Recommendation: `prioritize_content_cyclic_surface_alignment`

## Stable Cyclic Rescue Examples

| id | answer | content pred | cyclic pred | content answer rank | content answer margin | perm counts |
|---|---|---|---|---:|---:|---|
| `ACTAAP_2014_7_6` | `D` | `B` | `D` | 2 | -0.1426 | `{"D": 4}` |
| `MDSA_2011_4_8` | `A` | `C` | `A` | 2 | -0.0481 | `{"A": 4}` |
| `Mercury_183190` | `C` | `B` | `C` | 3 | -0.5905 | `{"C": 4}` |
| `Mercury_7010815` | `B` | `D` | `B` | 2 | -0.0739 | `{"B": 4}` |
| `Mercury_7017080` | `A` | `B` | `A` | 2 | -0.0293 | `{"A": 4}` |
| `Mercury_7194425` | `B` | `D` | `B` | 2 | -0.4370 | `{"B": 4}` |
| `Mercury_7212520` | `B` | `D` | `B` | 2 | -0.1756 | `{"B": 4}` |
| `NCEOGA_2013_5_28` | `C` | `A` | `C` | 2 | -0.5201 | `{"C": 4}` |

## Unrescued Content Loss Examples

| id | answer | content pred | cyclic pred | content answer rank | content answer margin |
|---|---|---|---|---:|---:|
| `MCAS_2013_5_17` | `C` | `A` | `D` | 2 | -0.3034 |
| `Mercury_400540` | `A` | `D` | `D` | 2 | -0.0395 |
