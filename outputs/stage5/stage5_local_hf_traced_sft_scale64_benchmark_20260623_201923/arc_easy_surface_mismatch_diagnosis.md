# MCQ Surface-Mismatch Diagnosis - arc_easy_scale64

- Content delta: `-7` (68/128 vs 75/128)
- Candidate cyclic vs candidate content delta: `+32`
- Content losses: `10`
- Content losses rescued by cyclic: `8` (0.800)
- Stable cyclic rescues: `8` (0.800)
- Unrescued content losses: `2` (0.200)
- Order-sensitive content losses: `0` (0.000)
- Recommendation: `prioritize_content_cyclic_surface_alignment`

## Stable Cyclic Rescue Examples

| id | answer | content pred | cyclic pred | content answer rank | content answer margin | perm counts |
|---|---|---|---|---:|---:|---|
| `ACTAAP_2014_7_6` | `D` | `B` | `D` | 2 | -0.1730 | `{"D": 4}` |
| `Mercury_183190` | `C` | `B` | `C` | 3 | -0.9438 | `{"C": 4}` |
| `Mercury_7017080` | `A` | `B` | `A` | 2 | -0.0959 | `{"A": 4}` |
| `Mercury_7163940` | `A` | `D` | `A` | 2 | -0.5659 | `{"A": 4}` |
| `Mercury_7164658` | `B` | `C` | `B` | 2 | -0.3082 | `{"B": 4}` |
| `Mercury_7194425` | `B` | `D` | `B` | 2 | -0.1660 | `{"B": 4}` |
| `NCEOGA_2013_5_28` | `C` | `A` | `C` | 2 | -0.6062 | `{"C": 4}` |
| `NYSEDREGENTS_2014_8_25` | `C` | `A` | `C` | 2 | -0.4786 | `{"C": 4}` |

## Unrescued Content Loss Examples

| id | answer | content pred | cyclic pred | content answer rank | content answer margin |
|---|---|---|---|---:|---:|
| `MCAS_2013_5_17` | `C` | `A` | `D` | 2 | -0.4097 |
| `Mercury_400540` | `A` | `D` | `D` | 2 | -0.0263 |
