# MCQ Surface-Mismatch Diagnosis - arc_easy_direct_preserve_confirm_256

- Content delta: `-8` (140/256 vs 148/256)
- Candidate cyclic vs candidate content delta: `+63`
- Content losses: `16`
- Content losses rescued by cyclic: `14` (0.875)
- Stable cyclic rescues: `8` (0.500)
- Unrescued content losses: `2` (0.125)
- Order-sensitive content losses: `6` (0.375)
- Recommendation: `prioritize_content_cyclic_surface_alignment`

## Stable Cyclic Rescue Examples

| id | answer | content pred | cyclic pred | content answer rank | content answer margin | perm counts |
|---|---|---|---|---:|---:|---|
| `ACTAAP_2014_7_6` | `D` | `B` | `D` | 2 | -0.1814 | `{"D": 4}` |
| `MDSA_2011_4_8` | `A` | `C` | `A` | 2 | -0.0893 | `{"A": 4}` |
| `Mercury_183190` | `C` | `B` | `C` | 3 | -0.6495 | `{"C": 4}` |
| `Mercury_7010815` | `B` | `D` | `B` | 2 | -0.0378 | `{"B": 4}` |
| `Mercury_7017080` | `A` | `B` | `A` | 2 | -0.0360 | `{"A": 4}` |
| `Mercury_7194425` | `B` | `D` | `B` | 2 | -0.4069 | `{"B": 4}` |
| `Mercury_7212520` | `B` | `D` | `B` | 2 | -0.3887 | `{"B": 4}` |
| `NCEOGA_2013_5_28` | `C` | `A` | `C` | 2 | -0.6119 | `{"C": 4}` |

## Unrescued Content Loss Examples

| id | answer | content pred | cyclic pred | content answer rank | content answer margin |
|---|---|---|---|---:|---:|
| `MCAS_2013_5_17` | `C` | `A` | `D` | 2 | -0.1082 |
| `Mercury_400540` | `A` | `D` | `D` | 2 | -0.0350 |
