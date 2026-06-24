# MCQ Surface-Mismatch Diagnosis - arc_easy_stage5_surface_alignment_repair_content_cyclic_20260624_010142_benchmark

- Content delta: `-8` (140/256 vs 148/256)
- Candidate cyclic vs candidate content delta: `+66`
- Content losses: `14`
- Content losses rescued by cyclic: `12` (0.857)
- Stable cyclic rescues: `7` (0.500)
- Unrescued content losses: `2` (0.143)
- Order-sensitive content losses: `5` (0.357)
- Recommendation: `prioritize_content_cyclic_surface_alignment`

## Stable Cyclic Rescue Examples

| id | answer | content pred | cyclic pred | content answer rank | content answer margin | perm counts |
|---|---|---|---|---:|---:|---|
| `ACTAAP_2014_7_6` | `D` | `B` | `D` | 2 | -0.1325 | `{"D": 4}` |
| `MDSA_2011_4_8` | `A` | `C` | `A` | 2 | -0.0462 | `{"A": 4}` |
| `Mercury_183190` | `C` | `A` | `C` | 3 | -0.5307 | `{"C": 4}` |
| `Mercury_7010815` | `B` | `D` | `B` | 2 | -0.0501 | `{"B": 4}` |
| `Mercury_7194425` | `B` | `D` | `B` | 2 | -0.4343 | `{"B": 4}` |
| `Mercury_7212520` | `B` | `D` | `B` | 2 | -0.1272 | `{"B": 4}` |
| `NCEOGA_2013_5_28` | `C` | `A` | `C` | 2 | -0.5323 | `{"C": 4}` |

## Unrescued Content Loss Examples

| id | answer | content pred | cyclic pred | content answer rank | content answer margin |
|---|---|---|---|---:|---:|
| `MCAS_2013_5_17` | `C` | `A` | `D` | 2 | -0.3024 |
| `Mercury_400540` | `A` | `D` | `D` | 2 | -0.0251 |
