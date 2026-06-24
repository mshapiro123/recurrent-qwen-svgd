# MCQ Surface-Mismatch Diagnosis - arc_easy_stage5_score_alignment_repair_content_route_20260624_benchmark

- Content delta: `-7` (139/256 vs 146/256)
- Candidate cyclic vs candidate content delta: `+65`
- Content losses: `13`
- Content losses rescued by cyclic: `11` (0.846)
- Stable cyclic rescues: `6` (0.462)
- Unrescued content losses: `2` (0.154)
- Order-sensitive content losses: `5` (0.385)
- Recommendation: `no_dominant_surface_failure_pattern`

## Stable Cyclic Rescue Examples

| id | answer | content pred | cyclic pred | content answer rank | content answer margin | perm counts |
|---|---|---|---|---:|---:|---|
| `ACTAAP_2014_7_6` | `D` | `B` | `D` | 2 | -0.0977 | `{"D": 4}` |
| `MDSA_2011_4_8` | `A` | `C` | `A` | 2 | -0.0449 | `{"A": 4}` |
| `Mercury_7010815` | `B` | `D` | `B` | 2 | -0.0342 | `{"B": 4}` |
| `Mercury_7194425` | `B` | `D` | `B` | 2 | -0.4476 | `{"B": 4}` |
| `Mercury_7212520` | `B` | `D` | `B` | 2 | -0.1804 | `{"B": 4}` |
| `NCEOGA_2013_5_28` | `C` | `A` | `C` | 2 | -0.5541 | `{"C": 4}` |

## Unrescued Content Loss Examples

| id | answer | content pred | cyclic pred | content answer rank | content answer margin |
|---|---|---|---|---:|---:|
| `MCAS_2013_5_17` | `C` | `A` | `D` | 2 | -0.3292 |
| `Mercury_400540` | `A` | `D` | `D` | 2 | -0.0300 |
