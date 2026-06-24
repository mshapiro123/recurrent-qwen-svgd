# MCQ Order-Sensitivity Diagnosis - arc_easy_stage5_score_alignment_repair_content_route_20260624_benchmark

- Content delta: `-7` (139/256 vs 146/256)
- Candidate order-sensitive rows: `87/256` (0.340)
- Content losses order-sensitive: `5/13` (0.385)
- Content losses rescued by cyclic aggregation: `11/13` (0.846)
- Loss rate on base-correct order-sensitive rows: `0.119`
- Loss rate on base-correct order-stable rows: `0.077`
- Order-sensitivity loss-rate lift: `0.042`
- Recommendation: `diagnose_content_route_scoring_or_prompt_alignment_before_more_distillation`

## Loss Examples

| id | answer | base content | recurrent content | recurrent cyclic | order-sensitive | cyclic rescue | perm counts |
|---|---|---|---|---|---:|---:|---|
| `ACTAAP_2014_7_6` | `D` | `D` | `B` | `D` | False | True | `{"D": 4}` |
| `LEAP_2011_8_10436` | `B` | `B` | `A` | `B` | True | True | `{"B": 3, "C": 1}` |
| `MCAS_2003_5_35` | `D` | `D` | `A` | `D` | True | True | `{"B": 2, "D": 2}` |
| `MCAS_2013_5_17` | `C` | `C` | `A` | `D` | False | False | `{"D": 4}` |
| `MDSA_2011_4_8` | `A` | `A` | `C` | `A` | False | True | `{"A": 4}` |
| `Mercury_400540` | `A` | `A` | `D` | `D` | False | False | `{"D": 4}` |
| `Mercury_7010815` | `B` | `B` | `D` | `B` | False | True | `{"B": 4}` |
| `Mercury_7013090` | `B` | `B` | `C` | `B` | True | True | `{"B": 3, "D": 1}` |
| `Mercury_7114940` | `C` | `C` | `A` | `C` | True | True | `{"A": 1, "C": 3}` |
| `Mercury_7194425` | `B` | `B` | `D` | `B` | False | True | `{"B": 4}` |
| `Mercury_7212520` | `B` | `B` | `D` | `B` | False | True | `{"B": 4}` |
| `NCEOGA_2013_5_28` | `C` | `C` | `A` | `C` | False | True | `{"C": 4}` |
