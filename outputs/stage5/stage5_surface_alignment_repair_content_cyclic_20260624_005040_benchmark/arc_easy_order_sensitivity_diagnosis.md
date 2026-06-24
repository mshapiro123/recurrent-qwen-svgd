# MCQ Order-Sensitivity Diagnosis - arc_easy_stage5_surface_alignment_repair_content_cyclic_20260624_005040_benchmark

- Content delta: `-10` (138/256 vs 148/256)
- Candidate order-sensitive rows: `86/256` (0.336)
- Content losses order-sensitive: `5/15` (0.333)
- Content losses rescued by cyclic aggregation: `13/15` (0.867)
- Loss rate on base-correct order-sensitive rows: `0.119`
- Loss rate on base-correct order-stable rows: `0.094`
- Order-sensitivity loss-rate lift: `0.025`
- Recommendation: `diagnose_content_route_scoring_or_prompt_alignment_before_more_distillation`

## Loss Examples

| id | answer | base content | recurrent content | recurrent cyclic | order-sensitive | cyclic rescue | perm counts |
|---|---|---|---|---|---:|---:|---|
| `ACTAAP_2014_7_6` | `D` | `D` | `B` | `D` | False | True | `{"D": 4}` |
| `LEAP_2011_8_10436` | `B` | `B` | `A` | `B` | True | True | `{"B": 3, "C": 1}` |
| `MCAS_2003_5_35` | `D` | `D` | `A` | `D` | True | True | `{"B": 2, "D": 2}` |
| `MCAS_2013_5_17` | `C` | `C` | `A` | `D` | False | False | `{"D": 4}` |
| `MDSA_2011_4_8` | `A` | `A` | `C` | `A` | False | True | `{"A": 4}` |
| `Mercury_183190` | `C` | `C` | `B` | `C` | False | True | `{"C": 4}` |
| `Mercury_400540` | `A` | `A` | `D` | `D` | False | False | `{"D": 4}` |
| `Mercury_7010815` | `B` | `B` | `D` | `B` | False | True | `{"B": 4}` |
| `Mercury_7013090` | `B` | `B` | `C` | `B` | True | True | `{"B": 3, "D": 1}` |
| `Mercury_7017080` | `A` | `A` | `B` | `A` | False | True | `{"A": 4}` |
| `Mercury_7114940` | `C` | `C` | `A` | `C` | True | True | `{"A": 1, "C": 3}` |
| `Mercury_7194425` | `B` | `B` | `D` | `B` | False | True | `{"B": 4}` |
