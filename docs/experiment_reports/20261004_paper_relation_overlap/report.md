# Paper relation intersection with the complete split-overlap screen

The paper lists 16 source reference/rescan relations. The official validation evaluation contains 110 relations across 47 reference environments.
The source-by-validation audit covers 752 relevant train-reference/validation-reference pairs. At the fixed geometric screen, 3 candidates intersect the paper's actual source references and evaluated validation references; 0 meet the fixed semantic-coherence rule.

| Paper source reference | Evaluated validation reference | Eval relations | Fitness | RMSE (m) | Source/target coverage | Matched vertices | Fixed semantic rule |
|---|---|---:|---:|---:|---:|---:|---|
| `5630cfc9-12bf-2860-84ed-5bb189f0e94e` | `5630cfcf-12bf-2860-8784-83d28a611a83` | 4 | 0.414 | 0.059 | 3.7%/4.1% | 2511 | no |
| `6bde6055-9162-246f-8f44-e3efc21c9e14` | `b1f23308-d255-2761-94da-981d962c6bf8` | 1 | 0.528 | 0.043 | 12.0%/11.5% | 3380 | no |
| `6bde6055-9162-246f-8f44-e3efc21c9e14` | `0cac7540-8d6f-2d13-8eee-36ba2a428e3f` | 2 | 0.419 | 0.051 | 5.2%/8.0% | 1478 | no |

This is an intersection with the 16 source relations listed in the paper and the 110 official-validation relations. It does not imply other official-train references were used by the paper's method, and registration outcomes are post hoc.

Spatial-block sensitivity for the direct overlap pair: `outputs/reports/20261004_paper_relation_overlap/spatial_blocks/report.md`.
