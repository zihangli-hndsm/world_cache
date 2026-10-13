# Official 3RScan validation: frozen ViT-G and geometry controls

Scored 110 reference/rescan relations in 47 official validation environments. The bootstrap unit is the environment; all rescans from a sampled environment remain grouped.

| Arm | Query denominator | Correct semantic inliers / query (pooled) | Semantic precision (pooled) | Coverage, environment macro (95% CI) | Strict pose successes by relation |
|---|---:|---:|---:|---:|---:|
| FCGF | 106137 | 3.3% | 28.8% | 3.4% [2.4%, 4.6%] | 0/110 |
| FPFH mutual | 87258 | 0.4% | 47.2% | 0.4% [0.3%, 0.5%] | 0/110 |
| FPFH one-way | 87258 | 0.8% | 32.5% | 0.9% [0.6%, 1.2%] | 0/110 |
| ViT-G | 106138 | 14.4% | 76.5% | 15.6% [12.0%, 19.7%] | 2/110 |

## Paired environment-macro differences

| Comparison | Difference | 95% environment bootstrap CI | Environments |
|---|---:|---:|---:|
| ViT-G_minus_FPFH one-way:semantic_inlier_coverage | 14.8% | [11.2%, 18.7%] | 47 |
| ViT-G_minus_FPFH one-way:semantic_precision_among_inliers | 40.1% | [31.3%, 48.8%] | 47 |
| ViT-G_minus_FPFH one-way:inlier_coverage | 16.9% | [13.5%, 20.7%] | 47 |
| ViT-G_minus_FPFH one-way:strict_pose_success | 2.1% | [0.0%, 5.3%] | 47 |
| ViT-G_minus_FPFH mutual:semantic_inlier_coverage | 15.2% | [11.7%, 19.2%] | 47 |
| ViT-G_minus_FPFH mutual:semantic_precision_among_inliers | 29.1% | [19.6%, 38.6%] | 47 |
| ViT-G_minus_FPFH mutual:inlier_coverage | 18.6% | [15.1%, 22.4%] | 47 |
| ViT-G_minus_FPFH mutual:strict_pose_success | 2.1% | [0.0%, 5.3%] | 47 |
| ViT-G_minus_FCGF:semantic_inlier_coverage | 12.2% | [8.8%, 15.9%] | 47 |
| ViT-G_minus_FCGF:semantic_precision_among_inliers | 43.0% | [33.2%, 53.0%] | 47 |
| ViT-G_minus_FCGF:inlier_coverage | 8.1% | [4.8%, 11.7%] | 47 |
| ViT-G_minus_FCGF:strict_pose_success | 2.1% | [0.0%, 5.3%] | 47 |

Per-method query denominators differ for geometry baselines because their sampled point sets differ. Paired comparisons are environment-macro differences, not same-query paired effects.

## Subsequent cross-split scan-content screen

A complete post-hoc geometry/annotation audit registered all 18,095 official train-reference/validation-reference pairs and 129 same-environment controls. At fitness >=0.40 / RMSE <=0.08 m, 86 cross-split candidates were found; the screen recovered 87/129 positive controls (67.4%). Five geometry-selected candidates meet the descriptive semantic-coherence rule, compared with 80/87 controls. This includes a five-train-reference cluster aligned to validation reference `4fbad31e`; the pairwise results are clustered evidence, not independent site-identity tests. The audit also confirms strong repeated-content evidence for study source reference `6bde6055-9162-246f-8f44-e3efc21c9e14` and validation reference `b1f23308-d255-2761-94da-981d962c6bf8`. Excluding that flagged validation environment leaves the ViT-G--FCGF macro coverage difference at +12.4 points (95% CI [+8.9,+16.1]) across 46 environments, but this does not establish physical-site independence. Details: `outputs/reports/20261003_3rscan_cross_split_overlap/train_full_complete/`.

The direct intersection with the paper's 16 source references and the 110
evaluated validation relations covers 752 pairs. Three pass the fixed geometry
cutoff and none passes the fixed semantic-coherence rule. For the one
paper-source pair with at least 10% coverage, exact RGB agreement is 91.8%,
but the matched-label semantic lift is only 3.3--5.0 points. Its 0.5 m spatial
bootstrap finds matches in 96/189 occupied blocks; the top five account for
24.2% of matches. A same-environment control for this validation reference has
94.6% exact RGB agreement and larger semantic lifts, so exact RGB agreement
alone does not distinguish cross-split from same-environment overlap. These
spatial intervals are descriptive because registration was selected on the
full clouds. Relation-level and spatial reports:
`outputs/reports/20261004_paper_relation_overlap/`.

The five full-screen pairs passing the fixed semantic-coherence rule target
two validation environments (4fbad31e and c7895f7c). Excluding both entire
environments leaves 45 environments / 105 relations. The ViT-G--FCGF
environment-macro coverage difference is +12.4 points (95% environment
bootstrap CI [+9.0,+16.3]), with ViT-G higher in 43/45 environments. This is
a more conservative, post-hoc sensitivity analysis based on the same geometry
and annotation screen; it does not certify the remaining environments as
independent physical sites. Strict pose success is 2/105 for ViT-G and 0/105
for FCGF; the environment-macro difference is +2.2 points (95% bootstrap CI
[0.0,5.6]), which does not reliably distinguish the methods. Full result:
outputs/reports/20261004_validation_leaveoverlap_out_strict/.

An exhaustive leave-one/two-environment stress test over the full validation
table remains positive in all 47 single removals and all 1,081 two-environment
removals. The worst two-environment deletion leaves a +10.6-point macro gap.
Because this is the minimum selected from observed outcomes, it is descriptive
and has no confidence interval. Per-combination values:
`outputs/reports/20261004_validation_leaveoverlap_out_strict/environment_removal_sensitivity.csv`.

## Physical-instance results (ViT-G)

- physical_candidate_recall: 56.4%
- physical_inlier_coverage: 13.0%
- physical_precision_among_inliers: 68.3%
