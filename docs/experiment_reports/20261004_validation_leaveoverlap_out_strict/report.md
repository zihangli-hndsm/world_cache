# Validation sensitivity excluding geometric overlap candidates

This post-hoc analysis excludes `4fbad31e-465b-2a5d-84b7-c0ddea978db4`, `c7895f7c-339c-2d13-819f-3bb0b26c91f6` from the frozen official validation environment table after a cross-split geometry/annotation overlap screen.

- Environments / relations: 45 / 105
- ViT-G macro correct-semantic inlier coverage: 15.8%
- FCGF macro correct-semantic inlier coverage: 3.4%
- Paired difference: +12.4 percentage points (95% environment-bootstrap CI [+9.0, +16.3])
- Environment wins / ties / losses: 43 / 0 / 2
- Strict pose successes: ViT-G 2/105, FCGF 0/105
- Environment-macro strict pose success: ViT-G 2.2%, FCGF 0.0%; paired gap +2.2 points (95% environment-bootstrap CI [+0.0, +5.6])

This was selected after observing both the validation metrics and the geometry screen. It is an exploratory robustness check and does not establish site independence for the remaining environments.

## Exhaustive environment-removal stress test

Across the full 47-environment table, we enumerated every one- and two-environment removal and recomputed the same unweighted environment-macro coverage difference. This asks whether a small number of environments can determine the sign or size of the observed contrast; it is not a bootstrap confidence interval and uses no new inference.

- Removing any one environment (47 combinations): minimum gap +11.37 pp after excluding `d7d40d4e-7a5d-2b36-97e7-34324c52ac42`; 5th percentile +11.69 pp; all 47 combinations remain positive.
- Removing any two environments (1,081 combinations): minimum gap +10.64 pp after excluding `d7d40d4e-7a5d-2b36-97e7-34324c52ac42`, `fcf66d7b-622d-291c-86b8-7db96aebcee3`; 5th percentile +11.46 pp; all 1,081 combinations remain positive.

The minimum is selected from the observed validation outcomes, so report it as a descriptive stress-test bound rather than an inferential guarantee. Per-combination values are in `environment_removal_sensitivity.csv`.
