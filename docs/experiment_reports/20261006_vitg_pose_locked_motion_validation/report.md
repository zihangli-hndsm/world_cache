# Pose-locked secondary-motion validation

The validation manifest contains 110 official relations across 47 environment IDs. Hyperparameters were selected on the 16 source relations only and frozen before this validation analysis: four candidate modes, a 0.15 m residual improvement margin, at least 32 independent secondary-mode inliers, top-20 ViT-G candidates, a 0.10 m residual threshold, 1,500 global RANSAC iterations, 1,000 secondary-mode iterations, and seed 17. The official rigid-instance annotations are used for scoring after inference.

## Transfer coverage

Among 14,088 annotated moved-rigid query patches, the global-pose baseline correctly transfers 2,876 (20.4%); the pose-locked motion branch transfers 3,808 (27.0%). This is 932 rescues and 0 lost baseline-correct queries. The pooled coverage difference is +6.6%; the environment-macro difference is +8.2% (95% environment-bootstrap CI [+4.0%, +13.9%]). The environment-bootstrap CI for the pooled difference is [+4.5%, +9.5%]. Coverage intervals use 20,000 environment resamples with seed 10062026. The method improves coverage in 20 environments, ties in 22, and is worse in 0; five environments have no annotated moved-rigid queries.

## Assignment precision and scene pose

The branch assigns 7,180 moved-rigid queries, compared with 4,309 for global RANSAC. Pooled physical-instance precision changes from 66.7% to 53.0% (-13.7%; 95% environment-bootstrap CI [-20.6%, -3.3%]; 20,000 resamples, seed 10062027). Thus the frozen rule expands coverage but accepts lower-precision assignments. Strict global-pose successes remain 2/110; each method arm uses the same baseline transform by construction.

## Scope

This is a frozen train-selected branch evaluated on the official validation relations; no validation labels select modes or assignments. The validation split is reused from the correspondence audit. A later geometry/annotation screen found strong repeated-content evidence between one source and one validation reference, so treat the result as official-split evidence for a coverage/precision trade-off in pose-locked local transfer, not independent-site generalization.

Machine-readable outputs: `summary.json`, `per_relation.csv`, `per_environment.csv`, and `run_manifest.csv`. Per-relation diagnostics and baseline-reproduction deltas are under `relations/`.
