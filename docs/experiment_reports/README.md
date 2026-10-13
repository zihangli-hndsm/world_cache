# Selected experimental reports

A compact, reviewable evidence set for the paper's main claims and their limits. Original report text is preserved where possible; aggregate results, environment-level tables, and protocol records are included when useful. Per-query dumps, descriptors, prediction binaries, downloaded data, model weights, and machine-specific session state are excluded.

## Milestones

| Report | Why it matters | Key result and boundary |
|---|---|---|
| [`20261003_vitg_validation/`](20261003_vitg_validation/aggregate/report.md) | Frozen ViT-G evaluation and geometry controls on the complete official validation split | 110 relations / 47 environments; +12.2 pp environment-macro semantic-inlier coverage vs FCGF (95% environment bootstrap CI +8.8 to +15.9). Geometry methods use different query populations, so this is an environment-level contrast, not a same-query effect. |
| [`20261004_vitg_validation_candidate_audit/`](20261004_vitg_validation_candidate_audit/report.md) | Same-query, same-top-20 semantic vs linked physical-instance candidate audit | 98,221 queries; macro recall 77.7% vs 62.8%, gap +14.9 pp (CI +10.0 to +20.3). Post-hoc availability analysis, not independent-site evidence. |
| [`20261003_3rscan_cross_split_overlap/`](20261003_3rscan_cross_split_overlap/train_full_complete/report.md) | Checks repeated content across official train/validation environment IDs | Finds repeated-content indicators; automated geometry/annotation screening is descriptive and does not prove physical-site identity. See [`leave-overlap sensitivity`](20261004_validation_leaveoverlap_out_strict/report.md) for its post-hoc selection caveat. |
| [`20261008_scene_transform_unit_audit/`](20261008_scene_transform_unit_audit/report.md) | Corrects a confirmed official metadata translation-unit error | Raw translations are in metres; 5 cm static-surface support improves in 110/110 relations. Corrected strict pose success is 14/110 vs 2/110 before correction; correspondence predictions are unchanged. |
| [`20261009_dino_multiview_source_gate/`](20261009_dino_multiview_source_gate/report.md), [`fit uncertainty`](20261009_dino_source_gate_fit_uncertainty/report.md) | Records source-fitted selective gate and nested uncertainty analysis | Frozen validation macro precision / correct coverage: 87.4% / 29.4%; unequal acceptance counts and previously explored annotated data make this exploratory, not independent confirmation. |
| [`20261006_vitg_pose_locked_motion_validation/`](20261006_vitg_pose_locked_motion_validation/report.md) | Measures a frozen local-motion transfer extension | Moved-rigid coverage rises while pooled physical-instance precision falls 66.7% to 53.0%; correspondence utility and global pose recovery are separate outcomes. |
| [`rgbd_projection_diagnostic/`](rgbd_projection_diagnostic/report.md) | Documents the MultiScan projection bug and evidence-based fix | Correct principal-point indexing lets the original fixed frames pass the unchanged 5 cm gate. |
| [`independent_rgbd_rescan_validation/`](independent_rgbd_rescan_validation/report.md) | Native RGB-D room-rescan experiment under storage limits | 17/17 predeclared relations pass projection and identity checks; DINO exceeds matched RGB on recall and correct coverage. This is a deterministic 17-environment prefix and must be described as quota-limited exploratory evidence, not broad/full validation. |

## Interpretation and access

These snapshots do not make every analysis confirmatory. Read each report's limitations and post-hoc notes alongside the headline value. The full-split geometry comparison has method-specific query populations; candidate audits and overlap exclusions are post-hoc; MultiScan covers only the deterministic quota subset.

MultiScan RGB-D archives are not included. Reproduction requires separately meeting that dataset's access terms and supplying external data and model weights. The bundled `independent_rgbd_rescan_validation/download_state.json` contains no credentials or local paths.

`SOURCES.tsv` maps every included item to the source path, original SHA-256, published SHA-256, and handling.
