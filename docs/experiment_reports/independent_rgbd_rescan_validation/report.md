# MultiScan native RGB-D rescan: quota-limited exploratory run

Date: 2026-10-12 UTC
Conclusion: **exploratory evidence only; this does not complete the full independent validation population.**

## Scope and selection

The frozen official repository revision is `c62c9aad850a8638ac3e42605926a65707600125` (`3dlg-hcvc/MultiScan`). The original frozen population contains 279 relations across 100 environments and 254 unique scan archives (359,011,333,493 compressed bytes). The user reported 50 GB of quota remaining and requested a smaller, quick subset. Before projection checks or model predictions, the run froze the longest lexicographically sorted prefix of complete rooms that fits a 40,000,000,000-byte budget. This selected 17 environments, 17 two-scan relations, 34 archives, and 39,744,374,300 bytes. No room was selected or excluded using projection or model outcomes.

The 262 remaining relations in 83 environments were not downloaded or evaluated; they are itemized in [`quota_subset_exclusions.csv`](quota_subset_exclusions.csv). `scene_00069` remains excluded because it was used to diagnose and repair the projection bug. The complete original population remains in [`candidate_pairs.csv`](candidate_pairs.csv), and the pre-registered quota subset is [`quota_subset_manifest.json`](quota_subset_manifest.json). Since the evaluated rooms are a deterministic prefix rather than a random or representative sample, the intervals below describe only this subset and cannot correct for ordering or selection effects.

## Data quality and eligibility

The previously found principal-point indexing bug is fixed in the frozen code: column-major camera intrinsics were decoded correctly, but the old projection read `K[2,0]`/`K[2,1]`; it now reads `K[0,2]`/`K[1,2]`, consistent with the official MultiScan format and reconstruction bridge. No cross-scan GT alignment, labels, retrieval scores, or manually chosen successful frames were used to fit a transform. The original scene-00069 regression test remains recorded in [`../rgbd_projection_diagnostic/report.md`](../rgbd_projection_diagnostic/report.md).

All 34 scans in the fixed quota subset were checked with the frozen uniform 72-frame rule. **34/34 scans and 17/17 relations pass the original 5 cm gate**: per-scan medians of frame-level median absolute depth error range from 0.60 cm to 2.52 cm (median 1.18 cm). Median valid-depth fraction is 100%; median mesh-hit coverage of valid depth is 99.98% (range 95.89–100%). Across sampled frames, the median fraction of overlapping pixels within 5 cm is 90.4% (frame-distribution p05–p95: 59.9–98.9%). Tail errors remain: the median frame-level pixel p95 residual is 9.0 cm, and its p95 across frames is 61.0 cm. The 5 cm gate was not changed, and no frame was dropped. Full per-scan and per-frame distributions are in [`projection_preflight/`](projection_preflight/).

Persistent object labels and annotated PLY IDs were audited separately for every fixed relation. All 17/17 have shared cross-scan persistent physical-instance IDs, with a minimum of 14 and median of 37 shared IDs. Raw scan-local integer IDs were not equated. Annotation projection maps a mean 88.1% of sampled query points to an identity; points without a label count as incorrect in the reported recall metrics. The per-environment identity audit is in [`identity_preflight/`](identity_preflight/).

## Frozen evaluation and label isolation

We reused the frozen configuration: DINOv2 ViT-G/14, weight revision `dd22bb6485eeb29d67273f76d983061298bdc7f0`, `model.safetensors` SHA256 `949077749c7a414f3443a007cf021d085bc11deb0b08959a6e32e170b1e8e359`, blocks 2/20/40, 224×224 input, up to 20 uniformly spaced frames per scan including endpoints, all valid 14×14 patch-center depth points, top-20 candidates, 0.10 m consensus threshold, 1,500 RANSAC iterations, seed 17. There was no training, threshold choice, or tuning on these scans.

The prediction runner reads only RGB video, native depth, stream metadata, and within-scan camera poses. It does not open annotation, object-ID, annotated-mesh, or cross-scan-alignment members. It saves all 17 relation prediction files before the separate scoring runner opens annotation and mesh members. The RGB control uses the identical query/reference point pools and per-patch mean RGB descriptors. No pose result is reported.

## Results

All values are environment-macro percentages. Bootstrap intervals resample the 17 environments (10,000 draws, seed 20261011); frames and patches are not treated as independent samples. Recall uses all depth-valid query patch points; missing labels and unmapped identities count as incorrect.

| Metric | DINOv2 ViT-G | RGB patch mean |
|---|---:|---:|
| Semantic Recall@20 | 82.8 (79.4–86.1) | 68.9 (65.5–72.2) |
| Physical-instance Recall@20 | 79.1 (75.0–82.8) | 60.1 (56.4–63.8) |
| Semantic minus physical Recall@20 | +3.8 pp (2.6–5.0) | +8.8 pp (7.3–10.5) |
| Consensus accepted physical-instance precision | 77.8 (74.1–81.4) | 74.8 (69.3–80.0) |
| Consensus accepted-query coverage | 35.4 (29.2–41.2) | 3.0 (2.3–3.9) |
| Correct accepted-query coverage | 27.5 (22.5–32.1) | 2.3 (1.7–2.9) |

On the same environment clusters, DINO minus RGB is +13.9 pp semantic Recall@20 (95% CI 12.8–15.1) and +18.9 pp physical-instance Recall@20 (17.2–20.6). DINO minus RGB accepted precision is +3.0 pp (−1.0–6.6), which is inconclusive at this sample size; the difference in correct accepted-query coverage is +25.2 pp (20.6–29.7). Per-environment values, all paired intervals, and the plotted summary are in [`results/`](results/).

## Runtime and reproducibility

The run used the existing `worldcache` environment (Python 3.9.23, PyTorch 2.5.1+cu124, timm 1.0.29, NumPy 1.26.4, Pillow 11.3.0) and an NVIDIA A100 80 GB. No environment was rebuilt and no package was installed. The 34 archive download took 134 seconds; the 72-frame projection preflight took 418 seconds in aggregate; identity qualification took 10.4 seconds.

Measured matching-stage totals across 17 relations: data decode (depth plus selected RGB video decode) 910.7 s; patch-grid geometry preparation 8.4 s; GPU ViT features 29.9 s, plus shared model load 22.9 s; CPU top-20 ranking 80.9 s; CPU RANSAC consensus 218.1 s. Post-prediction label projection and scoring took 38.6 s; the 10,000-draw environment bootstrap took 2.3 s; the scoring command wall time was 42.8 s. Relation-level timings are in [`predictions/run_summary.csv`](predictions/run_summary.csv), and scoring timings are in [`results/summary.json`](results/summary.json).

Reproduce from the workspace root after HF authentication and with the selected archives present:

```bash
conda run --no-capture-output -n worldcache python docs/experiment_reports/independent_rgbd_rescan_validation/download_frozen_archives.py --prefix-cap-bytes 40000000000
conda run --no-capture-output -n worldcache python docs/experiment_reports/independent_rgbd_rescan_validation/audit_frozen_subset_projection.py --sample-count 72
conda run --no-capture-output -n worldcache python docs/experiment_reports/independent_rgbd_rescan_validation/audit_frozen_subset_identities.py
conda run --no-capture-output -n worldcache python docs/experiment_reports/independent_rgbd_rescan_validation/run_frozen_subset_matching.py
conda run --no-capture-output -n worldcache python docs/experiment_reports/independent_rgbd_rescan_validation/score_frozen_subset_predictions.py
conda run --no-capture-output -n worldcache python docs/experiment_reports/independent_rgbd_rescan_validation/make_quota_subset_artifacts.py
```

The bundled [`download_state.json`](download_state.json) is an empty resumable template; the recorded commands and scoring preflight guard correction are summarized in [`run_log.md`](run_log.md). A first full-download command was stopped as soon as the user clarified that only 50 GB of quota remained. It was not resumed; the fixed quota subset was downloaded instead.

## Limitations and conclusion

This is a genuine native RGB-D room-rescan experiment on 17 separate MultiScan environments, and the frozen 5 cm projection gate and physical-identity qualification passed on all selected relations. The observed DINO results are promising relative to the matched RGB control. However, only 17/100 frozen environments and 17/279 relations were scored. The lexicographic-prefix subset may reflect dataset ordering, the environment count is modest, and the bootstrap covers only within-subset cluster variation. The result should be described as **quota-limited exploratory evidence**, not broad or complete independent validation. The untested 83 environments and their 262 relations remain limitations. No manuscript source was modified.
