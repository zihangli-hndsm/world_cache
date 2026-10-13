# Source-frozen DINO multi-view acceptance gate

## Protocol

We selected one absolute cosine-similarity threshold using only eight source reference groups. Among observed assignment scores, the rule chooses the threshold with maximum source macro correct coverage subject to source macro precision of at least 90%. We then froze that threshold and applied it unchanged to the 110 validation relations. Labels are read only for source threshold fitting and post-hoc scoring; inference artifacts record `labels_read: false`. Source and validation reference-group IDs are disjoint.

## Results

| Split | Reference groups | Accepted | Correct | Macro precision | Macro correct coverage | Pooled precision | Pooled correct coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Source fit | 8 | 350 | 311 | 90.0% | 33.9% | 88.9% | 33.5% |
| Frozen validation transfer | 47 | 4134 | 3563 | 87.4% | 29.4% | 86.2% | 26.8% |

Frozen cutoff: **0.789155**. Validation reference-group bootstrap 95% percentile intervals, conditional on this fitted cutoff: macro precision 84.3%–90.2%; macro correct coverage 26.1%–32.8%.

Against the saved REMIND-style one-to-one map on the same 47 reference groups, the frozen DINO gate has macro precision/coverage of 87.4%/29.4%, versus 81.5%/11.2%. The gate accepts 4,134 masks (3,563 correct), compared with 1,554 accepted (1,231 correct) for REMIND. Paired macro differences are +5.8 pp precision (two-outcome-adjusted interval [+1.4,+11.1]) and +18.2 pp correct coverage (adjusted interval [+15.8,+20.7]). This is not an equal-acceptance comparison: the DINO gate accepts more than twice as many links, the association procedures differ, and both receive privileged masks on a previously explored benchmark.

As a source-only stability check, we refit the same rule in eight leave-one-reference-group-out folds. A fold is feasible only when its seven training groups contain a cutoff meeting the same 90% macro precision floor. Across feasible folds, 8/8 had such a cutoff; their thresholds ranged from 0.785403 to 0.796130. Held-out source-group precision averaged 90.5% and coverage 33.2%; 5/8 held-out groups reached 90% precision. These folds are a stability diagnostic on previously explored source data, not independent replications.

The paired intervals above condition on the selected cutoff. A separate 10,000-draw nested source/validation reference-group bootstrap refits the threshold in every draw; all source fits were feasible. Its cutoff median is 0.78918 (95% percentile range 0.77387--0.82836). Nested paired differences are +6.0 pp precision (two-outcome-adjusted percentile interval [+1.0,+12.9]) and +17.8 pp correct coverage ([+6.6,+22.8]). See `outputs/reports/20261009_dino_source_gate_fit_uncertainty/` for details. These results use the full validation cohort; the separate overlap-excluded refit sensitivity has a precision interval spanning zero.

## Limits

This is exploratory source-fit calibration with only eight source groups, not a distribution-free precision guarantee. Although the validation labels did not choose the cutoff, the same 3RScan benchmark and masks were examined in earlier analyses, so the transfer is not independent-site confirmation. Both sides use privileged annotation masks. The matcher has no track memory or query updates. The original validation intervals condition on the selected cutoff; the separate nested bootstrap propagates source-fit and validation-group resampling but remains a small-cluster sensitivity analysis.
