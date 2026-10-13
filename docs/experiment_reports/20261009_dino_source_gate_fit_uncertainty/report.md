# Nested source-fit uncertainty for the DINO cosine gate

We resampled the 8 source reference UUIDs with replacement, refit the maximum-coverage threshold subject to 90% source macro precision, then independently resampled the 47 validation UUIDs and evaluated the paired DINO-minus-REMIND effects at that draw's fitted threshold. There were 10000 joint draws; 10000 had a feasible source fit. Excluded source UUIDs: []. Excluded validation UUIDs: [].

The fitted cutoff median is 0.789177, with 95% percentile range [0.773875, 0.828364]. The nested paired precision difference median is +6.0 pp (two-outcome-adjusted 95% range [+1.0, +12.9]); correct-coverage difference is +17.8 pp (adjusted range [+6.6, +22.8]).

This nested bootstrap propagates source-fit and validation-group resampling, but eight source groups are too few for a strong finite-sample guarantee. Precision averages only over sampled groups with at least one DINO accept; the validation cohort is previously explored, uses privileged masks, and has known possible cross-split content overlap.
