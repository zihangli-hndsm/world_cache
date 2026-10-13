# ViT-G semantic vs physical-instance candidate recall

Post-hoc audit of the 110 official 3RScan validation relations (47 environments). Both hit indicators are evaluated on the same descriptor-derived top-20 list. The primary denominator is the identical set of queries whose physical objectId has an unambiguous cross-scan mapping. This is a candidate-availability analysis before spatial consensus, not an independent validation split.

- Matched evaluable queries: 98221
- Pooled semantic candidate recall: 73.0%
- Pooled physical-instance candidate recall: 56.4%
- Environment-macro semantic recall: 77.7%
- Environment-macro physical-instance recall: 62.8%
- Paired environment-macro gap (semantic minus physical): +14.9 pp (95% cluster bootstrap CI [+10.0, +20.3] pp)
- Environments with positive / tied / negative gap: 43 / 4 / 0
- Joint candidate outcomes, pooled (both / semantic-only / physical-only / neither): 56.4% / 16.6% / 0.0% / 27.0%
- Semantic-only availability, environment-macro: 14.9%; physical-only availability: 0.0%

The bootstrap resamples environments and keeps all relations within each sampled environment together. This post-hoc diagnostic characterizes this validation set; it does not establish cross-site generalization or a causal descriptor-specific effect.
