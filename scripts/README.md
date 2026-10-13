# Scripts

Scripts are grouped by filename prefix and are intended to be run from the
repository root. They are research utilities rather than a single CLI. Check
each script's `--help` and source header for required inputs and output paths.
Dataset roots, checkpoints, and generated result files must be supplied
locally; none are part of the release.

## Paper's 3RScan source/target study

- `evaluate_3rscan_spatial_consensus.py` evaluates a prepared scan pair;
- `generate_supplementary_tables.py` regenerates tables from the frozen
  experiment outputs.

## Expanded full-validation code

The repository also includes a separate 110-relation official validation
pipeline, which is not the 17-relation source/held-out paper snapshot:

- `download_3rscan_validation.py` downloads its frozen scan list;
- `run_3rscan_validation_protocol.py` runs the relation-level evaluation;
- `aggregate_3rscan_validation.py` aggregates relation outputs.

## Baselines and diagnostic studies

`evaluate_3rscan_*`, `run_*validation*`, `audit_*`, `analyze_*`,
`aggregate_*`, and `summarize_*` contain protocol variants, controls, and
statistical analyses reported in or considered during the study. Several are
exploratory or depend on optional upstream repositories. They should not be
interpreted as additional official benchmark results without checking the
associated protocol and limitations in the manuscript.

## Paper assets

`verify_submission_consistency.py` checks a prepared submission bundle.
These commands require their documented result inputs and are not standalone
data downloaders. `make -C paper verify` compiles the checked-in TeX and
figures; it does not reproduce numerical results without the external
experiment snapshot.
