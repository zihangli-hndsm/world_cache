# WorldCache

Reproducible code and diagnostics for frozen vision-transformer reuse and
selective semantic correspondence across real RGB-D scans.

The current paper artifact is the selective-correspondence study in
[`paper/`](paper/). It uses frozen multi-scale DINOv2 tokens, top-$K$ candidate
retrieval, and 3-D spatial consensus to separate semantic correspondence
quality, rigid-pose validity, and abstention coverage.

## Current headline result

On the frozen 3RScan audit (16 source relations and one held-out target), the
ViT-G protocol reaches 68.0% target coverage, 93.5% retained semantic
accuracy, and 1.54 degrees rotation error. A source-fit residual selector reaches
22.1% target coverage and 96.6% retained semantic accuracy. Its target semantic
mIoU is 89.5% conditional on retained queries (19.7% over all fixed queries
when abstentions count as false negatives), and its conditional instance mIoU is
90.7%. The source audit also contains candidate-set,
wrong-consensus, and wrong-pose failures; this repository does not claim
universal pose-free registration or dense correspondence.

## Quick verification

The project environment is Python 3.9 and is described in
[`environment.yml`](environment.yml). With the environment installed:

```bash
cd /path/to/world_cache
PYTHONPATH=. conda run -n worldcache pytest -q
cd paper
make verify
```

For importing the lightweight package without installing the optional vision
stack, use the editable project metadata:

```bash
conda run -n worldcache python -m pip install --no-deps -e .
```

The optional `vision` extra is declared in `pyproject.toml`; the CUDA-pinned
environment in `environment.yml` remains the authoritative path for the full
paper experiments.

The current checks pass 38 tests. `paper/make verify` compiles the main paper
and supplement, regenerates their tables and figures, rebuilds the anonymous
PDF package and reproducibility bundle, and runs the numerical consistency
verifier. The generated bundle is under `outputs/reports/`; it is not required
for normal code use.

Public pull requests run the CPU-compatible regression subset in
`.github/workflows/ci.yml`. The two environment tests that require CUDA and
Habitat-Sim remain part of the full conda-environment check, not the hosted CI
job.

## Reproducing the paper

Start with [`paper/README.md`](paper/README.md), which lists the authoritative
reports, evaluator commands, controls, and artifact boundary. The paper uses
external 3RScan frame archives, annotations, DINOv2 weights, and the published
FCGF checkpoint; those large or externally licensed inputs are intentionally
not bundled. Dataset paths and provenance are documented in
[`DATASETS.md`](DATASETS.md).

The anonymous submission sources are the two TeX files and the local figure
bundle under `paper/`. Date-stamped release bundles are intentionally omitted
from this dated snapshot; generate a fresh manifest only after the paper and
its artifact provenance have been frozen.

- [`paper/main.tex`](paper/main.tex)
- [`paper/supplementary.tex`](paper/supplementary.tex)

## Repository layout

- `src/worldcache/`: reusable geometry, backbone, retrieval, and cache modules;
- `scripts/`: evaluators, audits, table/figure generation, and verification;
- `tests/`: unit and protocol regression tests;
- `paper/`: compile-checked paper text, figures, and generated tables;
- `outputs/reports/`: authoritative reports and release diagnostics;
- `third_party/`: small compatibility adapters with their upstream SPDX notices.

## Scope and data policy

Raw datasets, model checkpoints, experiment caches, and machine-specific paths
are excluded from the public release. Results that require them state the
external input explicitly. The release includes lightweight diagnostics and
the code needed to inspect or reproduce the reported protocols when those
inputs are supplied. Session logs and prompt material are ignored for new
checkouts; before creating a public git commit, manually curate any historical
tracked state files as well. The exact publish/curate boundary is listed in
[`PUBLIC_RELEASE_CHECKLIST.md`](PUBLIC_RELEASE_CHECKLIST.md).

## License

Project code and documentation are released under the MIT License in
[`LICENSE`](LICENSE), except where a file or dependency carries its own SPDX
notice. Dataset terms and pretrained-model terms remain governed by their
respective providers.
