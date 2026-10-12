# WorldCache

Research code and diagnostics for frozen visual correspondence across real
RGB-D scans. The released paper snapshot in [`paper/`](paper/) studies how
semantic retrieval, physical-instance identity, rigid-pose recovery, and
selective coverage differ under cross-scan domain shift.

On the paper's 16-source / one-held-out 3RScan study, a source-fit residual
selector retains 22.1% of target queries at 96.6% semantic accuracy; target
semantic mIoU is 89.5% conditional on retained queries and 19.7% when
abstentions count as false negatives. These are paper-specific results, not a
claim of universal pose-free registration. The manuscript describes the
protocol and its limits.

## Setup

The reference environment is Python 3.9 (`environment.yml`). The full
experiments use CUDA-enabled PyTorch. After creating the environment:

```bash
python -m pip install --no-deps -e .
PYTHONPATH=. python -m pytest -q --ignore=tests/test_environment.py
```

GitHub Actions runs the CPU-compatible regression suite on pushes and pull
requests. The paper build instructions are in [`paper/README.md`](paper/README.md).
Run the supported protocol from the repository root; its datasets and model
weights must be provided locally.

## 3RScan data

The upstream downloader is [`download.py`](download.py). Review and accept the
provider's terms before downloading. Fetch one scan with:

```bash
python download.py --out_dir /path/to/authorized/3rscan --id SCAN_ID
```

The optional expanded full-validation downloader uses the frozen scan-ID manifest in
`outputs/reports/20260929_validation_access/validation_scan_manifest.csv`:

```bash
THREERSCAN_ROOT=/path/to/authorized/3rscan
PYTHONPATH=scripts python scripts/download_3rscan_validation.py \
  --data-root "$THREERSCAN_ROOT"
```

Other external inputs and access notes are listed in [`DATASETS.md`](DATASETS.md).
No dataset payloads, checkpoints, or experiment caches are included.

## Layout and license

- `src/worldcache/`: reusable geometry, backbone, retrieval, and cache modules;
- `scripts/`: evaluators, audits, aggregation, and data helpers;
- `tests/`: unit and protocol regression tests;
- `paper/`: paper snapshot, figures, and generated tables.

WorldCache-authored code is MIT-licensed. Dataset utilities, datasets, model
weights, and optional upstream baselines retain their own terms. See
[`third_party/README.md`](third_party/README.md) for provenance and license
boundaries.
