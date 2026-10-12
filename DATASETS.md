# Data and model inputs

This repository contains no RGB-D dataset payloads, pretrained checkpoints, or
model outputs. Obtain each input from its provider, comply with its terms, and
store it outside the Git checkout. Do not redistribute provider data unless
the applicable license explicitly permits it.

## 3RScan

The upstream downloader is [`download.py`](download.py); its header links to
the provider's terms and describes the data's CC BY-NC-SA 4.0 terms. Each user
must review and accept the current terms independently. The frozen
`validation_scan_manifest.csv` covers 157 scans for the expanded 110-relation
validation protocol. The paper snapshot currently in `paper/` reports a
separate 17-relation source/held-out study; do not conflate these populations.
Use the bounded helper to download only the required validation files:

```bash
THREERSCAN_ROOT=/path/to/authorized/3rscan
PYTHONPATH=scripts python scripts/download_3rscan_validation.py \
  --data-root "$THREERSCAN_ROOT"
```

The helper fetches `sequence.zip`, `labels.instances.annotated.v2.ply`, and
`semseg.v2.json` for listed scans. The official downloader supports a single
scan through `--id`; avoid an unrestricted download unless required.

## Other evaluated datasets

Experiments and diagnostics in the paper also refer to ReplicaCAD, Rescan,
SceneDiff, TU Wien ObChange, MultiScan, ScanNet/ScanNet++, and related
benchmark data. These sources have separate access conditions and are not
mirrored here. Follow the providers' current terms and paper citation
requirements. Some protocols require only selected annotations, while others
require RGB, depth, poses, or model outputs; see script `--help` and the paper
protocol before downloading.

## Model weights and upstream baselines

DINOv2, FCGF, BUFFER-X, MASt3R/SegMASt3R, and other pretrained weights are
external inputs and are not bundled. Their providers' licenses apply. Optional
baseline source locations, revisions, and license notes are listed in
[`third_party/README.md`](third_party/README.md). SegMASt3R's upstream code is
non-commercially licensed and is not included in this release.
