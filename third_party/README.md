# Optional upstream research code

The directories below may exist as local clones for comparison experiments.
They are excluded from this repository because they are upstream projects,
not WorldCache source. Install or clone them separately when reproducing the
corresponding optional experiment, and follow each provider's terms.

The checked-in `warpconvnet_fcgf.py` adapter is separately marked
`Apache-2.0` in its SPDX header; that license applies to the adapter.

| Local checkout | Upstream | Local revision | License note | WorldCache entry points |
|---|---|---|---|---|
| `BUFFER-X/` | [MIT-SPARK/BUFFER-X](https://github.com/MIT-SPARK/BUFFER-X) | `b56af186cdfef30b691a4866dbb5f5a55852f285` | Upstream checkout contains its MIT license. | `audit_bufferx_runtime.py`, `evaluate_bufferx_native_pair.py` |
| `crossover/` | [GradientSpaces/CrossOver](https://github.com/GradientSpaces/CrossOver) | `e46dce2b35b053c124aaf90240d950463f6c997f` | Upstream checkout contains its MIT license. | `analyze_crossover_rgb_pairwise.py` |
| `remind_reid/` | [cvar-vision-dl/remind-reid-tracker](https://github.com/cvar-vision-dl/remind-reid-tracker) | Not recorded in this workspace | Local checkout contains an MIT license; pin a commit before documenting a reproduction. | `evaluate_remind_style_3rscan.py` |
| `segmast3r/` | [SegMASt3R/segmast3r](https://github.com/SegMASt3R/segmast3r) | `cd6583ec507ad6365a9e30bc94192e254d2d0002` | Upstream README states CC BY-NC 4.0; this is non-commercial and does not qualify as an OSI-approved open-source license. Keep its code and weights out of the WorldCache release. | `infer_segmast3r_segment_pair.py`, `probe_rescan_mast3r_alignment.py` |

The current `.gitignore` excludes these checkout directories. The local copies
remain available on this machine. A release that needs a baseline should
document its upstream revision and setup instructions without vendoring the
upstream tree unless its license and any included assets have been reviewed.
