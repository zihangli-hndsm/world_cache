# Curated run record

- Dataset: `3dlg-hcvc/MultiScan`, pinned revision `c62c9aad850a8638ac3e42605926a65707600125`; no archive contents are included. Access is gated by dataset terms; reproductions require independent authorization.
- Frozen subset: 17 complete rooms / 17 relations, from the deterministic lexicographic prefix under a 40,000,000,000-byte archive cap. The subset was frozen before projection and model outcomes; it covers 17/100 candidate environments and 17/279 candidate relations.
- Model: DINOv2 ViT-G/14; weight revision `dd22bb6485eeb29d67273f76d983061298bdc7f0`, SHA-256 `949077749c7a414f3443a007cf021d085bc11deb0b08959a6e32e170b1e8e359`; blocks 2/20/40; 224 input; top-20; consensus threshold 0.10 m; 1,500 iterations; seed 17.
- Projection preflight: uniform 72-frame sample per scan; original 5 cm gate unchanged. All 17 relations passed; no frame removed.
- Scoring: 10,000 environment-cluster bootstrap draws, seed 20261011. Predictions were saved before the separate scoring process opened labels and meshes.
- Measured totals: archive download 134 s; projection 418 s; identity qualification 10.4 s; RGB/depth decode 910.7 s; geometry preparation 8.4 s; GPU features 29.9 s plus 22.9 s model load; CPU ranking 80.9 s; CPU consensus 218.1 s; scoring 38.6 s; bootstrap 2.3 s.
- Scope: quota-limited exploratory result, not full-split or broad independent-site evidence; deterministic-prefix ordering may affect coverage.

The report folder includes environment-level outcomes, per-relation metrics, frame-level projection results, timings, frozen subset manifest, and evaluation scripts. Raw RGB-D, annotations, checkpoints, prediction arrays, caches, machine-specific paths, credentials, and the original session log are omitted. `download_state.json` is a blank resumable state template.
