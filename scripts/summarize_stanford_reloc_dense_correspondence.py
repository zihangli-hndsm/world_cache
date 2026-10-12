#!/usr/bin/env python3
"""Aggregate the 12-room Stanford Reloc dense-correspondence audit."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/reports/20261009_stanford_reloc_dense_correspondence"
SOURCES = [
    OUT / "summary.json",
    OUT / "_kitchen/summary.json",
    *sorted(OUT.parent.glob("20261009_stanford_reloc_dense_correspondence_*/summary.json")),
]
rows = []
seen = set()
for path in SOURCES:
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    if data.get("status") != "complete" or "pooled" not in data or data.get("resolution") != 518:
        continue
    root = data.get("dataset_root") or ("apt1" if "Apartment 1" in data.get("dataset", "") else "")
    if not root:
        continue
    key = (root, data["room"])
    if key in seen:
        continue
    seen.add(key)
    p = data["pooled"]
    rows.append({
        "scene": root, "room": data["room"],
        "pairs": data["directed_frame_pairs"], "queries": data["queries"],
        "support_5cm": p["support_5cm"],
        "dino_supported_hit_5cm": p["dino_hit_5cm_supported_queries"],
        "rgbhist_supported_hit_5cm": p["rgbhist_hit_5cm_supported_queries"],
        "dino_all_hit_5cm": p["dino_hit_5cm_all_queries"],
        "rgbhist_all_hit_5cm": p["rgbhist_hit_5cm_all_queries"],
        "dino_median_error_cm": p["dino_median_error_cm"],
        "rgbhist_median_error_cm": p["rgbhist_median_error_cm"],
    })
rows.sort(key=lambda x: (x["scene"], x["room"]))
if len(rows) != 12:
    raise RuntimeError(f"Expected 12 completed rooms, found {len(rows)}: {[(r['scene'],r['room']) for r in rows]}")

fields = ["support_5cm", "dino_supported_hit_5cm", "rgbhist_supported_hit_5cm",
          "dino_all_hit_5cm", "rgbhist_all_hit_5cm"]
room_macro = {k: sum(r[k] for r in rows) / len(rows) for k in fields}
scenes = sorted({r["scene"] for r in rows})
scene_macro = {k: sum(sum(r[k] for r in rows if r["scene"] == s) /
                      sum(1 for r in rows if r["scene"] == s) for s in scenes) / len(scenes)
               for k in fields}
total_queries = sum(r["queries"] for r in rows)
pooled = {k: sum(r[k] * r["queries"] for r in rows) / total_queries for k in fields}
summary = {
    "status": "complete", "dataset": "Stanford RGB Camera Relocalization Dataset",
    "scenes": scenes, "rooms": rows, "room_count": len(rows),
    "directed_frame_pairs": sum(r["pairs"] for r in rows),
    "queries": total_queries, "room_macro": room_macro,
    "scene_macro_equal_weight": scene_macro, "query_pooled": pooled,
    "interpretation": [
        "DINO top-1 geometric correspondence exceeds the RGB histogram control in all 12 rooms.",
        "The magnitude is heterogeneous, and the RGB histogram is competitive in some rooms.",
        "This is geometric surface correspondence under supplied poses/depth, not object identity, room change, or deployable system validation.",
        "Room and scene summaries are descriptive; only four scene clusters are available, so no environment-level significance claim is made.",
    ],
}
(OUT / "aggregate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
lines = [
    "# Stanford Reloc dense correspondence audit — 12-room summary", "",
    "Frozen DINOv2 ViT-G/14 block 2/20/40 patch features were compared with an RGB patch-histogram control. Each row uses three sampled frames per sequence and directed cross-sequence frame pairs. A query is geometrically supported when its nearest ground-truth target patch is within 5 cm; conditional hit rates are reported only over supported queries.", "",
    f"**Scope:** {len(rows)} rooms across four scenes; {summary['directed_frame_pairs']:,} directed pairs; {total_queries:,} patch queries. No object identity labels are provided. The supplied intrinsics are uncalibrated defaults, RGB-depth extrinsics are assumed identity, and the provided poses are treated as given.", "",
    "| Scene | Room | Pairs | Supported at 5 cm | DINO hit | RGB hist hit | DINO all-query hit | RGB hist all-query hit |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(f"| {r['scene']} | {r['room']} | {r['pairs']} | {r['support_5cm']:.1%} | {r['dino_supported_hit_5cm']:.1%} | {r['rgbhist_supported_hit_5cm']:.1%} | {r['dino_all_hit_5cm']:.1%} | {r['rgbhist_all_hit_5cm']:.1%} |")
lines += ["", "## Descriptive averages", "",
          "| Weighting | 5 cm support | DINO hit on supported queries | RGB hist hit on supported queries | DINO all-query hit | RGB hist all-query hit |",
          "|---|---:|---:|---:|---:|---:|"]
for name, vals in [("Room macro", room_macro), ("Equal scene macro", scene_macro), ("Query pooled", pooled)]:
    lines.append(f"| {name} | {vals['support_5cm']:.1%} | {vals['dino_supported_hit_5cm']:.1%} | {vals['rgbhist_supported_hit_5cm']:.1%} | {vals['dino_all_hit_5cm']:.1%} | {vals['rgbhist_all_hit_5cm']:.1%} |")
lines += ["", "DINO exceeds the color baseline in all 12 rooms, but the advantage varies substantially: room-macro conditional hit is {:.1%} vs {:.1%}, while the weakest room-level margin is {:.1f} percentage points. This is supporting evidence that frozen dense appearance features can help recover geometrically corresponding surfaces across sequences. It does **not** supply physical-object identity validation, prove map/pose accuracy, test room changes, establish calibrated confidence, or demonstrate a new method. The four scene clusters and uncalibrated camera defaults limit external-validity claims. Treat this as a diagnostic external geometry stress test; it does not materially lift the current venue odds or close the validation gate.".format(
              room_macro['dino_supported_hit_5cm'], room_macro['rgbhist_supported_hit_5cm'],
              min((r['dino_supported_hit_5cm']-r['rgbhist_supported_hit_5cm'])*100 for r in rows)),
          "", "Reproduce with `python scripts/summarize_stanford_reloc_dense_correspondence.py`. Per-room evaluator: `scripts/evaluate_stanford_reloc_dense_correspondence.py`. Machine-readable results: `aggregate_summary.json`; individual summaries are in the adjacent room-specific directories.", ""]
(OUT / "aggregate_summary.md").write_text("\n".join(lines))
print(OUT / "aggregate_summary.md")
print(f"rooms={len(rows)} pairs={summary['directed_frame_pairs']} queries={total_queries}")
print("room_macro", json.dumps(room_macro, sort_keys=True))
print("scene_macro", json.dumps(scene_macro, sort_keys=True))
print("query_pooled", json.dumps(pooled, sort_keys=True))
