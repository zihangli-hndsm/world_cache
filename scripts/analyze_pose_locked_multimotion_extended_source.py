#!/usr/bin/env python3
"""Source-only LOPO audit of pose-locked multi-motion on extra train pairs.

This is deliberately separate from the 17-pair headline protocol.  It loads
the original 16 source caches plus locally complete extra relations, fits the
same label-free inference branch, and selects its support/margin settings
using relation-held-out source transfer labels only.  There is no target row
and no target scoring in this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_3rscan_multimotion_rescue_mechanisms import (
    MARGINS_M,
    MIN_SUPPORTS,
    MODE_COUNTS,
    predict,
    read_pair,
    relation_bootstrap,
)


def environment_groups(input_dirs: list[Path], metadata_path: Path) -> dict[str, dict[str, str]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    by_scan: dict[str, tuple[str, str]] = {}
    for scene in metadata:
        env = str(scene["reference"])
        split = str(scene.get("type", "unknown"))
        by_scan[env] = (env, split)
        for scan in scene.get("scans", []):
            by_scan[str(scan["reference"])] = (env, split)
    result: dict[str, dict[str, str]] = {}
    for input_dir in input_dirs:
        for cache_path in sorted((input_dir / "descriptor_candidates").glob("*.npz")):
            pair = cache_path.stem
            with np.load(cache_path) as cache:
                ref = str(cache["reference_id"].item())
                rescan = str(cache["rescan_id"].item())
            if ref not in by_scan or rescan not in by_scan:
                raise ValueError(f"metadata is missing scans for {pair}")
            ref_env, ref_split = by_scan[ref]
            rescan_env, rescan_split = by_scan[rescan]
            if ref_env != rescan_env or ref_split != rescan_split:
                raise ValueError(f"environment/split mismatch for {pair}")
            result[pair] = {"environment_id": ref_env, "official_split": ref_split}
    return result


def choose_lopo(data_by_pair: dict[str, dict[str, object]]) -> tuple[pd.DataFrame, dict[str, object]]:
    pairs = sorted(data_by_pair)
    if len(pairs) < 3:
        raise ValueError("extended source audit requires at least three relations")
    rows: list[dict[str, object]] = []
    for heldout in pairs:
        training = [pair for pair in pairs if pair != heldout]
        scores: dict[tuple[int, float, int], int] = {}
        for config in ((m, g, s) for m in MODE_COUNTS for g in MARGINS_M for s in MIN_SUPPORTS):
            scores[config] = sum(int(predict(data_by_pair[pair], *config)[0].sum()) for pair in training)
        best = max(scores.values())
        chosen = min(
            (config for config, score in scores.items() if score == best),
            key=lambda config: (config[0], -config[2], -config[1]),
        )
        correct, selected_mode, _, _ = predict(data_by_pair[heldout], *chosen)
        data = data_by_pair[heldout]
        moved = data["moved"]
        baseline = data["baseline_correct"]
        rows.append({
            "pair": heldout,
            "modes": chosen[0],
            "margin_m": chosen[1],
            "min_independent_support": chosen[2],
            "training_relations": len(training),
            "training_correct_transfers": best,
            "moved_queries": int(moved.sum()),
            "baseline_correct": int(baseline.sum()),
            "method_correct": int(correct.sum()),
            "rescued": int((correct & ~baseline).sum()),
            "lost": int((baseline & ~correct).sum()),
            "selected_secondary_queries": int((selected_mode > 0).sum()),
            "baseline_coverage": float(baseline.sum() / moved.sum()) if moved.any() else np.nan,
            "method_coverage": float(correct.sum() / moved.sum()) if moved.any() else np.nan,
        })
    outcomes = pd.DataFrame(rows)
    return outcomes, {"pairs": pairs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pair_dirs: dict[str, Path] = {}
    for input_dir in args.input_dir:
        for cache_path in sorted((input_dir / "descriptor_candidates").glob("*.npz")):
            pair = cache_path.stem
            # The primary 17-pair directory also contains the predesignated
            # target cache.  This script is source-only by construction.
            if pair == "target":
                continue
            if pair in pair_dirs:
                raise ValueError(f"duplicate pair cache: {pair}")
            pair_dirs[pair] = input_dir
    data_by_pair = {
        pair: read_pair(pair_dirs[pair], pair, args.annotation_root, seed=args.seed)
        for pair in sorted(pair_dirs)
    }
    outcomes, meta = choose_lopo(data_by_pair)
    groups = environment_groups(args.input_dir, args.metadata)
    outcomes["environment_id"] = outcomes.pair.map(lambda pair: groups[pair]["environment_id"])
    outcomes["official_split"] = outcomes.pair.map(lambda pair: groups[pair]["official_split"])
    if outcomes.environment_id.nunique() != len(outcomes):
        raise ValueError("source relations share environment groups")
    outcomes["delta"] = outcomes.method_correct - outcomes.baseline_correct
    outcomes["delta_pp"] = np.where(
        outcomes.moved_queries > 0,
        100.0 * outcomes.delta / outcomes.moved_queries,
        np.nan,
    )
    outcomes.to_csv(args.output_dir / "source_lopo_outcomes.csv", index=False)
    bootstrap = relation_bootstrap(outcomes)
    bootstrap.to_csv(args.output_dir / "relation_bootstrap_ci.csv", index=False)

    eligible = outcomes[outcomes.moved_queries > 0]
    summary = {
        "seed": args.seed,
        "source_relations": len(outcomes),
        "eligible_source_relations": len(eligible),
        "official_split": sorted(outcomes.official_split.unique().tolist()),
        "source_moved_rigid_queries": int(eligible.moved_queries.sum()),
        "baseline_correct": int(eligible.baseline_correct.sum()),
        "method_correct": int(eligible.method_correct.sum()),
        "coverage_delta_pp": float(100.0 * eligible.delta.sum() / eligible.moved_queries.sum()),
        "rescued": int(eligible.rescued.sum()),
        "lost": int(eligible.lost.sum()),
        "nonnegative_source_relations": int((eligible.delta >= 0).sum()),
        "positive_source_relations": int((eligible.delta > 0).sum()),
        "relation_bootstrap_weighted_ci95_pp": [
            float(bootstrap.iloc[0].ci95_lower), float(bootstrap.iloc[0].ci95_upper)
        ],
        "relation_bootstrap_macro_ci95_pp": [
            float(bootstrap.iloc[1].ci95_lower), float(bootstrap.iloc[1].ci95_upper)
        ],
        "target_used_for_selection": False,
        "input_pairs": meta["pairs"],
        "method": "global top-20 scene pose locked; source-LOPO selected secondary motion switch",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Extended source-only pose-locked multi-motion audit",
        "",
        f"RANSAC seed {args.seed}; {len(outcomes)} source relations, "
        f"{len(eligible)} with linked moved-object queries. The branch uses the "
        "same frozen candidate cache, source-only LOPO selector, and global "
        "top-20 scene-pose lock as the 16-source audit. No target relation is "
        "included or scored.",
        "",
        f"On {int(eligible.moved_queries.sum()):,} linked queries, global RANSAC "
        f"transfers {int(eligible.baseline_correct.sum()):,} correctly and the "
        f"pose-locked branch transfers {int(eligible.method_correct.sum()):,} "
        f"({summary['coverage_delta_pp']:+.2f} pp). Rescued/lost: "
        f"{summary['rescued']}/{summary['lost']}; nonnegative relations: "
        f"{summary['nonnegative_source_relations']}/{len(eligible)}. Weighted "
        f"relation-bootstrap 95% CI: [{summary['relation_bootstrap_weighted_ci95_pp'][0]:+.2f}, "
        f"{summary['relation_bootstrap_weighted_ci95_pp'][1]:+.2f}] pp; macro CI: "
        f"[{summary['relation_bootstrap_macro_ci95_pp'][0]:+.2f}, "
        f"{summary['relation_bootstrap_macro_ci95_pp'][1]:+.2f}] pp.",
        "",
        "| Pair | Queries | Baseline | Method | Gain (pp) | Modes | Margin | Support |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in outcomes.itertuples(index=False):
        lines.append(
            f"| {row.pair} | {row.moved_queries} | {row.baseline_correct} | {row.method_correct} | "
            f"{row.delta_pp:+.2f} | {row.modes} | {row.margin_m:.2f} | {row.min_independent_support} |"
        )
    lines += [
        "",
        "The two extra relations were selected because their raw train inputs "
        "are locally complete; this is an availability-selected train audit, "
        "not an independent-site validation. The scene pose is frozen to the "
        "global baseline, so additional pose loss is zero by construction.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
