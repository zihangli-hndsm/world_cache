#!/usr/bin/env python3
"""Run matched ViT-L/ViT-G physical-instance candidate audits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evaluate_3rscan_spatial_consensus import main as evaluate_pair


PAIRS = {
    "20260917_3rscan_pair02_s1": "pair02",
    "20260917_3rscan_pair03_s1": "pair03",
    "20260917_3rscan_pair04_s1": "pair04",
    "20260917_3rscan_pair05_s1": "pair05",
    "20260917_3rscan_target_pair_s1": "target",
    "20260921_3rscan_train_0cac": "0cac",
    "20260921_3rscan_train_20c993": "20c993",
    "20260921_3rscan_train_4aca_extra": "4aca",
    "20260921_3rscan_train_6bde": "6bde",
    "20260921_3rscan_train_95be": "95be",
    "20260921_3rscan_semantic_5630": "5630",
    "20260921_3rscan_semantic_751a": "751a",
    "20260921_3rscan_semantic_c670": "c670",
    "20260921_3rscan_semantic_chairs": "chairs",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--vitl-root", type=Path, required=True)
    parser.add_argument("--vitg-root", type=Path, required=True)
    parser.add_argument("--model", choices=("large", "giant", "both"), default="both")
    parser.add_argument("--top-k", type=int, nargs="+", default=[20])
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--iterations", type=int, default=1500)
    args = parser.parse_args()

    specs = {
        "large": ("large", (2, 12, 24), args.vitl_root, None),
        "giant": ("giant", (2, 20, 40), args.vitg_root, None),
    }
    requested = ("large", "giant") if args.model == "both" else (args.model,)
    pairs = tuple(PAIRS.items())
    if args.source_only:
        pairs = tuple((pair, short) for pair, short in pairs if "target" not in pair)
    for selected_model in requested:
        model, layers, output_root, cached_g_pattern = specs[selected_model]
        for pair, short in pairs:
            pair_run = (
                args.repo_root / cached_g_pattern.format(short=short)
                if cached_g_pattern
                else args.repo_root / "outputs/runs" / pair
            )
            output_dir = output_root / pair
            status_path = output_dir / "status.json"
            if status_path.is_file():
                print(f"SKIP existing {model} {pair}", flush=True)
                continue
            print(f"RUN {model} {pair}", flush=True)
            sys.argv = [
                "evaluate_3rscan_spatial_consensus.py",
                "--pair-run", str(pair_run),
                "--annotation-root", str(args.annotation_root),
                "--output-dir", str(output_dir),
                "--model-size", model,
                "--layers", *(str(layer) for layer in layers),
                "--top-k", *(str(k) for k in args.top_k),
                "--threshold", "0.10",
                "--iterations", str(args.iterations),
                "--save-query-diagnostics",
                "--save-candidate-set-audit",
                "--save-physical-instance-candidate-audit",
            ]
            evaluate_pair()


if __name__ == "__main__":
    main()
