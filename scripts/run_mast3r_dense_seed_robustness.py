#!/usr/bin/env python3
"""Extract one fixed dense MASt3R candidate cache per matched relation, then rerun RANSAC seeds."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--mast3r-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("outputs/runs/20261003_mast3r_dense_seed_cache"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/runs/20261003_mast3r_dense_seed_sweep"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 41])
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--checkpoint", default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
    args = parser.parse_args()
    root = args.root.resolve()
    dataset = args.dataset_root.resolve()
    metadata = args.metadata.resolve()
    mast3r = args.mast3r_root.resolve()
    cache_root = args.cache_root if args.cache_root.is_absolute() else root / args.cache_root
    output_root = args.output_root if args.output_root.is_absolute() else root / args.output_root
    audit = pd.read_csv(root / "outputs/reports/20261003_mast3r_dense_train_audit/per_relation.csv")
    paired = audit[audit.dino_audit_available.astype(bool)]
    if len(paired) != 14:
        raise ValueError(f"expected 14 paired relations, got {len(paired)}")
    for item in paired.itertuples(index=False):
        source_run = str(item.source_run)
        pair_run = root / str(item.pair_run)
        cache_dir = cache_root / source_run
        candidate_cache = cache_dir / "candidate_cache.npz"
        extraction_output = cache_dir / "seed_17_extracted"
        if candidate_cache.is_file():
            print(f"reusing candidate cache: {source_run}", flush=True)
            continue
        cache_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, str(root / "scripts/evaluate_3rscan_mast3r_baseline.py"),
            "--pair-run", str(pair_run), "--annotation-root", str(dataset),
            "--metadata", str(metadata), "--output-dir", str(extraction_output),
            "--candidate-cache", str(candidate_cache), "--mast3r-root", str(mast3r),
            "--checkpoint", args.checkpoint, "--match-mode", "dense_topk",
            "--top-k", str(args.top_k), "--image-size", str(args.image_size),
            "--seed", str(args.seeds[0]), "--threshold", str(args.threshold),
            "--iterations", str(args.iterations),
        ]
        print(f"extracting fixed candidates for {source_run}: {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=root, env=os.environ.copy(), check=True)
        if not candidate_cache.is_file():
            raise RuntimeError(f"candidate-cache output missing after successful evaluator: {candidate_cache}")
    consensus_command = [
        sys.executable, str(root / "scripts/evaluate_mast3r_consensus_seeds.py"),
        "--cache-root", str(cache_root), "--output-root", str(output_root),
        "--seeds", *[str(x) for x in args.seeds],
        "--threshold", str(args.threshold), "--iterations", str(args.iterations),
    ]
    print(f"rerunning consensus only: {' '.join(consensus_command)}", flush=True)
    subprocess.run(consensus_command, cwd=root, env=os.environ.copy(), check=True)


if __name__ == "__main__":
    main()
