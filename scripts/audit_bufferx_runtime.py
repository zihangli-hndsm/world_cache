#!/usr/bin/env python3
"""Check upstream BUFFER-X CUDA operators and complete pretrained state loading."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("third_party/BUFFER-X"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/20261008_bufferx_preflight"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo))
    import torch
    import pointnet2_ops.pointnet2_utils as pointnet
    from knn_cuda import KNN
    from torch_batch_svd import svd
    from config import make_cfg
    from models.BUFFERX import BufferX

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    device = torch.device("cuda:0")
    points = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [4., 0., 0.]]], device=device)
    indices = pointnet.furthest_point_sample(points, 3)
    gathered = pointnet.gather_operation(points.transpose(1, 2).contiguous(), indices)
    torch.testing.assert_close(gathered.transpose(1, 2), points[torch.arange(1, device=device)[:, None], indices.long()])
    if len(torch.unique(indices)) != 3:
        raise AssertionError("FPS did not return three distinct points")
    queries = torch.tensor([[[.1, 0., 0.], [3.9, 0., 0.]]], device=device)
    distances, neighbors = KNN(k=1, transpose_mode=True)(points, queries)
    if neighbors.flatten().tolist() != [0, 3]:
        raise AssertionError("upstream KNN disagrees with exact nearest neighbors")
    torch.testing.assert_close(distances.flatten(), torch.tensor([.1, .1], device=device), atol=1e-5, rtol=1e-5)
    torch.manual_seed(17)
    matrix = torch.randn(8, 3, 3, device=device)
    u, singular, v = svd(matrix)
    reconstruction = u @ torch.diag_embed(singular) @ v.transpose(-1, -2)
    torch.testing.assert_close(reconstruction, matrix, atol=1e-4, rtol=1e-4)

    cfg = make_cfg("3DMatch", repo / "unused_dataset_root")
    cfg.stage = "test"
    cfg.match.pose_estimator = "ransac"
    cfg.match.enable_early_exit = False
    model = BufferX(cfg)
    weights = {}
    records = []
    for stage in cfg.train.all_stage:
        path = repo / "snapshot" / "threedmatch" / stage / "best.pth"
        state = torch.load(path, map_location="cpu", weights_only=True)
        selected = {k: value for k, value in state.items() if k.startswith(stage + ".")}
        if not selected:
            raise ValueError(f"checkpoint has no {stage} parameters")
        if set(weights) & set(selected):
            raise ValueError("overlapping checkpoint stages")
        weights.update(selected)
        records.append({"stage": stage, "keys": len(selected), "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    expected_keys = set(model.state_dict())
    if set(weights) != expected_keys:
        raise ValueError(f"incomplete model state: missing={sorted(expected_keys-set(weights))}, extra={sorted(set(weights)-expected_keys)}")
    model.load_state_dict(weights, strict=True)
    model.to(device).eval()
    result = {
        "status": "pass", "repo_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "operators": ["FPS/gather", "exact KNN", "batched SVD reconstruction"],
        "model_state_keys": len(expected_keys), "checkpoints": records,
        "pose_solver": "ransac", "early_exit": False,
        "scope": "runtime feasibility only; no scene-pair inference or benchmark result",
    }
    (output / "runtime_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
