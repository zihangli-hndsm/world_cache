#!/usr/bin/env python3
"""Run official BUFFER-X on native RGB-D geometry; read GT only after inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import subprocess
import time

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=Path('third_party/BUFFER-X'))
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    import open3d as o3d
    import torch
    from config import make_cfg
    from models.BUFFERX import BufferX
    from utils.tools import sphericity_based_voxel_analysis

    manifest = json.loads((args.geometry_dir / 'manifest.json').read_text())
    if manifest['coordinate_convention'] != 'native_scan_frames' or manifest['annotation_or_cross_scan_transform_used']:
        raise ValueError('unaligned native geometry required')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    o3d.utility.random.seed(args.seed)
    with np.load(args.geometry_dir / 'native_geometry.npz') as geometry:
        clouds = []
        for arm in ('rescan', 'reference'):
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(geometry[arm + '_points'])
            clouds.append(cloud)
    voxel, sphericity, pca_aligned = sphericity_based_voxel_analysis(*clouds)
    arrays = [np.asarray(cloud.voxel_down_sample(voxel).points).astype(np.float32) for cloud in clouds]
    for array in arrays:
        np.random.shuffle(array)
    cfg = make_cfg('3DMatch', args.repo.resolve() / 'unused_dataset_root')
    cfg.stage = 'test'
    cfg.match.pose_estimator = 'ransac'
    cfg.match.enable_early_exit = False
    cfg.test.enable_timing = True
    if min(map(len, arrays)) < max(cfg.patch.num_fps, cfg.patch.num_points_radius_estimate):
        raise ValueError('insufficient distinct downsampled points for upstream FPS')
    model = BufferX(cfg)
    weights = {}
    for stage in cfg.train.all_stage:
        state = torch.load(args.repo / 'snapshot' / 'threedmatch' / stage / 'best.pth', map_location='cpu', weights_only=True)
        weights.update({k: v for k, v in state.items() if k.startswith(stage + '.')})
    model.load_state_dict(weights, strict=True)
    model.cuda().eval()
    data = {'src_fds_pcd': torch.from_numpy(arrays[0]).cuda(),
            'tgt_fds_pcd': torch.from_numpy(arrays[1]).cuda(),
            'is_aligned_to_global_z': cfg.patch.is_aligned_to_global_z,
            'src_id': manifest['rescan'], 'tgt_id': manifest['reference']}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        result = model(data)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if result is None:
        raise RuntimeError('upstream model returned no pose')
    pose, times, inliers, mutual, inlier_ind, scales = result
    pose = np.asarray(pose)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError('invalid predicted pose')
    # Persist the prediction before opening any ground-truth alignment.
    np.save(args.output_dir / 'predicted_rescan_to_reference.npy', pose)
    config = json.loads((Path(manifest['pair_run']) / 'config.json').read_text())
    gt = np.asarray(config['rescan_to_reference'])
    relative = pose[:3, :3] @ gt[:3, :3].T
    rotation = np.degrees(np.arccos(np.clip((np.trace(relative)-1)/2, -1, 1)))
    translation = np.linalg.norm(pose[:3, 3] - gt[:3, 3])
    record = {'status': 'complete', 'reference': manifest['reference'], 'rescan': manifest['rescan'],
              'repo_commit': subprocess.check_output(['git', '-C', str(args.repo), 'rev-parse', 'HEAD'], text=True).strip(),
              'geometry_sha256': hashlib.sha256((args.geometry_dir / 'native_geometry.npz').read_bytes()).hexdigest(),
              'seed': args.seed, 'geometry_dir': str(args.geometry_dir), 'voxel_size_m': voxel,
              'sphericity': float(sphericity), 'pca_alignment_flag': bool(pca_aligned),
              'model_alignment_flag': bool(cfg.patch.is_aligned_to_global_z),
              'input_points_src_tgt': list(map(len, arrays)), 'pose_solver': 'ransac',
              'early_exit': False, 'pose_refine': bool(cfg.test.pose_refine),
              'elapsed_seconds': elapsed, 'upstream_times_seconds': list(map(float, times)),
              'inliers': int(inliers), 'mutual_inliers': int(mutual), 'inlier_ind': int(inlier_ind),
              'scales_used': int(scales), 'rotation_error_degrees': float(rotation),
              'translation_error_m': float(translation), 'predicted_pose': pose.tolist(),
              'scope': 'single relation pose estimate; no aggregate or transfer performance claim',
              'gt_used_for_inference': False}
    (args.output_dir / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
