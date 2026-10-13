#!/usr/bin/env python3
"""Feasibility probe: reconstruct a Rescan RGB sequence with local MASt3R.

Uses RGB only. The annotated PLY is not read here; a separate geometry-only
alignment step can compare this exported point cloud with scan geometry.
"""
import argparse
import importlib
import pathlib
import sys
import types

import numpy as np
import torch


def install_vendored_namespace():
    root = pathlib.Path(__file__).resolve().parents[1]
    mast3r_root = root / "third_party/segmast3r/mast3r_src"
    sys.path[:0] = [str(root / "third_party/segmast3r"), str(mast3r_root)]
    import mast3r
    shim = types.ModuleType("mast3r_src")
    shim.__path__ = [str(mast3r_root)]
    sys.modules["mast3r_src"] = shim
    for name in ("utils", "cloud_opt", "fast_nn"):
        sys.modules[f"mast3r_src.{name}"] = importlib.import_module(f"mast3r.{name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--niter1", type=int, default=200)
    ap.add_argument("--niter2", type=int, default=100)
    args = ap.parse_args()
    install_vendored_namespace()
    from mast3r.model import AsymmetricMASt3R
    from dust3r.image_pairs import make_pairs
    from dust3r.utils.image import load_images
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment

    frames = sorted(pathlib.Path(args.frames).glob("*.jpg"))
    if len(frames) < 3:
        raise SystemExit(f"Need >=3 frames; found {len(frames)}")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "cache"
    cache.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AsymmetricMASt3R.from_pretrained(
        "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    ).to(device).eval()
    imgs = load_images([str(p) for p in frames], size=512, verbose=True)
    pairs = make_pairs(imgs, scene_graph=f"swin-{args.window}-noncyclic",
                       prefilter=None, symmetrize=True)
    print(f"frames={len(frames)} pairs={len(pairs)} device={device}", flush=True)
    scene = sparse_global_alignment(
        [str(p) for p in frames], pairs, str(cache), model,
        device=device, shared_intrinsics=True,
        lr1=0.15, niter1=args.niter1, lr2=0.01, niter2=args.niter2,
        opt_depth=True, matching_conf_thr=5.,
    )
    poses = scene.get_im_poses().detach().cpu().numpy()
    intrinsics = np.stack([k.detach().cpu().numpy() for k in scene.intrinsics])
    focals = scene.get_focals().detach().cpu().numpy()
    centers = poses[:, :3, 3]
    np.savez_compressed(out / "mast3r_reconstruction.npz",
                        frame_paths=np.array([p.name for p in frames]),
                        cam2world=poses, intrinsics=intrinsics, focals=focals,
                        centers=centers)
    pts, _, confs = scene.get_dense_pts3d(clean_depth=False)
    xyz_parts, rgb_parts, conf_parts = [], [], []
    for im, p, c in zip(scene.imgs, pts, confs):
        p = p.detach().cpu().numpy().reshape(*im.shape[:2], 3)[::8, ::8]
        c = c.detach().cpu().numpy().reshape(*im.shape[:2])[::8, ::8]
        rgb = im[::8, ::8]
        valid = np.isfinite(p).all(axis=-1) & np.isfinite(c) & (c > 1)
        xyz_parts.append(p[valid])
        rgb_parts.append(rgb[valid])
        conf_parts.append(c[valid])
    np.savez_compressed(out / "mast3r_points.npz",
                        xyz=np.concatenate(xyz_parts).astype(np.float32),
                        rgb=np.concatenate(rgb_parts).astype(np.float32),
                        confidence=np.concatenate(conf_parts).astype(np.float32))
    print(f"pose_center_span_mast3r={np.linalg.norm(centers.max(0)-centers.min(0)):.4f}")
    print(f"saved_frames={len(poses)} sampled_points={sum(map(len, xyz_parts))}")


if __name__ == "__main__":
    main()
