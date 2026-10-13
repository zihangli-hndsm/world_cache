#!/usr/bin/env python3
"""Cross-session YCB-label retrieval from native ObChange RGB-D views.

The official point annotations are projected into RGB frames using recorded
camera calibration and TF, then depth-gated. This is a privileged-mask,
fine-grained object-label retrieval stress test; YCB labels are not physical
specimen IDs. Bags are streamed and never copied into the result directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_obchange_rgbd_pose import (  # noqa: E402
    collect_bag_inputs,
    map_to_camera,
    stamp_ns,
)
from worldcache.backbone.dinov2 import extract_block_tokens, load_dinov2

MEAN = np.asarray([0.485, 0.456, 0.406], np.float32)
STD = np.asarray([0.229, 0.224, 0.225], np.float32)


def load_labeled_points(pcd_dir: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    all_points, labels = [], defaultdict(list)
    offset = 0
    for pcd in sorted(pcd_dir.glob("*/merged_plane_clouds_ds002.pcd")):
        cloud = o3d.io.read_point_cloud(str(pcd))
        points = np.asarray(cloud.points, dtype=np.float32)
        all_points.append(points)
        anno = pcd.with_name(pcd.stem + "_GT.anno")
        if anno.exists():
            for line in anno.read_text(errors="replace").splitlines():
                parts = line.split()
                if len(parts) < 2:
                    continue
                # Annotation indices are local to the corresponding PCD.
                idx_values = [int(x) for x in parts[1:] if x.lstrip("-").isdigit()]
                idx = np.asarray(idx_values, dtype=np.int64)
                idx = idx[(idx >= 0) & (idx < len(points))]
                if len(idx):
                    labels[parts[0]].append(idx + offset)
        offset += len(points)
    points = np.concatenate(all_points, axis=0)
    return points, {k: np.unique(np.concatenate(v)) for k, v in labels.items()}


def project(points: np.ndarray, map_to_cam: np.ndarray, camera: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # map_to_camera() follows the stored TF parent->child convention and
    # returns camera-to-map; invert it before projecting map-frame points.
    cam_from_map = np.linalg.inv(map_to_cam)
    xyz = points @ cam_from_map[:3, :3].T + cam_from_map[:3, 3]
    z = xyz[:, 2]
    good = np.isfinite(xyz).all(1) & (z > .2) & (z < 8.)
    xyz, z = xyz[good], z[good]
    x, y = xyz[:, 0] / z, xyz[:, 1] / z
    d = camera["D"]
    if camera["distortion_model"] == "plumb_bob" and d.size >= 5:
        k1, k2, p1, p2, k3 = d[:5]
        r2 = x*x + y*y
        radial = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
        x, y = x*radial + 2*p1*x*y + p2*(r2+2*x*x), y*radial + p1*(r2+2*y*y) + 2*p2*x*y
    K = camera["K"]
    u = np.rint(K[0, 0]*x + K[0, 2]).astype(np.int32)
    v = np.rint(K[1, 1]*y + K[1, 2]).astype(np.int32)
    inside = (u >= 0) & (u < camera["width"]) & (v >= 0) & (v < camera["height"])
    return u[inside], v[inside], z[inside]


def crop_view(rgb: np.ndarray, mask: np.ndarray) -> tuple[torch.Tensor, np.ndarray] | None:
    ys, xs = np.where(mask)
    if len(xs) < 80:
        return None
    h, w = mask.shape
    pad = max(8, int(.18 * max(xs.max()-xs.min()+1, ys.max()-ys.min()+1)))
    x0, x1 = max(0, xs.min()-pad), min(w, xs.max()+pad+1)
    y0, y1 = max(0, ys.min()-pad), min(h, ys.max()+pad+1)
    im = Image.fromarray(rgb[y0:y1, x0:x1]).convert("RGB")
    ma = Image.fromarray((mask[y0:y1, x0:x1].astype(np.uint8)*255))
    scale = 224 / max(im.size)
    nw, nh = max(1, round(im.width*scale)), max(1, round(im.height*scale))
    im = im.resize((nw, nh), Image.Resampling.BICUBIC)
    ma = ma.resize((nw, nh), Image.Resampling.NEAREST)
    canvas = np.round(MEAN*255).astype(np.uint8)
    arr = np.broadcast_to(canvas, (224,224,3)).copy()
    m = np.zeros((224,224), bool)
    x, y = (224-nw)//2, (224-nh)//2
    m[y:y+nh,x:x+nw] = np.asarray(ma) > 0
    image = np.asarray(im)
    arr[y:y+nh,x:x+nw] = np.where(m[y:y+nh,x:x+nw,None], image, arr[y:y+nh,x:x+nw])
    tensor = torch.from_numpy(arr.copy()).permute(2,0,1).float()/255
    tensor = (tensor-torch.from_numpy(MEAN)[:,None,None])/torch.from_numpy(STD)[:,None,None]
    return tensor, m


def color_feature(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.concatenate([np.histogram(rgb[:,:,c][mask], bins=16, range=(0,256))[0] for c in range(3)]).astype(np.float32)
    values /= max(values.sum(), 1.)
    return values / max(np.linalg.norm(values), 1e-12)


@torch.inference_mode()
def extract_session(bag: Path, pcd_dir: Path, device: torch.device, samples: int) -> dict[str, dict[str, np.ndarray]]:
    camera, rgbs, depths, dynamic, static = collect_bag_inputs(bag, samples)
    points, label_indices = load_labeled_points(pcd_dir)
    # Only labels with enough annotated points can form a measurable projected mask.
    label_indices = {k:v for k,v in label_indices.items() if len(v) >= 30}
    model = load_dinov2("giant", 224, device)
    feats, colors, meta = defaultdict(list), defaultdict(list), defaultdict(list)
    for frame_i, (rgb_ts, rgb) in enumerate(rgbs):
        dt, dep = min(depths, key=lambda row: abs(row[0]-rgb_ts))
        if abs(dt-rgb_ts) > 40_000_000:
            continue
        pose = map_to_camera(rgb_ts, dynamic, static)
        if pose is None:
            continue
        valid_depth = (dep > 200) & (dep < 8000)
        views, masks, color_features, names = [], [], [], []
        for label, indices in label_indices.items():
            u,v,z = project(points[indices], pose, camera)
            visible = valid_depth[v,u] & (np.abs(dep[v,u].astype(np.float32)/1000-z) < .05)
            if visible.sum() < 80:
                continue
            mask = np.zeros((camera["height"],camera["width"]), bool)
            mask[v[visible],u[visible]] = True
            # Fill tiny projection gaps while retaining the depth-validated support.
            try:
                from scipy.ndimage import binary_dilation
                mask = binary_dilation(mask, iterations=2)
            except ImportError:
                pass
            view = crop_view(rgb, mask)
            if view is None:
                continue
            tensor, cropmask = view
            views.append(tensor); masks.append(cropmask); color_features.append(color_feature(rgb, mask)); names.append(label)
        if not views:
            continue
        captured = extract_block_tokens(model, torch.stack(views).to(device), (2,20,40))
        tokens = torch.cat([F.normalize(captured[layer][:, model.num_prefix_tokens:], dim=2) for layer in (2,20,40)], dim=2)
        weights = F.avg_pool2d(torch.from_numpy(np.stack(masks).astype(np.float32))[:,None], 14,14).flatten(1).to(tokens.device)
        desc = (tokens*weights[:,:,None]).sum(1)/weights.sum(1).clamp_min(1e-6)[:,None]
        desc = F.normalize(desc, dim=1).cpu().numpy().astype(np.float32)
        for j,label in enumerate(names):
            feats[label].append(desc[j]); colors[label].append(color_features[j]); meta[label].append(frame_i)
        print(f"frame {frame_i+1}/{len(rgbs)}: {len(names)} labeled masks", flush=True)
    del model
    result = {}
    for label in sorted(feats):
        if len(feats[label]) >= 2:
            result[label] = {"dino":np.stack(feats[label]), "color":np.stack(colors[label]), "frame":np.asarray(meta[label])}
    return result


def compare(source: dict, query: dict, min_views: int) -> dict:
    common = sorted(set(source)&set(query))
    common = [x for x in common if len(source[x]["dino"])>=min_views and len(query[x]["dino"])>=min_views]
    if len(common) < 2:
        raise RuntimeError(f"fewer than two labels have >= {min_views} views in both sessions: {common}")
    def score(metric: str) -> dict:
        bank = {lab: source[lab][metric].mean(0) for lab in common}
        bank = {lab: vec/(np.linalg.norm(vec)+1e-12) for lab,vec in bank.items()}
        rows=[]
        for lab in common:
            candidates=np.stack([bank[x] for x in common])
            sims=query[lab][metric]@candidates.T
            pred=np.argmax(sims,axis=1)
            rows.extend((lab, common[int(i)]) for i in pred)
        per_label={lab:float(np.mean([a==b for a,b in rows if a==lab])) for lab in common}
        acc=np.mean([a==b for a,b in rows])
        return {"micro_top1":float(acc),"macro_label_top1":float(np.mean(list(per_label.values()))),"chance":1/len(common),"per_label_top1":per_label,"query_views":len(rows)}
    return {"common_labels":common,"n_labels":len(common),"source_views":{x:len(source[x]["dino"]) for x in common},"query_views":{x:len(query[x]["dino"]) for x in common},"dino":score("dino"),"rgb_histogram_control":score("color")}


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--source-bag",type=Path,required=True); p.add_argument("--source-pcd",type=Path,required=True)
    p.add_argument("--query-bag",type=Path,required=True); p.add_argument("--query-pcd",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--samples",type=int,default=24)
    p.add_argument("--device",default="cuda"); p.add_argument("--min-views",type=int,default=2)
    a=p.parse_args(); device=torch.device(a.device)
    print("extracting source session",flush=True); src=extract_session(a.source_bag,a.source_pcd,device,a.samples)
    print("extracting query session",flush=True); qry=extract_session(a.query_bag,a.query_pcd,device,a.samples)
    result={"dataset":"TU Wien ObChange","protocol":"source/query are separate recorded sessions; point labels projected into native RGB with TF/intrinsics and depth gated at 5cm; DINOv2-G/14 blocks 2/20/40, annotation-derived object masks; query top-1 over labels shared by both sessions; YCB label retrieval, not specimen identity","source_bag":a.source_bag.name,"query_bag":a.query_bag.name,"samples_per_session":a.samples,"result":compare(src,qry,a.min_views),"per_session_view_counts":{"source":{k:len(v['dino']) for k,v in src.items()},"query":{k:len(v['dino']) for k,v in qry.items()}}}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
