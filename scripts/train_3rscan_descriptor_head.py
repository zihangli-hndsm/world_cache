"""Train a lightweight cross-scan descriptor head on annotated 3RScan pairs.

The DINOv2 backbone stays frozen.  Training examples are built from RGB-D
patch tokens and the official 3RScan global instance IDs: a rescan token is
matched against nearby reference tokens, with the same global ID as the
positive and other nearby IDs as hard negatives.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, TensorDataset

from worldcache.backbone.dinov2 import extract_block_tokens, load_dinov2
from worldcache.geometry.projection import backproject_pixels
from worldcache.reuse.descriptor import DescriptorProjection, save_descriptor_projection


def load_observation(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path) as data:
        return {key: torch.from_numpy(data[key].copy()) for key in data.files}


def preprocess(rgb: torch.Tensor, device: torch.device) -> torch.Tensor:
    image = rgb.permute(2, 0, 1).float().div(255.0).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
    return ((image - mean) / std).unsqueeze(0)


def patch_world_points(observation: dict[str, torch.Tensor], patch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = observation["depth"].shape
    yy, xx = torch.meshgrid(
        torch.arange(patch_size / 2, height, patch_size),
        torch.arange(patch_size / 2, width, patch_size),
        indexing="ij",
    )
    pixels = torch.stack((xx, yy), dim=-1).reshape(-1, 2).float()
    depth = observation["depth"][pixels[:, 1].long(), pixels[:, 0].long()]
    valid = torch.isfinite(depth) & (depth > 0)
    return backproject_pixels(pixels[valid], depth[valid], observation["intrinsics"], observation["camera_to_world"]), valid


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    return (homogeneous @ matrix.T)[:, :3]


def read_mesh(root: Path, scan_id: str) -> tuple[cKDTree, np.ndarray, np.ndarray]:
    import re

    payload = (root / scan_id / "labels.instances.annotated.v2.ply").read_bytes()
    header_end = payload.index(b"end_header\n") + len(b"end_header\n")
    header = payload[:header_end].decode("ascii")
    vertex_count = int(re.search(r"element vertex (\d+)", header).group(1))
    values = np.fromstring(payload[header_end:].decode("ascii"), sep=" ", dtype=np.float64)
    vertices = values[: vertex_count * 11].reshape(vertex_count, 11)
    return (
        cKDTree(vertices[:, :3].astype(np.float32)),
        vertices[:, 6].astype(np.int64),
        vertices[:, 7].astype(np.int64),
    )


@torch.no_grad()
def extract_side(
    pair_run: Path,
    entries: list[dict[str, object]],
    backbone: torch.nn.Module,
    layer: int,
    device: torch.device,
    tree: cKDTree,
    global_ids: np.ndarray,
    rescan_to_reference: np.ndarray | None,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    points_all: list[torch.Tensor] = []
    tokens_all: list[torch.Tensor] = []
    ids_all: list[np.ndarray] = []
    for entry in entries:
        observation = load_observation(pair_run / str(entry["path"]))
        points, valid = patch_world_points(observation, patch_size=14)
        tokens = extract_block_tokens(
            backbone, preprocess(observation["rgb"], device), [layer]
        )[layer][0, backbone.num_prefix_tokens :].cpu()[valid]
        points_np = points.numpy()
        if rescan_to_reference is not None:
            points_np = transform_points(points_np, np.linalg.inv(rescan_to_reference))
        _, nearest = tree.query(points_np, k=1)
        ids = global_ids[nearest]
        keep = ids > 0
        points_all.append(torch.from_numpy(points_np[keep]).float())
        tokens_all.append(tokens[keep].float())
        ids_all.append(ids[keep])
    return torch.cat(points_all), torch.cat(tokens_all), np.concatenate(ids_all)


@torch.no_grad()
def collect_examples(
    pair_run: Path,
    annotation_root: Path,
    backbone: torch.nn.Module,
    layer: int,
    device: torch.device,
    radius: float,
    negatives: int,
    max_queries: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    config = json.loads((pair_run / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((pair_run / "manifest.json").read_text(encoding="utf-8"))
    ref_tree, _, ref_global = read_mesh(annotation_root, config["reference"])
    rescan_tree, _, rescan_global = read_mesh(annotation_root, config["rescan"])
    ref_points, ref_tokens, ref_ids = extract_side(
        pair_run, manifest["reference"], backbone, layer, device, ref_tree, ref_global, None
    )
    rescan_points, rescan_tokens, rescan_ids = extract_side(
        pair_run, manifest["rescan"], backbone, layer, device, rescan_tree, rescan_global,
        None,
    )
    ref_np = ref_points.numpy()
    rescan_np = rescan_points.numpy()
    tree = cKDTree(ref_np)
    rng = random.Random(seed)
    order = list(range(len(rescan_points)))
    rng.shuffle(order)
    order = order[:max_queries]
    ref_by_id: dict[int, list[int]] = {}
    for index, global_id in enumerate(ref_ids):
        ref_by_id.setdefault(int(global_id), []).append(index)

    query_rows: list[torch.Tensor] = []
    candidate_rows: list[torch.Tensor] = []
    usable = 0
    for query_index in order:
        global_id = int(rescan_ids[query_index])
        nearby = tree.query_ball_point(rescan_np[query_index], radius)
        positives = [index for index in nearby if int(ref_ids[index]) == global_id]
        if not positives:
            # A same-instance reference token may be farther than the local
            # hard-negative radius.  Keep the training task local by skipping it.
            continue
        negatives_local = [index for index in nearby if int(ref_ids[index]) != global_id]
        if not negatives_local:
            continue
        positives.sort(key=lambda index: float(np.linalg.norm(ref_np[index] - rescan_np[query_index])))
        selected = negatives_local[:]
        rng.shuffle(selected)
        selected = selected[:negatives]
        if len(selected) < negatives:
            # Fill with globally sampled different instances so every example
            # has a fixed-size contrastive candidate set.
            while len(selected) < negatives:
                candidate = rng.randrange(len(ref_ids))
                if int(ref_ids[candidate]) != global_id and candidate not in selected:
                    selected.append(candidate)
        candidate_indices = [positives[0], *selected]
        query_rows.append(rescan_tokens[query_index])
        candidate_rows.append(ref_tokens[candidate_indices])
        usable += 1
    if not query_rows:
        raise RuntimeError(f"no usable contrastive examples in {pair_run}")
    stats = {
        "rescan_tokens": len(rescan_points),
        "usable_examples": usable,
        "pairs_with_global_positive": sum(int(int(x) in ref_by_id) for x in rescan_ids),
    }
    return torch.stack(query_rows), torch.stack(candidate_rows), stats


def train_projection(
    queries: torch.Tensor,
    candidates: torch.Tensor,
    hidden_width: int,
    output_width: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    temperature: float,
    seed: int,
) -> tuple[DescriptorProjection, list[float]]:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DescriptorProjection(queries.shape[-1], hidden_width, output_width).to(device)
    loader = DataLoader(TensorDataset(queries, candidates), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history: list[float] = []
    model.train()
    for _ in range(epochs):
        losses = []
        for batch_queries, batch_candidates in loader:
            batch_queries = batch_queries.to(device)
            batch_candidates = batch_candidates.to(device)
            query_embedding = model(batch_queries)
            candidate_embedding = model(batch_candidates.reshape(-1, batch_candidates.shape[-1]))
            candidate_embedding = candidate_embedding.reshape(batch_candidates.shape[0], batch_candidates.shape[1], -1)
            logits = torch.einsum("bd,bkd->bk", query_embedding, candidate_embedding) / temperature
            loss = torch.nn.functional.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long, device=device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
    return model.eval(), history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pair-run", type=Path, nargs="+", required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-size", choices=("small", "base", "large"), default="large")
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--radius", type=float, default=0.50)
    parser.add_argument("--negatives", type=int, default=15)
    parser.add_argument("--max-queries-per-pair", type=int, default=2500)
    parser.add_argument("--hidden-width", type=int, default=256)
    parser.add_argument("--output-width", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolution = json.loads((args.train_pair_run[0] / "config.json").read_text(encoding="utf-8"))["resolution"]
    backbone = load_dinov2(args.model_size, resolution, device)
    query_sets = []
    candidate_sets = []
    pair_stats = []
    for pair_run in args.train_pair_run:
        queries, candidates, stats = collect_examples(
            pair_run, args.annotation_root, backbone, args.layer, device,
            args.radius, args.negatives, args.max_queries_per_pair, args.seed,
        )
        query_sets.append(queries)
        candidate_sets.append(candidates)
        pair_stats.append({"pair_run": str(pair_run), **stats})
        print(json.dumps(pair_stats[-1]))
    queries = torch.cat(query_sets)
    candidates = torch.cat(candidate_sets)
    model, history = train_projection(
        queries, candidates, args.hidden_width, args.output_width, args.epochs,
        args.batch_size, args.learning_rate, args.temperature, args.seed,
    )
    checkpoint = args.output_dir / "projection.pt"
    save_descriptor_projection(model.cpu(), str(checkpoint))
    summary = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "examples": len(queries),
        "candidate_count": int(candidates.shape[1]),
        "input_width": int(queries.shape[-1]),
        "pair_stats": pair_stats,
        "loss_history": history,
        "config": {key: str(value) for key, value in vars(args).items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
