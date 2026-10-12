"""Frozen DINO segment pooling on the identical anonymous segment inputs."""
import hashlib
import argparse
import json
import os
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_3rscan_instance_correspondence import preprocess
from worldcache.backbone.dinov2 import extract_block_tokens, load_dinov2


@torch.inference_mode()
def pool(model, pack, device):
    captured = extract_block_tokens(model, preprocess(torch.from_numpy(pack['rgb'].copy()), device), (2, 20, 40))
    blocks = [F.normalize(captured[layer][0, model.num_prefix_tokens:], dim=1) for layer in (2, 20, 40)]
    tokens = F.normalize(torch.cat(blocks, dim=1), dim=1)
    feature = tokens.T.reshape(1, tokens.shape[1], 16, 16)
    feature = F.interpolate(feature, size=(224, 224), mode='bilinear', align_corners=False)[0].flatten(1)
    masks = torch.from_numpy(pack['masks'].copy()).float().flatten(1)
    if len(masks) == 0:
        return np.empty((0, tokens.shape[1]), np.float32)
    pooled = masks @ feature.T / masks.sum(1, keepdim=True).clamp_min(1)
    return F.normalize(pooled, dim=1).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, default=Path('outputs/reports/20261008_segmast3r_feasibility'))
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    torch.manual_seed(17)
    output = args.output_root
    frozen = json.loads((output / 'frozen_segment_population.json').read_text())
    device = torch.device('cuda')
    model = load_dinov2('giant', 224, device)
    rows = []
    descriptor_cache = {}
    for relation in frozen['population']:
        root = output / 'uniform20_inputs' / relation['run_id']
        manifest = json.loads((root / 'manifest.json').read_text())
        destination = output / 'dino_segment_predictions' / relation['run_id']
        destination.mkdir(parents=True, exist_ok=True)
        manifest_hash = hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest()
        summary_path = destination / 'inference_summary.json'
        if summary_path.exists():
            previous = json.loads(summary_path.read_text())
            if previous.get('input_manifest_sha256') == manifest_hash and previous.get('status') == 'complete':
                rows.append(dict(run_id=relation['run_id'], status='complete', frame_pairs=len(previous['records'])))
                (output / 'dino_inference_manifest.json').write_text(json.dumps(rows, indent=2) + '\n')
                print(f"reused{len(rows)}/{len(frozen['population'])} {relation['run_id']}", flush=True)
                continue
        records = []
        for ref, query in zip(manifest['records']['reference'], manifest['records']['rescan']):
            descriptors = []
            for record in [ref, query]:
                path = Path(record['input'])
                if hashlib.sha256(path.read_bytes()).hexdigest() != record['input_sha256']:
                    raise ValueError('anonymous input hash mismatch')
                key = record['input_sha256']
                if key not in descriptor_cache:
                    with np.load(path) as pack:
                        descriptor_cache[key] = pool(model, pack, device)
                descriptors.append(descriptor_cache[key])
            # Query rows / reference columns; candidate scores precede labels.
            scores = descriptors[1] @ descriptors[0].T
            selected = np.full(len(descriptors[1]), -1, np.int64)
            mutual = np.zeros(len(selected), bool)
            if min(scores.shape) > 0:
                selected = scores.argmax(1)
                reverse = scores.argmax(0)
                mutual = reverse[selected] == np.arange(len(selected))
            name = f"ref{ref['frame_id']:06d}_query{query['frame_id']:06d}"
            prediction = destination / (name + '_anonymous_predictions.npz')
            np.savez_compressed(prediction, scores=scores, query_to_reference=selected, mutual_assigned=mutual)
            records.append(dict(reference_frame=ref['frame_id'], query_frame=query['frame_id'],
                                reference_masks=scores.shape[1], query_masks=scores.shape[0],
                                predictions=str(prediction)))
        summary = dict(status='complete', run_id=relation['run_id'], records=records,
                       input_manifest_sha256=manifest_hash,
                       backbone='vit_giant_patch14_dinov2.lvd142m', layers=[2, 20, 40],
                       pooling='Normalized per-block tokens concatenated and normalized; bilinear to224; mask average; L2',
                       torch_version=torch.__version__, identity_targets_read=False, semantic_labels_read=False)
        (destination / 'inference_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        rows.append(dict(run_id=relation['run_id'], status='complete', frame_pairs=len(records)))
        (output / 'dino_inference_manifest.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(f"{len(rows)}/{len(frozen['population'])} {relation['run_id']} {len(records)} pairs", flush=True)


if __name__ == '__main__':
    main()
