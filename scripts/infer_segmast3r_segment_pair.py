#!/usr/bin/env python3
"""Run pinned official segment inference on anonymous prepared masks only."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path('third_party/segmast3r'))
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--input-manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--input-preprocessing', choices=['official-demo', 'official-dataset'], default='official-demo')
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    from configs.default import cfg
    from model_infer import MASt3RSegFeatInfer
    from src.models.mast3r_segfeat.diff_masked_pooling import masked_average_pooling

    cfg.merge_from_file(str(args.repo / 'configs/config_eval_spp_resz.yaml'))
    model = MASt3RSegFeatInfer(cfg)
    weights = torch.load(args.weights, map_location='cpu', weights_only=True)
    model.load_state_dict(weights['state_dict'], strict=True)
    del weights
    device = torch.device('cuda')
    model.prepare(device)
    manifest = json.loads(args.input_manifest.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for reference, query in zip(manifest['records']['reference'], manifest['records']['rescan']):
        views, masks, hashes = [], [], []
        for record in [reference, query]:
            path = Path(record['input'])
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != record['input_sha256']:
                raise ValueError('prepared input changed')
            hashes.append(digest)
            with np.load(path) as pack:
                # Official demo uses OpenCV INTER_LINEAR; preserve that exactly.
                if args.input_preprocessing == 'official-demo':
                    import cv2
                    resized = cv2.resize(pack['rgb'], (512, 512), interpolation=cv2.INTER_LINEAR)
                else:
                    from PIL import Image
                    resized = np.asarray(Image.fromarray(pack['rgb']).resize((512, 512), Image.Resampling.BILINEAR)).copy()
                image = torch.from_numpy(resized).permute(2, 0, 1).float()[None] / 255
                if args.input_preprocessing == 'official-dataset':
                    image = (image - .5) / .5
                mask = torch.from_numpy(pack['masks'].copy()).float()[None]
            mask = F.interpolate(mask, size=(512, 512), mode='nearest') if mask.shape[1] else torch.zeros((1, 0, 512, 512))
            views.append(image.to(device))
            masks.append(mask.to(device))
        name = f"ref{reference['frame_id']:06d}_query{query['frame_id']:06d}"
        if min(mask.shape[1] for mask in masks) == 0:
            matches = np.full(masks[0].shape[1], -1, dtype=np.int64)
            scores = np.zeros((masks[0].shape[1]+1, masks[1].shape[1]+1))
            status = 'empty_mask_input'
        else:
            with torch.inference_mode():
                result = model.infer_pair(views[0], views[1], masks[0], masks[1])
                view0 = dict(img=views[0], true_shape=torch.tensor([[512, 512]], device=device), instance=['view1_0'])
                view1 = dict(img=views[1], true_shape=torch.tensor([[512, 512]], device=device), instance=['view2_0'])
                desc0, desc1 = model.forward(view0, view1)
                pooled0 = masked_average_pooling(desc0.permute(0, 3, 1, 2), masks[0])
                pooled1 = masked_average_pooling(desc1.permute(0, 3, 1, 2), masks[1])
                scores = model.feature_matcher(pooled0, pooled1).exp()[0].cpu().numpy()
                matches = result[0].cpu().numpy()
                # Reproduce the official reciprocal + dustbin decisions.
                forward, backward = scores[:-1].argmax(1), scores[:, :-1].argmax(0)
                reproduced = np.full(len(forward), -1, np.int64)
                for i, j in enumerate(forward):
                    if j < len(backward) and backward[j] == i:
                        reproduced[i] = j
                np.testing.assert_array_equal(matches, reproduced)
            status = 'complete'
        path = args.output_dir / (name + '_anonymous_predictions.npz')
        np.savez_compressed(path, reference_to_query=matches, scores_with_dustbin=scores)
        records.append(dict(reference_frame=reference['frame_id'], query_frame=query['frame_id'],
                            status=status, reference_masks=masks[0].shape[1], query_masks=masks[1].shape[1],
                            matched_segments=int((matches >= 0).sum()), input_hashes=hashes,
                            predictions=str(path)))
        print(records[-1], flush=True)
    summary = dict(status='complete', frame_pairs=len(records), records=records,
                   input_preprocessing=args.input_preprocessing,
                   identity_targets_read=False, semantic_labels_read=False,
                   scope='Official segmentation-assisted inference; scientific correctness not scored here')
    (args.output_dir / 'inference_summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
