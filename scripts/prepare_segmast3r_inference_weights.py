#!/usr/bin/env python3
"""Range-read pinned official SegMASt3R tensors, excluding optimizer state.

Tensor metadata is decoded with a restricted, non-executing unpickler. ZIP
member CRCs and metadata hashes are checked. Values, strides, offsets, dtype
and shared storage are preserved; no reduced-precision conversion is applied.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import pickle
from pathlib import Path
from zipfile import ZipFile

import torch

from stream_rescan_archive_subset import HTTPRangeReader

REVISION = "769237c4f6e8efc244c7a262db54828398e916da"
URL = f"https://huggingface.co/rjayanti/segmast3r/resolve/{REVISION}/segmast3r_spp.ckpt"


class Config(dict):
    pass


class Storage:
    pass


def rebuild(storage, offset, size, stride, requires_grad, hooks, metadata=None):
    return dict(storage=storage, offset=offset, shape=size, stride=stride)


class MetadataReader(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("collections", "OrderedDict"): collections.OrderedDict,
            ("__builtin__", "set"): set,
            ("torch", "FloatStorage"): Storage,
            ("torch._utils", "_rebuild_tensor_v2"): rebuild,
            ("yacs.config", "CfgNode"): Config,
        }
        if (module, name) not in allowed:
            raise ValueError(f"unsupported pickle class {module}.{name}")
        return allowed[module, name]

    def persistent_load(self, value):
        if value[0] != "storage" or value[1] is not Storage:
            raise ValueError("expected float32 storage metadata")
        return dict(key=value[2], numel=value[4])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    reader = HTTPRangeReader(URL)
    hashes, state, storages = [], collections.OrderedDict(), {}
    with ZipFile(reader) as archive:
        meta_name, = [n for n in archive.namelist() if n.endswith("/data.pkl")]
        raw = archive.read(meta_name)
        expected = args.evidence_dir / "checkpoint_data.pkl"
        if expected.exists() and expected.read_bytes() != raw:
            raise ValueError("pinned metadata differs from inspected checkpoint")
        meta = MetadataReader(io.BytesIO(raw)).load()["state_dict"]
        root = meta_name.rsplit("/", 1)[0]
        for index, (name, descriptor) in enumerate(meta.items(), 1):
            storage = descriptor["storage"]
            key = storage["key"]
            if key not in storages:
                member = f"{root}/data/{key}"
                data = archive.read(member)
                if len(data) != storage["numel"] * 4:
                    raise ValueError(f"storage size mismatch: {member}")
                hashes.append(dict(member=member, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
                storages[key] = torch.frombuffer(bytearray(data), dtype=torch.float32)
            state[name] = torch.as_strided(storages[key], descriptor["shape"], descriptor["stride"], descriptor["offset"])
            if index % 100 == 0:
                print(f"loaded {index}/{len(meta)} tensors; network={reader.bytes_downloaded} bytes", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial")
    torch.save({"state_dict": state}, temporary)
    temporary.replace(args.output)
    digest = hashlib.sha256()
    with args.output.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    report = dict(status="complete", source_url=URL, revision=REVISION,
                  metadata_sha256=hashlib.sha256(raw).hexdigest(), tensors=len(state),
                  downloaded_bytes=reader.bytes_downloaded, output=str(args.output),
                  output_bytes=args.output.stat().st_size, output_sha256=digest.hexdigest(),
                  storage_hashes=hashes, scope="Exact official inference tensors; optimizer state omitted")
    (args.evidence_dir / "inference_weights_provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"saved {len(state)} exact tensors to {args.output}", flush=True)


if __name__ == "__main__":
    main()
