#!/usr/bin/env python3
"""Plan or fetch only the preselected MultiScan two-scan pilot pair.

The dataset is gated under CC BY-NC terms. Accept those terms on Hugging Face
and authenticate with ``hf auth login`` before passing ``--download``. This
script never accepts credentials as command-line arguments and does not
extract archives. The release revision and expected file sizes are pinned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


REPO_ID = "3dlg-hcvc/MultiScan"
REVISION = "c62c9aad850a8638ac3e42605926a65707600125"
FILES = {
    "scans/scene_00069_00.zip": {
        "size": 281_587_310,
        "repo_oid": "68be568b00a78a17d0aa3c294c7051b612256a35",
    },
    "scans/scene_00069_01.zip": {
        "size": 300_208_021,
        "repo_oid": "6d2ceb320473ffd7f77fb0ddcd630fc47b706427",
    },
}
TOTAL_BYTES = sum(item["size"] for item in FILES.values())
DEFAULT_OUTPUT = Path("data/multiscan_scene_00069")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--download",
        action="store_true",
        help="download the two pinned files; default behavior only prints the plan",
    )
    args = parser.parse_args()

    print(f"repository: {REPO_ID}")
    print(f"revision: {REVISION}")
    print(f"output: {args.output_dir}")
    for filename, metadata in FILES.items():
        print(f"{filename}: {metadata['size']} bytes; repo oid {metadata['repo_oid']}")
    print(f"compressed total: {TOTAL_BYTES} bytes ({TOTAL_BYTES / 2**30:.3f} GiB)")
    if not args.download:
        print("plan only; pass --download after accepting the dataset terms and logging in")
        return

    try:
        from huggingface_hub import get_token, hf_hub_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required; install it in the active environment") from exc

    token = get_token()
    if not token:
        raise SystemExit("no cached Hugging Face login; run `hf auth login` after accepting the dataset terms")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(args.output_dir).free
    if free_bytes < int(TOTAL_BYTES * 1.10):
        raise SystemExit(
            f"insufficient filesystem free space for the compressed pair: "
            f"need at least {int(TOTAL_BYTES * 1.10)} bytes, have {free_bytes}"
        )

    local_files = []
    for filename, metadata in FILES.items():
        resolved = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
                revision=REVISION,
                token=token,
                local_dir=str(args.output_dir),
            )
        )
        actual_size = resolved.stat().st_size
        if actual_size != metadata["size"]:
            raise RuntimeError(
                f"unexpected size for {filename}: expected {metadata['size']}, got {actual_size}"
            )
        local_files.append({
            "filename": filename,
            "path": str(resolved),
            "size": actual_size,
            "sha256": sha256(resolved),
            "repo_oid": metadata["repo_oid"],
        })
        print(f"downloaded {filename}: {actual_size} bytes")

    manifest = {
        "repository": REPO_ID,
        "revision": REVISION,
        "scene": "scene_00069",
        "license": "CC BY-NC 4.0; terms accepted by user outside this script",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": local_files,
        "archives_extracted": False,
    }
    manifest_path = args.output_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
