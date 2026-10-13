"""Build a portable anonymized submission bundle from the SHA256 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/reports/20261009_submission_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/20261009_submission_bundle.zip"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    output_path = args.output if args.output.is_absolute() else root / args.output
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("format") != "portable_sha256_manifest_v2":
        raise ValueError("submission manifest is not complete")
    try:
        manifest_member = manifest_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("manifest must be inside the submission root") from exc
    member_names = {entry["path"] for entry in manifest["files"]}
    if manifest_member in member_names:
        raise ValueError("manifest must not list itself as an input")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as archive:
        for entry in manifest["files"]:
            relative = Path(entry["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe manifest path: {relative}")
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            content = path.read_bytes()
            if len(content) != entry["bytes"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
                raise ValueError(f"manifest hash mismatch: {relative}")
            archive.write(path, arcname=relative.as_posix())
        archive.write(manifest_path, arcname=manifest_member)
    with ZipFile(output_path) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ValueError(f"corrupt bundle member: {corrupt}")
        for entry in manifest["files"]:
            data = archive.read(entry["path"])
            if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise ValueError(f"bundle hash mismatch: {entry['path']}")
        if archive.read(manifest_member) != manifest_path.read_bytes():
            raise ValueError("bundled manifest differs from generated manifest")
    print(f"wrote and verified {len(manifest['files']) + 1} files to {output_path}")


if __name__ == "__main__":
    main()
