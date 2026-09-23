"""Build a portable anonymized submission bundle from the SHA256 manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/reports/20261008_submission_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/20261008_submission_bundle.zip"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    output_path = args.output if args.output.is_absolute() else root / args.output
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("submission manifest is not complete")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as archive:
        for entry in manifest["files"]:
            relative = Path(entry["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe manifest path: {relative}")
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            archive.write(path, arcname=relative.as_posix())
        archive.write(manifest_path, arcname="outputs/reports/20261008_submission_manifest.json")
    print(f"wrote {len(manifest['files']) + 1} files to {output_path}")


if __name__ == "__main__":
    main()
