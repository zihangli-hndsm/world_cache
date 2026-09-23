"""Build the venue-facing anonymized PDF submission bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/20261008_anonymous_submission_pdfs.zip"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    files = {
        "main.pdf": root / "paper/main.pdf",
        "supplementary.pdf": root / "paper/supplementary.pdf",
    }
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for arcname, path in files.items():
            archive.write(path, arcname=arcname)
    print(f"wrote {len(files)} files to {output}")


if __name__ == "__main__":
    main()
