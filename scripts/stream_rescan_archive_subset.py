#!/usr/bin/env python3
"""Inspect or range-extract only selected Rescan annotation PLYs from its ZIP.

The official archive is large. This reader uses HTTP byte ranges so listing the
central directory and reading a small set of members does not download the
entire archive. The default action is an inventory; pass --extract to write
the selected PLYs under the requested data root.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict, OrderedDict
import io
from pathlib import Path, PurePosixPath
import shutil
from urllib.request import Request, urlopen
from zipfile import ZipFile


DEFAULT_URL = "https://rescan.cs.princeton.edu/assets/rescan_dataset.zip"
DEFAULT_CANDIDATES = Path("outputs/reports/20261002_rescan_rerank_all_source_b_m/color_rerank_candidates.csv")
CHUNK_BYTES = 4 * 1024 * 1024


class HTTPRangeReader(io.BufferedIOBase):
    """Small seekable HTTP reader with bounded aligned-chunk caching."""

    def __init__(self, url: str, cache_chunks: int = 8) -> None:
        super().__init__()
        request = Request(url, method="HEAD")
        with urlopen(request, timeout=60) as response:
            self.size = int(response.headers["Content-Length"])
            self.etag = response.headers.get("ETag")
            self.range_supported = "bytes" in response.headers.get("Accept-Ranges", "").lower()
        if not self.range_supported:
            raise RuntimeError("server does not advertise byte-range support")
        self.url = url
        self.cache_chunks = cache_chunks
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        self.position = 0
        self.bytes_downloaded = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self.position = position
        return position

    def _chunk(self, chunk_index: int) -> bytes:
        if chunk_index in self.cache:
            self.cache.move_to_end(chunk_index)
            return self.cache[chunk_index]
        start = chunk_index * CHUNK_BYTES
        end = min(self.size, start + CHUNK_BYTES) - 1
        headers = {"Range": f"bytes={start}-{end}"}
        if self.etag:
            headers["If-Range"] = self.etag
        request = Request(self.url, headers=headers)
        with urlopen(request, timeout=120) as response:
            if response.status != 206:
                raise RuntimeError(f"server ignored range request at byte {start}")
            actual = response.headers.get("Content-Range", "")
            if not actual.startswith(f"bytes {start}-{end}/"):
                raise RuntimeError(f"unexpected Content-Range: {actual}")
            data = response.read()
        if len(data) != end - start + 1:
            raise RuntimeError(f"short byte-range response at {start}: {len(data)} bytes")
        self.bytes_downloaded += len(data)
        self.cache[chunk_index] = data
        self.cache.move_to_end(chunk_index)
        while len(self.cache) > self.cache_chunks:
            self.cache.popitem(last=False)
        return data

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.size:
            return b""
        if size is None or size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        result = bytearray()
        remaining = size
        while remaining:
            chunk_index = self.position // CHUNK_BYTES
            chunk = self._chunk(chunk_index)
            offset = self.position - chunk_index * CHUNK_BYTES
            take = min(remaining, len(chunk) - offset)
            result.extend(chunk[offset:offset + take])
            self.position += take
            remaining -= take
        return bytes(result)


def selected_members(candidate_csv: Path) -> dict[str, set[str]]:
    required: dict[str, set[str]] = defaultdict(set)
    with candidate_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            scene = row["scene"]
            required[scene].add(f"{row['source_scan']}.ply")
            required[scene].add(f"{row['target_scan']}.ply")
    if not required:
        raise ValueError("candidate CSV contains no object/scan rows")
    return dict(required)


def member_key(name: str) -> tuple[str, str] | None:
    parts = PurePosixPath(name).parts
    for index in range(len(parts) - 2):
        if parts[index + 1] == "gt_segmentation" and parts[-1].endswith(".ply"):
            return parts[index], parts[-1]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-url", default=DEFAULT_URL)
    parser.add_argument("--candidate-csv", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "data/worldcache_datasets/rescan_subset",
    )
    parser.add_argument("--extract", action="store_true", help="extract selected PLY files; otherwise only report required sizes")
    args = parser.parse_args()
    required = selected_members(args.candidate_csv)
    expected = {(scene, filename) for scene, names in required.items() for filename in names}
    found: dict[tuple[str, str], object] = {}

    remote = HTTPRangeReader(args.archive_url)
    with ZipFile(remote) as archive:
        for info in archive.infolist():
            key = member_key(info.filename)
            if key in expected:
                found[key] = info
        missing = sorted(expected - found.keys())
        if missing:
            raise FileNotFoundError(f"archive lacks {len(missing)} requested PLYs; examples={missing[:5]}")
        ordered = sorted(found.items())
        total_compressed = sum(info.compress_size for _, info in ordered)
        total_uncompressed = sum(info.file_size for _, info in ordered)
        print(f"archive_bytes={remote.size}")
        print(f"selected_files={len(ordered)} selected_compressed_bytes={total_compressed} selected_uncompressed_bytes={total_uncompressed}")
        print(f"scenes={len(required)} archive_bytes_fetched={remote.bytes_downloaded}")
        for (scene, filename), info in ordered:
            print(f"{scene}/{filename}\tcompressed={info.compress_size}\tuncompressed={info.file_size}")
        if not args.extract:
            print("inventory_only=true; pass --extract after reviewing the byte totals")
            return

        for (scene, filename), info in ordered:
            destination = args.data_root / scene / "gt_segmentation" / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".partial")
            with archive.open(info) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            temporary.replace(destination)
            print(f"extracted={destination}")


if __name__ == "__main__":
    main()
