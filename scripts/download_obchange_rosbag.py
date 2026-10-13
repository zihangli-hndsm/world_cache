#!/usr/bin/env python3
"""Extract one ObChange ROS bag using HTTP range reads from its official ZIP.

The public TU Wien record is CC BY 4.0. This downloader reads one requested
member from the record's room archive without downloading the full multi-GB
archive, then verifies its official size and CRC32.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
import zlib
import zipfile
from collections import OrderedDict
from pathlib import Path


RECORD_API = "https://researchdata.tuwien.ac.at/api/records/y3ggy-hxp10"
USER_AGENT = "world-cache ObChange research evaluation/1.0"


class HTTPRangeFile:
    """Seekable read-only file backed by cached HTTP byte-range requests."""

    def __init__(self, url: str, size: int, block_size: int = 8 * 1024 * 1024, cache_blocks: int = 3):
        self.url = url
        self.size = int(size)
        self.block_size = int(block_size)
        self.cache_blocks = int(cache_blocks)
        self.position = 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        self.bytes_fetched = 0

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            target = offset
        elif whence == os.SEEK_CUR:
            target = self.position + offset
        elif whence == os.SEEK_END:
            target = self.size + offset
        else:
            raise ValueError(f"unknown seek mode: {whence}")
        if target < 0:
            raise ValueError("negative seek position")
        self.position = min(int(target), self.size)
        return self.position

    def _block(self, start: int) -> bytes:
        if start in self.cache:
            self.cache.move_to_end(start)
            return self.cache[start]
        end = min(start + self.block_size, self.size) - 1
        request = urllib.request.Request(
            self.url,
            headers={"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end}"},
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            if response.status != 206:
                raise RuntimeError(
                    f"server did not honor range request: HTTP {response.status}"
                )
            expected_range = f"bytes {start}-{end}/{self.size}"
            if response.headers.get("Content-Range") != expected_range:
                raise RuntimeError(
                    f"unexpected Content-Range {response.headers.get('Content-Range')!r}; "
                    f"expected {expected_range!r}"
                )
            data = response.read()
        if len(data) != end - start + 1:
            raise RuntimeError(f"short range response at byte {start}: got {len(data)} bytes")
        self.bytes_fetched += len(data)
        self.cache[start] = data
        self.cache.move_to_end(start)
        while len(self.cache) > self.cache_blocks:
            self.cache.popitem(last=False)
        return data

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.size - self.position
        size = min(int(size), self.size - self.position)
        if size <= 0:
            return b""
        chunks = []
        remaining = size
        while remaining:
            block_start = (self.position // self.block_size) * self.block_size
            block = self._block(block_start)
            offset = self.position - block_start
            chunk = block[offset : offset + remaining]
            if not chunk:
                break
            chunks.append(chunk)
            self.position += len(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--room-archive", required=True, help="for example Kitchen.zip")
    parser.add_argument("--member", required=True, help="for example GH30_kitchen/scene6.bag")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-mib", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    record = fetch_json(RECORD_API)
    rights = record.get("metadata", {}).get("rights", [])
    if not any(item.get("id") == "cc-by-4.0" for item in rights):
        raise RuntimeError(f"official dataset record no longer reports CC BY 4.0: {rights}")
    file_listing = fetch_json(record["links"]["files"])
    archive_meta = next(
        (item for item in file_listing["entries"] if item["key"] == args.room_archive),
        None,
    )
    if archive_meta is None:
        raise FileNotFoundError(f"no {args.room_archive!r} in the official record")
    container = fetch_json(archive_meta["links"]["container"])
    member_meta = next(
        (item for item in container["entries"] if item["key"] == args.member),
        None,
    )
    if member_meta is None:
        raise FileNotFoundError(f"no {args.member!r} in {args.room_archive}")

    print(
        json.dumps(
            {
                "record_doi": "10.48436/y3ggy-hxp10",
                "license": "CC BY 4.0",
                "archive": args.room_archive,
                "archive_bytes": archive_meta["size"],
                "member": args.member,
                "member_bytes": member_meta["size"],
                "compressed_member_bytes": member_meta["compressed_size"],
                "crc32": f"{int(member_meta['crc']):08x}",
            },
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        return

    remote = HTTPRangeFile(
        archive_meta["links"]["content"],
        int(archive_meta["size"]),
        block_size=max(1, args.block_mib) * 1024 * 1024,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    crc = 0
    written = 0
    try:
        with zipfile.ZipFile(remote) as archive:
            info = archive.getinfo(args.member)
            if info.file_size != int(member_meta["size"]):
                raise RuntimeError("ZIP member size differs from the official container API")
            if info.CRC != int(member_meta["crc"]):
                raise RuntimeError("ZIP member CRC differs from the official container API")
            with archive.open(info, "r") as source, partial.open("wb") as destination:
                while True:
                    chunk = source.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    destination.write(chunk)
                    crc = zlib.crc32(chunk, crc)
                    written += len(chunk)
        if written != int(member_meta["size"]) or (crc & 0xFFFFFFFF) != int(member_meta["crc"]):
            raise RuntimeError(
                f"extracted member failed integrity check: bytes={written}, "
                f"crc={crc & 0xFFFFFFFF:08x}"
            )
        os.replace(partial, output)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            {
                "output": str(output),
                "verified_bytes": written,
                "verified_crc32": f"{crc & 0xFFFFFFFF:08x}",
                "range_bytes_fetched": remote.bytes_fetched,
                "outer_archive_size": archive_meta["size"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
