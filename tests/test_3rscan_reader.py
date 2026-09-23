import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from worldcache.datasets.threerscan import ThreeRScanPair, ThreeRScanSequence


def _png(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def make_scan(root: Path, scan_id: str, pose: np.ndarray) -> None:
    scan_dir = root / scan_id
    scan_dir.mkdir()
    with zipfile.ZipFile(scan_dir / "sequence.zip", "w") as archive:
        archive.writestr("_info.txt", "m_depthShift = 1000\nm_calibrationColorIntrinsic = " + " ".join(map(str, np.eye(4).reshape(-1))))
        archive.writestr("frames/color/0.jpg", _png(np.full((2, 3, 3), 80, dtype=np.uint8)))
        archive.writestr("frames/depth/0.png", _png(np.full((2, 3), 2000, dtype=np.uint16)))
        archive.writestr("frames/pose/0.txt", "\n".join(" ".join(map(str, row)) for row in pose))


def test_reads_frame_and_applies_global_transform(tmp_path: Path) -> None:
    pose = np.eye(4, dtype=np.float32)
    make_scan(tmp_path, "ref", pose)
    sequence = ThreeRScanSequence("ref", tmp_path)
    frame = sequence.read_frame(0)
    assert sequence.frame_ids() == [0]
    assert frame["rgb"].shape == (2, 3, 3)
    assert np.allclose(frame["depth"], 2.0)
    shift = np.eye(4, dtype=np.float32)
    shift[0, 3] = 1.5
    transformed = sequence.read_frame(0, shift)
    assert np.isclose(transformed["camera_to_world"][0, 3], 1.5)


def test_pair_reads_official_metadata_shape(tmp_path: Path) -> None:
    make_scan(tmp_path, "ref", np.eye(4, dtype=np.float32))
    make_scan(tmp_path, "cur", np.eye(4, dtype=np.float32))
    transform = np.eye(4, dtype=np.float32)
    # 3RScan.json scene transforms encode translations in millimetres.
    transform[2, 3] = 250.0
    metadata = [{"reference": "ref", "scans": [{"reference": "cur", "transform": transform.T.reshape(-1).tolist()}]}]
    metadata_path = tmp_path / "3RScan.json"
    metadata_path.write_text(json.dumps(metadata))
    pair = ThreeRScanPair.from_metadata(tmp_path, metadata_path, "ref", "cur")
    assert pair.reference.scan_id == "ref"
    assert np.isclose(pair.rescan_to_reference[2, 3], 0.25)


def test_reads_monolithic_official_sample_layout(tmp_path: Path) -> None:
    depth = np.full((2, 3), 2000, dtype=">u2")
    pgm = b"P5\n3 2\n65535\n" + depth.tobytes()
    pose = np.eye(4, dtype=np.float32)
    with zipfile.ZipFile(tmp_path / "3RScan.v2.zip", "w") as archive:
        prefix = "scan/sequence/"
        info = "\n".join([
            "m_depthShift = 1000",
            "m_colorWidth = 3",
            "m_colorHeight = 2",
            "m_depthWidth = 3",
            "m_depthHeight = 2",
            "m_calibrationColorIntrinsic = 1 0 1 0 0 1 1 0 0 0 1 0 0 0 0 1",
            "m_calibrationDepthIntrinsic = 1 0 1 0 0 1 1 0 0 0 1 0 0 0 0 1",
            "m_calibrationDepthExtrinsic = 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1",
        ])
        archive.writestr(prefix + "_info.txt", info)
        archive.writestr(prefix + "frame-000000.color.jpg", _png(np.full((2, 3, 3), 80, dtype=np.uint8)))
        archive.writestr(prefix + "frame-000000.depth.pgm", pgm)
        archive.writestr(prefix + "frame-000000.pose.txt", "\n".join(" ".join(map(str, row)) for row in pose))
    sequence = ThreeRScanSequence("scan", tmp_path)
    frame = sequence.read_frame(0)
    assert sequence.frame_ids() == [0]
    assert frame["rgb"].shape == (2, 3, 3)
    assert frame["depth"].shape == (2, 3)
    assert np.allclose(frame["depth"], 2.0)


def test_reads_direct_official_sequence_layout(tmp_path: Path) -> None:
    with zipfile.ZipFile(tmp_path / "sequence.zip", "w") as archive:
        info = "\n".join([
            "m_depthShift = 1000",
            "m_calibrationColorIntrinsic = 1 0 1 0 0 1 1 0 0 0 1 0 0 0 0 1",
        ])
        archive.writestr("_info.txt", info)
        archive.writestr("frame-000000.color.jpg", _png(np.full((2, 3, 3), 80, dtype=np.uint8)))
        archive.writestr("frame-000000.depth.pgm", b"P5\n3 2\n65535\n" + np.full((2, 3), 2000, dtype=">u2").tobytes())
        archive.writestr("frame-000000.pose.txt", "\n".join(" ".join(map(str, row)) for row in np.eye(4, dtype=np.float32)))
    scan_dir = tmp_path / "scan"
    scan_dir.mkdir()
    (scan_dir / "sequence.zip").write_bytes((tmp_path / "sequence.zip").read_bytes())
    sequence = ThreeRScanSequence("scan", tmp_path)
    frame = sequence.read_frame(0)
    assert sequence.frame_ids() == [0]
    assert np.allclose(frame["depth"], 2.0)
