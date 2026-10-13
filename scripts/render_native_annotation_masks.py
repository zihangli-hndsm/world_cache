"""CPU z-buffer for privileged scan-local annotation masks.

Released triangle meshes and camera calibration supply segmentation only.
Mixed-label triangles are marked unknown; unlabeled surfaces still occlude.
Triangles crossing the near plane are omitted and recorded as a limitation.
"""
import re
from pathlib import Path

import numpy as np
from numba import njit


def read_faces(path: Path):
    payload = path.read_bytes()
    end = payload.index(b'end_header\n') + len(b'end_header\n')
    header = payload[:end].decode('ascii')
    vertices = int(re.search(r'element vertex (\d+)', header).group(1))
    faces = int(re.search(r'element face (\d+)', header).group(1))
    data = np.fromstring(payload[end:].decode('ascii'), sep=' ', dtype=np.float64)
    records = data[vertices * 11:].reshape(faces, 4)
    if not np.all(records[:, 0] == 3):
        raise ValueError('expected triangular annotation mesh')
    return records[:, 1:].astype(np.int64)


@njit
def rasterize(uv, camera_z, triangles, vertex_ids, height, width):
    depth = np.full((height, width), np.inf)
    labels = np.zeros((height, width), np.int64)
    for face in triangles:
        i, j, k = face
        z0, z1, z2 = camera_z[i], camera_z[j], camera_z[k]
        if min(z0, z1, z2) <= .01:
            continue
        x0, y0 = uv[i]
        x1, y1 = uv[j]
        x2, y2 = uv[k]
        low_x = max(0, int(np.floor(min(x0, x1, x2))))
        high_x = min(width - 1, int(np.ceil(max(x0, x1, x2))))
        low_y = max(0, int(np.floor(min(y0, y1, y2))))
        high_y = min(height - 1, int(np.ceil(max(y0, y1, y2))))
        determinant = (y1-y2)*(x0-x2) + (x2-x1)*(y0-y2)
        if abs(determinant) < 1e-12:
            continue
        label = vertex_ids[i] if vertex_ids[i] == vertex_ids[j] == vertex_ids[k] else 0
        for y in range(low_y, high_y+1):
            for x in range(low_x, high_x+1):
                a = ((y1-y2)*(x-x2) + (x2-x1)*(y-y2))/determinant
                b = ((y2-y0)*(x-x2) + (x0-x2)*(y-y2))/determinant
                c = 1-a-b
                if min(a, b, c) < -1e-8:
                    continue
                inverse_depth = a/z0 + b/z1 + c/z2
                if inverse_depth <= 0:
                    continue
                z = 1/inverse_depth
                if z < depth[y, x]:
                    depth[y, x] = z
                    labels[y, x] = label
    return labels, depth


def render(points, vertex_ids, triangles, observation, resolution=224):
    pose = observation['camera_to_world']
    camera = (points - pose[:3, 3]) @ pose[:3, :3]
    intrinsic = observation['intrinsics'].copy()
    intrinsic[0] *= resolution / observation['rgb'].shape[1]
    intrinsic[1] *= resolution / observation['rgb'].shape[0]
    projected = camera @ intrinsic.T
    denominator = np.where(abs(projected[:, 2:3]) > 1e-12, projected[:, 2:3], 1e-12)
    uv = projected[:, :2] / denominator
    return rasterize(uv, camera[:, 2], triangles, vertex_ids, resolution, resolution)
