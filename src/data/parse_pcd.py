"""
Binary PCD parser — no open3d/pypcd dependency.
Builds numpy dtype dynamically from PCD header FIELDS/SIZE/TYPE lines.
"""
import numpy as np
from pathlib import Path


def _build_dtype(fields, sizes, types):
    """Build numpy dtype from PCD header FIELDS/SIZE/TYPE lists."""
    type_map = {'F': 'f', 'I': 'i', 'U': 'u'}
    dt = []
    for name, size, t in zip(fields, sizes, types):
        char = type_map.get(t.upper(), 'f')
        dt.append((name, f'<{char}{size}'))
    return np.dtype(dt)


def _parse_header(f):
    """Read PCD header lines until DATA binary. Returns metadata dict."""
    meta = {}
    while True:
        line = f.readline()
        if isinstance(line, bytes):
            line = line.decode('utf-8', errors='replace')
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        key, *rest = line.split()
        key = key.upper()
        if key == 'DATA':
            meta['data'] = rest[0].lower()
            break
        meta[key] = rest
    return meta


def load_pcd(path):
    """
    Load a binary PCD file.

    Returns a structured numpy array with named fields matching the PCD FIELDS header.
    """
    path = Path(path)
    with open(path, 'rb') as f:
        meta = _parse_header(f)
        data_start = f.tell()
        raw = f.read()

    if meta.get('data') != 'binary':
        raise ValueError(f"Only binary PCD supported, got: {meta.get('data')}")

    fields = meta['FIELDS']
    sizes = [int(s) for s in meta['SIZE']]
    types = meta['TYPE']
    n_points = int(meta['POINTS'][0])

    dtype = _build_dtype(fields, sizes, types)
    points = np.frombuffer(raw, dtype=dtype, count=n_points)
    return points


def load_pcd_xyz(path):
    """
    Load binary PCD and return (N, 3) float32 array of x, y, z coordinates.
    """
    pts = load_pcd(path)
    xyz = np.column_stack([
        pts['x'].astype(np.float32),
        pts['y'].astype(np.float32),
        pts['z'].astype(np.float32),
    ])
    return xyz


def load_pcd_road_seg(path):
    """
    Load a road-segmentation PCD (x y z label fields).
    label field: int16 (SIZE=2, TYPE=I).

    Returns:
        xyz: (N, 3) float32
        labels: (N,) int16
    """
    pts = load_pcd(path)
    xyz = np.column_stack([
        pts['x'].astype(np.float32),
        pts['y'].astype(np.float32),
        pts['z'].astype(np.float32),
    ])
    labels = pts['label'].astype(np.int16)
    return xyz, labels
