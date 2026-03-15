"""
LiDAR → camera image-plane projection + height map densification.

Coordinate conventions (RSRD dataset):
  LiDAR frame: X=lateral, Y=forward (negative values in front), Z=up
  Camera frame: X=right, Y=down, Z=forward (depth)

Projection:
  P_cam = R @ P_lidar + T       (extrinsics from config)
  depth = P_cam[2]              (camera-frame Z = depth)
  u = fx * P_cam[0] / depth + cx
  v = fy * P_cam[1] / depth + cy
  height_map[v, u] = P_lidar[2]  (store LiDAR Z = road elevation)
"""
import re
import numpy as np
import cv2
from scipy.interpolate import LinearNDInterpolator


def get_calib_for_sequence(seq_name: str, config: dict) -> dict:
    """
    Select the correct calibration set for a sequence based on its date prefix.

    seq_name examples: '2023-03-17-07-48-37', '20230317074850.000', 'test'
    Returns a calibration dict with keys: fx, fy, cx, cy, R (3x3), T (3,).
    Falls back to Set 2 (20230408) if no match found.
    """
    # Extract 8-digit date from sequence name (YYYYMMDD)
    m = re.search(r'(\d{4})-?(\d{2})-?(\d{2})', str(seq_name))
    date_key = None
    if m:
        date_key = m.group(1) + m.group(2) + m.group(3)

    calib_dict = config['calibration']
    if date_key and date_key in calib_dict:
        raw = calib_dict[date_key]
    else:
        # fallback: pick the first key
        raw = next(iter(calib_dict.values()))

    R = np.array(raw['R'], dtype=np.float64).reshape(3, 3)
    T = np.array(raw['T'], dtype=np.float64)
    return {
        'fx': float(raw['fx']),
        'fy': float(raw['fy']),
        'cx': float(raw['cx']),
        'cy': float(raw['cy']),
        'R': R,
        'T': T,
    }


def _scale_calib(calib: dict, src_w: int, src_h: int, dst_w: int, dst_h: int) -> dict:
    """Scale intrinsics from source to destination resolution."""
    sx = dst_w / src_w
    sy = dst_h / src_h
    return {
        'fx': calib['fx'] * sx,
        'fy': calib['fy'] * sy,
        'cx': calib['cx'] * sx,
        'cy': calib['cy'] * sy,
        'R': calib['R'],
        'T': calib['T'],
    }


def project_to_image(
    points: np.ndarray,
    calib: dict,
    H: int,
    W: int,
    road_labels=None,
    depth_min: float = 1.0,
    depth_max: float = 30.0,
) -> np.ndarray:
    """
    Project (N, 3) LiDAR xyz points onto an H×W image plane.

    Returns a float32 (H, W) sparse height map where each valid pixel
    contains the LiDAR Z value (height above sensor mounting plane).
    Invalid pixels are NaN.

    Args:
        points:      (N, 3) float32/64 array — LiDAR x, y, z
        calib:       dict with fx, fy, cx, cy, R (3x3), T (3,)
        H, W:        output image height/width (already at target resolution)
        road_labels: optional (N,) int array; if provided, keep only label==1
        depth_min:   minimum valid depth in meters
        depth_max:   maximum valid depth in meters
    """
    pts = points.astype(np.float64)  # (N, 3)

    # Optional road-label filter
    if road_labels is not None:
        mask = road_labels == 1
        pts = pts[mask]

    if len(pts) == 0:
        return np.full((H, W), np.nan, dtype=np.float32)

    # Z-range filter: keep near-ground points only
    # 10th percentile (lowest = most below sensor) + 0.3m tolerance
    z_thresh = np.percentile(pts[:, 2], 10) + 0.3
    pts = pts[pts[:, 2] < z_thresh]

    if len(pts) == 0:
        return np.full((H, W), np.nan, dtype=np.float32)

    R = calib['R']   # (3, 3)
    T = calib['T']   # (3,)
    fx, fy = calib['fx'], calib['fy']
    cx, cy = calib['cx'], calib['cy']

    # P_cam = R @ P_lidar + T  →  shape (3, N)
    P_cam = (R @ pts.T) + T[:, None]   # (3, N)

    depth = P_cam[2, :]                 # camera Z = depth

    # Keep points in front of camera within depth range
    valid = (depth >= depth_min) & (depth <= depth_max)
    P_cam = P_cam[:, valid]
    pts_valid = pts[valid]
    depth = depth[valid]

    if len(depth) == 0:
        return np.full((H, W), np.nan, dtype=np.float32)

    u = (fx * P_cam[0, :] / depth + cx).astype(np.float32)
    v = (fy * P_cam[1, :] / depth + cy).astype(np.float32)

    # Pixel indices
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)

    in_bounds = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui = ui[in_bounds]
    vi = vi[in_bounds]
    depth_valid = depth[in_bounds]
    z_lidar = pts_valid[in_bounds, 2].astype(np.float32)

    # Build sparse height map: where multiple points map to the same pixel,
    # keep the one with smallest depth (closest return).
    height_map = np.full((H, W), np.nan, dtype=np.float32)
    depth_map = np.full((H, W), np.inf, dtype=np.float32)

    # Vectorised approach: sort by depth ascending so closer points win
    order = np.argsort(depth_valid)
    ui, vi, depth_valid, z_lidar = ui[order], vi[order], depth_valid[order], z_lidar[order]

    # Process in reverse (furthest first) so closer overwrite further
    for i in range(len(ui) - 1, -1, -1):
        r, c = vi[i], ui[i]
        d = depth_valid[i]
        if d < depth_map[r, c]:
            depth_map[r, c] = d
            height_map[r, c] = z_lidar[i]

    return height_map


def densify_height_map(sparse_map: np.ndarray) -> np.ndarray:
    """
    Two-phase densification of a sparse height map.

    Phase 1: cv2.inpaint (Navier-Stokes, radius=5) for scan-line gaps.
    Phase 2: scipy LinearNDInterpolator for large uncovered regions.

    NaN values outside any data coverage (sky/edges) remain NaN.

    Args:
        sparse_map: (H, W) float32 with NaN where no data

    Returns:
        (H, W) float32 dense map, NaN only where no interpolation possible
    """
    H, W = sparse_map.shape
    valid_mask = ~np.isnan(sparse_map)

    if valid_mask.sum() < 10:
        return sparse_map.copy()

    # ---- Phase 1: cv2.inpaint for small gaps ----
    # inpaint needs uint8 mask (255 = missing) and float32 image
    inpaint_mask = (~valid_mask).astype(np.uint8) * 255

    # Replace NaN with 0 for inpaint input
    img_for_inpaint = np.where(valid_mask, sparse_map, 0.0).astype(np.float32)

    # cv2.inpaint works on 8-bit or 16-bit; we scale to [0, 65535] for precision
    z_min = sparse_map[valid_mask].min()
    z_max = sparse_map[valid_mask].max()
    z_range = z_max - z_min if (z_max - z_min) > 1e-6 else 1.0

    img_norm = ((img_for_inpaint - z_min) / z_range * 65535).astype(np.float32)
    img_norm_u16 = np.clip(img_norm, 0, 65535).astype(np.uint16)

    inpainted_u16 = cv2.inpaint(img_norm_u16, inpaint_mask, inpaintRadius=5,
                                flags=cv2.INPAINT_NS)
    phase1 = (inpainted_u16.astype(np.float32) / 65535.0) * z_range + z_min

    # Determine where phase 1 actually filled (was missing before)
    phase1_filled = phase1.copy()
    phase1_filled[valid_mask] = sparse_map[valid_mask]  # original data takes priority

    # Check if there are still large regions uncovered after phase 1
    # (inpaint only works well for small gaps; large gaps remain poorly filled)
    # We use scipy for remaining large gaps identified as connected components > 500px
    still_missing = np.isnan(sparse_map)  # phase1 fills all pixels (no NaN from inpaint)
    # Actually cv2 inpaint fills everything, but we still want scipy for large-gap quality
    # Phase 1 result already has no NaN. Use scipy for large contiguous missing regions.

    # Find contiguous missing regions
    n_labels, missing_labels = cv2.connectedComponents(inpaint_mask)
    large_gap_mask = np.zeros((H, W), dtype=bool)
    for label_id in range(1, n_labels):
        region = missing_labels == label_id
        if region.sum() > 500:
            large_gap_mask |= region

    if large_gap_mask.sum() == 0:
        return phase1_filled

    # ---- Phase 2: scipy LinearNDInterpolator for large gaps ----
    ys, xs = np.where(valid_mask)
    vals = sparse_map[ys, xs]

    try:
        # Build interpolator: input coords are (col, row) = (x, y)
        interp = LinearNDInterpolator(np.column_stack([xs, ys]), vals)
        # Query coords for large-gap pixels
        rows, cols = np.where(large_gap_mask)
        q_vals = interp(np.column_stack([cols, rows]))

        dense = phase1_filled.copy()
        filled = ~np.isnan(q_vals)
        dense[rows[filled], cols[filled]] = q_vals[filled]
        return dense
    except Exception:
        return phase1_filled


def project_and_densify(
    points: np.ndarray,
    calib: dict,
    out_H: int,
    out_W: int,
    src_H: int = 1080,
    src_W: int = 1920,
    road_labels=None,
    depth_min: float = 1.0,
    depth_max: float = 30.0,
):
    """
    Project LiDAR to image, densify, return (sparse, dense) at out_H×out_W.

    Scales intrinsics to out_H×out_W before projection so the output
    is already at training resolution.

    Returns:
        sparse: (out_H, out_W) float32 — raw projection, NaN where empty
        dense:  (out_H, out_W) float32 — after densification, NaN at unreach edges
    """
    scaled_calib = _scale_calib(calib, src_W, src_H, out_W, out_H)
    sparse = project_to_image(
        points, scaled_calib, out_H, out_W,
        road_labels=road_labels,
        depth_min=depth_min,
        depth_max=depth_max,
    )
    dense = densify_height_map(sparse)
    return sparse, dense
