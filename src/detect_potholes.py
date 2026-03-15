"""
Post-process a predicted height map to detect potholes.

Potholes are localized depressions below the local road plane.

Usage (as module):
    from src.detect_potholes import detect_potholes
    mask, boxes, residual = detect_potholes(height_map, cfg)

CLI:
    python src/detect_potholes.py --height_map path/to/height.npy --config config.yaml
"""
import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


def detect_potholes(
    height_map: np.ndarray,
    cfg: dict,
) -> Tuple[np.ndarray, List[Tuple], np.ndarray]:
    """
    Detect potholes in a dense height map.

    Strategy:
      1. Gaussian blur of height map → local road plane estimate
      2. residual = blur - height  (positive = depression below local plane)
      3. Threshold + morphological cleanup → binary mask
      4. Connected components → bounding boxes with confidence

    Args:
        height_map: (H, W) float32, predicted height in meters (denormalized)
                    NaN values are handled by interpolation before processing.
        cfg:        config dict with detection section

    Returns:
        mask:     (H, W) bool — pothole pixels
        boxes:    list of (x1, y1, x2, y2, confidence) tuples
        residual: (H, W) float32 — depression depth map (positive = pothole)
    """
    det = cfg['detection']
    sigma      = det['local_plane_sigma']
    threshold  = det['pothole_depth_threshold']
    min_area   = det['min_pothole_area_px']
    kern_size  = det['morph_kernel_size']

    H, W = height_map.shape

    # Fill NaN with local mean for plane estimation
    h = height_map.copy()
    nan_mask = np.isnan(h)
    if nan_mask.any():
        # Simple fill: use median of valid pixels
        fill_val = float(np.nanmedian(h))
        h[nan_mask] = fill_val

    # Local road plane via Gaussian blur
    local_plane = gaussian_filter(h, sigma=sigma)

    # Residual: positive = depression (pothole), negative = bump
    residual = local_plane - h

    # Zero out originally-NaN regions (no data → no detection)
    residual[nan_mask] = 0.0

    # Threshold
    binary = (residual > threshold).astype(np.uint8)

    # Morphological open (remove noise) then close (fill gaps)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kern_size, kern_size)
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Connected components
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    mask = np.zeros((H, W), dtype=bool)
    boxes = []

    for label_id in range(1, n_labels):  # skip background (0)
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        component = labels == label_id
        mask |= component

        x1 = stats[label_id, cv2.CC_STAT_LEFT]
        y1 = stats[label_id, cv2.CC_STAT_TOP]
        w  = stats[label_id, cv2.CC_STAT_WIDTH]
        h_ = stats[label_id, cv2.CC_STAT_HEIGHT]
        x2, y2 = x1 + w, y1 + h_

        # Confidence: mean residual depth in the component
        confidence = float(residual[component].mean())
        boxes.append((x1, y1, x2, y2, confidence))

    # Sort by confidence descending
    boxes.sort(key=lambda b: b[4], reverse=True)

    return mask, boxes, residual


def visualize_detections(
    image: np.ndarray,
    mask: np.ndarray,
    boxes: List[Tuple],
    residual: np.ndarray,
    out_path: str = None,
):
    """
    Overlay pothole detections on the image.

    Args:
        image:    (H, W, 3) uint8 RGB image at any resolution
        mask:     (H, W) bool — must match image resolution
        boxes:    list of (x1, y1, x2, y2, confidence)
        residual: (H, W) float32 depression depth map
        out_path: save path (show if None)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(image)
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[mask] = [1.0, 0.0, 0.0, 0.5]  # red semi-transparent
    axes[1].imshow(overlay)
    for (x1, y1, x2, y2, conf) in boxes:
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='yellow', facecolor='none'
        )
        axes[1].add_patch(rect)
        axes[1].text(x1, y1 - 2, f'{conf*100:.1f}cm', color='yellow', fontsize=8)
    axes[1].set_title(f'Potholes ({len(boxes)} detected)')
    axes[1].axis('off')

    vmax = np.nanpercentile(np.abs(residual), 99)
    im = axes[2].imshow(residual, cmap='RdYlGn', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=axes[2], label='Depression depth (m)')
    axes[2].set_title('Residual (depression map)')
    axes[2].axis('off')

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--height_map', required=True, help='Path to .npy height map')
    parser.add_argument('--image',      default=None,  help='Optional RGB image for overlay')
    parser.add_argument('--config',     default='config.yaml')
    parser.add_argument('--output',     default='results/pothole_detection.png')
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    height_map = np.load(args.height_map)
    mask, boxes, residual = detect_potholes(height_map, cfg)

    print(f"Detected {len(boxes)} potholes:")
    for i, (x1, y1, x2, y2, conf) in enumerate(boxes):
        print(f"  [{i+1}] bbox=({x1},{y1},{x2},{y2}) depth={conf*100:.1f}cm")

    # Load image for visualization
    if args.image:
        import cv2 as _cv2
        img = _cv2.imread(args.image)
        img_rgb = _cv2.cvtColor(img, _cv2.COLOR_BGR2RGB)
        # Resize to match height map if needed
        H, W = height_map.shape
        if img_rgb.shape[:2] != (H, W):
            img_rgb = _cv2.resize(img_rgb, (W, H))
    else:
        H, W = height_map.shape
        img_rgb = np.zeros((H, W, 3), dtype=np.uint8)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    visualize_detections(img_rgb, mask, boxes, residual, out_path=args.output)
    print(f"Saved visualization: {args.output}")


if __name__ == '__main__':
    main()
