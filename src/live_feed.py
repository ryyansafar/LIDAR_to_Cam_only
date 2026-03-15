"""
Live pothole detection from webcam, RTSP stream, or any OpenCV-readable source.

Usage:
    # Webcam (default camera)
    python src/live_feed.py --checkpoint checkpoints/best.pt

    # Specific webcam index
    python src/live_feed.py --checkpoint checkpoints/best.pt --source 0

    # IP camera / RTSP stream
    python src/live_feed.py --checkpoint checkpoints/best.pt --source "rtsp://192.168.1.100/stream"

    # Video file (plays frame by frame with detection)
    python src/live_feed.py --checkpoint checkpoints/best.pt --source road_video.mp4

Controls (while running):
    Q         — quit
    S         — save current frame as PNG
    H         — toggle height map overlay
    D         — toggle pothole detection overlay
    +/-       — adjust pothole detection threshold on the fly
    SPACE     — pause / resume
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.height_net import HeightNet
from src.detect_potholes import detect_potholes
from src.utils import get_device


# ─── Colormap: height value → BGR color ───────────────────────────────────────

def height_to_colormap(height_map: np.ndarray) -> np.ndarray:
    """Convert float32 height map to a BGR uint8 image using viridis colormap."""
    h = height_map.copy()
    nan_mask = np.isnan(h)
    h[nan_mask] = np.nanmin(h) if not nan_mask.all() else 0

    lo, hi = np.percentile(h, 2), np.percentile(h, 98)
    if hi - lo < 1e-4:
        hi = lo + 0.1
    norm = np.clip((h - lo) / (hi - lo), 0, 1)

    # Apply matplotlib viridis via cv2 colormap
    gray_u8 = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(gray_u8, cv2.COLORMAP_VIRIDIS)

    # Black out pixels with no data
    colored[nan_mask] = 0
    return colored


def draw_potholes(frame_bgr: np.ndarray, mask: np.ndarray,
                  boxes: list, residual: np.ndarray) -> np.ndarray:
    """Overlay pothole mask and bounding boxes on a BGR frame."""
    out = frame_bgr.copy()

    # Semi-transparent red overlay on pothole pixels
    if mask.any():
        overlay = out.copy()
        overlay[mask] = [0, 0, 220]   # red in BGR
        cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)

    # Bounding boxes + labels
    for (x1, y1, x2, y2, conf) in boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        label = f"Pothole {conf*100:.0f}cm"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 255), -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def build_display(orig_bgr: np.ndarray, height_bgr: np.ndarray,
                  detection_bgr: np.ndarray, fps: float,
                  n_potholes: int, threshold: float,
                  show_height: bool, show_detect: bool) -> np.ndarray:
    """
    Side-by-side display: [Camera] [Height Map] [Detections]
    or single panel depending on toggles.
    """
    H, W = orig_bgr.shape[:2]

    # Status bar at top of original frame
    status = orig_bgr.copy()
    cv2.rectangle(status, (0, 0), (W, 28), (30, 30, 30), -1)
    cv2.putText(status, f"FPS: {fps:.1f}  |  Potholes: {n_potholes}  |"
                f"  Threshold: {threshold*100:.0f}cm  |"
                f"  [H]eight {'ON' if show_height else 'OFF'}"
                f"  [D]etect {'ON' if show_detect else 'OFF'}  |  [Q]uit",
                (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    panels = [status]
    if show_height:
        hpanel = height_bgr.copy()
        cv2.putText(hpanel, "Height Map", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        panels.append(hpanel)
    if show_detect:
        dpanel = detection_bgr.copy()
        cv2.putText(dpanel, "Detections", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        panels.append(dpanel)

    return np.hstack(panels)


# ─── Main loop ────────────────────────────────────────────────────────────────

def run_live(args):
    device = get_device(args.device)

    # Load model + config from checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = HeightNet(pretrained=False).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    cfg = ckpt['cfg']

    height_mean = cfg['training']['height_mean']
    height_std  = cfg['training']['height_std']
    W, H = cfg['data']['input_resize']   # 640, 360

    # Patch detection threshold if provided via CLI
    if args.threshold is not None:
        cfg['detection']['pothole_depth_threshold'] = args.threshold

    # Open video source
    source = args.source
    if source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open source: {args.source}")
        return

    orig_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Source: {args.source}  ({orig_W}x{orig_H})")
    print(f"Model input: {W}x{H}")
    print(f"Controls: Q=quit  S=save  H=height  D=detect  +/-=threshold  SPACE=pause")

    mean_t = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std_t  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    prev_height = None
    ema_alpha   = 0.3          # temporal smoothing (0=no update, 1=no smoothing)
    show_height = True
    show_detect = True
    paused      = False
    threshold   = cfg['detection']['pothole_depth_threshold']
    frame_idx   = 0
    fps_avg     = 0.0

    while True:
        if not paused:
            ret, frame_bgr = cap.read()
            if not ret:
                if isinstance(source, str) and source != '0':
                    print("Stream ended.")
                    break
                continue

            t_start = time.perf_counter()

            # ── Preprocess ──────────────────────────────────────────────────
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(frame_rgb).resize((W, H), Image.BILINEAR)
            img_t = TF.to_tensor(pil)                    # (3,H,W) [0,1]
            img_t = (img_t - mean_t) / std_t
            img_t = img_t.unsqueeze(0).to(device)        # (1,3,H,W)

            # ── Inference ───────────────────────────────────────────────────
            with torch.no_grad():
                pred = model(img_t)

            height_map = pred.squeeze().cpu().numpy()    # (H,W) normalized
            height_map = height_map * height_std + height_mean  # denormalized (meters)

            # ── Temporal EMA smoothing ───────────────────────────────────────
            if prev_height is None:
                prev_height = height_map
            else:
                height_map = ema_alpha * height_map + (1 - ema_alpha) * prev_height
                prev_height = height_map

            # ── Pothole detection ────────────────────────────────────────────
            cfg['detection']['pothole_depth_threshold'] = threshold
            mask, boxes, residual = detect_potholes(height_map, cfg)

            # ── Scale back to original resolution ───────────────────────────
            sx, sy = orig_W / W, orig_H / H
            mask_orig = cv2.resize(mask.astype(np.uint8),
                                   (orig_W, orig_H), interpolation=cv2.INTER_NEAREST).astype(bool)
            boxes_orig = [(int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy), c)
                          for x1, y1, x2, y2, c in boxes]
            height_bgr  = cv2.resize(height_to_colormap(height_map),
                                     (orig_W, orig_H), interpolation=cv2.INTER_LINEAR)

            # ── Build panels ─────────────────────────────────────────────────
            detection_bgr = draw_potholes(frame_bgr, mask_orig, boxes_orig, residual)

            elapsed = time.perf_counter() - t_start
            fps_inst = 1.0 / max(elapsed, 1e-6)
            fps_avg  = 0.9 * fps_avg + 0.1 * fps_inst if fps_avg > 0 else fps_inst

            display = build_display(
                frame_bgr, height_bgr, detection_bgr,
                fps_avg, len(boxes_orig), threshold,
                show_height, show_detect
            )
            frame_idx += 1

        cv2.imshow("LiDAR-free Pothole Detection", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"capture_{frame_idx:05d}.png"
            cv2.imwrite(fname, display)
            print(f"Saved: {fname}")
        elif key == ord('h'):
            show_height = not show_height
        elif key == ord('d'):
            show_detect = not show_detect
        elif key == ord('+') or key == ord('='):
            threshold = min(threshold + 0.01, 0.5)
            print(f"Threshold: {threshold*100:.0f}cm")
        elif key == ord('-'):
            threshold = max(threshold - 0.01, 0.01)
            print(f"Threshold: {threshold*100:.0f}cm")
        elif key == ord(' '):
            paused = not paused
            print("Paused" if paused else "Resumed")

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Live pothole detection")
    parser.add_argument('--checkpoint', default='checkpoints/best.pt')
    parser.add_argument('--source',    default='0',
                        help='Camera index (0,1...), video file, or RTSP URL')
    parser.add_argument('--device',    default=None, choices=['cuda', 'mps', 'cpu'])
    parser.add_argument('--threshold', type=float, default=None,
                        help='Pothole depth threshold in meters (default: from config)')
    args = parser.parse_args()
    run_live(args)


if __name__ == '__main__':
    main()
