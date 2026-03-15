"""
Inference: predict height map from a single image or video, detect potholes.

Single image:
    python src/inference.py --image path/to/img.jpg --checkpoint checkpoints/best.pt

Video:
    python src/inference.py --video path/to/video.mp4 --checkpoint checkpoints/best.pt --output out.mp4
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import cv2
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.height_net import HeightNet
from src.detect_potholes import detect_potholes, visualize_detections


_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = HeightNet(pretrained=False).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    cfg = ckpt['cfg']
    return model, cfg


def preprocess_image(img_rgb: np.ndarray, W: int = 640, H: int = 360) -> torch.Tensor:
    """Resize, normalize, return (1, 3, H, W) tensor."""
    pil = Image.fromarray(img_rgb)
    pil = pil.resize((W, H), Image.BILINEAR)
    t = TF.to_tensor(pil)
    t = TF.normalize(t, _MEAN, _STD)
    return t.unsqueeze(0)


def predict_height(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    device: torch.device,
    height_mean: float,
    height_std: float,
) -> np.ndarray:
    """Return denormalized height map (H, W) float32."""
    with torch.no_grad():
        pred = model(img_tensor.to(device))
    height = pred.squeeze().cpu().numpy()
    return height * height_std + height_mean


def run_single_image(args, model, cfg, device):
    height_mean = cfg['training']['height_mean']
    height_std  = cfg['training']['height_std']
    W, H = cfg['data']['input_resize']

    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        print(f"Could not load image: {args.image}")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_H, orig_W = img_rgb.shape[:2]

    img_t = preprocess_image(img_rgb, W, H)
    height_map = predict_height(model, img_t, device, height_mean, height_std)

    # Detect potholes
    mask, boxes, residual = detect_potholes(height_map, cfg)

    print(f"Detected {len(boxes)} potholes")
    for i, (x1, y1, x2, y2, conf) in enumerate(boxes):
        print(f"  [{i+1}] bbox=({x1},{y1},{x2},{y2}) depth={conf*100:.1f}cm")

    # Resize back to original for visualization
    h_orig = cv2.resize(height_map, (orig_W, orig_H), interpolation=cv2.INTER_LINEAR)
    m_orig = cv2.resize(mask.astype(np.uint8), (orig_W, orig_H),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
    r_orig = cv2.resize(residual, (orig_W, orig_H), interpolation=cv2.INTER_LINEAR)

    # Scale boxes
    sx, sy = orig_W / W, orig_H / H
    boxes_orig = [(int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy), c) for x1,y1,x2,y2,c in boxes]

    out_path = args.output or f"{Path(args.image).stem}_pothole.png"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    visualize_detections(img_rgb, m_orig, boxes_orig, r_orig, out_path=out_path)
    print(f"Saved: {out_path}")

    # Also save the raw height map
    npy_path = Path(out_path).with_suffix('.npy')
    np.save(npy_path, height_map)
    print(f"Height map saved: {npy_path}")


def run_video(args, model, cfg, device):
    height_mean = cfg['training']['height_mean']
    height_std  = cfg['training']['height_std']
    W, H = cfg['data']['input_resize']
    ema_alpha = 0.3  # temporal EMA smoothing factor

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Could not open video: {args.video}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS)
    orig_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = args.output or f"{Path(args.video).stem}_pothole.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (orig_W, orig_H))

    print(f"Processing {total} frames at {fps:.1f} fps → {out_path}")

    prev_height = None
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_t = preprocess_image(frame_rgb, W, H)
        height_map = predict_height(model, img_t, device, height_mean, height_std)

        # Temporal EMA smoothing
        if prev_height is None:
            prev_height = height_map
        else:
            height_map = ema_alpha * height_map + (1 - ema_alpha) * prev_height
            prev_height = height_map

        mask, boxes, residual = detect_potholes(height_map, cfg)

        # Scale boxes to original resolution
        sx, sy = orig_W / W, orig_H / H
        m_orig = cv2.resize(mask.astype(np.uint8), (orig_W, orig_H),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
        boxes_orig = [(int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy), c)
                      for x1, y1, x2, y2, c in boxes]

        # Draw on frame
        out_frame = frame_rgb.copy()
        # Red overlay for pothole mask
        red = np.zeros_like(out_frame)
        red[:, :, 0] = 255
        alpha = 0.4
        out_frame[m_orig] = (out_frame[m_orig] * (1 - alpha) + red[m_orig] * alpha).astype(np.uint8)

        # Bounding boxes
        for (x1, y1, x2, y2, conf) in boxes_orig:
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(out_frame, f'{conf*100:.0f}cm',
                        (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 0), 2)

        cv2.putText(out_frame, f'Frame {frame_idx} | Potholes: {len(boxes)}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        writer.write(cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR))
        frame_idx += 1

        if frame_idx % 50 == 0:
            print(f"  {frame_idx}/{total} frames processed")

    cap.release()
    writer.release()
    print(f"Done. Output: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='HeightNet inference')
    parser.add_argument('--checkpoint', default='checkpoints/best.pt')
    parser.add_argument('--config',     default=None,
                        help='config.yaml (auto-loaded from checkpoint if not provided)')
    parser.add_argument('--image',  default=None, help='Single image path')
    parser.add_argument('--video',  default=None, help='Video file path')
    parser.add_argument('--output', default=None, help='Output file path')
    args = parser.parse_args()

    if not args.image and not args.video:
        parser.error('Provide --image or --video')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model, cfg = load_model(args.checkpoint, device)

    # Optionally override config
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    if args.image:
        run_single_image(args, model, cfg, device)
    else:
        run_video(args, model, cfg, device)


if __name__ == '__main__':
    main()
