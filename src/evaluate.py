"""
Evaluate HeightNet on the test set and save visualizations.

Usage:
    python src/evaluate.py --config config.yaml --checkpoint checkpoints/best.pt
    python src/evaluate.py --config config.yaml --checkpoint checkpoints/best.pt --save_vis
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import HeightMapDataset, collate_fn
from src.models.height_net import HeightNet
from src.train import compute_metrics
from src.utils import get_device


def denormalize(t, mean, std):
    return t * std + mean


def save_visualization(img_t, pred_t, gt_t, mask_t, out_path, mean, std):
    """Save side-by-side: RGB | predicted height | GT height | error map."""
    _mean = np.array([0.485, 0.456, 0.406])
    _std  = np.array([0.229, 0.224, 0.225])

    # De-normalize image
    img = img_t.cpu().numpy().transpose(1, 2, 0)  # (H,W,3)
    img = (img * _std + _mean).clip(0, 1)

    pred = denormalize(pred_t.squeeze().cpu().numpy(), mean, std)
    gt   = denormalize(gt_t.squeeze().cpu().numpy(),   mean, std)
    mask = mask_t.squeeze().cpu().numpy().astype(bool)

    # Error (only on valid pixels)
    err = np.abs(pred - gt)
    err[~mask] = np.nan
    gt_vis = gt.copy()
    gt_vis[~mask] = np.nan

    vmin = np.nanpercentile(np.where(mask, gt, np.nan), 2)
    vmax = np.nanpercentile(np.where(mask, gt, np.nan), 98)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img)
    axes[0].set_title('Input RGB')

    im1 = axes[1].imshow(pred, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title('Predicted Height')
    plt.colorbar(im1, ax=axes[1], label='m')

    im2 = axes[2].imshow(gt_vis, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[2].set_title('GT Height (sparse)')
    plt.colorbar(im2, ax=axes[2], label='m')

    im3 = axes[3].imshow(err, cmap='hot', vmin=0, vmax=0.3)
    axes[3].set_title('|Error|')
    plt.colorbar(im3, ax=axes[3], label='m')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()


def evaluate(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device(args.device)

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = HeightNet(pretrained=False).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    height_mean = cfg['training']['height_mean']
    height_std  = cfg['training']['height_std']

    test_ds = HeightMapDataset(cfg, split='test', augment=False)
    if len(test_ds) == 0:
        print("No test samples found. Check precomputed/test/ directory.")
        return

    loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg['training']['num_workers'],
        collate_fn=collate_fn,
    )
    print(f"Evaluating on {len(test_ds)} test frames...")

    results_dir = Path('results')
    vis_dir = results_dir / 'vis'
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc='Eval')):
            images  = batch['image'].to(device)
            heights = batch['height'].to(device)
            masks   = batch['mask'].to(device)

            pred = model(images)
            m = compute_metrics(pred, heights, masks, height_mean, height_std)
            all_metrics.append(m)

            if args.save_vis and i < args.max_vis:
                out_path = vis_dir / f'frame_{i:04d}.png'
                save_visualization(
                    images[0], pred[0], heights[0], masks[0],
                    out_path, height_mean, height_std
                )

    # Aggregate metrics
    mae    = np.mean([m['mae']    for m in all_metrics])
    rmse   = np.mean([m['rmse']   for m in all_metrics])
    delta1 = np.mean([m['delta1'] for m in all_metrics])

    print(f"\n{'='*40}")
    print(f"Test Set Results ({len(all_metrics)} frames)")
    print(f"  MAE:    {mae:.4f} m")
    print(f"  RMSE:   {rmse:.4f} m")
    print(f"  delta1: {delta1*100:.2f}%")
    print(f"{'='*40}")

    # Save metrics CSV
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / 'metrics.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame', 'mae', 'rmse', 'delta1'])
        writer.writeheader()
        for i, m in enumerate(all_metrics):
            writer.writerow({'frame': i, **m})

    # Append summary
    with open(results_dir / 'metrics.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['MEAN', f'{mae:.4f}', f'{rmse:.4f}', f'{delta1:.4f}'])

    print(f"\nMetrics saved to results/metrics.csv")
    if args.save_vis:
        print(f"Visualizations saved to results/vis/ ({min(args.max_vis, len(test_ds))} frames)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     default='config.yaml')
    parser.add_argument('--checkpoint', default='checkpoints/best.pt')
    parser.add_argument('--save_vis',   action='store_true', help='Save visualization PNGs')
    parser.add_argument('--max_vis',    type=int, default=50, help='Max visualizations to save')
    parser.add_argument('--device',     default=None, choices=['cuda', 'mps', 'cpu'],
                        help='Force a specific device (default: auto-select best)')
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
