"""
Training script for HeightNet.

Usage:
    python src/train.py --config config.yaml
    python src/train.py --config config.yaml --resume checkpoints/last.pt
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import HeightMapDataset, collate_fn
from src.models.height_net import HeightNet


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def berhu_loss(pred, gt, mask):
    """
    Reverse Huber (BerHu) loss on valid pixels.
    delta = 0.2 * max(|e|) per batch.
    """
    diff = pred - gt
    abs_diff = torch.abs(diff)

    # Compute per-batch delta
    with torch.no_grad():
        max_val = abs_diff[mask.bool()].max() if mask.bool().any() else torch.tensor(1.0)
        delta = 0.2 * max_val.clamp(min=1e-6)

    l1_part  = abs_diff.clamp(max=delta)
    l2_part  = (diff ** 2 + delta ** 2) / (2.0 * delta)

    # BerHu: L1 when |e|<=delta, L2-style when |e|>delta
    loss_map = torch.where(abs_diff <= delta, abs_diff, l2_part)

    valid_pixels = mask.bool()
    if not valid_pixels.any():
        return torch.tensor(0.0, device=pred.device)
    return loss_map[valid_pixels].mean()


def edge_aware_smoothness(pred, image):
    """
    Edge-aware smoothness loss.
    L = mean(|∂h/∂x| * exp(-|∂I/∂x|) + |∂h/∂y| * exp(-|∂I/∂y|))
    image: (B, 3, H, W) — used to detect edges
    """
    # Gradients of predicted height
    dh_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    dh_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])

    # Gradients of image (grayscale)
    gray = image.mean(dim=1, keepdim=True)
    dI_dx = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
    dI_dy = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])

    w_x = torch.exp(-dI_dx)
    w_y = torch.exp(-dI_dy)

    return (dh_dx * w_x).mean() + (dh_dy * w_y).mean()


def total_loss(pred, gt, mask, image, cfg):
    w_berhu  = cfg['training']['loss_berhu_weight']
    w_smooth = cfg['training']['loss_smooth_weight']
    w_l1     = cfg['training']['loss_l1_weight']

    l_berhu  = berhu_loss(pred, gt, mask)
    l_smooth = edge_aware_smoothness(pred, image)

    valid = mask.bool()
    l_l1 = (torch.abs(pred - gt)[valid]).mean() if valid.any() else torch.tensor(0.0, device=pred.device)

    return w_berhu * l_berhu + w_smooth * l_smooth + w_l1 * l_l1, {
        'berhu':  l_berhu.item(),
        'smooth': l_smooth.item(),
        'l1':     l_l1.item(),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred, gt, mask, height_mean, height_std):
    """Compute MAE, RMSE, delta1 in de-normalized (meter) space."""
    valid = mask.bool()
    if not valid.any():
        return {'mae': 0.0, 'rmse': 0.0, 'delta1': 0.0}

    # De-normalize
    p = pred[valid] * height_std + height_mean
    g = gt[valid]   * height_std + height_mean

    diff = torch.abs(p - g)
    mae  = diff.mean().item()
    rmse = torch.sqrt((diff ** 2).mean()).item()

    ratio = torch.max(p / g.clamp(min=1e-6), g / p.clamp(min=1e-6))
    delta1 = (ratio < 1.25).float().mean().item()

    return {'mae': mae, 'rmse': rmse, 'delta1': delta1}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Datasets
    full_ds = HeightMapDataset(cfg, split='train', augment=True)
    val_size = max(1, int(len(full_ds) * 0.1))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))
    val_ds.dataset.augment = False  # no augmentation for val

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=cfg['training']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['training']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"Train: {len(train_ds)} frames, Val: {len(val_ds)} frames")

    # Model
    model = HeightNet(pretrained=True).to(device)

    # Optimizer with differential LRs
    param_groups = model.get_param_groups(cfg['training']['lr'])
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=cfg['training']['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['training']['epochs']
    )

    ckpt_dir = Path(cfg['training']['checkpoint_dir'])
    ckpt_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=str(ckpt_dir / 'logs'))

    start_epoch = 0
    best_val_mae = float('inf')
    height_mean = cfg['training']['height_mean']
    height_std  = cfg['training']['height_std']

    # Resume
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_val_mae = ckpt.get('best_val_mae', float('inf'))
        print(f"Resumed from epoch {start_epoch}")

    freeze_epochs = cfg['training']['freeze_encoder_epochs']

    for epoch in range(start_epoch, cfg['training']['epochs']):
        # Encoder freeze schedule
        if epoch < freeze_epochs:
            model.freeze_encoder(partial=True)
        elif epoch == freeze_epochs:
            model.unfreeze_all()
            print(f"Epoch {epoch}: unfreezing all encoder layers")

        # --- Train ---
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            images  = batch['image'].to(device)
            heights = batch['height'].to(device)
            masks   = batch['mask'].to(device)

            pred = model(images)
            loss, loss_parts = total_loss(pred, heights, masks, images, cfg)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # --- Validate ---
        model.eval()
        val_metrics = {'mae': 0.0, 'rmse': 0.0, 'delta1': 0.0}
        val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                images  = batch['image'].to(device)
                heights = batch['height'].to(device)
                masks   = batch['mask'].to(device)

                pred = model(images)
                loss, _ = total_loss(pred, heights, masks, images, cfg)
                val_loss += loss.item()

                m = compute_metrics(pred, heights, masks, height_mean, height_std)
                for k in val_metrics:
                    val_metrics[k] += m[k]
                n_val += 1

        val_loss /= max(n_val, 1)
        for k in val_metrics:
            val_metrics[k] /= max(n_val, 1)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{cfg['training']['epochs']} "
            f"| train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"| MAE={val_metrics['mae']:.4f}m RMSE={val_metrics['rmse']:.4f}m "
            f"delta1={val_metrics['delta1']*100:.1f}% "
            f"| {elapsed:.0f}s"
        )

        # TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val',   val_loss,   epoch)
        writer.add_scalar('Metrics/MAE',    val_metrics['mae'],    epoch)
        writer.add_scalar('Metrics/RMSE',   val_metrics['rmse'],   epoch)
        writer.add_scalar('Metrics/delta1', val_metrics['delta1'], epoch)
        writer.add_scalar('LR/encoder', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('LR/decoder', optimizer.param_groups[1]['lr'], epoch)

        # Save checkpoints
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_mae': best_val_mae,
            'cfg': cfg,
        }
        torch.save(state, ckpt_dir / 'last.pt')

        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            torch.save(state, ckpt_dir / 'best.pt')
            print(f"  ✓ New best MAE: {best_val_mae:.4f}m — saved best.pt")

    writer.close()
    print(f"\nTraining complete. Best val MAE: {best_val_mae:.4f}m")
    print(f"Best checkpoint: {ckpt_dir / 'best.pt'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--resume', default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
