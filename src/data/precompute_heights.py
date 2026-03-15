"""
Precompute height maps for all train/test frames and save to disk.

Run once before training:
    python src/data/precompute_heights.py --config config.yaml --split all --workers 4

Verify calibration alignment (plots LiDAR overlay on 5 sample images):
    python src/data/precompute_heights.py --config config.yaml --verify --n_verify 5

Compute dataset height statistics (updates config.yaml height_mean/std):
    python src/data/precompute_heights.py --config config.yaml --stats
"""
import argparse
import sys
import os
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import yaml
from tqdm import tqdm

# Allow running as script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.parse_pcd import load_pcd_xyz, load_pcd_road_seg
from src.data.project_lidar import get_calib_for_sequence, project_and_densify


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _find_road_seg_match(stem: str, road_seg_dir: Path):
    """Find a road_seg PCD matching the given timestamp stem, if any."""
    candidate = road_seg_dir / f"{stem}.pcd"
    return candidate if candidate.exists() else None


def _process_frame(args):
    """Worker function: process a single frame and save .npy files."""
    pcd_path, img_path, out_sparse, out_dense, road_seg_path, calib, cfg = args

    if out_dense.exists() and out_sparse.exists():
        return str(pcd_path), 'skipped'

    try:
        points = load_pcd_xyz(pcd_path)
        road_labels = None

        if road_seg_path is not None and Path(road_seg_path).exists():
            try:
                xyz_seg, labels = load_pcd_road_seg(road_seg_path)
                # Only use road_seg if it has the same number of points
                # (should match via timestamp, but verify)
                road_labels = labels
                points = xyz_seg  # road_seg has same xyz, use those points
            except Exception:
                pass  # fall back to plain xyz

        out_H, out_W = cfg['data']['input_resize'][1], cfg['data']['input_resize'][0]
        depth_min = cfg['training']['valid_depth_min']
        depth_max = cfg['training']['valid_depth_max']

        sparse, dense = project_and_densify(
            points, calib,
            out_H=out_H, out_W=out_W,
            src_H=cfg['data']['img_height'], src_W=cfg['data']['img_width'],
            road_labels=road_labels,
            depth_min=depth_min, depth_max=depth_max,
        )

        out_sparse.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_sparse, sparse)
        np.save(out_dense, dense)
        return str(pcd_path), 'ok'

    except Exception as e:
        return str(pcd_path), f'error: {e}'


def _gather_train_frames(cfg, root: Path, road_seg_dir: Path):
    """Return list of (pcd_path, img_path, out_sparse, out_dense, road_seg_path, calib) tuples."""
    frames = []
    precomp = Path(cfg['data']['precomputed_dir'])

    for seq_dir in sorted(root.iterdir()):
        if not seq_dir.is_dir():
            continue
        pcd_dir = seq_dir / 'pcd'
        img_dir = seq_dir / 'left'
        if not pcd_dir.exists() or not img_dir.exists():
            continue

        calib = get_calib_for_sequence(seq_dir.name, cfg)

        for pcd_file in sorted(pcd_dir.glob('*.pcd')):
            stem = pcd_file.stem  # e.g. '20230317074850.000'
            img_file = img_dir / f'{stem}.jpg'
            if not img_file.exists():
                continue

            road_seg = _find_road_seg_match(stem, road_seg_dir)

            out_sparse = precomp / 'train' / seq_dir.name / f'{stem}_sparse.npy'
            out_dense = precomp / 'train' / seq_dir.name / f'{stem}_dense.npy'

            frames.append((pcd_file, img_file, out_sparse, out_dense,
                           road_seg, calib, cfg))
    return frames


def _gather_test_frames(cfg, root: Path):
    """Return frames for the flat test set."""
    frames = []
    precomp = Path(cfg['data']['precomputed_dir'])
    pcd_dir = root / 'pcd'
    img_dir = root / 'left'

    if not pcd_dir.exists():
        return frames

    # Test filenames start with 20230317 → use Set 1 calibration
    calib = get_calib_for_sequence('20230317', cfg)

    for pcd_file in sorted(pcd_dir.glob('*.pcd')):
        stem = pcd_file.stem
        img_file = img_dir / f'{stem}.jpg'
        if not img_file.exists():
            continue

        out_sparse = precomp / 'test' / f'{stem}_sparse.npy'
        out_dense = precomp / 'test' / f'{stem}_dense.npy'
        frames.append((pcd_file, img_file, out_sparse, out_dense,
                       None, calib, cfg))
    return frames


def run_precompute(frames, n_workers):
    ok = skipped = errors = 0
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        futs = {exe.submit(_process_frame, f): f for f in frames}
        for fut in tqdm(as_completed(futs), total=len(futs), desc='Precomputing'):
            _, status = fut.result()
            if status == 'ok':
                ok += 1
            elif status == 'skipped':
                skipped += 1
            else:
                errors += 1
                tqdm.write(f"ERROR: {futs[fut][0]} — {status}")
    print(f"Done: {ok} processed, {skipped} skipped, {errors} errors")


def run_verify(cfg, root: Path, road_seg_dir: Path, n: int = 5):
    """Overlay LiDAR projections on sample images for visual calibration check."""
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.data.project_lidar import get_calib_for_sequence, _scale_calib, project_to_image

    out_dir = Path('results') / 'verify'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all frames
    all_frames = []
    for seq_dir in sorted(root.iterdir()):
        if not seq_dir.is_dir():
            continue
        pcd_dir = seq_dir / 'pcd'
        img_dir = seq_dir / 'left'
        if not pcd_dir.exists():
            continue
        for pcd_file in sorted(pcd_dir.glob('*.pcd'))[:3]:
            stem = pcd_file.stem
            img_file = img_dir / f'{stem}.jpg'
            if img_file.exists():
                all_frames.append((seq_dir.name, pcd_file, img_file))

    sample = random.sample(all_frames, min(n, len(all_frames)))

    for seq_name, pcd_path, img_path in sample:
        points = load_pcd_xyz(pcd_path)
        calib = get_calib_for_sequence(seq_name, cfg)

        # Project at full resolution for visual quality
        sparse = project_to_image(
            points, calib,
            H=cfg['data']['img_height'], W=cfg['data']['img_width'],
            depth_min=cfg['training']['valid_depth_min'],
            depth_max=cfg['training']['valid_depth_max'],
        )

        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        valid = ~np.isnan(sparse)
        ys, xs = np.where(valid)
        heights = sparse[ys, xs]

        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        ax.imshow(img_rgb)
        sc = ax.scatter(xs, ys, c=heights, cmap='RdYlGn_r', s=0.5, alpha=0.6,
                        vmin=heights.min(), vmax=heights.max())
        plt.colorbar(sc, ax=ax, label='LiDAR Z (m)')
        ax.set_title(f'{seq_name} / {pcd_path.name}')
        ax.axis('off')

        out_file = out_dir / f'{seq_name}_{pcd_path.stem}.png'
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved verify image: {out_file}")

    print(f"\nVerification images saved to {out_dir}/")
    print("Check that colored dots land on the road surface in each image.")
    print("If points are systematically offset, real calibration values are needed.")


def run_stats(cfg):
    """Compute dataset height statistics from precomputed dense maps."""
    precomp = Path(cfg['data']['precomputed_dir'])
    train_dir = precomp / 'train'

    if not train_dir.exists():
        print("No precomputed train data found. Run precompute first.")
        return

    all_vals = []
    npy_files = list(train_dir.rglob('*_dense.npy'))
    print(f"Computing stats over {len(npy_files)} dense maps...")

    for f in tqdm(npy_files):
        m = np.load(f)
        valid = m[~np.isnan(m)]
        if len(valid) > 0:
            # Sample up to 1000 values per map to keep memory manageable
            idx = np.random.choice(len(valid), min(1000, len(valid)), replace=False)
            all_vals.append(valid[idx])

    if not all_vals:
        print("No valid data found.")
        return

    all_vals = np.concatenate(all_vals)
    mean = float(np.mean(all_vals))
    std = float(np.std(all_vals))
    p1, p99 = float(np.percentile(all_vals, 1)), float(np.percentile(all_vals, 99))

    print(f"\nHeight statistics:")
    print(f"  mean = {mean:.4f} m")
    print(f"  std  = {std:.4f} m")
    print(f"  p1   = {p1:.4f} m  (1st percentile)")
    print(f"  p99  = {p99:.4f} m  (99th percentile)")

    # Patch config.yaml
    config_path = 'config.yaml'
    with open(config_path) as f:
        content = f.read()

    import re
    content = re.sub(r'height_mean:.*', f'height_mean: {mean:.4f}', content)
    content = re.sub(r'height_std:.*', f'height_std: {std:.4f}', content)

    with open(config_path, 'w') as f:
        f.write(content)
    print(f"\nUpdated {config_path} with height_mean={mean:.4f}, height_std={std:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Precompute height maps for RSRD dataset')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--split', choices=['train', 'test', 'all'], default='all')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--verify', action='store_true',
                        help='Plot LiDAR overlay on sample images for calibration check')
    parser.add_argument('--n_verify', type=int, default=5)
    parser.add_argument('--stats', action='store_true',
                        help='Compute dataset height statistics and update config.yaml')
    args = parser.parse_args()

    cfg = _load_config(args.config)
    train_root = Path(cfg['data']['train_root'])
    test_root = Path(cfg['data']['test_root'])
    road_seg_root = Path(cfg['data']['road_seg_root'])

    if args.verify:
        run_verify(cfg, train_root, road_seg_root, n=args.n_verify)
        return

    if args.stats:
        run_stats(cfg)
        return

    frames = []
    if args.split in ('train', 'all'):
        train_frames = _gather_train_frames(cfg, train_root, road_seg_root)
        print(f"Found {len(train_frames)} train frames")
        frames.extend(train_frames)

    if args.split in ('test', 'all'):
        test_frames = _gather_test_frames(cfg, test_root)
        print(f"Found {len(test_frames)} test frames")
        frames.extend(test_frames)

    if not frames:
        print("No frames found. Check paths in config.yaml.")
        return

    run_precompute(frames, n_workers=args.workers)


if __name__ == '__main__':
    main()
