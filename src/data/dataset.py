"""
PyTorch Dataset for height map prediction.

Returns dicts with:
  image:  (3, H, W) float32 tensor — ImageNet-normalized
  height: (1, H, W) float32 tensor — normalized height (NaN→0)
  mask:   (1, H, W) float32 tensor — 1 where height is valid
"""
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image


# ImageNet normalization constants
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


class HeightMapDataset(Dataset):
    """
    Args:
        cfg:       loaded config dict
        split:     'train' or 'test'
        augment:   apply random augmentation (train only)
    """

    def __init__(self, cfg: dict, split: str = 'train', augment: bool = False):
        self.cfg = cfg
        self.split = split
        self.augment = augment
        self.W, self.H = cfg['data']['input_resize']  # 640, 360
        self.height_mean = cfg['training']['height_mean']
        self.height_std  = cfg['training']['height_std']

        precomp = Path(cfg['data']['precomputed_dir'])
        self.is_flat = (split == 'test')

        self.samples = []  # list of (img_path, dense_npy_path)

        if self.is_flat:
            img_dir  = Path(cfg['data']['test_root']) / 'left'
            npy_dir  = precomp / 'test'
            self._index_flat(img_dir, npy_dir)
        else:
            img_root = Path(cfg['data']['train_root'])
            npy_root = precomp / 'train'
            self._index_sequences(img_root, npy_root)

    def _index_sequences(self, img_root: Path, npy_root: Path):
        for seq_dir in sorted(img_root.iterdir()):
            if not seq_dir.is_dir():
                continue
            img_dir = seq_dir / 'left'
            npy_dir = npy_root / seq_dir.name
            if not img_dir.exists() or not npy_dir.exists():
                continue
            for dense_npy in sorted(npy_dir.glob('*_dense.npy')):
                stem = dense_npy.name.replace('_dense.npy', '')
                img = img_dir / f'{stem}.jpg'
                sparse_npy = dense_npy.parent / f'{stem}_sparse.npy'
                if img.exists() and sparse_npy.exists():
                    self.samples.append((img, dense_npy, sparse_npy))

    def _index_flat(self, img_dir: Path, npy_dir: Path):
        if not img_dir.exists() or not npy_dir.exists():
            return
        for dense_npy in sorted(npy_dir.glob('*_dense.npy')):
            stem = dense_npy.name.replace('_dense.npy', '')
            img = img_dir / f'{stem}.jpg'
            sparse_npy = dense_npy.parent / f'{stem}_sparse.npy'
            if img.exists() and sparse_npy.exists():
                self.samples.append((img, dense_npy, sparse_npy))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, dense_path, sparse_path = self.samples[idx]

        # Load image
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.W, self.H), Image.BILINEAR)

        # Load height maps
        # dense: GT values (interpolated, for smooth supervision)
        # sparse: loss mask (only real LiDAR measurements are trusted)
        dense = np.load(dense_path)   # (H, W) float32
        sparse = np.load(sparse_path) # (H, W) float32 with NaN where no LiDAR data
        valid_mask = ~np.isnan(sparse)

        # Augmentation (train only)
        if self.augment:
            img, dense, valid_mask = self._augment(img, dense, valid_mask)

        # Normalize image
        img_t = TF.to_tensor(img)  # (3, H, W), [0, 1]
        img_t = TF.normalize(img_t, _MEAN, _STD)

        # Normalize height
        height = dense.copy()
        height[~valid_mask] = 0.0
        height = (height - self.height_mean) / (self.height_std + 1e-6)
        height[~valid_mask] = 0.0  # re-zero after norm

        height_t = torch.from_numpy(height).unsqueeze(0)  # (1, H, W)
        mask_t   = torch.from_numpy(valid_mask.astype(np.float32)).unsqueeze(0)

        return {
            'image':  img_t,
            'height': height_t,
            'mask':   mask_t,
        }

    def _augment(self, img: Image.Image, dense: np.ndarray, valid_mask: np.ndarray):
        """Random horizontal flip + color jitter (RGB only)."""
        import random

        # Random horizontal flip
        if random.random() < 0.5:
            img = TF.hflip(img)
            dense = np.fliplr(dense).copy()
            valid_mask = np.fliplr(valid_mask).copy()

        # Color jitter on image only
        if random.random() < 0.8:
            img = TF.adjust_brightness(img, 1.0 + random.uniform(-0.3, 0.3))
        if random.random() < 0.5:
            img = TF.adjust_contrast(img, 1.0 + random.uniform(-0.2, 0.2))
        if random.random() < 0.5:
            img = TF.adjust_saturation(img, 1.0 + random.uniform(-0.2, 0.2))

        return img, dense, valid_mask


def collate_fn(batch):
    """
    Default collate that handles any remaining NaN by zeroing them.
    Mask already encodes validity, so loss ignores those pixels.
    """
    images  = torch.stack([b['image']  for b in batch])
    heights = torch.stack([b['height'] for b in batch])
    masks   = torch.stack([b['mask']   for b in batch])

    # Safety: replace any stray NaN/Inf
    heights = torch.nan_to_num(heights, nan=0.0, posinf=0.0, neginf=0.0)

    return {'image': images, 'height': heights, 'mask': masks}
