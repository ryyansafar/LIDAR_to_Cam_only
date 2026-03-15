# LiDAR-to-Camera Height Prediction for Pothole Detection

Train a model on paired LiDAR + camera data from the [RSRD dataset](https://thu-rsxd.com/rsrd/) so that at **inference time only a camera image is needed** to produce a dense road surface height map. Potholes are detected as localized depressions in the predicted height map — no LiDAR required at deployment.

## Overview

```
Camera Image (1920×1080) → HeightNet → Height Map (640×360) → Pothole Mask + Bboxes
```

**HeightNet**: U-Net with pretrained ResNet-50 encoder (43.9M parameters).
**Training signal**: Sparse LiDAR ground truth projected onto the camera image plane.
**Calibration**: Real intrinsics and extrinsics from the RSRD dataset toolkit (stored in `calibration_files/`).

## Dataset

RSRD-dense dataset (not included — download from [thu-rsxd.com/rsrd](https://thu-rsxd.com/rsrd/)):
- **Train**: 2,493 frames across 27 sequences (LiDAR PCD + left camera JPG)
- **Test**: 300 frames (flat structure)
- Resolution: 1920×1080, training downscaled to 640×360

## Quick Start

```bash
pip install -r requirements.txt

# Step 1: Precompute height maps from LiDAR (run once, ~10 min with 4 workers)
python src/data/precompute_heights.py --config config.yaml --split all --workers 4

# Step 2: Verify calibration alignment visually
python src/data/precompute_heights.py --config config.yaml --verify

# Step 3: Update height statistics in config.yaml
python src/data/precompute_heights.py --config config.yaml --stats

# Step 4: Train
python src/train.py --config config.yaml

# Step 5: Evaluate on test set
python src/evaluate.py --checkpoint checkpoints/best.pt --save_vis

# Step 6: Inference on a new image
python src/inference.py --image path/to/road.jpg --checkpoint checkpoints/best.pt
```

> **macOS note**: If you get an OpenMP error, prepend `KMP_DUPLICATE_LIB_OK=TRUE` to any command.

## Project Structure

```
├── config.yaml                   # All parameters: calibration, training, detection
├── calibration_files/            # Real RSRD calibration PKL files (from RSRD_dev_toolkit)
├── requirements.txt
├── CLAUDE.md                     # Detailed technical documentation
└── src/
    ├── data/
    │   ├── parse_pcd.py          # Binary PCD parser (no open3d)
    │   ├── project_lidar.py      # LiDAR → camera projection + densification
    │   ├── dataset.py            # PyTorch Dataset
    │   └── precompute_heights.py # Run-once: precompute all height maps to disk
    ├── models/
    │   └── height_net.py         # U-Net with ResNet-50 encoder
    ├── train.py                  # Training loop (BerHu + edge-smooth + L1 loss)
    ├── evaluate.py               # Evaluation: MAE, RMSE, delta1 + visualizations
    ├── detect_potholes.py        # Height map → pothole mask + bounding boxes
    └── inference.py              # Single image or video inference
```

## Calibration

Two calibration sets sourced from `calibration_files/*.pkl` (RSRD_dev_toolkit):

| Date sequences | fx = fy | cx | cy |
|---|---|---|---|
| 2023-03-17, 2023-03-21 | 2024.51 | 1033.38 | 498.57 |
| 2023-04-06, 2023-04-08, 2023-04-09 | 2022.60 | 1037.42 | 500.82 |

Calibration is auto-selected based on the sequence date prefix.

## Projection Math

Standard pinhole model with LiDAR→camera extrinsics:
```
P_cam = R @ P_lidar + T
depth = P_cam[2]
u = fx * P_cam[0] / depth + cx
v = fy * P_cam[1] / depth + cy
height_map[v, u] = P_lidar[2]   # store LiDAR Z = road elevation
```

## Pothole Detection

```python
local_plane = gaussian_filter(height_map, sigma=25)
residual = local_plane - height_map   # positive = depression below local plane
pothole_mask = residual > 0.05        # 5cm threshold (configurable in config.yaml)
```

## Expected Results

With real calibration and 2,493 training frames:
- Target validation MAE < 0.10 m within 20 epochs
- Pothole depressions ≥ 5 cm detectable
- Inference: ~30ms/frame on GPU, ~300ms on CPU

## References

- [RSRD Dataset](https://thu-rsxd.com/rsrd/)
- [RSRD Dev Toolkit](https://github.com/ztsrxh/RSRD_dev_toolkit)
- [RSRD Paper (Scientific Data 2024)](https://www.nature.com/articles/s41597-024-03261-9)
