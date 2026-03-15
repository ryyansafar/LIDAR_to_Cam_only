#!/bin/bash
# ============================================================
# Remote GPU Setup Script
# Run this on the remote Nvidia machine to get everything ready
# ============================================================

set -e  # stop on any error

echo "=== Step 1: Installing Python dependencies ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
pip install -r requirements.txt -q
echo "Done."

echo ""
echo "=== Step 2: Checking GPU ==="
python3 -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'VRAM: {vram:.1f} GB')
    if vram >= 16:
        print('Tip: You can set batch_size=8 in config.yaml for faster training')
    elif vram >= 8:
        print('Tip: batch_size=4 (default) is fine for your GPU')
"

echo ""
echo "=== Step 3: Verifying code works ==="
python3 -c "
import sys
sys.path.insert(0, '.')
from src.models.height_net import HeightNet
import torch
m = HeightNet(pretrained=False)
x = torch.randn(1, 3, 360, 640)
print('Model output shape:', m(x).shape)
print('Model OK.')
"

echo ""
echo "=== All set! ==="
echo ""
echo "Next: upload your data (see README for rsync command), then run:"
echo "  python3 src/data/precompute_heights.py --config config.yaml --split all --workers 8"
echo "  python3 src/train.py --config config.yaml"
