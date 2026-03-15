"""Shared utilities."""
import torch


def get_device(override: str = None) -> torch.device:
    """
    Auto-select best available device, or use override.

    Priority: CUDA (Nvidia) > MPS (Apple Silicon) > CPU
    Override values: 'cuda', 'mps', 'cpu'
    """
    if override:
        device = torch.device(override)
        print(f"Using device: {device} (manual override)")
        return device

    if torch.cuda.is_available():
        device = torch.device('cuda')
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Using device: CUDA — {name} ({vram:.1f} GB VRAM)")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using device: MPS (Apple Silicon GPU)")
    else:
        device = torch.device('cpu')
        print("Using device: CPU (no GPU found)")

    return device
