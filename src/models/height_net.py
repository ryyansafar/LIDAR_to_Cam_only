"""
HeightNet: U-Net with ResNet-50 encoder for height map prediction.

Encoder (pretrained ResNet-50) → Decoder (bilinear upsample + skip connections)
Input:  (B, 3, 360, 640)
Output: (B, 1, 360, 640)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DoubleConv(nn.Module):
    """Two 3×3 convs with BN + ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    """Upsample to match skip size, concatenate skip, then DoubleConv."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        # Upsample to match skip spatial dims exactly (handles non-power-of-2)
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class HeightNet(nn.Module):
    """
    ResNet-50 encoder + 4-stage decoder.

    Feature map sizes at 640×360 input:
      skip1: 320×180  [64ch]  — after conv1 (before maxpool)
      skip2: 160×90   [256ch] — after layer1
      skip3: 80×45    [512ch] — after layer2
      skip4: 40×23    [1024ch]— after layer3
      bottleneck: 20×12 [2048ch] — after layer4
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Encoder
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )

        self.enc_stem   = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.enc_pool   = backbone.maxpool
        self.enc_layer1 = backbone.layer1   # → 256ch
        self.enc_layer2 = backbone.layer2   # → 512ch
        self.enc_layer3 = backbone.layer3   # → 1024ch
        self.enc_layer4 = backbone.layer4   # → 2048ch

        # Decoder
        # up4: bottleneck(2048) + skip4(1024) → 512
        self.dec4 = DecoderBlock(2048, 1024, 512)
        # up3: 512 + skip3(512) → 256
        self.dec3 = DecoderBlock(512, 512, 256)
        # up2: 256 + skip2(256) → 128
        self.dec2 = DecoderBlock(256, 256, 128)
        # up1: 128 + skip1(64) → 64
        self.dec1 = DecoderBlock(128, 64, 64)

        # Final upsample to input resolution (640×360) + prediction head
        self.head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),   # (B, 1, H, W) — raw, unbounded
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # Encoder
        s1 = self.enc_stem(x)       # (B, 64, H/2, W/2)
        s2 = self.enc_layer1(self.enc_pool(s1))   # (B, 256, H/4, W/4)
        s3 = self.enc_layer2(s2)    # (B, 512, H/8, W/8)
        s4 = self.enc_layer3(s3)    # (B, 1024, H/16, W/16)
        z  = self.enc_layer4(s4)    # (B, 2048, H/32, W/32)

        # Decoder
        d = self.dec4(z,  s4)       # → 512ch, 40×23
        d = self.dec3(d,  s3)       # → 256ch, 80×45
        d = self.dec2(d,  s2)       # → 128ch, 160×90
        d = self.dec1(d,  s1)       # → 64ch,  320×180

        # Upsample back to input resolution
        d = F.interpolate(d, size=(H, W), mode='bilinear', align_corners=False)
        return self.head(d)          # (B, 1, H, W)

    def get_param_groups(self, base_lr: float):
        """
        Return two optimizer param groups for differential learning rates:
          - encoder: base_lr * 0.1  (fine-tune carefully)
          - decoder: base_lr         (train from scratch)
        """
        encoder_params = (
            list(self.enc_stem.parameters()) +
            list(self.enc_pool.parameters()) +
            list(self.enc_layer1.parameters()) +
            list(self.enc_layer2.parameters()) +
            list(self.enc_layer3.parameters()) +
            list(self.enc_layer4.parameters())
        )
        decoder_params = (
            list(self.dec4.parameters()) +
            list(self.dec3.parameters()) +
            list(self.dec2.parameters()) +
            list(self.dec1.parameters()) +
            list(self.head.parameters())
        )
        return [
            {'params': encoder_params, 'lr': base_lr * 0.1},
            {'params': decoder_params, 'lr': base_lr},
        ]

    def freeze_encoder(self, partial: bool = True):
        """
        Freeze encoder layers for warm-up phase.
        partial=True: freeze stem + layer1 + layer2 only.
        partial=False: unfreeze all.
        """
        modules = [self.enc_stem, self.enc_pool, self.enc_layer1, self.enc_layer2]
        for m in modules:
            for p in m.parameters():
                p.requires_grad = not partial
        # layer3/4 always trainable (lower-level features, higher LR penalty)
        for m in [self.enc_layer3, self.enc_layer4]:
            for p in m.parameters():
                p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True
