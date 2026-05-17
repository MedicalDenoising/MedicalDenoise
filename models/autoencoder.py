"""
Medical Image Denoising - Convolutional Autoencoder with U-Net Skip Connections
Designed for MRI/CT Brain Tumor Scan Denoising
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double convolution block with BatchNorm and ReLU"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """Encoder block: ConvBlock + MaxPool, returns both for skip connections"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, dropout)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        features = self.conv(x)
        pooled = self.pool(features)
        return features, pooled


class DecoderBlock(nn.Module):
    """Decoder block: Upsample + skip connection + ConvBlock"""
    def __init__(self, in_channels, skip_channels, out_channels, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch (if input not divisible by 2)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetAutoencoder(nn.Module):
    """
    U-Net style Convolutional Autoencoder for Medical Image Denoising.
    
    Architecture:
        Encoder: 4 levels of downsampling with skip connections
        Bottleneck: Dense feature extraction
        Decoder: 4 levels of upsampling with skip connections
    
    Args:
        in_channels (int): Input channels (1 for grayscale MRI/CT, 3 for RGB)
        out_channels (int): Output channels (same as input)
        features (list): Feature sizes at each encoder level
        dropout (float): Dropout rate for regularization
    """
    def __init__(self, in_channels=1, out_channels=1,
                 features=[64, 128, 256, 512], dropout=0.1):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # Build Encoder
        prev_channels = in_channels
        for feat in features:
            self.encoders.append(EncoderBlock(prev_channels, feat, dropout))
            prev_channels = feat

        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2, dropout)

        # Build Decoder (reverse)
        rev_features = list(reversed(features))
        bottleneck_channels = features[-1] * 2
        prev_channels = bottleneck_channels
        for feat in rev_features:
            self.decoders.append(DecoderBlock(prev_channels, feat, feat, dropout))
            prev_channels = feat

        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], out_channels, kernel_size=1),
            nn.Sigmoid()  # Output in [0, 1] range
        )

    def forward(self, x):
        skips = []

        # Encode
        for encoder in self.encoders:
            skip, x = encoder(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decode with skip connections
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return self.final_conv(x)


class LightweightDenoiser(nn.Module):
    """
    Lightweight variant for faster training/inference on limited hardware.
    3-level encoder-decoder with skip connections.
    """
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_channels, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(128, 256)

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)

        self.out = nn.Sequential(
            nn.Conv2d(32, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)


def get_model(model_type='unet', **kwargs):
    """Factory function to get model by name"""
    models = {
        'unet': UNetAutoencoder,
        'lightweight': LightweightDenoiser,
    }
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")
    return models[model_type](**kwargs)


if __name__ == '__main__':
    # Quick architecture test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNetAutoencoder(in_channels=1, out_channels=1).to(device)
    x = torch.randn(2, 1, 256, 256).to(device)
    out = model(x)
    print(f"UNetAutoencoder | Input: {x.shape} → Output: {out.shape}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    model_light = LightweightDenoiser().to(device)
    out2 = model_light(x)
    print(f"LightweightDenoiser | Input: {x.shape} → Output: {out2.shape}")
