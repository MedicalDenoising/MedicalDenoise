"""
Loss Functions for Medical Image Denoising
Includes: MSE, MAE, SSIM Loss, Perceptual Loss, Combined Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────
# SSIM Loss
# ─────────────────────────────────────────────

def gaussian_kernel(kernel_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Create a 2D Gaussian kernel for SSIM computation"""
    coords = torch.arange(kernel_size, dtype=torch.float32)
    coords -= kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel = g[:, None] * g[None, :]
    return kernel


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) based loss.
    
    SSIM measures perceptual similarity by comparing:
    - Luminance (mean intensity)
    - Contrast (standard deviation)
    - Structure (cross-correlation)
    
    Loss = 1 - SSIM (range [0, 1], lower is better)
    """
    def __init__(self, kernel_size: int = 11, sigma: float = 1.5,
                 k1: float = 0.01, k2: float = 0.03, channels: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.k1 = k1
        self.k2 = k2

        # Register Gaussian kernel as buffer (non-trainable)
        kernel = gaussian_kernel(kernel_size, sigma)
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
        self.register_buffer('kernel', kernel)
        self.channels = channels
        self.padding = kernel_size // 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        C1 = (self.k1) ** 2
        C2 = (self.k2) ** 2

        # Local means
        mu_x = F.conv2d(pred, self.kernel, padding=self.padding, groups=self.channels)
        mu_y = F.conv2d(target, self.kernel, padding=self.padding, groups=self.channels)

        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y

        # Local variances and covariance
        sigma_x = F.conv2d(pred ** 2, self.kernel, padding=self.padding, groups=self.channels) - mu_x_sq
        sigma_y = F.conv2d(target ** 2, self.kernel, padding=self.padding, groups=self.channels) - mu_y_sq
        sigma_xy = F.conv2d(pred * target, self.kernel, padding=self.padding, groups=self.channels) - mu_xy

        # SSIM map
        numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
        ssim_map = numerator / (denominator + 1e-8)

        return 1.0 - ssim_map.mean()


# ─────────────────────────────────────────────
# Edge-Aware Loss (gradient-based)
# ─────────────────────────────────────────────

class EdgeLoss(nn.Module):
    """
    Edge-preserving loss using Sobel gradients.
    Critical for medical imaging where lesion boundaries matter.
    """
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.unsqueeze(0).unsqueeze(0))
        self.register_buffer('sobel_y', sobel_y.unsqueeze(0).unsqueeze(0))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        def gradient(x):
            gx = F.conv2d(x, self.sobel_x, padding=1)
            gy = F.conv2d(x, self.sobel_y, padding=1)
            return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

        pred_edges = gradient(pred)
        target_edges = gradient(target)
        return F.l1_loss(pred_edges, target_edges)


# ─────────────────────────────────────────────
# Combined Loss
# ─────────────────────────────────────────────

class CombinedDenoisingLoss(nn.Module):
    """
    Weighted combination of multiple losses for optimal denoising:
    
    L = α·MSE + β·SSIM + γ·MAE + δ·Edge
    
    Default weights tuned for MRI brain denoising.
    """
    def __init__(self, mse_weight: float = 0.5,
                 ssim_weight: float = 0.3,
                 mae_weight: float = 0.1,
                 edge_weight: float = 0.1,
                 channels: int = 1):
        super().__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.mae_weight = mae_weight
        self.edge_weight = edge_weight

        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.ssim = SSIMLoss(channels=channels)
        self.edge = EdgeLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        losses = {}
        total = 0.0

        if self.mse_weight > 0:
            losses['mse'] = self.mse(pred, target)
            total += self.mse_weight * losses['mse']

        if self.ssim_weight > 0:
            losses['ssim'] = self.ssim(pred, target)
            total += self.ssim_weight * losses['ssim']

        if self.mae_weight > 0:
            losses['mae'] = self.mae(pred, target)
            total += self.mae_weight * losses['mae']

        if self.edge_weight > 0:
            losses['edge'] = self.edge(pred, target)
            total += self.edge_weight * losses['edge']

        losses['total'] = total
        return total, losses


def get_loss_fn(loss_type: str = 'combined', **kwargs):
    """Factory to get loss function by name"""
    loss_map = {
        'mse': nn.MSELoss,
        'mae': nn.L1Loss,
        'ssim': SSIMLoss,
        'combined': CombinedDenoisingLoss,
    }
    if loss_type not in loss_map:
        raise ValueError(f"Unknown loss type: {loss_type}")
    return loss_map[loss_type](**kwargs)


if __name__ == '__main__':
    pred = torch.rand(2, 1, 256, 256)
    target = torch.rand(2, 1, 256, 256)

    ssim_loss = SSIMLoss()
    combined_loss = CombinedDenoisingLoss()

    print(f"SSIM Loss: {ssim_loss(pred, target):.4f}")
    total, breakdown = combined_loss(pred, target)
    print(f"Combined Loss: {total:.4f}")
    for k, v in breakdown.items():
        print(f"  {k}: {v:.4f}")
