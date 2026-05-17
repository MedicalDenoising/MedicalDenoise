"""
Evaluation Metrics for Medical Image Denoising
Implements: PSNR, SSIM, MSE, MAE, SNR
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List


# ─────────────────────────────────────────────
# Core Metrics
# ─────────────────────────────────────────────

def compute_psnr(pred: torch.Tensor, target: torch.Tensor,
                 data_range: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio (PSNR) in dB.
    
    PSNR = 10 · log10(MAX² / MSE)
    
    Higher is better. Typical values:
        > 40 dB : Excellent (near-lossless)
        30-40 dB: Good
        < 30 dB : Noticeable degradation
    
    Args:
        pred: Predicted/denoised image tensor [B, C, H, W] in [0, 1]
        target: Ground truth image tensor [B, C, H, W] in [0, 1]
        data_range: Maximum value of the data (1.0 for normalized images)
    Returns:
        Mean PSNR in dB across the batch
    """
    mse = F.mse_loss(pred, target, reduction='none')
    mse = mse.mean(dim=[1, 2, 3])  # Per-image MSE

    # Avoid log(0)
    mse = torch.clamp(mse, min=1e-10)
    psnr = 10.0 * torch.log10(data_range ** 2 / mse)
    return psnr.mean().item()


def _gaussian_kernel_1d(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor,
                 kernel_size: int = 11, sigma: float = 1.5,
                 k1: float = 0.01, k2: float = 0.03,
                 data_range: float = 1.0) -> float:
    """
    Structural Similarity Index Measure (SSIM).
    
    SSIM ∈ [-1, 1]. Higher is better.
    1.0 means identical images.
    
    Measures three components:
        - Luminance comparison
        - Contrast comparison  
        - Structural comparison
    
    Args:
        pred: Predicted image [B, C, H, W]
        target: Ground truth image [B, C, H, W]
    Returns:
        Mean SSIM across batch
    """
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2

    channels = pred.shape[1]

    # Build 2D Gaussian kernel
    k1d = _gaussian_kernel_1d(kernel_size, sigma).to(pred.device)
    kernel = k1d[:, None] * k1d[None, :]
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
    pad = kernel_size // 2

    mu_x = F.conv2d(pred, kernel, padding=pad, groups=channels)
    mu_y = F.conv2d(target, kernel, padding=pad, groups=channels)

    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x = F.conv2d(pred ** 2, kernel, padding=pad, groups=channels) - mu_x2
    sigma_y = F.conv2d(target ** 2, kernel, padding=pad, groups=channels) - mu_y2
    sigma_xy = F.conv2d(pred * target, kernel, padding=pad, groups=channels) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sigma_x + sigma_y + C2) + 1e-8)

    return ssim_map.mean().item()


def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return F.mse_loss(pred, target).item()


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return F.l1_loss(pred, target).item()


def compute_snr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Signal-to-Noise Ratio (SNR) in dB"""
    signal_power = (target ** 2).mean()
    noise_power = ((pred - target) ** 2).mean()
    if noise_power < 1e-10:
        return float('inf')
    return (10.0 * torch.log10(signal_power / noise_power)).item()


# ─────────────────────────────────────────────
# Metrics Tracker
# ─────────────────────────────────────────────

@dataclass
class MetricsTracker:
    """
    Running tracker for all denoising metrics.
    Use during evaluation to accumulate and report results.
    """
    psnr_values: List[float] = field(default_factory=list)
    ssim_values: List[float] = field(default_factory=list)
    mse_values: List[float] = field(default_factory=list)
    mae_values: List[float] = field(default_factory=list)
    snr_values: List[float] = field(default_factory=list)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Compute and store metrics for a batch"""
        with torch.no_grad():
            self.psnr_values.append(compute_psnr(pred, target))
            self.ssim_values.append(compute_ssim(pred, target))
            self.mse_values.append(compute_mse(pred, target))
            self.mae_values.append(compute_mae(pred, target))
            self.snr_values.append(compute_snr(pred, target))

    def summary(self) -> Dict[str, float]:
        """Return mean of all tracked metrics"""
        def mean(lst): return float(np.mean(lst)) if lst else 0.0
        return {
            'PSNR (dB)': mean(self.psnr_values),
            'SSIM': mean(self.ssim_values),
            'MSE': mean(self.mse_values),
            'MAE': mean(self.mae_values),
            'SNR (dB)': mean(self.snr_values),
        }

    def report(self):
        """Pretty-print metrics summary"""
        s = self.summary()
        print("\n" + "=" * 45)
        print("  DENOISING EVALUATION METRICS")
        print("=" * 45)
        for name, value in s.items():
            bar = "█" * int(min(value, 50) / 2) if 'dB' not in name else ""
            print(f"  {name:<15} {value:>8.4f}  {bar}")
        print("=" * 45)
        return s

    def reset(self):
        self.psnr_values.clear()
        self.ssim_values.clear()
        self.mse_values.clear()
        self.mae_values.clear()
        self.snr_values.clear()


# ─────────────────────────────────────────────
# Baseline Metrics (noisy vs clean)
# ─────────────────────────────────────────────

def evaluate_baseline(noisy: torch.Tensor, clean: torch.Tensor) -> Dict[str, float]:
    """
    Compute metrics for noisy input (before denoising).
    Useful as baseline to measure improvement.
    """
    return {
        'Baseline PSNR (dB)': compute_psnr(noisy, clean),
        'Baseline SSIM': compute_ssim(noisy, clean),
        'Baseline MSE': compute_mse(noisy, clean),
    }


if __name__ == '__main__':
    # Test with synthetic data
    clean = torch.rand(4, 1, 256, 256)
    noisy = torch.clamp(clean + torch.randn_like(clean) * 0.1, 0, 1)

    print("=== Baseline (noisy vs clean) ===")
    baseline = evaluate_baseline(noisy, clean)
    for k, v in baseline.items():
        print(f"  {k}: {v:.4f}")

    tracker = MetricsTracker()
    tracker.update(noisy, clean)
    tracker.report()
