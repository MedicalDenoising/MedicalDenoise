"""
Visualization utilities for Medical Image Denoising
Saves comparison grids: Noisy | Denoised | Ground Truth
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Optional

import torch


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert [1, H, W] or [H, W] tensor to [H, W] numpy array"""
    arr = t.detach().cpu().float()
    if arr.ndim == 3:
        arr = arr.squeeze(0)
    return arr.numpy()


def save_comparison_grid(noisy: torch.Tensor,
                         pred: torch.Tensor,
                         clean: torch.Tensor,
                         save_path: str or Path = 'comparison.png',
                         title: Optional[str] = None):
    """
    Save a 3-column comparison grid: Noisy | Predicted | Clean
    
    Args:
        noisy: Noisy input [B, C, H, W] or [C, H, W]
        pred:  Model output [B, C, H, W] or [C, H, W]
        clean: Ground truth [B, C, H, W] or [C, H, W]
        save_path: Where to save the PNG
        title: Optional super-title
    """
    # Handle batch dimension
    if noisy.ndim == 4:
        noisy = noisy[0]
        pred = pred[0]
        clean = clean[0]

    n_img = tensor_to_numpy(noisy)
    p_img = tensor_to_numpy(pred)
    c_img = tensor_to_numpy(clean)

    from utils.metrics import compute_psnr, compute_ssim
    psnr_noisy = compute_psnr(noisy.unsqueeze(0), clean.unsqueeze(0))
    ssim_noisy = compute_ssim(noisy.unsqueeze(0), clean.unsqueeze(0))
    psnr_pred = compute_psnr(pred.unsqueeze(0), clean.unsqueeze(0))
    ssim_pred = compute_ssim(pred.unsqueeze(0), clean.unsqueeze(0))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#0a0a0a')

    labels = [
        f'Noisy Input\nPSNR: {psnr_noisy:.2f} dB | SSIM: {ssim_noisy:.4f}',
        f'Denoised Output\nPSNR: {psnr_pred:.2f} dB | SSIM: {ssim_pred:.4f}',
        'Ground Truth\n(Clean Image)',
    ]
    images = [n_img, p_img, c_img]
    colors = ['#ff6b6b', '#51cf66', '#74c0fc']

    for ax, img, label, color in zip(axes, images, labels, colors):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(label, color=color, fontsize=11, fontweight='bold', pad=10)
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
            spine.set_visible(True)

    if title:
        fig.suptitle(title, color='white', fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Viz] Saved comparison to {save_path}")


def plot_training_history(history: dict, save_path: str = 'results/training_history.png'):
    """
    Plot training curves: Loss, PSNR, SSIM over epochs.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('#0f1117')

    metrics = [
        ('Loss', 'train_loss', 'val_loss', '#ff6b6b', '#ffa8a8'),
        ('PSNR (dB)', None, 'val_psnr', '#51cf66', None),
        ('SSIM', None, 'val_ssim', '#74c0fc', None),
    ]

    for ax, (name, train_key, val_key, val_color, train_color) in zip(axes, metrics):
        ax.set_facecolor('#1a1d27')
        epochs = range(1, len(history.get(val_key, [])) + 1)

        if train_key and train_key in history:
            ax.plot(range(1, len(history[train_key]) + 1), history[train_key],
                    color=train_color, linewidth=2, label='Train', alpha=0.8)

        if val_key in history:
            ax.plot(epochs, history[val_key],
                    color=val_color, linewidth=2.5, label='Val', alpha=0.95)

        ax.set_title(name, color='white', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch', color='#adb5bd')
        ax.tick_params(colors='#adb5bd')
        ax.spines['bottom'].set_color('#495057')
        ax.spines['left'].set_color('#495057')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(facecolor='#2d3142', edgecolor='none', labelcolor='white')
        ax.grid(True, alpha=0.15, color='white')

    plt.suptitle('Training History — Medical Image Denoising', color='white',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Viz] Training history saved to {save_path}")


def visualize_noise_comparison(image: torch.Tensor,
                                save_path: str = 'results/noise_types.png'):
    """
    Visualize different noise types applied to the same image.
    Educational figure for reports/demos.
    """
    from data.dataset import NOISE_TYPES

    noise_configs = [
        ('Original', None, None),
        ('Gaussian (σ=0.1)', 'gaussian', 0.10),
        ('Gaussian (σ=0.2)', 'gaussian', 0.20),
        ('Rician (σ=0.08)', 'rician', 0.08),
        ('Salt & Pepper (2%)', 'salt_pepper', 0.02),
        ('Mixed', 'mixed', 0.05),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor('#0a0a0a')
    axes = axes.flatten()

    clean = image.squeeze(0) if image.ndim == 4 else image
    clean_np = tensor_to_numpy(clean)

    for ax, (label, noise_type, level) in zip(axes, noise_configs):
        if noise_type is None:
            img_np = clean_np
        else:
            noise_fn = NOISE_TYPES[noise_type]
            noisy = noise_fn(clean, level)
            img_np = tensor_to_numpy(noisy)

        ax.imshow(img_np, cmap='gray', vmin=0, vmax=1)
        ax.set_title(label, color='white', fontsize=10, fontweight='bold')
        ax.axis('off')
        ax.set_facecolor('#111')

    plt.suptitle('Noise Types in Medical Imaging', color='white',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
    plt.close()
    print(f"[Viz] Noise comparison saved to {save_path}")
