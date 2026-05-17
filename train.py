"""
Training Script for Medical Image Denoising Autoencoder
Supports: checkpointing, early stopping, LR scheduling, TensorBoard logging
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# Project imports
sys.path.insert(0, str(Path(__file__).parent))
from models.autoencoder import get_model
from models.losses import get_loss_fn, CombinedDenoisingLoss
from data.dataset import get_dataloaders
from utils.metrics import MetricsTracker, evaluate_baseline
from utils.visualize import save_comparison_grid


# ─────────────────────────────────────────────
# Training Config
# ─────────────────────────────────────────────

def get_default_config():
    return {
        # Data
        'image_dir': None,
        'use_synthetic': True,
        'synthetic_samples': 3000,
        'image_size': 256,
        'noise_type': 'gaussian',   # gaussian | rician | salt_pepper | mixed
        'noise_level': 0.1,
        'batch_size': 8,
        'num_workers': 2,

        # Model
        'model_type': 'unet',       # unet | lightweight
        'in_channels': 1,
        'features': [64, 128, 256, 512],
        'dropout': 0.1,

        # Training
        'epochs': 50,
        'lr': 1e-3,
        'weight_decay': 1e-5,
        'loss_type': 'combined',
        'scheduler': 'cosine',      # cosine | plateau | none
        'early_stopping_patience': 10,

        # Output
        'output_dir': 'results',
        'checkpoint_dir': 'checkpoints',
        'save_every': 5,
        'log_every': 10,
    }


# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────

class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n[Trainer] Using device: {self.device}")

        # Directories
        self.output_dir = Path(config['output_dir'])
        self.ckpt_dir = Path(config['checkpoint_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Build components
        self._build_data()
        self._build_model()
        self._build_optimizer()
        self._build_loss()

        # State
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {'train_loss': [], 'val_loss': [], 'val_psnr': [], 'val_ssim': []}

    def _build_data(self):
        cfg = self.config
        self.loaders = get_dataloaders(
            image_dir=cfg.get('image_dir'),
            image_size=cfg['image_size'],
            batch_size=cfg['batch_size'],
            noise_type=cfg['noise_type'],
            noise_level=cfg['noise_level'],
            num_workers=cfg['num_workers'],
            use_synthetic=cfg['use_synthetic'],
            synthetic_samples=cfg.get('synthetic_samples', 2000),
        )

    def _build_model(self):
        cfg = self.config
        model_kwargs = {
            'in_channels': cfg['in_channels'],
            'out_channels': cfg['in_channels'],
        }
        if cfg['model_type'] == 'unet':
            model_kwargs['features'] = cfg.get('features', [64, 128, 256, 512])
            model_kwargs['dropout'] = cfg.get('dropout', 0.1)

        self.model = get_model(cfg['model_type'], **model_kwargs).to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Model] {cfg['model_type'].upper()} | Parameters: {total_params:,}")

    def _build_optimizer(self):
        cfg = self.config
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg['lr'],
            weight_decay=cfg.get('weight_decay', 1e-5)
        )

        scheduler_type = cfg.get('scheduler', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg['epochs'], eta_min=1e-6)
        elif scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5)
        else:
            self.scheduler = None

    def _build_loss(self):
        loss_type = self.config.get('loss_type', 'combined')
        if loss_type == 'combined':
            self.loss_fn = CombinedDenoisingLoss(channels=self.config['in_channels'])
        elif loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'mae':
            self.loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss: {loss_type}")
        self.loss_fn = self.loss_fn.to(self.device)
        print(f"[Loss] Using: {loss_type}")

    # ─── Training ─────────────────────────────

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        loader = self.loaders['train']

        for batch_idx, (noisy, clean) in enumerate(loader):
            noisy = noisy.to(self.device, non_blocking=True)
            clean = clean.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            pred = self.model(noisy)

            if isinstance(self.loss_fn, CombinedDenoisingLoss):
                loss, _ = self.loss_fn(pred, clean)
            else:
                loss = self.loss_fn(pred, clean)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % self.config.get('log_every', 10) == 0:
                print(f"  Batch [{batch_idx}/{len(loader)}] Loss: {loss.item():.4f}", end='\r')

        return total_loss / len(loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        metrics = MetricsTracker()
        loader = self.loaders['val']

        for noisy, clean in loader:
            noisy = noisy.to(self.device, non_blocking=True)
            clean = clean.to(self.device, non_blocking=True)
            pred = self.model(noisy)

            if isinstance(self.loss_fn, CombinedDenoisingLoss):
                loss, _ = self.loss_fn(pred, clean)
            else:
                loss = self.loss_fn(pred, clean)

            total_loss += loss.item()
            metrics.update(pred.cpu(), clean.cpu())

        avg_loss = total_loss / len(loader)
        summary = metrics.summary()
        return avg_loss, summary

    # ─── Main Train Loop ──────────────────────

    def train(self):
        cfg = self.config
        patience = cfg.get('early_stopping_patience', 10)
        save_every = cfg.get('save_every', 5)

        print(f"\n{'='*60}")
        print(f"  TRAINING: {cfg['model_type'].upper()} | {cfg['epochs']} epochs")
        print(f"  Noise: {cfg['noise_type']} (level={cfg['noise_level']})")
        print(f"{'='*60}\n")

        for epoch in range(self.start_epoch, cfg['epochs']):
            t0 = time.time()

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss, val_metrics = self.validate()

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            lr = self.optimizer.param_groups[0]['lr']
            elapsed = time.time() - t0

            # Log
            psnr = val_metrics['PSNR (dB)']
            ssim = val_metrics['SSIM']
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_psnr'].append(psnr)
            self.history['val_ssim'].append(ssim)

            print(f"Epoch [{epoch+1:03d}/{cfg['epochs']}] "
                  f"Train: {train_loss:.4f} | "
                  f"Val: {val_loss:.4f} | "
                  f"PSNR: {psnr:.2f}dB | "
                  f"SSIM: {ssim:.4f} | "
                  f"LR: {lr:.2e} | "
                  f"{elapsed:.1f}s")

            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch, val_loss, tag=f'epoch_{epoch+1}')

            # Best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_loss, tag='best')
                print(f"  ✓ New best model saved (val_loss={val_loss:.5f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"\n[EarlyStopping] No improvement for {patience} epochs. Stopping.")
                    break

        self.save_history()
        print(f"\n[Training complete] Best val loss: {self.best_val_loss:.5f}")
        return self.history

    # ─── Evaluation ───────────────────────────

    @torch.no_grad()
    def evaluate_test(self, checkpoint: str = 'best'):
        """Full evaluation on test set with visualization"""
        self.load_checkpoint(tag=checkpoint)
        self.model.eval()

        metrics = MetricsTracker()
        baseline_psnrs, baseline_ssims = [], []

        for i, (noisy, clean) in enumerate(self.loaders['test']):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            pred = self.model(noisy)

            metrics.update(pred.cpu(), clean.cpu())

            # Baseline
            from utils.metrics import compute_psnr, compute_ssim
            baseline_psnrs.append(compute_psnr(noisy.cpu(), clean.cpu()))
            baseline_ssims.append(compute_ssim(noisy.cpu(), clean.cpu()))

            # Save first few comparison images
            if i < 5:
                save_comparison_grid(
                    noisy.cpu(), pred.cpu(), clean.cpu(),
                    save_path=self.output_dir / f'test_sample_{i}.png'
                )

        print("\n[Test Evaluation Results]")
        import numpy as np
        print(f"  Baseline  PSNR: {np.mean(baseline_psnrs):.2f} dB | SSIM: {np.mean(baseline_ssims):.4f}")
        summary = metrics.report()
        return summary

    # ─── Checkpoint I/O ───────────────────────

    def save_checkpoint(self, epoch: int, val_loss: float, tag: str = 'latest'):
        path = self.ckpt_dir / f'checkpoint_{tag}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
        }, path)

    def load_checkpoint(self, tag: str = 'best'):
        path = self.ckpt_dir / f'checkpoint_{tag}.pt'
        if not path.exists():
            print(f"[Checkpoint] {path} not found, using current weights.")
            return
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.start_epoch = ckpt['epoch'] + 1
        print(f"[Checkpoint] Loaded from {path} (epoch {ckpt['epoch']})")

    def save_history(self):
        path = self.output_dir / 'training_history.json'
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"[History] Saved to {path}")


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Train Medical Image Denoising Autoencoder')
    parser.add_argument('--image_dir', type=str, default=None)
    parser.add_argument('--use_synthetic', action='store_true', default=True)
    parser.add_argument('--model_type', type=str, default='unet', choices=['unet', 'lightweight'])
    parser.add_argument('--noise_type', type=str, default='gaussian')
    parser.add_argument('--noise_level', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--loss_type', type=str, default='combined')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--evaluate', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = get_default_config()
    config.update(vars(args))

    trainer = Trainer(config)

    if args.evaluate:
        trainer.evaluate_test()
    else:
        history = trainer.train()
        trainer.evaluate_test()
