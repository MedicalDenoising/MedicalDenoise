"""
Dataset utilities for Medical Image Denoising
Supports: BraTS (Brain Tumor MRI), synthetic noise generation, and custom datasets
"""

import os
import random
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ─────────────────────────────────────────────
# Noise Generation
# ─────────────────────────────────────────────

def add_gaussian_noise(image: torch.Tensor, std: float = 0.1) -> torch.Tensor:
    """Add Gaussian (white) noise — most common in MRI"""
    noise = torch.randn_like(image) * std
    return torch.clamp(image + noise, 0.0, 1.0)


def add_salt_pepper_noise(image: torch.Tensor, prob: float = 0.02) -> torch.Tensor:
    """Add salt-and-pepper noise — common in low-dose CT"""
    noisy = image.clone()
    salt = torch.rand_like(image) < prob / 2
    pepper = torch.rand_like(image) < prob / 2
    noisy[salt] = 1.0
    noisy[pepper] = 0.0
    return noisy


def add_rician_noise(image: torch.Tensor, std: float = 0.05) -> torch.Tensor:
    """
    Add Rician noise — physically accurate model for MRI magnitude images.
    MRI signal magnitude follows Rician distribution due to quadrature detection.
    """
    n1 = torch.randn_like(image) * std
    n2 = torch.randn_like(image) * std
    return torch.clamp(torch.sqrt((image + n1) ** 2 + n2 ** 2), 0.0, 1.0)


def add_poisson_noise(image: torch.Tensor, scale: float = 50.0) -> torch.Tensor:
    """Add Poisson (shot) noise — dominant in low-dose CT"""
    scaled = image * scale
    noisy = torch.poisson(scaled) / scale
    return torch.clamp(noisy, 0.0, 1.0)


def add_mixed_noise(image: torch.Tensor,
                    gaussian_std: float = 0.05,
                    sp_prob: float = 0.01) -> torch.Tensor:
    """Combined Gaussian + Salt-Pepper noise (realistic clinical scenario)"""
    image = add_gaussian_noise(image, gaussian_std)
    image = add_salt_pepper_noise(image, sp_prob)
    return image


NOISE_TYPES = {
    'gaussian': add_gaussian_noise,
    'salt_pepper': add_salt_pepper_noise,
    'rician': add_rician_noise,
    'poisson': add_poisson_noise,
    'mixed': add_mixed_noise,
}


# ─────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────

class MedicalImageDataset(Dataset):
    """
    General-purpose dataset for medical image denoising.
    
    Loads clean images from a directory and generates noisy versions on-the-fly.
    Supports: PNG, JPG, JPEG, TIFF, BMP formats.
    
    Args:
        image_dir (str): Path to directory containing clean images
        image_size (int): Resize images to (image_size x image_size)
        noise_type (str): Type of noise to apply
        noise_level (float): Noise intensity (std for Gaussian, prob for S&P)
        augment (bool): Apply data augmentation during training
        split (str): 'train', 'val', or 'test'
    """
    EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}

    def __init__(self, image_dir: str, image_size: int = 256,
                 noise_type: str = 'gaussian', noise_level: float = 0.1,
                 augment: bool = False, split: str = 'train'):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.augment = augment and (split == 'train')

        self.image_paths = sorted([
            p for p in self.image_dir.rglob('*')
            if p.suffix.lower() in self.EXTENSIONS
        ])

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No images found in {image_dir}. "
                f"Supported formats: {self.EXTENSIONS}"
            )

        print(f"[Dataset] Found {len(self.image_paths)} images in '{image_dir}' ({split})")

        # Base transform: resize + to tensor
        self.base_transform = T.Compose([
            T.Resize((image_size, image_size), antialias=True),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def _augment(self, image: torch.Tensor) -> torch.Tensor:
        """Random augmentation: flip, rotation, brightness jitter"""
        if random.random() > 0.5:
            image = TF.hflip(image)
        if random.random() > 0.5:
            image = TF.vflip(image)
        angle = random.uniform(-15, 15)
        image = TF.rotate(image, angle)
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            image = torch.clamp(image * factor, 0.0, 1.0)
        return image

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')   # Grayscale
        clean = self.base_transform(image)           # [1, H, W], float32 in [0,1]

        if self.augment:
            clean = self._augment(clean)

        # Generate noisy version
        noise_fn = NOISE_TYPES.get(self.noise_type, add_gaussian_noise)
        noisy = noise_fn(clean, self.noise_level)

        return noisy, clean


class SyntheticBrainDataset(Dataset):
    """
    Synthetic brain phantom dataset for testing without real data.
    Generates simple circular/elliptical shapes mimicking brain cross-sections.
    Useful for quick debugging and architecture validation.
    """
    def __init__(self, num_samples: int = 1000, image_size: int = 256,
                 noise_type: str = 'rician', noise_level: float = 0.08):
        self.num_samples = num_samples
        self.image_size = image_size
        self.noise_type = noise_type
        self.noise_level = noise_level

    def _generate_brain_phantom(self) -> torch.Tensor:
        """Generate a synthetic brain-like phantom image"""
        size = self.image_size
        img = np.zeros((size, size), dtype=np.float32)
        cx, cy = size // 2, size // 2

        # Outer skull ellipse
        Y, X = np.ogrid[:size, :size]
        skull_mask = ((X - cx) / (cx * 0.85)) ** 2 + ((Y - cy) / (cy * 0.9)) ** 2 <= 1
        img[skull_mask] = 0.3

        # Gray matter
        gm_mask = ((X - cx) / (cx * 0.75)) ** 2 + ((Y - cy) / (cy * 0.8)) ** 2 <= 1
        img[gm_mask] = 0.6

        # White matter
        wm_mask = ((X - cx) / (cx * 0.55)) ** 2 + ((Y - cy) / (cy * 0.6)) ** 2 <= 1
        img[wm_mask] = 0.8

        # Random tumor (with 40% probability)
        if random.random() < 0.4:
            tx = cx + random.randint(-40, 40)
            ty = cy + random.randint(-40, 40)
            tr = random.randint(15, 35)
            tumor_mask = (X - tx) ** 2 + (Y - ty) ** 2 <= tr ** 2
            img[tumor_mask] = random.uniform(0.2, 0.95)

        # Slight random intensity variation
        img += np.random.normal(0, 0.02, img.shape)
        img = np.clip(img, 0, 1)

        return torch.from_numpy(img).unsqueeze(0)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        clean = self._generate_brain_phantom()
        noise_fn = NOISE_TYPES.get(self.noise_type, add_rician_noise)
        noisy = noise_fn(clean, self.noise_level)
        return noisy, clean


# ─────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────

def get_dataloaders(image_dir: str = None,
                    image_size: int = 256,
                    batch_size: int = 8,
                    noise_type: str = 'gaussian',
                    noise_level: float = 0.1,
                    val_split: float = 0.15,
                    test_split: float = 0.10,
                    num_workers: int = 2,
                    use_synthetic: bool = False,
                    synthetic_samples: int = 2000):
    """
    Create train/val/test DataLoaders.
    
    Args:
        image_dir: Path to image directory (None if use_synthetic=True)
        use_synthetic: Use SyntheticBrainDataset instead of real images
        ...
    Returns:
        dict with 'train', 'val', 'test' DataLoaders
    """
    if use_synthetic:
        full_dataset = SyntheticBrainDataset(
            num_samples=synthetic_samples,
            image_size=image_size,
            noise_type=noise_type,
            noise_level=noise_level
        )
    else:
        full_dataset = MedicalImageDataset(
            image_dir=image_dir,
            image_size=image_size,
            noise_type=noise_type,
            noise_level=noise_level,
            augment=True,
        )

    total = len(full_dataset)
    n_test = int(total * test_split)
    n_val = int(total * val_split)
    n_train = total - n_val - n_test

    train_set, val_set, test_set = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"[DataLoader] Train: {n_train} | Val: {n_val} | Test: {n_test}")

    loaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True),
        'test': DataLoader(test_set, batch_size=1, shuffle=False,
                           num_workers=num_workers, pin_memory=True),
    }
    return loaders


if __name__ == '__main__':
    # Test with synthetic data
    loaders = get_dataloaders(use_synthetic=True, synthetic_samples=100, batch_size=4)
    noisy, clean = next(iter(loaders['train']))
    print(f"Batch shapes — Noisy: {noisy.shape}, Clean: {clean.shape}")
    print(f"Value range — min: {noisy.min():.3f}, max: {noisy.max():.3f}")
