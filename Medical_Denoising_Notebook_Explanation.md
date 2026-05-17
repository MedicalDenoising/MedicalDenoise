# Medical Image Denoising & Reconstruction — Complete Notebook Explanation

> **Notebook:** `medical-denoising.ipynb`
> **Platform:** Kaggle (T4 x2 GPU)
> **Task:** Train a ResUNet with noise conditioning, residual learning, and SE attention to denoise brain MRI scans across 5 noise types

---

## Table of Contents

1. [Cell 0 — Title & Project Overview (Markdown)](#cell-0--title--project-overview)
2. [Cell 1 — Step 1 Heading (Markdown)](#cell-1--step-1-heading)
3. [Cell 2 — Step 1: Installation & wandb Login (Code)](#cell-2--step-1-installation--wandb-login)
4. [Cell 3 — Dataset Directory Check (Code)](#cell-3--dataset-directory-check)
5. [Cell 4 — Step 2 Heading (Markdown)](#cell-4--step-2-heading)
6. [Cell 5 — Step 2: Imports & Configuration (Code)](#cell-5--step-2-imports--configuration)
7. [Cell 6 — Step 3 Heading (Markdown)](#cell-6--step-3-heading)
8. [Cell 7 — Step 3: Dataset Auto-Detection & Loading (Code)](#cell-7--step-3-dataset-auto-detection--loading)
9. [Cell 8 — Step 4 Heading (Markdown)](#cell-8--step-4-heading)
10. [Cell 9 — Step 4: 5 Noise Functions + Median Filter + Visualization (Code)](#cell-9--step-4-5-noise-functions--median-filter--visualization)
11. [Cell 10 — Step 5 Heading (Markdown)](#cell-10--step-5-heading)
12. [Cell 11 — Step 5: Dataset Classes + Split + DataLoaders (Code)](#cell-11--step-5-dataset-classes--split--dataloaders)
13. [Cell 12 — Step 6 Heading (Markdown)](#cell-12--step-6-heading)
14. [Cell 13 — Step 6: Model Architecture (Code)](#cell-13--step-6-model-architecture)
15. [Cell 14 — Step 7 Heading (Markdown)](#cell-14--step-7-heading)
16. [Cell 15 — Step 7: Loss Functions (Code)](#cell-15--step-7-loss-functions)
17. [Cell 16 — Step 8 Heading (Markdown)](#cell-16--step-8-heading)
18. [Cell 17 — Step 8: GPU Metrics & Evaluation (Code)](#cell-17--step-8-gpu-metrics--evaluation)
19. [Cell 18 — Step 9 Heading (Markdown)](#cell-18--step-9-heading)
20. [Cell 19 — Step 9: Training Loop (Code)](#cell-19--step-9-training-loop)
21. [Cell 20 — Step 10 Heading (Markdown)](#cell-20--step-10-heading)
22. [Cell 21 — Step 10: Training Curves (Code)](#cell-21--step-10-training-curves)
23. [Cell 22 — Step 11 Heading (Markdown)](#cell-22--step-11-heading)
24. [Cell 23 — Step 11: Comprehensive Evaluation (Code)](#cell-23--step-11-comprehensive-evaluation)
25. [Cell 24 — Step 12 Heading (Markdown)](#cell-24--step-12-heading)
26. [Cell 25 — Step 12: Visual Comparison Grid (Code)](#cell-25--step-12-visual-comparison-grid)
27. [Cell 26 — Step 13 Heading (Markdown)](#cell-26--step-13-heading)
28. [Cell 27 — Step 13: Metrics Charts (Code)](#cell-27--step-13-metrics-charts)
29. [Cell 28 — Step 14 Heading (Markdown)](#cell-28--step-14-heading)
30. [Cell 29 — Step 14: Save Final Model (Code)](#cell-29--step-14-save-final-model)
31. [Cell 30 — Step 15 Heading (Markdown)](#cell-30--step-15-heading)
32. [Cell 31 — Step 15: Load Model & Denoise Function (Code)](#cell-31--step-15-load-model--denoise-function)
33. [Cell 32 — Step 15: Upload Widget for Interactive Denoising (Code)](#cell-32--step-15-upload-widget-for-interactive-denoising)
34. [Cell 33 — Step 16 Heading (Markdown)](#cell-33--step-16-heading)
35. [Cell 34 — Step 16: Verify Outputs (Code)](#cell-34--step-16-verify-outputs)
36. [Big Picture — How Everything Connects](#big-picture--how-everything-connects)

---

## Cell 0 — Title & Project Overview

**Type:** Markdown

This is the introductory cell that sets the context for the entire notebook. It defines four key aspects of the project:

- **Problem Statement:** Train an autoencoder architecture to clean noisy medical scans. The core idea is that medical images (MRI, CT) are corrupted by various types of noise during acquisition, transmission, or storage, and a neural network can learn to reverse this corruption.
- **Target Domain:** MRI and CT scan denoising specifically related to brain tumors. The focus on brain imaging is deliberate — brain MRI has unique noise characteristics (particularly Rician noise from the magnitude reconstruction of complex-valued MRI data) that differ from natural images.
- **Core Technique:** Convolutional Autoencoders with Skip Connections, which is the U-Net architecture. Skip connections are critical because they preserve fine-grained spatial information that would otherwise be lost during the encoder's downsampling process. Without skip connections, the decoder would have to reconstruct pixel-level details from heavily compressed bottleneck features alone, resulting in blurry outputs.
- **Advanced Evaluation:** The model is evaluated using PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index) across multiple noise types, not just training loss. This matters because low training loss does not guarantee perceptually good results — a model can achieve low MSE by producing blurry averages.

The cell also includes a detailed comparison table between the "old version" and this "improved version," covering the following improvements:

| Problem in Old Version | Fix in This Version |
|---|---|
| No noise type awareness | Noise type conditioning via learned embeddings |
| Reconstructs full image from scratch | Residual learning — predicts only the noise component |
| S&P noise gets blurred by convolutions | Median filter preprocessing for impulse noise |
| Random validation noise makes tracking impossible | Per-noise-type evaluation every epoch |
| MSE-dominant loss produces blurry outputs | Rebalanced loss with SSIM as primary component |
| All noise types from epoch 1 | 3-phase curriculum training |
| Plain U-Net convolutional blocks | ResUNet + Channel Attention (SE blocks) |
| BatchNorm unstable with small batches | GroupNorm (batch-size independent) |

---

## Cell 1 — Step 1 Heading

**Type:** Markdown

Simple section heading: **"Step 1: Installation & Environment Setup"**

---

## Cell 2 — Step 1: Installation & wandb Login

**Type:** Code (26 lines)

### What It Does

This cell performs three operations:
1. Installs required Python packages
2. Authenticates with Weights & Biases (wandb) for experiment tracking
3. Verifies the GPU environment

### Line-by-Line Breakdown

```python
!pip install -q wandb scikit-image kagglehub
```

- `wandb`: Weights & Biases — experiment tracking platform. Logs training loss, PSNR, SSIM, learning rate, and per-noise-type metrics in real-time. Essential for monitoring training remotely and comparing runs.
- `scikit-image`: Provides `peak_signal_noise_ratio` and `structural_similarity` functions for CPU-based final evaluation (used in Step 11).
- `kagglehub`: Kaggle's dataset/model management library.
- The `-q` flag suppresses verbose pip output.

```python
from kaggle_secrets import UserSecretsClient
import wandb
user_secrets = UserSecretsClient()
wandb_key = user_secrets.get_secret("wandb_api_key")
wandb.login(key=wandb_key)
```

Kaggle provides a secrets system (Settings → Secrets) to store API keys securely. The notebook retrieves the wandb API key from Kaggle secrets rather than hardcoding it. This is critical for security — the notebook can be shared publicly without exposing the key. The `wandb.login()` call authenticates the session, enabling all subsequent `wandb.log()` calls in the training loop.

```python
print("=" * 55)
print("  ENVIRONMENT CHECK")
...
print(f"  PyTorch:    {torch.__version__}")
print(f"  CUDA:       {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU:        {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    if torch.cuda.device_count() > 1:
        print(f"  GPU Count:  {torch.cuda.device_count()}")
print(f"  AMP:        {'Available' if hasattr(torch.amp, 'autocast') else 'Not available'}")
```

This diagnostic block confirms:
- **PyTorch version:** Must support `torch.amp.autocast` and the `weights_only` parameter in `torch.load()` (PyTorch 2.0+)
- **CUDA availability:** Training without GPU is not practical — a single epoch would take hours instead of minutes
- **GPU name and VRAM:** The T4 has 16GB VRAM. This determines maximum batch size and whether AMP is needed
- **Multi-GPU count:** If `torch.cuda.device_count() > 1`, the notebook could potentially use `DataParallel` (though this notebook does not implement it)
- **AMP support:** Automatic Mixed Precision requires CUDA and recent PyTorch. However, this notebook intentionally disables AMP for stability (configured in Step 2)

### Why This Matters

Running the full 100-epoch training without GPU would be completely impractical. The environment check catches issues early — if CUDA shows as `False`, the user knows to enable GPU acceleration in Kaggle settings before wasting time on a CPU run.

---

## Cell 3 — Dataset Directory Check

**Type:** Code (1 line)

```python
!ls /kaggle/input/datasets/camorednex/
```

### What It Does

This is a quick verification cell that lists the contents of the user's personal Kaggle dataset directory. The path `/kaggle/input/datasets/camorednex/` contains previously saved model checkpoints (like the epoch-30 checkpoint and the best model) that were uploaded as Kaggle datasets. This cell confirms that the resume checkpoint and the best model file are accessible before training or inference begins.

### Why It Matters

Kaggle's input system is read-only — files appear under `/kaggle/input/` only if they were added via the "+ Add Input" button. If the dataset wasn't added, the `resume_from` path in CONFIG would fail silently (or the training would start from epoch 1 instead of resuming). This cell provides an early sanity check.

---

## Cell 4 — Step 2 Heading

**Type:** Markdown

Section heading: **"Step 2: Imports & Configuration"**

---

## Cell 5 — Step 2: Imports & Configuration

**Type:** Code (99 lines)

### What It Does

This is the central configuration hub. It does two things: (1) imports every library used throughout the notebook, and (2) defines the `CONFIG` dictionary that controls all hyperparameters, paths, and training behavior.

### Imports Explained

**Core PyTorch:**
- `torch`, `torch.nn`, `torch.nn.functional`: The foundation for defining neural network layers, loss functions, and performing tensor operations. `F` is the functional API used for operations like `F.conv2d`, `F.mse_loss`, `F.interpolate` that don't require persistent state.
- `torch.utils.data.Dataset`, `DataLoader`: PyTorch's data pipeline. `Dataset` defines how to load and transform individual samples; `DataLoader` handles batching, shuffling, parallel loading, and memory pinning.
- `torchvision.transforms`, `torchvision.transforms.functional`: Image preprocessing — resize, flip, rotate, convert to tensor. The functional API (`TF`) is used for stateless operations like `TF.to_pil_image` during inference.

**Image Processing:**
- `PIL.Image`: Opens and converts image files. The `.convert("L")` call converts to grayscale, which is essential because MRI scans are single-channel.
- `skimage.metrics.peak_signal_noise_ratio`, `structural_similarity`: CPU-based reference implementations for image quality metrics. Used only for final evaluation because they are slower than GPU versions.

**Visualization & Utilities:**
- `matplotlib` with `Agg` backend: Headless rendering — required on Kaggle because there's no display. `plt.show()` renders inline in the notebook.
- `tqdm.auto`: Smart progress bars that work in both terminal and notebook environments.
- `glob`, `os`, `shutil`: File system operations — finding images, creating directories, copying checkpoints.
- `collections.Counter`, `defaultdict`: Counting class distributions and building grouped result dictionaries.

### The CONFIG Dictionary — Every Field Explained

```python
CONFIG = {
    # Data
    "image_size": 256,
    "batch_size": 8,
    "num_workers": 0,
```

- **`image_size: 256`**: All images are resized to 256×256 pixels. This is a standard choice for medical image denoising. Larger sizes (512) would preserve more detail but require 4× more memory per batch and 4× more computation per convolution. The U-Net architecture requires dimensions divisible by 16 (due to 4 pooling layers of stride 2), and 256 satisfies this.
- **`batch_size: 8`**: The number of samples processed before updating weights. 8 is conservative — on a T4 GPU with 16GB VRAM, you could push to 16, but smaller batches provide more frequent gradient updates (720 training images / 8 = 90 batches per epoch vs. 45 with batch_size=16), which helps convergence with curriculum training.
- **`num_workers: 0`**: Set to 0 on Kaggle because multi-worker data loading can cause issues with Kaggle's containerized environment. Data is loaded in the main process. This is slower but more reliable.

```python
    "noise_factor_max": 0.20,
```

- **`noise_factor_max: 0.20`**: The maximum noise level applied during training. For Gaussian noise, this means standard deviation up to 0.20 (on a [0,1] scale). For S&P, this is the maximum probability. The actual level for each sample is randomly drawn from `[0.02, 0.20]`, so the model sees varying noise intensities. This range covers both mild (barely visible) and severe (clearly degraded) noise.

```python
    "architecture": "ResUNet + Attention + Noise Conditioning",
    "in_channels": 1,
    "num_noise_types": 5,
    "features": [64, 128, 256, 512],
    "dropout": 0.05,
    "use_residual": True,
    "use_attention": True,
```

- **`architecture`**: A descriptive string for logging — not used programmatically.
- **`in_channels: 1`**: Grayscale input (1 channel). MRI scans are typically single-channel intensity images. If using RGB, this would be 3.
- **`num_noise_types: 5`**: The number of distinct noise categories the model must learn: Gaussian (0), Rician (1), Salt & Pepper (2), Poisson (3), Mixed (4). This controls the size of the noise type embedding.
- **`features: [64, 128, 256, 512]`**: Channel counts at each of the 4 encoder/decoder levels. More channels = more capacity to represent features, but also more parameters and slower computation. The progression doubles at each level, which is standard U-Net practice. The bottleneck has 1024 channels (512 × 2).
- **`dropout: 0.05`**: 5% dropout in convolutional blocks. Very low because residual learning already provides implicit regularization (the model can learn an identity mapping by outputting zero). Higher dropout would make the residual path unreliable.
- **`use_residual: True`**: The single most important flag. When True, the model learns to predict the NOISE (which is added to get the clean image: `denoised = input - noise_est`). When False, the model directly predicts the clean image (harder because image structure is more complex than noise patterns).
- **`use_attention: True`**: Enables SE (Squeeze-and-Excitation) channel attention blocks. These learn which feature channels to emphasize for each noise type, providing a form of adaptive feature selection.

```python
    "epochs": 100,
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    "warmup_epochs": 8,
    "scheduler": "cosine_warmup",
    "early_stopping_patience": 999,
    "use_amp": False,
    "grad_clip_norm": 0.5,
    "loss_cap": 0.5,
```

- **`epochs: 100`**: Total training epochs. With curriculum (30+30+40), each phase gets enough epochs to converge.
- **`learning_rate: 5e-4`**: The peak learning rate after warmup. This is conservative — earlier versions used 2e-3 but that caused instability. 5e-4 with cosine decay provides smooth, stable convergence.
- **`weight_decay: 1e-4`**: L2 regularization strength for AdamW. Prevents weights from growing too large, which reduces overfitting. AdamW decouples weight decay from the gradient update, unlike standard Adam where L2 regularization interacts with momentum.
- **`warmup_epochs: 8`**: During the first 8 epochs, the learning rate linearly increases from 0 to 5e-4. This prevents the optimizer from making large, destabilizing updates when the model is randomly initialized. The warmup period lets the model find a reasonable region of parameter space before applying full learning rate.
- **`scheduler: "cosine_warmup"`**: After warmup, the learning rate follows a cosine curve from 5e-4 down to near-zero. Cosine annealing is preferred over step decay because it provides smooth transitions and often finds better minima by spending more time at intermediate learning rates.
- **`early_stopping_patience: 999`**: Effectively disables early stopping. This was a critical fix — during curriculum phase transitions (e.g., epoch 30→31 when new noise types are introduced), the validation PSNR temporarily drops. With patience=20, the model would stop prematurely. Patience=999 ensures training continues through all curriculum phases.
- **`use_amp: False`**: Automatic Mixed Precision is intentionally disabled. While AMP (float16 forward pass, float32 weight update) typically cuts training time by 30-40%, it can cause numerical instability with SSIM loss computation and gradient overflow during curriculum transitions. The stability cost outweighs the speed benefit.
- **`grad_clip_norm: 0.5`**: After computing gradients, if the total gradient norm exceeds 0.5, all gradients are scaled down proportionally. This prevents "gradient explosion" — a single batch with extreme noise producing huge gradients that corrupt the model's weights.
- **`loss_cap: 0.5`**: If a single batch produces a loss value exceeding 0.5, that batch is skipped entirely. This handles edge cases where extreme noise combinations produce anomalous loss values that would destabilize training.

```python
    "resume_from": "/kaggle/input/datasets/camorednex/brain-mri-epoch30-checkpoint/checkpoint_epoch_30.pth",
```

- **`resume_from`**: Path to a checkpoint file for resuming training. This particular path points to a checkpoint saved at epoch 30 (end of Phase 1). The training loop uses this to restore model weights, optimizer state, and training history. If the file doesn't exist, training starts from epoch 1.

```python
    "curriculum": {
        "phase1_end": 30,
        "phase2_end": 60,
        "phase3_end": 100,
    },
```

- **Curriculum phases**: This is a core training strategy. Phase 1 (epochs 1-30) trains only on Gaussian noise — the simplest and most common noise type, giving the model a solid foundation. Phase 2 (epochs 31-60) adds Rician (MRI-specific) and Poisson (quantum) noise — these are physically motivated noise models for medical imaging. Phase 3 (epochs 61-100) introduces all 5 noise types including the challenging Salt & Pepper and Mixed noise. The progressive introduction prevents the model from being overwhelmed by difficult noise types before it has learned basic denoising skills.

```python
    "loss_type": "combined_v2",
    "mse_w": 0.4,
    "ssim_w": 0.15,
    "mae_w": 0.3,
    "edge_w": 0.15,
```

- **Loss weights**: These control the relative importance of each loss component. The weights sum to 1.0. In this stabilized version, Charbonnier (mse_w=0.4) and L1 (mae_w=0.3) are the primary drivers, with SSIM reduced to 0.15 and Edge at 0.15. The earlier version had SSIM at 0.4 as primary, but this caused NaN losses during early training. The current weights prioritize stable pixel-level accuracy (Charbonnier + L1 = 70%) while still incorporating structural (SSIM) and boundary (Edge) signals.

```python
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
```

- **Data splits**: 80% training (5,760 images), 10% validation (720 images), 10% test (720 images). No stratification is applied — the random shuffle should distribute tumor classes roughly evenly.

```python
    "checkpoint_dir": "/kaggle/working/",
    "local_backup_dir": "/kaggle/working/",
    "dataset": "Brain Tumor MRI (Kaggle)",
    "optimizer": "AdamW",
```

- **Output paths**: Both point to `/kaggle/working/` — the only writable directory in Kaggle. All checkpoints, plots, and saved models go here.

### Random Seeds and Device

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

Setting seeds to 42 ensures reproducibility: the same train/val/test split, the same noise patterns, the same weight initialization, and the same augmentation every time the notebook is run. This is critical for debugging and for comparing different hyperparameter settings.

---

## Cell 6 — Step 3 Heading

**Type:** Markdown

Section heading: **"Step 3: Download Brain Tumor MRI Dataset"**

---

## Cell 7 — Step 3: Dataset Auto-Detection & Loading

**Type:** Code (65 lines)

### What It Does

This cell locates the brain MRI dataset within Kaggle's input directory, collects all image file paths, and reports the class distribution. The dataset used is `masoudnickparvar/brain-tumor-mri-dataset` containing approximately 7,200 brain MRI images across 4 tumor classes.

### Three-Tier Dataset Detection

```python
dataset_path = None
for d in sorted(os.listdir('/kaggle/input')):
    candidate = os.path.join('/kaggle/input', d)
    if os.path.isdir(candidate):
        subdirs = []
        try:
            subdirs = os.listdir(candidate)
        except:
            continue
        if any(s.lower() in ['glioma', 'meningioma', 'notumor', 'pituitary',
                              'training', 'testing'] for s in subdirs):
            dataset_path = candidate
            break
```

**Tier 1:** Scans all directories under `/kaggle/input/` and checks if any contains known brain tumor class subfolders (glioma, meningioma, notumor, pituitary) or the Training/Testing directory structure. This is the most reliable detection method because it identifies the dataset by its content structure rather than its folder name.

```python
if dataset_path is None:
    possible_paths = [
        '/kaggle/input/brain-tumor-mri-dataset',
        '/kaggle/input/masoudnickparvar-brain-tumor-mri-dataset',
    ]
    for p in possible_paths:
        if os.path.exists(p):
            dataset_path = p
            break
```

**Tier 2:** If Tier 1 fails, tries known common paths. Kaggle's dataset slugs can vary depending on the uploader username and how the URL is formatted.

```python
if dataset_path is None:
    for d in sorted(os.listdir('/kaggle/input')):
        candidate = os.path.join('/kaggle/input', d)
        if os.path.isdir(candidate):
            dataset_path = candidate
            break
```

**Tier 3:** Last resort — uses the first available input directory. This handles edge cases where the dataset has an unexpected name but still contains valid images.

### Image Collection

```python
all_image_paths = []
for ext in ['*.jpg', '*.jpeg', '*.png']:
    all_image_paths.extend(glob.glob(os.path.join(dataset_path, '**', ext), recursive=True))
```

Uses `glob.glob` with `recursive=True` and the `**` pattern to search all subdirectories. The brain tumor MRI dataset has a structure like `Training/glioma/`, `Testing/meningioma/`, etc. — the recursive search finds all images regardless of how deeply nested they are.

### Class Distribution

```python
class_counts = Counter()
for p in all_image_paths:
    for part in p.split('/'):
        if part.lower() in ['glioma', 'meningioma', 'notumor', 'pituitary']:
            class_counts[part.lower()] += 1
            break
```

Extracts the tumor class name from each file path by looking for known directory names. This is a pragmatic approach — the class label is embedded in the directory structure rather than in a metadata file. The `Counter` tallies how many images belong to each class, which helps verify that the dataset was loaded correctly and check for severe class imbalance.

### The `all_image_paths` List

This list is the foundation for everything that follows. Every Dataset class (Step 5) and every evaluation function references this list. The `random.shuffle()` call ensures the 80/10/10 split gets a mixed distribution — without shuffling, all glioma images might end up in the training set and all pituitary images in the test set.

---

## Cell 8 — Step 4 Heading

**Type:** Markdown

Section heading: **"Step 4: Noise Generation — 5 Types with Median Preprocessing for S&P"**

The markdown explains that Salt & Pepper noise requires fundamentally different treatment than additive noise because convolutions blur S&P outliers rather than removing them. A median filter preprocessing step is provided specifically for S&P noise.

---

## Cell 9 — Step 4: 5 Noise Functions + Median Filter + Visualization

**Type:** Code (90 lines)

### What It Does

Defines 5 noise functions, a GPU-friendly median filter, and visualizes all noise types on a sample brain MRI image.

### Noise Type Mappings

```python
NOISE_TYPE_MAP = {'gaussian': 0, 'rician': 1, 'salt_pepper': 2, 'poisson': 3, 'mixed': 4}
NOISE_TYPE_NAMES = ['gaussian', 'rician', 'salt_pepper', 'poisson', 'mixed']
```

These two data structures provide bidirectional lookup between noise type names and integer indices. The indices (0-4) are used by the `NoiseConditioning` module to look up learned embeddings. `NOISE_TYPE_MAP` maps string → int (for training), and `NOISE_TYPE_NAMES` provides int → string (for display).

### Noise Function 1: Gaussian

```python
def add_gaussian_noise(image, std=0.1):
    return torch.clamp(image + torch.randn_like(image) * std, 0.0, 1.0)
```

**Theory:** Gaussian (additive white) noise is the most common noise model in signal processing. It arises from thermal noise in electronic sensors, quantization errors, and the Central Limit Theorem — when many independent noise sources contribute, their sum converges to a Gaussian distribution regardless of individual distributions. The model is `noisy = clean + N(0, σ²)` where σ controls the noise level.

**Implementation:** `torch.randn_like(image)` generates random values from a standard normal distribution N(0,1) with the same shape as the input tensor. Multiplying by `std` scales the standard deviation. `torch.clamp(..., 0, 1)` ensures pixel values stay within the valid [0,1] range — without this, some pixels could become negative or exceed 1.0, which would be invalid for image data.

**Medical context:** Gaussian noise appears in MRI from thermal fluctuations in the receiver coil electronics and from the digitization process in the analog-to-digital converter.

### Noise Function 2: Salt & Pepper

```python
def add_salt_pepper_noise(image, prob=0.02):
    noisy = image.clone()
    salt = torch.rand_like(image) < prob / 2
    pepper = torch.rand_like(image) < prob / 2
    noisy[salt] = 1.0
    noisy[pepper] = 0.0
    return noisy
```

**Theory:** Salt & Pepper (impulse) noise randomly sets pixels to either the maximum value (salt = white) or minimum value (pepper = black). Unlike Gaussian noise which affects every pixel slightly, S&P noise affects a small fraction of pixels completely. It's caused by transmission errors, faulty pixel sensors, or analog-to-digital converter errors. The parameter `prob` is the total probability of a pixel being affected — half goes to salt, half to pepper.

**Implementation:** Two independent random masks are generated using `torch.rand_like(image) < prob/2`. The boolean masks select pixels to set to 1.0 (salt) or 0.0 (pepper). The image is cloned first to avoid modifying the original. This noise is fundamentally different from additive noise because the corrupted pixels carry no information about the original values.

**Why S&P is hard for neural networks:** Convolutional filters work by computing weighted averages of neighboring pixels. For Gaussian noise, averaging reduces the noise because positive and negative deviations cancel out. For S&P noise, averaging a black (0.0) pixel with its white (1.0) neighbors produces a gray pixel — the outlier isn't removed, it's spread around. This is why a specialized median filter preprocessing step is needed.

### Noise Function 3: Rician

```python
def add_rician_noise(image, std=0.05):
    n1 = torch.randn_like(image) * std
    n2 = torch.randn_like(image) * std
    return torch.clamp(torch.sqrt((image + n1)**2 + n2**2), 0.0, 1.0)
```

**Theory:** Rician noise is the physically correct noise model for MRI magnitude images. MRI acquires complex-valued data (real and imaginary channels), both of which have independent Gaussian noise. The magnitude image (which is what radiologists see) is computed as `M = sqrt((signal + n1)² + n2²)`, where n1 and n2 are independent Gaussian noise in the real and imaginary channels respectively. The resulting magnitude follows a Rician distribution, which is signal-dependent: the noise variance is higher in bright regions (high signal) and lower in dark regions (low signal). This is different from Gaussian noise where the variance is constant.

**Implementation:** Two independent Gaussian noise tensors are generated. The magnitude formula `sqrt((image + n1)² + n2²)` exactly mirrors the MRI reconstruction process. The `torch.clamp` ensures valid pixel range.

**Medical context:** This is the most important noise model for MRI specifically. Any MRI denoising system that doesn't handle Rician noise is ignoring the dominant physical noise source in its target modality.

### Noise Function 4: Poisson

```python
def add_poisson_noise(image, scale=50.0):
    scaled = image * scale
    noisy = torch.poisson(scaled) / scale
    return torch.clamp(noisy, 0.0, 1.0)
```

**Theory:** Poisson noise (also called shot noise or quantum noise) arises from the discrete nature of photon counting. In X-ray and CT imaging, each pixel value represents the number of photons detected, which follows a Poisson distribution where the variance equals the mean: Var(X) = E(X). This means brighter regions (more photons) have higher absolute noise but lower relative noise. The `scale` parameter controls the effective photon count — higher scale means more photons and less relative noise (better signal-to-noise ratio).

**Implementation:** `torch.poisson()` generates Poisson-distributed random integers. Since Poisson requires non-negative integer inputs, the image (which is in [0,1]) is scaled up by `scale=50`, effectively treating pixel values as expected photon counts between 0 and 50. After adding noise, the result is scaled back to [0,1] range.

**Medical context:** Poisson noise is the dominant noise source in X-ray, CT, and PET imaging. It's also present in MRI when the signal-to-noise ratio is very low (e.g., diffusion-weighted imaging).

### Noise Function 5: Mixed

```python
def add_mixed_noise(image, gaussian_std=0.05, sp_prob=0.01):
    image = add_gaussian_noise(image, gaussian_std)
    image = add_salt_pepper_noise(image, sp_prob)
    return image
```

**Theory:** Real-world medical images rarely have a single clean noise type. A typical hospital scan might have thermal noise (Gaussian) from the electronics plus transmission errors (S&P) from the network. Mixed noise combines two noise types sequentially, representing the most realistic degradation scenario.

**Implementation:** Applies Gaussian noise first (affects all pixels), then S&P noise on top (affects random pixels completely). The Gaussian std (0.05) and S&P probability (0.01) are both moderate — this represents a realistic combined degradation.

### Median Filter Approximation

```python
def median_filter_approx(image, kernel_size=3):
    pad = kernel_size // 2
    shifts = []
    for di in range(-pad, pad+1):
        for dj in range(-pad, pad+1):
            shifts.append(F.pad(image, [pad, pad, pad, pad], mode='reflect')
                       [:, :, di:di+image.shape[2], dj:dj+image.shape[3]])
    stacked = torch.stack(shifts, dim=0)
    result = stacked.median(dim=0).values
    return result
```

**Theory:** The median filter is the mathematically optimal filter for impulse (Salt & Pepper) noise. Unlike the mean (which convolution computes), the median is robust to outliers — a single S&P pixel in a 3×3 neighborhood won't affect the median value at all, but it will shift the mean significantly. For a 3×3 window, 9 values are collected and the 5th (middle) value is selected.

**GPU implementation challenge:** Exact median requires sorting, which is expensive on GPU. This implementation collects all 9 spatially-shifted versions of the image (representing the 3×3 neighborhood around each pixel), stacks them, and uses `torch.median()` which is GPU-optimized.

**Reflect padding** (`mode='reflect'`) handles border pixels by mirroring the image at edges, avoiding artificial zero-padding that would create edge artifacts.

### Visualization

The visualization takes the first brain MRI image and shows 6 panels: Original, Gaussian, Rician, Salt & Pepper, Poisson, and Mixed. This serves as a sanity check — if any noise type looks wrong (e.g., Rician producing all-white images), something is broken. It also helps the user build intuition for what each noise type looks like visually.

---

## Cell 10 — Step 5 Heading

**Type:** Markdown

Section heading: **"Step 5: Dataset with Curriculum Noise & Noise Type Labels"**

Explains that the dataset now returns noise type index alongside noisy/clean pairs, enabling noise conditioning. Also introduces curriculum training that changes active noise types based on epoch.

---

## Cell 11 — Step 5: Dataset Classes + Split + DataLoaders

**Type:** Code (121 lines)

### What It Does

Defines two PyTorch `Dataset` classes and creates all the DataLoaders needed for training and evaluation.

### BrainMRIDatasetV2 (Training Dataset)

```python
class BrainMRIDatasetV2(Dataset):
    def __init__(self, image_paths, image_size=256, noise_factor_max=0.20,
                 augment=False, noise_types=None):
```

**Key design decisions:**

1. **Returns `(noisy, clean, noise_type_idx)`** — the third element (noise type index) is what enables noise conditioning in the model. Without it, the model would have no way to know which type of noise it should be removing.

2. **`set_noise_types()` method** — this is the curriculum mechanism. Between epochs, the training loop calls `train_ds.set_noise_types(get_curriculum_noise_types(epoch))` to change which noise types are active. In Phase 1, only Gaussian is active. Phase 2 adds Rician + Poisson. Phase 3 activates all 5. This is much more efficient than creating new Dataset objects every epoch.

3. **Augmentation pipeline** (training only):
   - `RandomHorizontalFlip(p=0.5)`: 50% chance of mirroring the image left-right. Brain anatomy is roughly symmetric, so this is safe.
   - `RandomRotation(10)`: Random rotation up to 10 degrees. Small rotations simulate patient positioning variation.
   - `ColorJitter(brightness=0.1)`: Random brightness variation of ±10%. Simulates differences in MRI acquisition parameters.
   - These augmentations prevent overfitting by showing the model different versions of each image every epoch.

4. **Noise selection and level randomization:**
   ```python
   noise_type = random.choice(self.noise_types)
   level = np.random.uniform(0.02, self.noise_factor_max)
   ```
   Each time a sample is accessed, a random noise type (from the active set) and a random noise level are chosen. This means the model sees different noise on the same image across epochs, which prevents memorization and forces it to learn general denoising patterns.

### BrainMRIEvalDataset (Evaluation Dataset)

```python
class BrainMRIEvalDataset(Dataset):
    def __init__(self, image_paths, image_size=256, noise_type='gaussian',
                 noise_level=0.10):
```

**Key difference from training dataset:** FIXED noise type and level. No randomness. This is essential for fair evaluation — if the noise varied between evaluation runs, you couldn't determine whether a PSNR improvement came from the model getting better or from getting an easier noise sample. The fixed noise level (0.10 for most types, 0.02 for S&P, 50.0 for Poisson) provides a consistent benchmark.

### Data Splitting

```python
n = len(all_image_paths)
n_train = int(n * CONFIG["train_split"])  # ~5760
n_val = int(n * CONFIG["val_split"])       # ~720

train_paths = all_image_paths[:n_train]
val_paths = all_image_paths[n_train:n_train + n_val]
test_paths = all_image_paths[n_train + n_val:]
```

Simple sequential split after shuffling. No stratification is applied because the classes are roughly balanced (each tumor class has ~1,500-2,000 images).

### DataLoader Creation

**5 per-noise-type validation loaders:**
```python
for nt in NOISE_TYPE_NAMES:
    ds = BrainMRIEvalDataset(val_paths, CONFIG["image_size"], nt, level)
    val_loaders[nt] = DataLoader(ds, batch_size=CONFIG["batch_size"],
                                shuffle=False, num_workers=CONFIG["num_workers"],
                                pin_memory=True)
```
Each noise type gets its own dedicated validation DataLoader. This enables per-noise-type PSNR/SSIM tracking — you can see exactly which noise types the model is struggling with at any point during training.

**Combined validation loader:** For quick overall validation on epochs where full per-type evaluation is skipped (every 5th epoch gets full evaluation).

**Training loader:** `shuffle=True` ensures random ordering each epoch. `drop_last=True` discards the last incomplete batch, which prevents a batch of size 1 from causing BatchNorm issues (though this model uses GroupNorm, so it's a precaution rather than a necessity). `pin_memory=True` enables faster GPU transfer by allocating pinned (non-swappable) memory.

---

## Cell 12 — Step 6 Heading

**Type:** Markdown

Section heading: **"Step 6: Improved Architecture — ResUNet + Attention + Noise Conditioning"**

Lists 6 key improvements: residual learning, noise type embedding, SE attention, GroupNorm, residual connections in DoubleConv, and median filter for S&P.

---

## Cell 13 — Step 6: Model Architecture

**Type:** Code (173 lines)

### What It Does

Defines the complete neural network architecture: `ResUNetDenoiser` with SE attention, noise conditioning, and residual learning. This is the most important cell in the notebook.

### SEBlock (Squeeze-and-Excitation)

```python
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 8), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 8), channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b, c)       # (B, C, H, W) → (B, C)
        y = self.excite(y).view(b, c, 1, 1)  # (B, C) → (B, C, 1, 1)
        return x * y.expand_as(x)             # Channel-wise reweighting
```

**Theory:** Not all feature channels contribute equally to denoising. For Gaussian noise, channels detecting smooth regions might be most important. For S&P noise, channels detecting local outliers are critical. SE attention lets the network learn "how much" each channel should contribute for a given input.

**Data flow:**
1. **Squeeze:** `AdaptiveAvgPool2d(1)` collapses the spatial dimensions (H×W) to a single value per channel. For a feature map of shape (B, 64, 256, 256), this produces (B, 64, 1, 1). This global average captures the overall "activation level" of each channel.
2. **Excite:** Two fully-connected layers with ReLU and Sigmoid. The first layer reduces dimensionality by `reduction=16` (e.g., 64 → 4), creating an information bottleneck that forces the network to learn compressed channel relationships. The second layer restores the original dimensionality (4 → 64). Sigmoid constrains outputs to [0, 1], creating per-channel "importance weights."
3. **Reweight:** Element-wise multiplication `x * y` scales each channel by its learned weight. Important channels get boosted (weight near 1.0), unimportant channels get suppressed (weight near 0.0).

**Parameter efficiency:** With `channels=64` and `reduction=16`, the SE block adds only `64×4 + 4×64 = 512` parameters — negligible compared to the convolutional layers. The `max(channels // reduction, 8)` ensures a minimum bottleneck size of 8, preventing the bottleneck from becoming too narrow for small channel counts.

### ResDoubleConv (The Building Block)

```python
class ResDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.05, use_attention=True):
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
        )
        self.se = SEBlock(out_ch) if use_attention else nn.Identity()
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        return out + residual
```

**This is the core building block used throughout the U-Net.** Each block performs:

1. **Two 3×3 convolutions** with GroupNorm and ReLU. The 3×3 kernel is the standard choice for spatial feature extraction — it captures local patterns while keeping parameter count manageable. Padding=1 preserves spatial dimensions.
2. **GroupNorm(8, out_ch):** Divides channels into 8 groups and normalizes within each group. Unlike BatchNorm (which normalizes across the batch dimension and behaves poorly with batch_size=8), GroupNorm's behavior is independent of batch size. This is critical for this project because the small batch size would make BatchNorm statistics unreliable.
3. **`bias=False`:** Bias is omitted because GroupNorm already includes an affine transformation (learnable scale and shift), making the convolutional bias redundant.
4. **Dropout2d:** Applies spatial dropout — drops entire channels with probability `p=0.05`. Very low dropout rate because residual learning already provides regularization.
5. **SE attention:** Applied after the second convolution, letting the block dynamically emphasize important channels.
6. **Internal residual connection:** `out + residual` — the block learns only the DELTA from input to output. If the input is already good enough, the block can learn near-zero changes (the convolutions can output values close to zero, making the output approximately equal to the input via the skip). This is much easier than learning the full transformation from scratch.
7. **1×1 residual projection:** When `in_ch != out_ch` (at resolution boundaries), a 1×1 convolution projects the residual to match the output channel count. When channels match, `nn.Identity()` passes through unchanged.

### NoiseConditioning

```python
class NoiseConditioning(nn.Module):
    def __init__(self, num_noise_types=5, embed_dim=16, spatial_size=256):
        self.embedding = nn.Embedding(num_noise_types, embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.spatial_size = spatial_size

    def forward(self, noise_idx, batch_size):
        emb = self.embedding(noise_idx)                    # (B, 16)
        emb = self.proj(emb)                                # (B, 16)
        emb = emb.unsqueeze(-1).unsqueeze(-1)               # (B, 16, 1, 1)
        emb = emb.expand(-1, -1, self.spatial_size, self.spatial_size)  # (B, 16, 256, 256)
        return emb
```

**Theory:** This is the KEY innovation of the architecture. Without noise conditioning, the model must infer the noise type from the image alone — which is ambiguous because moderate Gaussian and Poisson noise can look similar, and the model has no explicit signal about what to remove. With conditioning, you're essentially telling the model "this is Gaussian noise" and it can activate its Gaussian-specific denoising strategy.

**How it works:**
1. **Embedding lookup:** `nn.Embedding(5, 16)` maps each noise type index (0-4) to a 16-dimensional learnable vector. During training, these embeddings are optimized alongside the rest of the model. Similar noise types (e.g., Gaussian and Rician) should develop similar embeddings.
2. **Projection MLP:** Two linear layers with ReLU transform the embedding. This gives the network flexibility to learn non-trivial noise type representations — the raw embedding might not be the best input representation for the convolutions.
3. **Spatial expansion:** The embedding is reshaped from (B, 16) to (B, 16, 1, 1) and then expanded to (B, 16, 256, 256). Every spatial position gets the same embedding, which makes sense because the noise type affects the entire image uniformly.
4. **Concatenation with input:** The expanded embedding is concatenated with the input image along the channel dimension: (B, 1, 256, 256) + (B, 16, 256, 256) → (B, 17, 256, 256). The first convolutional layer takes 17 input channels instead of 1.

**Why spatial expansion rather than addition?** Concatenation lets the network learn how to combine the noise type information with the image, rather than forcing a fixed combination. The first convolution can learn to weight different embedding channels differently for different spatial regions.

### EncoderBlock

```python
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.05, use_attention=True):
        self.pool = nn.MaxPool2d(2)
        self.conv = ResDoubleConv(in_ch, out_ch, dropout, use_attention)

    def forward(self, x):
        skip = self.conv(x)     # Convolution BEFORE pooling
        pooled = self.pool(skip)
        return skip, pooled      # Return both skip and pooled
```

The encoder block performs two operations:
1. Apply `ResDoubleConv` to extract features — this result is saved as the **skip connection** for the decoder.
2. Apply `MaxPool2d(2)` to downsample by 2× — this reduces spatial resolution while preserving the most activated features.

**Why convolution before pooling?** The skip connection needs to capture full-resolution features. If pooling happened first, the skip would be at half resolution and the decoder would receive degraded features. By convolving first, the skip gets rich, full-resolution feature maps.

**MaxPool vs. strided convolution:** MaxPool2d(2) is used instead of strided convolution because it selects the maximum activation in each 2×2 window, which tends to preserve the strongest features. Strided convolution would compute a weighted average, which can dilute important features.

### DecoderBlock

```python
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.05, use_attention=True):
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ResDoubleConv(in_ch // 2 + skip_ch, out_ch, dropout, use_attention)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)
```

The decoder block performs three operations:
1. **Upsample:** `ConvTranspose2d` (transposed convolution) doubles the spatial resolution. This is learned upsampling — the network learns how to best fill in the missing pixels. Kernel_size=2 and stride=2 means each input pixel maps to a 2×2 output region.
2. **Skip connection concatenation:** The upsampled features are concatenated with the corresponding encoder skip connection along the channel dimension. For example, if the upsampled features have 256 channels and the skip has 512 channels, the concatenated result has 768 channels. This is how high-resolution spatial information from the encoder reaches the decoder.
3. **Convolution:** `ResDoubleConv` processes the concatenated features, learning to combine the coarse (upsampled) and fine (skip) information.

**Size mismatch handling:** Due to odd input dimensions or rounding, the upsampled features might differ from the skip connection by 1 pixel. `F.interpolate` with bilinear mode resolves this by resizing the upsampled features to exactly match the skip dimensions.

### ResUNetDenoiser (The Full Network)

```python
class ResUNetDenoiser(nn.Module):
    def __init__(self, in_channels=1, num_noise_types=5, noise_embed_dim=16,
                 features=[64, 128, 256, 512], dropout=0.05,
                 use_residual=True, use_attention=True, spatial_size=256):
```

**Complete data flow diagram:**

```
Input: noisy_image (B, 1, 256, 256) + noise_idx (B,)

  ┌─ NoiseConditioning: noise_idx → embedding (B, 16, 256, 256)
  │
  ├─ Concatenate: image + embedding → (B, 17, 256, 256)
  │
  ├─ enc0 [ResDoubleConv]: (B, 17, 256, 256) → s0: (B, 64, 256, 256)
  │   └─ MaxPool2d: → p0: (B, 64, 128, 128)
  │
  ├─ enc1 [EncoderBlock]: p0 → s1: (B, 128, 128, 128), p1: (B, 128, 64, 64)
  │
  ├─ enc2 [EncoderBlock]: p1 → s2: (B, 256, 64, 64), p2: (B, 256, 32, 32)
  │
  ├─ enc3 [EncoderBlock]: p2 → s3: (B, 512, 32, 32), p3: (B, 512, 16, 16)
  │
  ├─ bottleneck [ResDoubleConv]: (B, 512, 16, 16) → (B, 1024, 16, 16)
  │
  ├─ dec3 [DecoderBlock]: bottleneck + s3 → (B, 512, 32, 32)
  │
  ├─ dec2 [DecoderBlock]: d3 + s2 → (B, 256, 64, 64)
  │
  ├─ dec1 [DecoderBlock]: d2 + s1 → (B, 128, 128, 128)
  │
  ├─ dec0 [DecoderBlock]: d1 + s0 → (B, 64, 256, 256)
  │
  ├─ out_conv [Conv2d 1×1]: (B, 64, 256, 256) → noise_est (B, 1, 256, 256)
  │
  └─ Residual: denoised = input - noise_est → clamp [0, 1]
```

**Why `enc0` is a `ResDoubleConv` (not `EncoderBlock`):** The first block doesn't need a skip connection in the traditional U-Net sense — it receives the concatenated input + noise embedding and produces the first-level features. The skip connection for the decoder comes from the output of this block. Manual `F.max_pool2d(s0, 2)` is applied after to downsample.

**Bottleneck design:** The bottleneck doubles the channel count (512 → 1024) without changing spatial resolution. This provides maximum representational capacity at the most compressed spatial level, where the network needs to capture global context about the image structure.

**1×1 output convolution:** `nn.Conv2d(features[0], in_channels, kernel_size=1)` projects the 64-channel decoder output down to 1 channel (grayscale). A 1×1 kernel performs channel-wise combination without spatial mixing — at this point, the spatial features are already fully resolved.

**Residual learning implementation:**
```python
if self.use_residual:
    denoised = x - noise_est
    return torch.clamp(denoised, 0.0, 1.0)
else:
    return torch.sigmoid(noise_est)
```

When `use_residual=True`, the model outputs an estimate of the NOISE. The clean image is recovered by subtracting: `clean = noisy - noise_estimate`. This is much easier for the network because:
- Noise is simpler and more predictable than image structure
- Noise has known statistical properties (the conditioning tells the model what type)
- If the model outputs zero, the result equals the input (safe default)
- The network only needs to learn the "correction" rather than the full reconstruction

When `use_residual=False`, the model directly predicts the clean image through a Sigmoid activation (mapping to [0,1]). This is harder because the model must learn the full image structure.

**Shape test at the bottom** creates dummy inputs and verifies the output shape matches expectations before training begins. This catches dimension mismatches early.

---

## Cell 14 — Step 7 Heading

**Type:** Markdown

Section heading: **"Step 7: Rebalanced Loss Function"**

Explains the shift from MSE-primary to Charbonnier + SSIM + L1 + Edge combined loss. MSE dominance produces blurry outputs; Charbonnier is more robust to outliers (critical for S&P noise); SSIM preserves structure; Edge loss protects tumor boundaries.

---

## Cell 15 — Step 7: Loss Functions

**Type:** Code (102 lines)

### What It Does

Defines 4 loss component classes and combines them into `CombinedLossV2`.

### SSIMLoss

```python
class SSIMLoss(nn.Module):
    def forward(self, pred, target):
        C1, C2 = 0.01**2, 0.03**2
        mu_x = F.conv2d(pred, self.kernel, padding=self.padding, groups=self.channels)
        mu_y = F.conv2d(target, self.kernel, padding=self.padding, groups=self.channels)
        ...
        return 1.0 - (num / (den + 1e-8)).mean()
```

**Theory:** SSIM (Structural Similarity Index) was designed to match human perception of image quality. Unlike MSE which treats all pixels equally, SSIM considers three components:
- **Luminance:** Are the overall brightness levels similar?
- **Contrast:** Are the local intensity variations similar?
- **Structure:** Are the local patterns (edges, textures) similar?

The SSIM formula is: `SSIM(x,y) = (2μxμy + C1)(2σxy + C2) / (μx² + μy² + C1)(σx² + σy² + C2)`

Where:
- `μx, μy` = local means (computed via Gaussian-weighted convolution)
- `σx², σy²` = local variances
- `σxy` = local covariance
- `C1 = 0.01², C2 = 0.03²` = stability constants preventing division by zero

**Implementation as convolution:** All operations (means, variances, covariance) are computed using `F.conv2d` with a pre-computed Gaussian kernel. This means the entire SSIM computation runs on GPU with no CPU transfers — critical for training speed.

**Loss = 1 - SSIM:** Since SSIM ranges from -1 to 1 (with 1 being identical), the loss `1 - SSIM` is minimized when images are most similar. The `.mean()` averages over all spatial positions and the batch.

### CharbonnierLoss

```python
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff**2 + self.eps**2))
```

**Theory:** Charbonnier loss is a smooth approximation of L1 loss. It behaves like:
- **L2 (MSE)** for small errors: `sqrt(x² + ε²) ≈ ε` when `|x| << ε` — smooth gradients
- **L1** for large errors: `sqrt(x² + ε²) ≈ |x|` when `|x| >> ε` — robust to outliers

This hybrid behavior is ideal for medical image denoising because:
- For most pixels (small errors), it provides smooth, well-behaved gradients like MSE
- For S&P noise pixels (large errors = 1.0), it doesn't over-penalize like MSE (which would produce a gradient of 2.0 for a unit error) but instead provides a bounded gradient like L1 (gradient ≈ 1.0)
- The `eps=1e-3` prevents the gradient from becoming infinite at zero, which would happen with pure L1

### EdgeLoss

```python
class EdgeLoss(nn.Module):
    def forward(self, pred, target):
        def grad(x):
            return torch.sqrt(F.conv2d(x, self.sobel_x, padding=1)**2 +
                              F.conv2d(x, self.sobel_y, padding=1)**2 + 1e-8)
        return F.l1_loss(grad(pred), grad(target))
```

**Theory:** Tumor boundaries in MRI are the most diagnostically important features. Standard pixel-wise losses (MSE, L1) treat all pixels equally, but a 1-pixel shift in a tumor boundary is far more consequential than a 1-pixel intensity error in a homogeneous region.

**Sobel gradient computation:** The Sobel operators detect horizontal and vertical edges:
```
Sobel_x = [[-1, 0, 1],    Sobel_y = [[-1, -2, -1],
            [-2, 0, 2],               [ 0,  0,  0],
            [-1, 0, 1]]               [ 1,  2,  1]]
```

The gradient magnitude at each pixel is `sqrt(Gx² + Gy²)`, computed using `F.conv2d` for GPU efficiency. The loss then measures the L1 distance between the gradient maps of the predicted and target images — if the model blurs an edge, the gradient magnitude changes, and this loss penalizes it.

### CombinedLossV2

```python
class CombinedLossV2(nn.Module):
    def __init__(self, mse_w=0.2, ssim_w=0.4, mae_w=0.15, edge_w=0.25, channels=1):
        self.charb = CharbonnierLoss()
        self.mae = nn.L1Loss()
        self.ssim = SSIMLoss(channels=channels)
        self.edge = EdgeLoss()

    def forward(self, pred, target):
        losses, total = {}, 0.0
        if self.mse_w > 0:
            losses['charb'] = self.charb(pred, target); total += self.mse_w * losses['charb']
        if self.ssim_w > 0:
            losses['ssim'] = self.ssim(pred, target); total += self.ssim_w * losses['ssim']
        ...
        return total, losses
```

**Final loss formula** (with the actual CONFIG values):
```
L = 0.4 × Charbonnier + 0.15 × SSIM + 0.3 × L1 + 0.15 × Edge
```

The weights sum to 1.0. In this stabilized version:
- **Charbonnier (0.4)** is the primary driver for pixel-level accuracy, robust to S&P outliers
- **L1 (0.3)** provides strong absolute error signal — complements Charbonnier
- **SSIM (0.15)** is reduced from the earlier 0.4 to prevent NaN losses during early training
- **Edge (0.15)** preserves tumor boundaries

The function returns both the total loss (for backpropagation) and a dictionary of individual losses (for logging and debugging). This dual return is crucial — without it, you couldn't diagnose which loss component is causing problems.

---

## Cell 16 — Step 8 Heading

**Type:** Markdown

Section heading: **"Step 8: GPU-Accelerated Metrics & Per-Noise-Type Evaluation"**

---

## Cell 17 — Step 8: GPU Metrics & Evaluation

**Type:** Code (70 lines)

### What It Does

Defines GPU-accelerated PSNR and SSIM functions, and the per-noise-type evaluation function.

### gpu_psnr

```python
def gpu_psnr(pred, target, data_range=1.0):
    mse = F.mse_loss(pred, target, reduction='none').mean(dim=[1,2,3])
    mse = torch.clamp(mse, min=1e-10)
    psnr = 10.0 * torch.log10(data_range**2 / mse)
    return psnr.mean().item()
```

PSNR = 10 × log₁₀(MAX² / MSE). With normalized images in [0,1], MAX=1.0. The computation stays entirely on GPU — `F.mse_loss` with `reduction='none'` computes per-sample MSE, then `.mean(dim=[1,2,3])` averages over spatial dimensions but not the batch. `torch.clamp(mse, min=1e-10)` prevents `log10(0)` when predictions are perfect. The final `.item()` converts a single scalar to Python float.

**Speed advantage:** CPU-based PSNR (skimage) requires transferring entire batch tensors to CPU and converting to numpy — this can be 10-100× slower than staying on GPU.

### gpu_ssim

```python
def gpu_ssim(pred, target, data_range=1.0):
    kernel = criterion.ssim.kernel   # Reuse the kernel buffer
    ...
```

Reuses the Gaussian kernel buffer from the `SSIMLoss` instance — avoids recomputing or allocating a new kernel. Same SSIM formula as the loss, but returns the similarity value (not 1-SSIM).

### evaluate_per_noise_type

```python
@torch.no_grad()
def evaluate_per_noise_type(model, val_loaders, criterion, device, use_amp=False):
```

This function is a KEY improvement over naive evaluation. It iterates over each noise type's dedicated validation loader independently, computing PSNR, SSIM, and loss for each. This reveals exactly which noise types the model handles well and which need improvement. A single overall PSNR would hide problems — e.g., Gaussian at 40 dB but S&P at 15 dB would average to 27.5 dB, looking reasonable but masking a critical failure.

The `@torch.no_grad()` decorator disables gradient computation, cutting memory usage in half and speeding up inference.

---

## Cell 18 — Step 9 Heading

**Type:** Markdown

Section heading: **"Step 9: Training Loop with Curriculum + Per-Type Tracking"**

Describes the 3-phase curriculum strategy and explains why starting with all 5 noise types from epoch 1 is problematic (the model gets confused by fundamentally different noise patterns and learns poor representations for all of them).

---

## Cell 19 — Step 9: Training Loop

**Type:** Code (289 lines — the longest and most complex cell)

### What It Does

The main training loop. Orchestrates curriculum updates, forward/backward passes, validation, checkpointing, wandb logging, and early stopping.

### Curriculum Noise Selection

```python
def get_curriculum_noise_types(epoch):
    cur = CONFIG["curriculum"]
    if epoch <= cur["phase1_end"]:
        return ['gaussian']
    elif epoch <= cur["phase2_end"]:
        return ['gaussian', 'rician', 'poisson']
    else:
        return NOISE_TYPE_NAMES  # all 5
```

Maps epoch number to active noise types. This function is called at the start of each epoch to update the training dataset's noise distribution.

**Phase 1 rationale (Gaussian only):** Gaussian noise is the simplest and most common. By training exclusively on it for 30 epochs, the model develops strong basic denoising capabilities — learning to distinguish signal from noise, understanding spatial correlations, and building useful convolutional features. These are transferable skills for other noise types.

**Phase 2 rationale (+ Rician + Poisson):** Rician is MRI-specific and Poisson is quantum noise — both are additive (or signal-dependent) noise types that share mathematical similarities with Gaussian. The model can extend its Gaussian knowledge to these types without catastrophic forgetting.

**Phase 3 rationale (all 5):** S&P and Mixed noise are fundamentally different (impulse vs. additive). Introducing them last, when the model already has strong feature extraction capabilities, prevents them from destabilizing the learning process.

### Resume from Checkpoint

```python
if CONFIG.get("resume_from") and os.path.exists(CONFIG["resume_from"]):
    ckpt = torch.load(CONFIG["resume_from"], map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_val_psnr = ckpt.get("val_psnr", 0.0)
    history = ckpt.get("history", history)
```

Restores model weights, epoch counter, best PSNR, and training history from a checkpoint file. `weights_only=False` is required because the checkpoint contains numpy arrays and dictionaries that PyTorch 2.6+ would otherwise reject as potentially unsafe. The `map_location=DEVICE` ensures tensors are loaded to the correct device (GPU if available).

**Optimizer state restoration:**
```python
if "optimizer_state_dict" in ckpt:
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
```

This is critical — without restoring optimizer state, the AdamW momentum buffers would be reset, causing the optimizer to behave as if starting from scratch even though the model has partially-trained weights. This would lead to large, destabilizing updates.

**Scheduler advancement:**
```python
if start_epoch > 1:
    for _ in range(start_epoch - 1):
        scheduler.step()
```

The LR scheduler maintains an internal step counter. If resuming from epoch 30, the scheduler needs to be advanced 29 steps so that the learning rate at epoch 30 matches what it would have been in the original run. Without this, the scheduler would start from warmup LR instead of the correct epoch-30 LR.

### Learning Rate Schedule

```python
def get_lr_lambda(epoch):
    warmup = CONFIG["warmup_epochs"]
    total = CONFIG["epochs"]
    if epoch < warmup:
        return (epoch + 1) / warmup   # Linear warmup
    progress = (epoch - warmup) / (total - warmup)
    return 0.5 * (1 + np.cos(np.pi * progress))  # Cosine decay
```

**Epochs 1-8 (warmup):** LR linearly increases from `5e-4/8 ≈ 6.25e-5` to `5e-4`. This prevents the randomly-initialized model from making large, destructive updates in the first few epochs when gradients are noisy and uninformative.

**Epochs 9-100 (cosine decay):** LR follows a cosine curve from 5e-4 down to near-zero. The smooth decay allows the model to make large steps early (exploring the loss landscape) and small steps later (refining the solution). The cosine schedule is preferred over step decay because it spends more time at intermediate learning rates, which often leads to better minima.

### wandb Initialization

```python
wandb_run = wandb.init(
    project="medical-denoising-brain-mri",
    entity="camorednex-_",
    name=f"resunet-stable-lr{CONFIG['learning_rate']}-ssim{CONFIG['ssim_w']}",
    config=CONFIG,
    resume="allow",
    id="twqu38pi"
)
```

- `resume="allow"` with a fixed `id="twqu38pi"` ensures that if the training is interrupted and restarted, wandb resumes the existing run rather than creating a new one. This keeps all training metrics in a single continuous chart.
- `config=CONFIG` logs all hyperparameters to wandb for experiment tracking and comparison.
- The run name encodes key hyperparameters (learning rate, SSIM weight) for easy identification in the wandb dashboard.

### Training Loop (Per Epoch)

```python
for epoch in range(start_epoch, CONFIG["epochs"] + 1):
```

**Each epoch consists of:**

1. **Curriculum update:** `train_ds.set_noise_types(active_noise)` changes which noise types the training dataset generates. This is a lightweight operation — it just changes a list attribute.

2. **Training pass:** Iterates over all batches, computes forward pass, loss, backward pass, and weight update. Key safety mechanisms:
   - **NaN/Inf detection:** If loss is NaN or Inf, the batch is skipped entirely. This prevents corrupted gradients from destroying the model. A counter tracks how often this happens.
   - **Loss capping:** If loss exceeds 0.5, the batch is skipped. Anomalous loss values often indicate extreme noise or numerical issues that would produce destructive gradient updates.
   - **Gradient clipping:** `clip_grad_norm_(model.parameters(), max_norm=0.5)` scales all gradients down if their total norm exceeds 0.5. This prevents gradient explosion without changing the gradient direction.
   - **`optimizer.zero_grad(set_to_none=True)`:** Sets gradients to `None` instead of zero tensors, which is slightly more memory-efficient.

3. **Validation:** Full per-noise-type evaluation every 5 epochs (or on epoch 1 and 100), quick combined validation on other epochs. The frequency tradeoff balances evaluation thoroughness against training speed — per-type evaluation takes ~5× longer than combined.

4. **Learning rate scheduling:** `scheduler.step()` updates the learning rate according to the warmup+cosine schedule.

5. **History recording:** All metrics are appended to the `history` dictionary for plotting and analysis.

6. **wandb logging:** All metrics (train/val loss, PSNR, SSIM, per-type metrics, learning rate, phase) are logged to wandb for real-time monitoring.

7. **Best model checkpointing:** If the current overall PSNR exceeds the previous best, the model is saved with full state (model, optimizer, epoch, metrics, history). The `best_model.pth` file is overwritten each time, so only the best model is retained.

8. **Periodic checkpoints:** Every 10 epochs, a backup checkpoint is saved (e.g., `checkpoint_epoch_30.pth`). These provide recovery points in case the best model checkpoint is corrupted or if you want to analyze the model at intermediate training stages.

9. **Early stopping:** Checks if patience counter exceeds the configured threshold (999 — effectively disabled). With patience=999, the model trains for all 100 epochs regardless of validation PSNR trends.

---

## Cell 20 — Step 10 Heading

**Type:** Markdown

Section heading: **"Step 10: Training Curves & Per-Noise-Type Analysis"**

---

## Cell 21 — Step 10: Training Curves

**Type:** Code (46 lines)

### What It Does

Plots 6 subplots of training history in a 2×3 grid.

**Subplot 1 — Loss (Train vs. Val):** Shows whether the model is learning (both should decrease) and whether it's overfitting (val loss increases while train loss decreases). A gap between train and val loss is normal but shouldn't grow.

**Subplot 2 — Overall PSNR:** Should increase over training. The PSNR curve typically shows three phases matching the curriculum: steady improvement in Phase 1, a dip or plateau when Phase 2 introduces new noise types (epochs 31-35), then recovery and further improvement.

**Subplot 3 — Overall SSIM:** Same pattern as PSNR but measures structural similarity rather than pixel accuracy. SSIM is more sensitive to blurring than PSNR.

**Subplot 4 — PSNR per Noise Type:** 5 colored lines showing each noise type's PSNR independently. This is the most informative subplot — you can see Gaussian performing best, Rician close behind, and S&P potentially lagging. The curves for noise types introduced later (e.g., S&P in Phase 3) start from lower values and catch up.

**Subplot 5 — SSIM per Noise Type:** Same as above but for SSIM. Complements the PSNR view — a noise type might have high PSNR but low SSIM, indicating pixel-level accuracy but structural distortion.

**Subplot 6 — Learning Rate:** Shows the warmup+cosine decay schedule. Confirms that the scheduler is working correctly.

**Red dashed vertical lines** mark curriculum phase transitions at epochs 30 and 60, making it easy to see how phase changes affect all metrics.

---

## Cell 22 — Step 11 Heading

**Type:** Markdown

Section heading: **"Step 11: Comprehensive Evaluation — All Noise Types & Levels"**

---

## Cell 23 — Step 11: Comprehensive Evaluation

**Type:** Code (67 lines)

### What It Does

Evaluates the best model across 12 noise configurations at varying severity levels.

**12 evaluation configurations:**

| Config | Noise Type | Level | Rationale |
|--------|-----------|-------|-----------|
| Gaussian 0.05 | Gaussian | σ=0.05 | Mild noise |
| Gaussian 0.10 | Gaussian | σ=0.10 | Moderate noise |
| Gaussian 0.15 | Gaussian | σ=0.15 | Heavy noise |
| Gaussian 0.20 | Gaussian | σ=0.20 | Maximum training noise |
| Rician 0.05 | Rician | σ=0.05 | MRI-specific, mild |
| Rician 0.10 | Rician | σ=0.10 | MRI-specific, moderate |
| S&P 1% | Salt & Pepper | p=0.01 | Sparse impulse noise |
| S&P 3% | Salt & Pepper | p=0.03 | Dense impulse noise |
| S&P 5% | Salt & Pepper | p=0.05 | Very dense impulse noise |
| Poisson 30 | Poisson | scale=30 | Low photon count (noisier) |
| Poisson 50 | Poisson | scale=50 | Higher photon count (cleaner) |
| Mixed (G+SP) | Mixed | σ=0.05, p=0.01 | Combined degradation |

**Evaluation method:** For each configuration, the cell applies the specific noise to clean validation images, runs the model with the correct noise type conditioning, and computes PSNR and SSIM using the CPU-based skimage functions (more accurate than GPU approximations for final reporting).

**Output table** shows: Noise Type | PSNR Noisy | PSNR Denoised | SSIM | Delta PSNR. The Delta PSNR (improvement) is the key metric — it tells you how much the model actually helped. A delta of +10 dB means the model improved PSNR by 10 dB, which is a very significant improvement.

**The `all_results` list** stores all evaluation results as dictionaries, which are used by the Metrics Charts cell (Step 13).

---

## Cell 24 — Step 12 Heading

**Type:** Markdown

Section heading: **"Step 12: Visual Comparison Grid"**

---

## Cell 25 — Step 12: Visual Comparison Grid

**Type:** Code (45 lines)

### What It Does

Creates a 5×3 grid showing noisy input, denoised output, and clean ground truth for each of the 5 noise types.

**Layout:** Each row is one noise type. Each column is: Noisy Input | Denoised (with PSNR/SSIM overlay) | Clean Ground Truth.

**Implementation:** Takes one clean validation image, applies each noise type at standard levels (Gaussian 0.10, Rician 0.08, S&P 2%, Poisson 30, Mixed 0.05), runs the model, and visualizes the results. The PSNR/SSIM values are overlaid on the denoised image in green text with a black background for readability.

**Diagnostic value:** This visual comparison is essential because metrics alone don't tell the full story. A model might achieve 35 dB PSNR but still produce visually noticeable artifacts (e.g., checkerboard patterns from transposed convolutions). The side-by-side comparison reveals such issues that metrics miss.

---

## Cell 26 — Step 13 Heading

**Type:** Markdown

Section heading: **"Step 13: Metrics Charts"**

---

## Cell 27 — Step 13: Metrics Charts

**Type:** Code (45 lines)

### What It Does

Creates three analysis charts from the evaluation results.

**Chart 1 — Gaussian PSNR across sigma:** A line chart with two lines — "Noisy" (decreasing, because higher sigma = more noise = lower PSNR) and "Denoised" (higher, because the model removes noise). The vertical gap between the lines represents the model's contribution. Ideally, the gap should be roughly constant across all sigma values, indicating the model handles varying noise levels equally well.

**Chart 2 — Delta PSNR by noise type:** A bar chart showing the average PSNR improvement for each noise type. This quickly reveals which noise types the model excels at and which need improvement. Typically, Gaussian shows the largest improvement (because it's the simplest and was trained first), while S&P shows the smallest (because impulse noise is fundamentally different from additive noise).

**Chart 3 — SSIM by noise type:** A bar chart showing the average SSIM after denoising. Higher SSIM means better structural preservation. This complements the PSNR chart — a noise type might show moderate PSNR improvement but high SSIM, meaning the model preserves structure well even if some pixel-level noise remains.

---

## Cell 28 — Step 14 Heading

**Type:** Markdown

Section heading: **"Step 14: Save Final Model"**

---

## Cell 29 — Step 14: Save Final Model

**Type:** Code (28 lines)

### What It Does

Saves the final model, training history, and evaluation results.

**What's saved in `final_model.pth`:**
- `model_state_dict`: The trained model weights — all convolutional filters, normalization parameters, SE attention weights, noise embeddings, etc.
- `config`: The complete CONFIG dictionary — needed to recreate the model architecture when loading for inference
- `test_results`: The `all_results` list from Step 11's comprehensive evaluation
- `training_history`: The complete `history` dictionary with all per-epoch metrics

**Also saves `training_history.json`:** A JSON version of the history for analysis without PyTorch. Useful for plotting training curves in a separate analysis notebook or script.

**Note on `best_model.pth` vs. `final_model.pth`:** `best_model.pth` was saved during training whenever PSNR improved (contains optimizer state for resuming training). `final_model.pth` is saved after all evaluation and contains test results but not optimizer state (intended for inference only).

---

## Cell 30 — Step 15 Heading

**Type:** Markdown

Section heading: **"Step 15: Denoise Your Own Images"**

---

## Cell 31 — Step 15: Load Model & Denoise Function

**Type:** Code (53 lines)

### What It Does

Two things: (1) loads the trained model weights from the Kaggle dataset, and (2) defines a `denoise_image()` convenience function.

### Model Loading

```python
MODEL_PATH = "/kaggle/input/datasets/camorednex/brain-mri-best-model/best_model_medDenoise.pth"
state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(state if "model_state_dict" not in state else state["model_state_dict"])
model.eval()
```

**Two-format handling:** The checkpoint might be either:
- A full state dict (saved directly with `torch.save(model.state_dict(), ...)`)
- A dictionary with `model_state_dict` key (saved as part of a training checkpoint)

The ternary expression handles both cases. `weights_only=False` is required for PyTorch 2.6+ because the checkpoint may contain numpy arrays and other non-tensor objects.

`model.eval()` switches the model to evaluation mode, which disables dropout and switches GroupNorm to use running statistics.

### denoise_image Function

```python
def denoise_image(model, image_path, noise_type='gaussian', image_size=256, device=DEVICE):
```

**Step-by-step operation:**
1. Opens the image and converts to grayscale (`"L"` mode). MRI scans are inherently single-channel.
2. Resizes to 256×256 and converts to tensor. The model expects this specific input size.
3. Creates noise type index tensor for conditioning.
4. Runs inference with `torch.no_grad()` (no gradient computation).
5. Converts output back to PIL Image and resizes to original dimensions using bilinear interpolation.
6. Displays a 3-panel comparison: Original | Denoised | Difference (5× amplified). The difference map uses a "hot" colormap where bright areas indicate where the model made the largest changes. The 5× amplification makes subtle changes visible.

**Limitation:** The user must specify the correct noise type. If the actual noise is Rician but the user specifies Gaussian, the model's conditioning will be wrong and the denoising quality will be poor.

---

## Cell 32 — Step 15: Upload Widget for Interactive Denoising

**Type:** Code (51 lines)

### What It Does

Creates an interactive upload widget for denoising user-provided images directly in the Kaggle notebook.

### Why This Cell Exists

Kaggle doesn't support Google Colab's `files.upload()` function. The alternative is `ipywidgets.FileUpload`, which provides a file upload button inside the notebook. However, Kaggle's widget rendering has quirks — `plt.show()` doesn't render inside widget output areas, so plots must be saved as PNG files and displayed using `IPImage`.

### Widget Components

```python
upload = widgets.FileUpload(accept='image/*', multiple=True, description='Upload MRI/CT Scans')
btn = widgets.Button(description='Denoise', button_style='success')
out = widgets.Output()
```

- **FileUpload widget:** Accepts any image type (`accept='image/*'`), supports multiple file uploads (`multiple=True`). When the user selects files, the widget stores them as a list of dictionaries with `name` and `content` (bytes) keys.
- **Button widget:** Triggers the denoising process. `button_style='success'` renders it in green.
- **Output widget:** A dedicated output area where results are displayed. `clear_output(wait=True)` clears previous results when new images are processed.

### Denoising Logic

```python
def on_btn(b):
    with out:
        clear_output(wait=True)
        for f in upload.value:
            img = Image.open(io.BytesIO(f['content'])).convert("L")
            input_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
            
            # Try all 5 noise types, show ALL results
            input_np = input_tensor.cpu().squeeze().numpy()
            results = []
            for i in range(5):
                with torch.no_grad():
                    d = model(input_tensor, torch.tensor([i], device=DEVICE)).cpu().squeeze().numpy()
                results.append(d.clip(0,1))
```

**Why all 5 noise types?** Earlier versions tried to "auto-detect" the best noise type by picking the one with the least change (assuming the model would make the most change when the noise type matches). This was fundamentally flawed — the model might make the least change when the conditioning is wrong (because it does nothing). Showing all 5 results side by side lets the user visually compare and choose the best one.

**2×3 grid layout:** Original + 5 denoised versions (Gaussian, Rician, S&P, Poisson, Mixed). This gives the user maximum information to judge which noise type conditioning produces the best result.

**Save and display workaround:**
```python
save_path = f"/kaggle/working/{filename.rsplit('.',1)[0]}_result.png"
plt.savefig(save_path, dpi=100, bbox_inches='tight')
plt.close()
display(IPImage(filename=save_path))
```

Kaggle's widget output area doesn't render matplotlib figures with `plt.show()`. The workaround is to save the figure as a PNG file and display it using `IPython.display.Image`. This adds a small I/O overhead but produces reliable results.

---

## Cell 33 — Step 16 Heading

**Type:** Markdown

Section heading: **"Step 16: Verify Saved Outputs"**

---

## Cell 34 — Step 16: Verify Outputs

**Type:** Code (17 lines)

### What It Does

Lists all files in `/kaggle/working/` with their sizes, providing a final verification that all outputs were saved correctly.

**Typical outputs:**
- `best_model.pth` (~50-100 MB): The best model checkpoint based on PSNR
- `final_model.pth`: Post-evaluation model save with test results
- `training_history.json`: JSON export of all per-epoch metrics
- `training_history.png`: The 6-subplot training curves figure
- `denoising_results_all_noise.png`: The 5×3 visual comparison grid
- `metrics_charts.png`: The 3-panel metrics analysis chart
- `checkpoint_epoch_*.pth`: Periodic backup checkpoints (every 10 epochs)
- `*_result.png`: Denoising results from the upload widget

The cell also prints instructions for downloading outputs from Kaggle (via the Output tab or the right panel).

---

## Big Picture — How Everything Connects

```
Step 1: Install dependencies + wandb login + GPU check
   │
   ▼
Step 2: Import all libraries + define CONFIG (controls EVERYTHING)
   │
   ▼
Step 3: Load brain MRI dataset (7,200 images, auto-detect path)
   │
   ▼
Step 4: Define 5 noise functions + median filter for S&P
   │
   ▼
Step 5: Create Dataset classes (with curriculum noise + noise type labels)
   │       + DataLoaders (5 per-type val loaders + combined)
   │
   ▼
Step 6: Build ResUNetDenoiser (SE attention + noise conditioning + residual learning)
   │       ~8M parameters
   │
   ▼
Step 7: Define loss = 0.4×Charbonnier + 0.15×SSIM + 0.3×L1 + 0.15×Edge
   │
   ▼
Step 8: Define GPU metrics (PSNR/SSIM) + per-noise-type evaluation function
   │
   ▼
Step 9: TRAIN for 100 epochs with curriculum (G→G+R+P→All5)
   │       Safety nets: NaN skip, loss cap, grad clip, early stop=999
   │       Logging: wandb + periodic checkpoints
   │       Result: PSNR ~35.67 dB, SSIM ~0.9045
   │
   ▼
Step 10: Plot training curves (loss, PSNR, SSIM, per-type, LR)
   │
   ▼
Step 11: Comprehensive evaluation across 12 noise configurations
   │
   ▼
Step 12: Visual comparison grid (5 noise types × 3 columns)
   │
   ▼
Step 13: Metrics charts (Gaussian PSNR curve, delta PSNR bars, SSIM bars)
   │
   ▼
Step 14: Save final model + training history
   │
   ▼
Step 15: Load best model + denoise function + upload widget
   │       Shows all 5 noise type results side by side
   │
   ▼
Step 16: Verify all saved outputs
```

### The Six Core Innovations

1. **Noise Conditioning** — The model receives a learned embedding of the noise type, so it can activate noise-type-specific denoising strategies. Without this, the model must infer the noise type from the image alone, which is ambiguous and leads to suboptimal results.

2. **Residual Learning** — The model learns to predict the NOISE rather than the clean image. Since noise is simpler and more predictable than image structure, this makes the learning problem much easier. The denoised image is recovered by subtraction: `clean = noisy - noise_estimate`.

3. **Curriculum Training** — Noise types are introduced progressively (Gaussian → +Rician+Poisson → +S&P+Mixed). Starting with the simplest noise builds a solid feature extraction foundation that transfers to harder noise types.

4. **SE Channel Attention** — Squeeze-and-Excitation blocks learn which feature channels to emphasize for each input. Different noise types require different channel weightings — SE attention provides this adaptability with minimal parameter overhead.

5. **Per-Noise-Type Evaluation** — Separate PSNR/SSIM tracking for each noise type reveals exactly where the model excels and where it struggles. This enables targeted debugging (e.g., if S&P PSNR is low, you know to investigate the median filter or increase S&P training).

6. **Rebalanced Loss** — Charbonnier (robust to S&P outliers) replaces MSE as the primary component. SSIM preserves structural quality. Edge loss protects tumor boundaries. The combined loss prevents the blurry outputs that pure MSE produces.

### Training Safety Net Summary

| Safety Net | Threshold | Purpose |
|-----------|-----------|---------|
| SSIM weight reduction | When SSIM loss becomes NaN | Prevents training crash from numerical instability |
| AMP disabled | Always (use_amp=False) | Prevents float16 overflow/underflow in loss computation |
| Gradient clipping | max_norm=0.5 | Prevents gradient explosion during curriculum transitions |
| Loss capping | 0.5 | Skips anomalous batches that would destabilize training |
| Early stopping patience | 999 | Prevents premature stopping during curriculum phase changes |
| NaN/Inf detection | Every batch | Skips corrupted batches, tracks frequency |

### Key Model Specifications

| Property | Value |
|----------|-------|
| Architecture | ResUNet + SE Attention + Noise Conditioning |
| Parameters | ~8 million |
| Input | (B, 1, 256, 256) grayscale image + noise type index |
| Output | (B, 1, 256, 256) denoised image |
| Feature channels | [64, 128, 256, 512] + bottleneck 1024 |
| Normalization | GroupNorm (8 groups) |
| Dropout | 5% |
| Residual learning | Yes (denoised = input - noise_estimate) |
| Noise conditioning | 16-dim embedding, spatially expanded |
| Training result | PSNR ~35.67 dB, SSIM ~0.9045 |
