# 🧠 Medical Image Denoising — U-Net Convolutional Autoencoder

> Deep learning-based denoising of MRI/CT brain scans using U-Net style skip connections.  
> Evaluated with PSNR, SSIM, and MSE metrics.

---

## 📌 Project Overview

This project implements a **Convolutional Autoencoder with U-Net skip connections** for denoising medical scans — specifically targeting **Brain MRI / CT** images related to **Brain Tumor** diagnosis.

Noise in medical images degrades diagnostic quality. This model learns to reconstruct clean images from noisy inputs using an encoder-decoder architecture with residual skip connections.

---

## 🏗️ Architecture

```
Input (Noisy MRI/CT)
        ↓
[Encoder Block 1 → 64]  ─────────────────────────────────┐
        ↓ MaxPool                                         │ skip
[Encoder Block 2 → 128] ──────────────────────────────┐  │
        ↓ MaxPool                                      │  │
[Encoder Block 3 → 256] ───────────────────────────┐  │  │
        ↓ MaxPool                                   │  │  │
[Encoder Block 4 → 512] ────────────────────────┐  │  │  │
        ↓ MaxPool                                │  │  │  │
    [Bottleneck → 1024]                          │  │  │  │
        ↓ ConvTranspose                          │  │  │  │
[Decoder Block 4] ← skip ◄───────────────────── ┘  │  │  │
        ↓ ConvTranspose                             │  │  │
[Decoder Block 3] ← skip ◄──────────────────────── ┘  │  │
        ↓ ConvTranspose                                │  │
[Decoder Block 2] ← skip ◄─────────────────────────── ┘  │
        ↓ ConvTranspose                                   │
[Decoder Block 1] ← skip ◄────────────────────────────── ┘
        ↓
 [1×1 Conv + Sigmoid]
        ↓
Output (Denoised Image)
```

**Skip connections** preserve spatial details lost during downsampling — critical for maintaining tumor boundary accuracy.

---

## 🧪 Noise Models Supported

| Noise Type | Physical Source | Level |
|-----------|-----------------|-------|
| Gaussian | MRI thermal noise | σ = 0.05–0.2 |
| Rician | MRI magnitude detection | σ = 0.05–0.15 |
| Salt & Pepper | Sensor defects / CT | prob = 1–5% |
| Poisson | Low-dose CT shot noise | scale |
| Mixed | Realistic clinical | Combined |

---

## 📊 Evaluation Metrics

| Metric | What It Measures | Better |
|--------|-----------------|--------|
| **PSNR** | Peak Signal-to-Noise Ratio (dB) | Higher |
| **SSIM** | Structural Similarity [0–1] | Higher |
| **MSE** | Mean Squared Error | Lower |
| **MAE** | Mean Absolute Error | Lower |
| **SNR** | Signal-to-Noise Ratio (dB) | Higher |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train (synthetic brain data — no dataset needed)
```bash
python train.py --use_synthetic --epochs 50 --noise_type gaussian
```

### 3. Train with real MRI images
```bash
python train.py --image_dir /path/to/mri/images --epochs 100 --noise_type rician
```

### 4. Evaluate
```bash
python train.py --evaluate
```

### 5. Denoise new images
```bash
python inference.py --input scan.png --checkpoint checkpoints/checkpoint_best.pt
```

---

## 📁 Project Structure

```
medical-denoising/
├── models/
│   ├── autoencoder.py     # U-Net autoencoder architecture
│   └── losses.py          # MSE, SSIM, Edge, Combined loss functions
├── data/
│   └── dataset.py         # Dataset, noise generators, DataLoaders
├── utils/
│   ├── metrics.py         # PSNR, SSIM, MSE, MAE, SNR
│   └── visualize.py       # Comparison grids, training curves
├── train.py               # Main training script
├── inference.py           # Denoise new images
├── requirements.txt
└── README.md
```

---

## 📈 Sample Results

| Model | Noise | Baseline PSNR | Model PSNR | SSIM Improvement |
|-------|-------|--------------|-----------|-----------------|
| U-Net AE | Gaussian σ=0.1 | ~20 dB | ~32–36 dB | +0.15–0.25 |
| U-Net AE | Rician σ=0.08 | ~22 dB | ~33–37 dB | +0.12–0.22 |
| Lightweight | Mixed | ~18 dB | ~28–32 dB | +0.10–0.18 |

---

## 🧑‍💻 Team

**Project:** Medical Image Denoising & Reconstruction  
**Domain:** Deep Learning / Medical Imaging  
**Techniques:** Convolutional Autoencoders, Skip Connections (U-Net), PSNR/SSIM Evaluation

---

## 📄 License

MIT License — free for educational use.
