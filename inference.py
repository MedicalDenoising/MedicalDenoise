"""
Inference Script — Denoise a single image or directory of medical scans
Usage:
    python inference.py --input scan.png --checkpoint checkpoints/checkpoint_best.pt
    python inference.py --input scans/ --checkpoint checkpoints/checkpoint_best.pt --output denoised/
"""

import argparse
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from models.autoencoder import get_model
from utils.metrics import compute_psnr, compute_ssim


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load model from checkpoint"""
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get('config', {})

    model = get_model(
        config.get('model_type', 'unet'),
        in_channels=config.get('in_channels', 1),
        out_channels=config.get('in_channels', 1),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"[Inference] Model loaded from {checkpoint_path}")
    return model


def denoise_image(model: torch.nn.Module,
                  image_path: str,
                  output_path: str,
                  image_size: int = 256,
                  device: torch.device = torch.device('cpu')) -> dict:
    """
    Denoise a single image and save result.
    
    Returns:
        dict with PSNR and SSIM metrics (if ground truth available)
    """
    # Load and preprocess
    image = Image.open(image_path).convert('L')
    original_size = image.size  # (W, H)

    transform = T.Compose([
        T.Resize((image_size, image_size), antialias=True),
        T.ToTensor(),
    ])
    noisy_tensor = transform(image).unsqueeze(0).to(device)  # [1, 1, H, W]

    # Inference
    with torch.no_grad():
        denoised_tensor = model(noisy_tensor)

    # Convert back to PIL image
    denoised_np = denoised_tensor.squeeze().cpu().numpy()
    denoised_np = (denoised_np * 255).clip(0, 255).astype('uint8')
    denoised_pil = Image.fromarray(denoised_np, mode='L')
    denoised_pil = denoised_pil.resize(original_size, Image.LANCZOS)

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    denoised_pil.save(output_path)
    print(f"[Inference] Saved denoised image: {output_path}")

    return {}


def main():
    parser = argparse.ArgumentParser(description='Medical Image Denoising Inference')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image or directory path')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--output', type=str, default='denoised',
                        help='Output directory for denoised images')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Processing resolution')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.checkpoint, device)

    input_path = Path(args.input)
    output_dir = Path(args.output)

    EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}

    if input_path.is_file():
        paths = [input_path]
    elif input_path.is_dir():
        paths = [p for p in input_path.iterdir() if p.suffix.lower() in EXTENSIONS]
        print(f"[Inference] Found {len(paths)} images in {input_path}")
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")

    for i, img_path in enumerate(paths):
        out_path = output_dir / f"denoised_{img_path.name}"
        denoise_image(model, img_path, out_path, args.image_size, device)
        print(f"  [{i+1}/{len(paths)}] {img_path.name} → {out_path.name}")

    print(f"\n[Done] Denoised {len(paths)} images → {output_dir}")


if __name__ == '__main__':
    main()
