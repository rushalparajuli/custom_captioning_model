"""
Inference script for the trained CNN+Encoder+Decoder captioning model.
Loads the saved checkpoint and generates captions for images.

Usage:
  python inference.py --checkpoint checkpoints/best_model.pt --image path/to/image.jpg
  python inference.py --image path/to/image.jpg   # uses checkpoints/best_model.pt
"""

import argparse
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

from cnn_encoder_decoder_captioning import CNNEncoderDecoderCaptioning


# Same preprocessing as training (EfficientNet / ImageNet)
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_model(checkpoint_path, device=None):
    """
    Load the trained captioning model from a checkpoint.
    Returns (model, tokenizer, device).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config")
    if not config:
        raise ValueError(
            "Checkpoint has no 'config'. Use a checkpoint saved by the current training script."
        )

    model = CNNEncoderDecoderCaptioning(**config)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()

    try:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    except ImportError:
        tokenizer = None

    return model, tokenizer, device


def caption_image(model, image_input, tokenizer=None, device="cpu", max_length=50, temperature=0.7, top_k=50):
    """
    Generate a caption for one or more images.

    Args:
        model: Loaded CNNEncoderDecoderCaptioning model.
        image_input: Either a path (str/Path), a PIL Image, or a tensor (B, 3, 224, 224).
        tokenizer: GPT2Tokenizer (or similar with bos_token_id, eos_token_id, decode). If None, returns token IDs.
        device: Device to run on.
        max_length: Max caption length.
        temperature: Sampling temperature.
        top_k: Top-k sampling.

    Returns:
        If single image: string (if tokenizer) or list of token IDs. If batch: list of strings or list of lists of IDs.
    """
    transform = get_transform()

    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
        img = transform(img).unsqueeze(0)
    elif isinstance(image_input, Image.Image):
        img = transform(image_input).unsqueeze(0)
    elif isinstance(image_input, torch.Tensor):
        img = image_input
        if img.dim() == 3:
            img = img.unsqueeze(0)
    else:
        raise TypeError("image_input must be path, PIL Image, or tensor")

    img = img.to(device)
    bos_id = getattr(tokenizer, "bos_token_id", 1) if tokenizer else 1
    eos_id = getattr(tokenizer, "eos_token_id", 2) if tokenizer else 2

    generated = model.generate(
        img,
        start_token_id=bos_id,
        end_token_id=eos_id,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
    )

    out = []
    for i in range(generated.size(0)):
        ids = generated[i].cpu().tolist()
        if tokenizer and hasattr(tokenizer, "decode"):
            out.append(tokenizer.decode(ids, skip_special_tokens=True))
        else:
            out.append(ids)

    return out[0] if len(out) == 1 else out


def main():
    project_dir = Path(__file__).resolve().parent
    default_checkpoint = project_dir / "checkpoints" / "best_model.pt"

    parser = argparse.ArgumentParser(description="Run captioning inference with trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(default_checkpoint),
        help="Path to checkpoint (e.g. checkpoints/best_model.pt)",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to image file",
    )
    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.checkpoint, args.device)
    caption = caption_image(
        model,
        args.image,
        tokenizer=tokenizer,
        device=device,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(caption)


if __name__ == "__main__":
    main()
