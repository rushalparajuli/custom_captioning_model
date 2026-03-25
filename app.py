from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import torch
from torchvision import transforms
from PIL import Image
from transformers import GPT2Tokenizer

import io
import base64
import logging

from cnn_encoder_decoder_captioning import CNNEncoderDecoderCaptioning

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load model once at startup (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Path to best_model.pt in project checkpoints directory
CHECKPOINT_PATH = Path(__file__).resolve().parent / "checkpoints" / "best_model.pt"

# Same image preprocessing as training/inference (EfficientNet / ImageNet, 224x224)
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

try:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    config = ckpt.get("config")
    if not config:
        raise ValueError(
            "Checkpoint has no 'config'. Use a checkpoint saved by the training script."
        )
    model = CNNEncoderDecoderCaptioning(**config)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    logger.info("Model loaded successfully from checkpoints/best_model.pt")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

class ImageData(BaseModel):
    image_base64: str
    caption_mode: str = "consistent"

    @validator('image_base64')
    def validate_base64(cls, v):
        if not v:
            raise ValueError("image_base64 cannot be empty")
        
        # Check if it has data URL prefix
        if ',' not in v:
            raise ValueError("Invalid base64 format - missing data URL prefix")
        
        # Basic length check (prevent huge uploads)
        if len(v) > 10_000_000:  # ~10MB limit
            raise ValueError("Image too large (max 10MB)")
        
        return v

    @validator('caption_mode')
    def validate_caption_mode(cls, v):
        allowed_modes = {
            "consistent",
            "safe_diverse",
            "balanced_diverse",
            "creative_diverse",
        }
        if v not in allowed_modes:
            raise ValueError(f"caption_mode must be one of: {', '.join(sorted(allowed_modes))}")
        return v

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": str(device),
        "model_loaded": model is not None
    }

@app.post("/caption")
async def generate_caption(data: ImageData):
    try:
        # Decode base64 image
        try:
            # Split on first comma to handle data:image/jpeg;base64,... format
            base64_data = data.image_base64.split(",", 1)[1]
            image_bytes = base64.b64decode(base64_data)
        except (IndexError, base64.binascii.Error) as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid base64 image data: {str(e)}"
            )

        # Open and validate image
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image format: {str(e)}"
            )

        # Generate caption
        try:
            # Preprocess image (same as training/inference: 224x224, ImageNet normalize)
            pixel_values = image_transform(image).unsqueeze(0).to(device)

            bos_id = getattr(tokenizer, "bos_token_id", 1)
            eos_id = getattr(tokenizer, "eos_token_id", 2)

            if data.caption_mode == "consistent":
                output_ids = model.generate_beam(
                    pixel_values,
                    start_token_id=bos_id,
                    end_token_id=eos_id,
                    max_length=16,
                    beam_width=5,
                )
            elif data.caption_mode == "safe_diverse":
                output_ids = model.generate(
                    pixel_values,
                    start_token_id=bos_id,
                    end_token_id=eos_id,
                    max_length=16,
                    temperature=0.55,
                    top_k=20,
                )
            elif data.caption_mode == "balanced_diverse":
                output_ids = model.generate(
                    pixel_values,
                    start_token_id=bos_id,
                    end_token_id=eos_id,
                    max_length=18,
                    temperature=0.8,
                    top_k=40,
                )
            else:  # creative_diverse
                output_ids = model.generate(
                    pixel_values,
                    start_token_id=bos_id,
                    end_token_id=eos_id,
                    max_length=22,
                    temperature=1.05,
                    top_k=80,
                )
            # Trim at first EOS so we never show tokens after the caption (only if we have content before EOS)
            seq = output_ids[0]
            eos_pos = (seq == eos_id).nonzero(as_tuple=True)[0]
            if eos_pos.numel() > 0 and eos_pos[0].item() > 1:
                seq = seq[: eos_pos[0].item() + 1]
            caption = tokenizer.decode(seq, skip_special_tokens=True)
            if not caption.strip():
                caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            # Restrict to a single sentence: stop after first full stop (.)
            idx = caption.find(".")
            if idx != -1:
                caption = caption[: idx + 1].strip()
            caption = caption.strip()
            logger.info(f"Generated caption: {caption}")
            return {"caption": caption}
            
        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA out of memory")
            raise HTTPException(
                status_code=503,
                detail="Server overloaded - try again in a moment"
            )
        except Exception as e:
            logger.error(f"Model inference error: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to generate caption"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Internal server error"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)