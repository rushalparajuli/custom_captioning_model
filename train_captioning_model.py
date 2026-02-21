"""
Training Script for Image Captioning: CNN (EfficientNet B1) + Encoder + Decoder
Uses COCO 2017 dataset. CNN backbone: pre-trained EfficientNet B1.

Place COCO 2017 in this project directory (default --data_root is project_dir/coco2017):
  <project_dir>/
    coco2017/
      train2017/          # training images
      val2017/            # validation images
      annotations/
        captions_train2017.json
        captions_val2017.json
    checkpoints/          # best_model.pt and epoch checkpoints (created by training)

Run: python train_captioning_model.py   # uses ./coco2017 in project dir
Inference: python inference.py --checkpoint checkpoints/best_model.pt --image path/to.jpg

Kaggle: Add COCO 2017 dataset to notebook, then:
  python train_captioning_model.py --kaggle
  (Uses /kaggle/input/coco-2017-dataset, saves to /kaggle/working/checkpoints, smaller model by default.)
  If your dataset folder name differs: --kaggle --input_dataset YOUR_DATASET_SLUG

Requirements: torch, torchvision, transformers (for GPT2Tokenizer).
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
from pathlib import Path
from tqdm import tqdm
import argparse

# Kaggle: input dir is read-only, output goes to /kaggle/working/
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")

# CNN + Encoder + Decoder model (EfficientNet B1 backbone)
from cnn_encoder_decoder_captioning import CNNEncoderDecoderCaptioning
from karpathy_split import load_karpathy_split, build_coco2017_id_to_path_and_refs


class ImageCaptionDataset(Dataset):
    """
    Dataset for image captioning
    Expects data in COCO-style format:
    {
        "images": [{"id": 1, "file_name": "image1.jpg"}, ...],
        "annotations": [{"image_id": 1, "caption": "A dog playing"}, ...]
    }
    """
    def __init__(
        self, 
        image_dir, 
        annotation_file, 
        tokenizer,
        transform=None,
        max_length=77
    ):
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform or self.get_default_transform()
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        # Create image_id to filename mapping
        self.id_to_filename = {
            img['id']: img['file_name'] 
            for img in data['images']
        }
        
        # Store annotations (image_id, caption pairs)
        self.annotations = [
            (ann['image_id'], ann['caption'])
            for ann in data['annotations']
        ]
        self.pad_token_id = getattr(tokenizer, 'pad_token_id', getattr(tokenizer, 'eos_token_id', 0))

    def get_default_transform(self):
        """Image preprocessing for EfficientNet (ImageNet stats, 224x224)."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        image_id, caption = self.annotations[idx]
        
        # Load image
        img_path = self.image_dir / self.id_to_filename[image_id]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Tokenize caption (tokenizer.encode returns list of ids)
        if hasattr(self.tokenizer, 'encode'):
            tokens = self.tokenizer.encode(
                caption,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length - 2,
            )
        else:
            tokens = self.tokenizer.encode(caption)
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        pad_id = getattr(self.tokenizer, 'pad_token_id', self.tokenizer.eos_token_id)
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [pad_id] * (self.max_length - len(tokens))
        
        tokens = torch.tensor(tokens, dtype=torch.long)
        
        return image, tokens


class KarpathyImageCaptionDataset(Dataset):
    """
    Dataset for Karpathy split: (image_id, caption) pairs with paths from COCO 2017.
    id_to_path[image_id] = (subdir, filename) e.g. ("train2017", "000000000001.jpg").
    use_augmentation=True adds mild train-time augmentation (e.g. random horizontal flip).
    """

    def __init__(
        self,
        data_root,
        id_to_path,
        annotations,
        tokenizer,
        transform=None,
        max_length=77,
        use_augmentation=False,
    ):
        self.data_root = Path(data_root)
        self.id_to_path = id_to_path
        self.annotations = annotations  # list of (image_id, caption)
        self.tokenizer = tokenizer
        self.max_length = max_length
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self.get_transform_with_augmentation() if use_augmentation else self.get_default_transform()
        self.pad_token_id = getattr(tokenizer, "pad_token_id", getattr(tokenizer, "eos_token_id", 0))

    def get_default_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_transform_with_augmentation(self):
        """Mild augmentation: random resized crop, horizontal flip, small rotation, then normalize."""
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_id, caption = self.annotations[idx]
        subdir, filename = self.id_to_path[image_id]
        img_path = self.data_root / subdir / filename
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        if hasattr(self.tokenizer, "encode"):
            tokens = self.tokenizer.encode(
                caption,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length - 2,
            )
        else:
            tokens = self.tokenizer.encode(caption)
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        pad_id = self.pad_token_id
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]
        else:
            tokens = tokens + [pad_id] * (self.max_length - len(tokens))
        tokens = torch.tensor(tokens, dtype=torch.long)
        return image, tokens


class Trainer:
    """Training loop for image captioning model"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        max_epochs=10,
        save_dir='checkpoints',
        tokenizer=None,
        model_config=None,
        start_epoch=0,
        initial_best_val_loss=None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.max_epochs = max_epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.tokenizer = tokenizer
        self.model_config = model_config or {}
        self.start_epoch = start_epoch
        self.best_val_loss = float('inf') if initial_best_val_loss is None else initial_best_val_loss
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, captions in pbar:
            images = images.to(self.device)
            captions = captions.to(self.device)
            
            # Forward pass
            # Input is all tokens except last, target is all tokens except first
            logits = self.model(images, captions[:, :-1])
            
            # Calculate loss (ignore padding tokens)
            pad_id = getattr(self.train_loader.dataset, 'pad_token_id', 0)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                captions[:, 1:].reshape(-1),
                ignore_index=pad_id
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        for images, captions in tqdm(self.val_loader, desc='Validation'):
            images = images.to(self.device)
            captions = captions.to(self.device)
            
            logits = self.model(images, captions[:, :-1])
            pad_id = getattr(self.val_loader.dataset, 'pad_token_id', 0)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                captions[:, 1:].reshape(-1),
                ignore_index=pad_id
            )
            
            total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self):
        """Main training loop"""
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        if self.start_epoch > 0:
            print(f"Resuming from epoch {self.start_epoch + 1} (best val loss so far: {self.best_val_loss:.4f})")
        
        for epoch in range(self.start_epoch, self.max_epochs):
            print(f"\nEpoch {epoch + 1}/{self.max_epochs}")
            print("-" * 70)
            
            # Train
            train_loss = self.train_epoch()
            print(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate()
            print(f"Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, is_best=True)
                print(f"✓ New best model saved! (loss: {val_loss:.4f})")
            
            # Save regular checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, val_loss)
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint (includes config for inference)."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.model_config,
        }
        if is_best:
            path = self.save_dir / 'best_model.pt'
        else:
            path = self.save_dir / f'checkpoint_epoch_{epoch+1}.pt'
        torch.save(checkpoint, path)
    
    @torch.no_grad()
    def generate_sample_captions(self, num_samples=5):
        """Generate sample captions for visualization"""
        self.model.eval()
        
        # Get a batch from validation
        images, _ = next(iter(self.val_loader))
        images = images[:num_samples].to(self.device)
        
        bos_id = getattr(self.tokenizer, 'bos_token_id', 1) if self.tokenizer else 1
        eos_id = getattr(self.tokenizer, 'eos_token_id', 2) if self.tokenizer else 2
        generated = self.model.generate(
            images,
            start_token_id=bos_id,
            end_token_id=eos_id,
            max_length=50,
            temperature=0.7,
            top_k=50
        )
        print("\nSample Generated Captions:")
        print("-" * 70)
        for i, caption_ids in enumerate(generated):
            ids = caption_ids.cpu().tolist()
            if self.tokenizer is not None and hasattr(self.tokenizer, 'decode'):
                text = self.tokenizer.decode(ids, skip_special_tokens=True)
                print(f"Image {i+1}: {text}")
            else:
                print(f"Image {i+1}: {ids}")


def get_coco2017_paths(data_root):
    """Return (train_images, train_ann, val_images, val_ann) for COCO 2017."""
    root = Path(data_root)
    return (
        root / "train2017",
        root / "annotations" / "captions_train2017.json",
        root / "val2017",
        root / "annotations" / "captions_val2017.json",
    )


def main():
    project_dir = Path(__file__).resolve().parent
    default_data_root = str(project_dir / "coco2017")
    parser = argparse.ArgumentParser(description="Train CNN+Encoder+Decoder captioning on COCO 2017")
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Root directory of COCO 2017 (contains train2017/, val2017/, annotations/). Default: project/coco2017, or Kaggle input when --kaggle.",
    )
    parser.add_argument(
        "--kaggle",
        action="store_true",
        help="Use Kaggle paths: data from /kaggle/input/<input_dataset>, checkpoints to /kaggle/working/checkpoints. Implies smaller model unless overridden.",
    )
    parser.add_argument(
        "--input_dataset",
        type=str,
        default="coco-2017-dataset",
        help="Kaggle input dataset folder name under /kaggle/input (used when --kaggle). Default: coco-2017-dataset",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Use smaller model (fewer layers, smaller embed_dim, frozen backbone) for faster training and less memory.",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Default: 32, or 24 when --kaggle/--small")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=None, help="Default: 4, or 2 when --kaggle")
    parser.add_argument("--max_length", type=int, default=77)
    parser.add_argument("--save_dir", type=str, default=None, help="Checkpoint output directory (default: project/checkpoints or /kaggle/working/checkpoints when --kaggle)")
    parser.add_argument("--dummy", action="store_true", help="Run with dummy data (no COCO)")
    parser.add_argument("--karpathy", action="store_true", help="Use Karpathy train/val split (COCO 2017 images). Requires --karpathy_split.")
    parser.add_argument("--karpathy_split", type=str, default=None, help="Path to Karpathy split JSON (e.g. dataset_coco.json). Required when --karpathy.")
    args = parser.parse_args()

    # Resolve paths: Kaggle vs local
    if args.kaggle:
        data_root = args.data_root or str(KAGGLE_INPUT / args.input_dataset)
        save_dir = args.save_dir or str(KAGGLE_WORKING / "checkpoints")
        if args.workers is None:
            args.workers = 2
        if args.batch_size is None:
            args.batch_size = 24
        use_small = args.small or True  # default small on Kaggle
    else:
        data_root = args.data_root or default_data_root
        save_dir = args.save_dir or str(project_dir / "checkpoints")
        if args.workers is None:
            args.workers = 4
        if args.batch_size is None:
            args.batch_size = 32
        use_small = args.small

    # Hyperparameters: full or small model
    if use_small:
        config = {
            "vocab_size": 50257,
            "max_seq_len": args.max_length,
            "embed_dim": 512,
            "encoder_depth": 4,
            "decoder_depth": 4,
            "num_heads": 4,
            "dropout": 0.1,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": 0.01,
            "max_epochs": args.epochs,
            "pretrained_cnn": True,
            "freeze_cnn_backbone": True,
        }
    else:
        config = {
            "vocab_size": 50257,
            "max_seq_len": args.max_length,
            "embed_dim": 512,
            "encoder_depth": 6,
            "decoder_depth": 6,
            "num_heads": 8,
            "dropout": 0.1,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": 0.01,
            "max_epochs": args.epochs,
            "pretrained_cnn": True,
            "freeze_cnn_backbone": False,
        }

    # Tokenizer: GPT-2 (required)
    try:
        from transformers import GPT2Tokenizer
    except ImportError:
        raise ImportError("GPT2Tokenizer requires transformers. Install with: pip install transformers")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    if args.kaggle:
        print("Kaggle mode: data_root=%s, save_dir=%s, small_model=%s" % (data_root, save_dir, use_small))

    pad_token_id = getattr(tokenizer, "pad_token_id", getattr(tokenizer, "eos_token_id", 0))

    # Dataset: COCO 2017 or dummy
    if args.dummy or not Path(data_root).exists():
        if not args.dummy:
            print("COCO path not found; using dummy data. Use --dummy to suppress this.")

        class DummyDataset(Dataset):
            def __init__(self, size, max_len, vocab_size, pad_token_id, bos_id, eos_id):
                self.size = size
                self.max_length = max_len
                self.pad_token_id = pad_token_id
                self.vocab_size = vocab_size
                self.bos_id = bos_id
                self.eos_id = eos_id

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                image = torch.randn(3, 224, 224)
                caption = torch.randint(0, max(3, self.vocab_size), (self.max_length,))
                caption[0] = self.bos_id
                caption[-1] = self.eos_id
                return image, caption

        bos_id = getattr(tokenizer, "bos_token_id", 1)
        eos_id = getattr(tokenizer, "eos_token_id", 2)
        train_dataset = DummyDataset(
            50000, config["max_seq_len"], config["vocab_size"],
            pad_token_id, bos_id, eos_id
        )
        val_dataset = DummyDataset(
            5000, config["max_seq_len"], config["vocab_size"],
            pad_token_id, bos_id, eos_id
        )
    elif args.karpathy:
        if not args.karpathy_split:
            raise ValueError("--karpathy requires --karpathy_split <path to JSON>")
        train_ids, val_ids, test_ids = load_karpathy_split(args.karpathy_split)
        id_to_path, id_to_refs = build_coco2017_id_to_path_and_refs(data_root)
        available = set(id_to_path.keys())
        train_ids &= available
        val_ids &= available
        train_annotations = [(iid, cap) for iid in train_ids for cap in id_to_refs.get(iid, [])]
        val_annotations = [(iid, cap) for iid in val_ids for cap in id_to_refs.get(iid, [])]
        print("Karpathy split: train %d ids / %d pairs, val %d ids / %d pairs" % (
            len(train_ids), len(train_annotations), len(val_ids), len(val_annotations)))
        train_dataset = KarpathyImageCaptionDataset(
            data_root, id_to_path, train_annotations, tokenizer,
            max_length=config["max_seq_len"],
        )
        val_dataset = KarpathyImageCaptionDataset(
            data_root, id_to_path, val_annotations, tokenizer,
            max_length=config["max_seq_len"],
        )
    else:
        train_img, train_ann, val_img, val_ann = get_coco2017_paths(data_root)
        train_dataset = ImageCaptionDataset(
            train_img, train_ann, tokenizer, max_length=config["max_seq_len"]
        )
        val_dataset = ImageCaptionDataset(
            val_img, val_ann, tokenizer, max_length=config["max_seq_len"]
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=args.workers,
    )

    # Model: CNN (EfficientNet B1) + Encoder + Decoder
    model = CNNEncoderDecoderCaptioning(
        vocab_size=config["vocab_size"],
        max_seq_len=config["max_seq_len"],
        embed_dim=config["embed_dim"],
        encoder_depth=config["encoder_depth"],
        decoder_depth=config["decoder_depth"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        pretrained_cnn=config["pretrained_cnn"],
        freeze_cnn_backbone=config["freeze_cnn_backbone"],
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    model_config = {
        "vocab_size": config["vocab_size"],
        "max_seq_len": config["max_seq_len"],
        "embed_dim": config["embed_dim"],
        "encoder_depth": config["encoder_depth"],
        "decoder_depth": config["decoder_depth"],
        "num_heads": config["num_heads"],
        "dropout": config["dropout"],
        "pretrained_cnn": config["pretrained_cnn"],
        "freeze_cnn_backbone": config["freeze_cnn_backbone"],
    }

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        max_epochs=config["max_epochs"],
        tokenizer=tokenizer,
        save_dir=save_dir,
        model_config=model_config,
    )
    trainer.train()
    trainer.generate_sample_captions()


if __name__ == "__main__":
    main()
