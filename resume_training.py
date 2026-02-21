"""
Resume training from a saved checkpoint (e.g. best_model.pt).
Loads model, optimizer state, and continues training from the next epoch.

Usage:
  python resume_training.py
  python resume_training.py --checkpoint checkpoints/best_model.pt --epochs 30
  python resume_training.py --unfreeze_cnn   # fine-tune EfficientNet B1 (unfreeze backbone)
"""

import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from train_captioning_model import (
    Trainer,
    ImageCaptionDataset,
    KarpathyImageCaptionDataset,
    get_coco2017_paths,
    KAGGLE_INPUT,
    KAGGLE_WORKING,
)
from karpathy_split import load_karpathy_split, build_coco2017_id_to_path_and_refs
from cnn_encoder_decoder_captioning import CNNEncoderDecoderCaptioning


def load_checkpoint_for_resume(checkpoint_path, device=None, dropout_override=None):
    """
    Load checkpoint and return model, optimizer, epoch, val_loss, config.
    Config is the model config (for building model); training config (lr, batch_size, etc.)
    must be passed via args. If dropout_override is set, use it instead of checkpoint dropout.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt.get("config")
    if not config:
        raise ValueError("Checkpoint has no 'config'. Use a checkpoint saved by the training script.")
    dropout = dropout_override if dropout_override is not None else config["dropout"]
    config["dropout"] = dropout

    model = CNNEncoderDecoderCaptioning(
        vocab_size=config["vocab_size"],
        max_seq_len=config["max_seq_len"],
        embed_dim=config["embed_dim"],
        encoder_depth=config["encoder_depth"],
        decoder_depth=config["decoder_depth"],
        num_heads=config["num_heads"],
        dropout=dropout,
        pretrained_cnn=config.get("pretrained_cnn", True),
        freeze_cnn_backbone=config.get("freeze_cnn_backbone", True),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device)

    epoch = ckpt.get("epoch", 0)
    val_loss = ckpt.get("val_loss", float("inf"))
    optimizer_state = ckpt.get("optimizer_state_dict")

    return model, optimizer_state, epoch, val_loss, config


def main():
    project_dir = Path(__file__).resolve().parent
    default_checkpoint = project_dir / "checkpoints" / "best_model.pt"
    default_data_root = str(project_dir / "coco2017")

    parser = argparse.ArgumentParser(description="Resume captioning training from a checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(default_checkpoint),
        help="Path to checkpoint (e.g. checkpoints/best_model.pt)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Root of COCO 2017 (train2017/, val2017/, annotations/). Default: project/coco2017 or Kaggle input when --kaggle.",
    )
    parser.add_argument(
        "--kaggle",
        action="store_true",
        help="Use Kaggle paths for data and save_dir.",
    )
    parser.add_argument(
        "--input_dataset",
        type=str,
        default="coco-2017-dataset",
        help="Kaggle input dataset name (when --kaggle).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Checkpoint output directory (default: same as checkpoint dir or /kaggle/working/checkpoints when --kaggle).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Total number of epochs to train (training will run from resumed epoch + 1 up to this).",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--unfreeze_cnn",
        action="store_true",
        help="Fine-tune EfficientNet B1 backbone (unfreeze); default is to keep it frozen as in the checkpoint.",
    )
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--dummy", action="store_true", help="Use dummy data (no COCO).")
    parser.add_argument("--karpathy", action="store_true", help="Use Karpathy train/val split (COCO 2017). Requires --karpathy_split.")
    parser.add_argument("--karpathy_split", type=str, default=None, help="Path to Karpathy split JSON (e.g. dataset_coco.json). Required when --karpathy.")
    parser.add_argument("--dropout", type=float, default=None, help="Override dropout from checkpoint (e.g. 0.2 for stronger regularization).")
    args = parser.parse_args()

    # Paths
    if args.kaggle:
        data_root = args.data_root or str(KAGGLE_INPUT / args.input_dataset)
        save_dir = args.save_dir or str(KAGGLE_WORKING / "checkpoints")
        if args.workers == 4:
            args.workers = 2
    else:
        data_root = args.data_root or default_data_root
        if args.save_dir:
            save_dir = args.save_dir
        else:
            save_dir = str(Path(args.checkpoint).resolve().parent)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint: {args.checkpoint}")
    model, optimizer_state, start_epoch, best_val_loss, model_config = load_checkpoint_for_resume(
        args.checkpoint, device, dropout_override=args.dropout
    )
    if args.dropout is not None:
        print(f"Dropout overridden to {args.dropout}")
    if args.unfreeze_cnn:
        for param in model.encoder.backbone.features.parameters():
            param.requires_grad = True
        model_config["freeze_cnn_backbone"] = False
        print("EfficientNet B1 backbone unfrozen (fine-tuning enabled).")
    start_epoch += 1  # next epoch to run
    print(f"Resuming from epoch {start_epoch} (best val loss so far: {best_val_loss:.4f})")
    if start_epoch >= args.epochs:
        print(f"Checkpoint already at epoch {start_epoch}; --epochs is {args.epochs}. Nothing to do.")
        return

    # Tokenizer
    try:
        from transformers import GPT2Tokenizer
    except ImportError:
        raise ImportError("pip install transformers")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    max_length = model_config["max_seq_len"]

    # Datasets and loaders
    if args.dummy or not Path(data_root).exists():
        if not args.dummy:
            print("COCO path not found; using dummy data. Use --dummy to suppress.")
        pad_id = getattr(tokenizer, "pad_token_id", getattr(tokenizer, "eos_token_id", 0))
        bos_id = getattr(tokenizer, "bos_token_id", 1)
        eos_id = getattr(tokenizer, "eos_token_id", 2)

        class DummyDataset(torch.utils.data.Dataset):
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

        train_dataset = DummyDataset(
            50000, max_length, model_config["vocab_size"], pad_id, bos_id, eos_id
        )
        val_dataset = DummyDataset(
            5000, max_length, model_config["vocab_size"], pad_id, bos_id, eos_id
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
            data_root, id_to_path, train_annotations, tokenizer, max_length=max_length, use_augmentation=True
        )
        val_dataset = KarpathyImageCaptionDataset(
            data_root, id_to_path, val_annotations, tokenizer, max_length=max_length, use_augmentation=False
        )
    else:
        train_img, train_ann, val_img, val_ann = get_coco2017_paths(data_root)
        train_dataset = ImageCaptionDataset(
            train_img, train_ann, tokenizer, max_length=max_length
        )
        val_dataset = ImageCaptionDataset(
            val_img, val_ann, tokenizer, max_length=max_length
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    # Optimizer (restore state if present; re-init when backbone unfrozen)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=model_config.get("weight_decay", 0.01),
    )
    if optimizer_state is not None and not args.unfreeze_cnn:
        try:
            optimizer.load_state_dict(optimizer_state)
            print("Optimizer state restored from checkpoint.")
        except Exception as e:
            print(f"Could not load optimizer state: {e}. Using fresh optimizer.")
    elif args.unfreeze_cnn:
        print("Optimizer re-initialized (backbone parameters added).")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        max_epochs=args.epochs,
        save_dir=save_dir,
        tokenizer=tokenizer,
        model_config=model_config,
    )
    # Set resume state (works even if Trainer.__init__ doesn't accept these kwargs)
    trainer.start_epoch = start_epoch
    trainer.best_val_loss = best_val_loss

    # Run training loop from start_epoch (in case Trainer.train() doesn't support resume)
    print(f"Training on {trainer.device}")
    print(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    print(f"Resuming from epoch {trainer.start_epoch + 1} (best val loss so far: {trainer.best_val_loss:.4f})")
    for epoch in range(trainer.start_epoch, trainer.max_epochs):
        print(f"\nEpoch {epoch + 1}/{trainer.max_epochs}")
        print("-" * 70)
        train_loss = trainer.train_epoch()
        print(f"Train Loss: {train_loss:.4f}")
        val_loss = trainer.validate()
        print(f"Val Loss: {val_loss:.4f}")
        if val_loss < trainer.best_val_loss:
            trainer.best_val_loss = val_loss
            trainer.save_checkpoint(epoch, val_loss, is_best=True)
            print(f"✓ New best model saved! (loss: {val_loss:.4f})")
        if (epoch + 1) % 5 == 0:
            trainer.save_checkpoint(epoch, val_loss)
    trainer.generate_sample_captions()


if __name__ == "__main__":
    main()
