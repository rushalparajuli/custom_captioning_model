"""
Self-Critical Sequence Training (SCST) for image captioning.
Loads best_model.pt and fine-tunes by optimizing CIDEr directly (REINFORCE with beam baseline).

Usage (local):
  pip install pycocoevalcap
  python scst_captioning.py --checkpoint checkpoints/best_model.pt --data_root ./coco2017 --epochs 5

With Karpathy split (default beam_width=3):
  python scst_captioning.py --karpathy --karpathy_split /path/to/dataset_coco.json --data_root ./coco2017

Usage (Kaggle GPU):
  python scst_captioning.py --kaggle --checkpoint /kaggle/working/checkpoints/best_model.pt --epochs 3 --batch_size 16
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from cnn_encoder_decoder_captioning import CNNEncoderDecoderCaptioning
from karpathy_split import load_karpathy_split, build_coco2017_id_to_path_and_refs

# Kaggle paths
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")


def load_checkpoint(checkpoint_path, device=None):
    """Load model and config from checkpoint."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt.get("config")
    if not config:
        raise ValueError("Checkpoint has no 'config'.")
    model = CNNEncoderDecoderCaptioning(
        vocab_size=config["vocab_size"],
        max_seq_len=config["max_seq_len"],
        embed_dim=config["embed_dim"],
        encoder_depth=config["encoder_depth"],
        decoder_depth=config["decoder_depth"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        pretrained_cnn=config.get("pretrained_cnn", True),
        freeze_cnn_backbone=config.get("freeze_cnn_backbone", True),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device)
    return model, config


def sample_with_log_probs(model, images, bos_id, eos_id, max_length=50, temperature=1.0):
    """
    Sample captions and return (seq, log_prob_sum) where log_prob_sum is (B,) sum of log probs until EOS.
    """
    B = images.size(0)
    device = images.device
    encoder_output = model.encoder(images)
    seq = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    log_prob_sum = torch.zeros(B, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_length - 1):
        logits = model.decoder(seq, encoder_output)
        next_logits = logits[:, -1, :] / temperature
        log_probs = F.log_softmax(next_logits, dim=-1)
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        # log prob of the selected token (B, 1)
        log_prob_selected = log_probs.gather(1, next_token).squeeze(1)
        log_prob_sum = log_prob_sum + torch.where(finished, torch.zeros_like(log_prob_sum), log_prob_selected)
        finished = finished | (next_token.squeeze(1) == eos_id)
        seq = torch.cat([seq, next_token], dim=1)
        if finished.all():
            break
    return seq, log_prob_sum


def get_cider_per_image(gts, res):
    """gts: dict[image_id] = [ref1, ref2, ...], res: dict[image_id] = [pred]. Returns (mean_cider, list_of_scores)."""
    try:
        from pycocoevalcap.cider.cider import Cider
    except ImportError:
        raise ImportError("SCST requires pycocoevalcap. Install with: pip install pycocoevalcap")
    scorer = Cider()
    mean_score, scores = scorer.compute_score(gts, res)
    return mean_score, scores


class COCOSCSTDataset(Dataset):
    """COCO train dataset for SCST: (image_id, image_tensor). Refs loaded separately.
    items: list of (image_id, file_name) or (image_id, (subdir, file_name)) for Karpathy.
    """

    def __init__(self, image_dir, items, transform=None):
        self.image_dir = Path(image_dir)
        self.items = items
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        image_id, path_info = self.items[idx]
        if isinstance(path_info, (list, tuple)):
            subdir, file_name = path_info
            path = self.image_dir / subdir / file_name
        else:
            path = self.image_dir / path_info
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return image_id, img


def load_coco_refs_and_items(annotation_file):
    """Return image_id_to_refs and list of (image_id, file_name) for train."""
    with open(annotation_file, "r") as f:
        data = json.load(f)
    id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}
    image_id_to_refs = {}
    for ann in data["annotations"]:
        iid = ann["image_id"]
        if iid not in image_id_to_refs:
            image_id_to_refs[iid] = []
        image_id_to_refs[iid].append(ann["caption"])
    image_ids = sorted(image_id_to_refs.keys())
    items = [(iid, id_to_filename[iid]) for iid in image_ids]
    return image_id_to_refs, items


def collate_with_ids(batch):
    """batch: [(image_id, tensor), ...]. Return image_ids, stacked tensor."""
    ids = [b[0] for b in batch]
    imgs = torch.stack([b[1] for b in batch])
    return ids, imgs


def main():
    project_dir = Path(__file__).resolve().parent
    default_ckpt = project_dir / "checkpoints" / "best_model.pt"
    default_data_root = str(project_dir / "coco2017")

    parser = argparse.ArgumentParser(description="SCST: optimize CIDEr with self-critical training")
    parser.add_argument("--checkpoint", type=str, default=str(default_ckpt), help="Path to best_model.pt")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--kaggle", action="store_true", help="Use Kaggle paths")
    parser.add_argument("--input_dataset", type=str, default="coco-2017-dataset")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5, help="LR for SCST (often lower than CE)")
    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=None, help="Cap train samples (e.g. for Kaggle)")
    parser.add_argument("--karpathy", action="store_true", help="Use Karpathy train split (train+restval). Requires --karpathy_split.")
    parser.add_argument("--karpathy_split", type=str, default=None, help="Path to Karpathy split JSON (e.g. dataset_coco.json). Required when --karpathy.")
    parser.add_argument("--beam_width", type=int, default=3, help="Beam width for baseline caption (default 3).")
    args = parser.parse_args()

    if args.kaggle:
        data_root = args.data_root or str(KAGGLE_INPUT / args.input_dataset)
        save_dir = args.save_dir or str(KAGGLE_WORKING / "checkpoints")
        if args.workers == 2 and args.batch_size == 16:
            pass
    else:
        data_root = args.data_root or default_data_root
        save_dir = args.save_dir or str(Path(args.checkpoint).resolve().parent)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint: {args.checkpoint}")
    model, config = load_checkpoint(args.checkpoint, device)
    model.train()

    try:
        from transformers import GPT2Tokenizer
    except ImportError:
        raise ImportError("pip install transformers")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    bos_id = getattr(tokenizer, "bos_token_id", 1)
    eos_id = getattr(tokenizer, "eos_token_id", 2)

    data_root_path = Path(data_root)
    if args.karpathy:
        if not args.karpathy_split:
            raise ValueError("--karpathy requires --karpathy_split <path to JSON>")
        train_ids, val_ids, test_ids = load_karpathy_split(args.karpathy_split)
        id_to_path, id_to_refs = build_coco2017_id_to_path_and_refs(data_root)
        available = set(id_to_path.keys())
        train_ids &= available
        image_id_to_refs = {iid: id_to_refs[iid] for iid in train_ids if iid in id_to_refs}
        items = [(iid, id_to_path[iid]) for iid in sorted(train_ids)]
        if args.max_samples is not None:
            items = items[: args.max_samples]
        print(f"SCST training on Karpathy split: {len(items)} images")
        image_dir = data_root_path
    else:
        train_img = data_root_path / "train2017"
        train_ann = data_root_path / "annotations" / "captions_train2017.json"
        if not train_ann.exists():
            raise FileNotFoundError(f"Annotations not found: {train_ann}")
        image_id_to_refs, items = load_coco_refs_and_items(train_ann)
        if args.max_samples is not None:
            items = items[: args.max_samples]
        print(f"SCST training on {len(items)} images")
        image_dir = train_img

    dataset = COCOSCSTDataset(image_dir, items)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_with_ids,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.get("weight_decay", 0.01))
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_reward = 0.0
        n_batches = 0
        pbar = tqdm(loader, desc=f"SCST epoch {epoch+1}/{args.epochs}")
        for batch_ids, batch_images in pbar:
            batch_images = batch_images.to(device)
            B = batch_images.size(0)

            # Greedy baseline (no grad)
            with torch.no_grad():
                greedy_seqs = model.generate_beam(
                    batch_images,
                    start_token_id=bos_id,
                    end_token_id=eos_id,
                    max_length=args.max_length,
                    beam_width=args.beam_width,
                )
            greedy_captions = [
                tokenizer.decode(greedy_seqs[i].cpu().tolist(), skip_special_tokens=True)
                for i in range(B)
            ]

            # Sampled captions with log probs (with grad)
            sampled_seqs, log_prob_sum = sample_with_log_probs(
                model, batch_images, bos_id, eos_id, max_length=args.max_length, temperature=1.0
            )
            sampled_captions = [
                tokenizer.decode(sampled_seqs[i].cpu().tolist(), skip_special_tokens=True)
                for i in range(B)
            ]

            # CIDEr per image for sampled and greedy
            gts = {iid: image_id_to_refs[iid] for iid in batch_ids}
            res_sampled = {iid: [sampled_captions[i]] for i, iid in enumerate(batch_ids)}
            res_greedy = {iid: [greedy_captions[i]] for i, iid in enumerate(batch_ids)}
            _, cider_sampled = get_cider_per_image(gts, res_sampled)
            _, cider_greedy = get_cider_per_image(gts, res_greedy)

            # rewards: (B,) advantage = cider_sampled - cider_greedy
            reward_sampled = torch.tensor(cider_sampled, dtype=torch.float, device=device)
            reward_greedy = torch.tensor(cider_greedy, dtype=torch.float, device=device)
            rewards = reward_sampled - reward_greedy

            # SCST loss: - E[(r - b) * log pi] -> minimize - (r - b) * log_prob_sum
            loss = -(rewards * log_prob_sum).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            mean_reward = reward_sampled.mean().item()
            epoch_reward += mean_reward
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.3f}", "cider": f"{mean_reward:.2f}"})

        print(f"Epoch {epoch+1} mean CIDEr (sampled): {epoch_reward / max(n_batches, 1):.4f}")
        ckpt_path = Path(save_dir) / f"scst_epoch_{epoch+1}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        }, ckpt_path)
        print(f"Saved {ckpt_path}")

    # Save final best-name for inference
    final_path = Path(save_dir) / "best_model_scst.pt"
    torch.save({
        "epoch": args.epochs - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }, final_path)
    print(f"Saved {final_path}")


if __name__ == "__main__":
    main()
