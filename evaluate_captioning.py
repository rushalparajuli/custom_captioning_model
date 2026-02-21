"""
Evaluate the captioning model: load best_model.pt from checkpoints and compute BLEU-4 and CIDEr
on the COCO 2017 validation set. Uses beam search for caption generation (deterministic).

Usage:
  python evaluate_captioning.py
  python evaluate_captioning.py --checkpoint checkpoints/best_model.pt --data_root ./coco2017
  python evaluate_captioning.py --beam_width 5 --max_samples 5000

Requires: nltk (for BLEU-4). For CIDEr: pip install pycocoevalcap
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from cnn_encoder_decoder_captioning import CNNEncoderDecoderCaptioning
from inference import load_model, get_transform
from karpathy_split import load_karpathy_split, build_coco2017_id_to_path_and_refs


def load_coco_annotations(annotation_file):
    """
    Load COCO captions and return:
    - image_id_to_refs: dict[image_id] = [caption1, caption2, ...]
    - image_id_to_filename: dict[image_id] = file_name
    - list of (image_id, file_name) for deterministic iteration
    """
    with open(annotation_file, "r") as f:
        data = json.load(f)
    id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}
    image_id_to_refs = {}
    for ann in data["annotations"]:
        iid = ann["image_id"]
        if iid not in image_id_to_refs:
            image_id_to_refs[iid] = []
        image_id_to_refs[iid].append(ann["caption"])
    # Deterministic order by image id
    image_ids = sorted(image_id_to_refs.keys())
    items = [(iid, id_to_filename[iid]) for iid in image_ids]
    return image_id_to_refs, id_to_filename, items


class EvalImageDataset(Dataset):
    """
    Dataset that yields (image_id, image_tensor) for evaluation.
    items: list of (image_id, file_name) with image_dir = single folder, or
           list of (image_id, (subdir, file_name)) with image_dir = data_root (Karpathy).
    """

    def __init__(self, image_dir, items, transform=None):
        self.image_dir = Path(image_dir)
        self.items = items
        self.transform = transform or get_transform()

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


def tokenize_for_bleu(text):
    """Tokenize caption into words (lowercase, split on spaces)."""
    return text.lower().strip().split()


def compute_bleu4(list_of_references, hypotheses):
    """
    list_of_references: list of list of list of words, one per image
        e.g. [[ref1_tokens, ref2_tokens], [ref1_tokens], ...]
    hypotheses: list of list of words, one per image
    """
    try:
        from nltk.translate.bleu_score import corpus_bleu
        from nltk.translate.bleu_score import SmoothingFunction
    except ImportError:
        raise ImportError("BLEU requires nltk. Install with: pip install nltk && python -c \"import nltk; nltk.download('punkt_tab')\"")
    # BLEU-4 with uniform weights; smoothing for short captions
    weights = (0.25, 0.25, 0.25, 0.25)
    smoothing = SmoothingFunction().method1
    return corpus_bleu(list_of_references, hypotheses, weights=weights, smoothing_function=smoothing)


def compute_cider(gts, res):
    """
    gts: dict[image_id] = [ref_caption1, ref_caption2, ...]
    res: dict[image_id] = [pred_caption]
    Returns scalar CIDEr score (average).
    """
    try:
        from pycocoevalcap.cider.cider import Cider
    except ImportError:
        raise ImportError("CIDEr requires pycocoevalcap. Install with: pip install pycocoevalcap")
    scorer = Cider()
    score, _ = scorer.compute_score(gts, res)
    return score


def main():
    project_dir = Path(__file__).resolve().parent
    default_checkpoint = project_dir / "checkpoints" / "best_model.pt"
    default_data_root = project_dir / "coco2017"

    parser = argparse.ArgumentParser(description="Evaluate captioning model: BLEU-4 and CIDEr")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(default_checkpoint),
        help="Path to checkpoint (e.g. checkpoints/best_model.pt or best_model.pth)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(default_data_root),
        help="Root of COCO 2017 (val2017/, annotations/captions_val2017.json).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for caption generation.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max number of validation images to evaluate (default: all).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=50,
        help="Max caption length for generation.",
    )
    parser.add_argument(
        "--beam_width",
        type=int,
        default=5,
        help="Beam width for beam search (used for metric generation).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )
    parser.add_argument("--karpathy", action="store_true", help="Evaluate on Karpathy test split (COCO 2017). Requires --karpathy_split.")
    parser.add_argument("--karpathy_split", type=str, default=None, help="Path to Karpathy split JSON (e.g. dataset_coco.json). Required when --karpathy.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if args.karpathy:
        if not args.karpathy_split:
            raise ValueError("--karpathy requires --karpathy_split <path to JSON>")
        train_ids, val_ids, test_ids = load_karpathy_split(args.karpathy_split)
        id_to_path, id_to_refs = build_coco2017_id_to_path_and_refs(data_root)
        available = set(id_to_path.keys())
        test_ids &= available
        image_id_to_refs = {iid: id_to_refs[iid] for iid in test_ids if iid in id_to_refs}
        items = [(iid, id_to_path[iid]) for iid in sorted(test_ids)]
        if args.max_samples is not None:
            items = items[: args.max_samples]
        print("Karpathy test split: %d images" % len(items))
        eval_dir = data_root
    else:
        val_img_dir = data_root / "val2017"
        val_ann_file = data_root / "annotations" / "captions_val2017.json"
        if not val_ann_file.exists():
            raise FileNotFoundError(f"Annotations not found: {val_ann_file}")
        print("Loading COCO val annotations...")
        image_id_to_refs, id_to_filename, items = load_coco_annotations(val_ann_file)
        if args.max_samples is not None:
            items = items[: args.max_samples]
        print(f"Evaluating on {len(items)} images (COCO 2017 val).")
        eval_dir = val_img_dir

    print("Loading checkpoint:", args.checkpoint)
    model, tokenizer, device = load_model(args.checkpoint, args.device)
    model.eval()

    dataset = EvalImageDataset(eval_dir, items)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    bos_id = getattr(tokenizer, "bos_token_id", 1)
    eos_id = getattr(tokenizer, "eos_token_id", 2)
    res = {}  # image_id -> [pred_caption]
    list_of_references = []
    hypotheses_tokens = []

    for batch_image_ids, batch_images in tqdm(loader, desc="Generating captions (beam search)"):
        batch_images = batch_images.to(device)
        with torch.no_grad():
            generated = model.generate_beam(
                batch_images,
                start_token_id=bos_id,
                end_token_id=eos_id,
                max_length=args.max_length,
                beam_width=args.beam_width,
            )
        for i, image_id in enumerate(batch_image_ids.tolist()):
            ids = generated[i].cpu().tolist()
            pred = tokenizer.decode(ids, skip_special_tokens=True)
            res[image_id] = [pred]
            list_of_references.append([tokenize_for_bleu(c) for c in image_id_to_refs[image_id]])
            hypotheses_tokens.append(tokenize_for_bleu(pred))

    # BLEU-4
    bleu4 = compute_bleu4(list_of_references, hypotheses_tokens)
    print(f"BLEU-4: {bleu4:.4f}")

    # CIDEr (gts: image_id -> list of ref strings)
    gts = {iid: image_id_to_refs[iid] for iid in res}
    try:
        cider = compute_cider(gts, res)
        print(f"CIDEr:  {cider:.4f}")
    except ImportError as e:
        print(f"CIDEr:  (skip) {e}")

    print("Done.")


if __name__ == "__main__":
    main()
