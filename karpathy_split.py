"""
Karpathy split for COCO: load split JSON and map to COCO 2017 paths/captions.
Use with train_captioning_model.py --karpathy --karpathy_split <path>
and evaluate_captioning.py --karpathy --karpathy_split <path>

Karpathy split JSON format (e.g. from karpathy/neuraltalk2 or coco-caption):
  - List of {"split": "train"|"restval"|"val"|"test", ...}. train+restval used for training (~113k).
  - Or dict with "images" key containing that list.
Image IDs from the JSON are the same in COCO 2017; we map them to train2017/ or val2017/ via COCO 2017 annotations.
"""

import json
import re
from pathlib import Path


def _image_id_from_item(item):
    """Extract COCO image id from a Karpathy split item. Prefer cocoid/id (real COCO ID); imgid is often a local index."""
    if "cocoid" in item:
        return int(item["cocoid"])
    if "id" in item:
        return int(item["id"])
    if "imgid" in item:
        return int(item["imgid"])
    if "filename" in item:
        # e.g. "COCO_val2014_000000391895.jpg" or "train2014/COCO_train2014_000000123456.jpg"
        m = re.search(r"(\d{6,12})\.jpg", item["filename"])
        if m:
            return int(m.group(1))
    return None


def load_karpathy_split(json_path):
    """
    Load Karpathy split JSON and return train_ids, val_ids, test_ids (sets of int).
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Karpathy split file not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "root" in data and isinstance(data["root"], dict) and "images" in data["root"]:
            images = data["root"]["images"]
        elif "images" in data:
            images = data["images"]
        else:
            raise ValueError("Karpathy JSON must have 'images' or 'root']['images'")
    elif isinstance(data, list):
        images = data
    else:
        raise ValueError("Karpathy JSON must be a list of images or dict with 'images' key")
    train_ids, val_ids, test_ids = set(), set(), set()
    for item in images:
        iid = _image_id_from_item(item)
        if iid is None:
            continue
        split = (item.get("split") or "").strip().lower()
        if split in ("train", "restval"):
            train_ids.add(iid)
        elif split == "val":
            val_ids.add(iid)
        elif split == "test":
            test_ids.add(iid)
    return train_ids, val_ids, test_ids


def build_coco2017_id_to_path_and_refs(data_root):
    """
    Load COCO 2017 train and val annotations; return id_to_path and id_to_refs.
    id_to_path[image_id] = (subdir, filename) e.g. ("train2017", "000000000001.jpg")
    id_to_refs[image_id] = [caption1, caption2, ...]
    data_root should contain train2017/, val2017/, annotations/captions_train2017.json, annotations/captions_val2017.json
    """
    root = Path(data_root)
    train_ann = root / "annotations" / "captions_train2017.json"
    val_ann = root / "annotations" / "captions_val2017.json"
    if not train_ann.exists():
        raise FileNotFoundError(f"COCO 2017 train annotations not found: {train_ann}")
    if not val_ann.exists():
        raise FileNotFoundError(f"COCO 2017 val annotations not found: {val_ann}")
    id_to_path = {}
    id_to_refs = {}
    for ann_path, subdir in [(train_ann, "train2017"), (val_ann, "val2017")]:
        with open(ann_path, "r") as f:
            data = json.load(f)
        for img in data["images"]:
            iid = img["id"]
            id_to_path[iid] = (subdir, img["file_name"])
        for ann in data["annotations"]:
            iid = ann["image_id"]
            if iid not in id_to_refs:
                id_to_refs[iid] = []
            id_to_refs[iid].append(ann["caption"])
    return id_to_path, id_to_refs
