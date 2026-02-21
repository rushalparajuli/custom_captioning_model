# Image Captioning: CNN + Transformer Encoder–Decoder

Image captioning model that generates natural-language descriptions for images. It uses a **pre-trained EfficientNet B1** backbone for visual features, a **Transformer encoder** on top of the CNN, and a **GPT-style decoder** with cross-attention to produce captions. The vocabulary is the GPT-2 tokenizer (50,257 tokens). Training and evaluation use the **COCO 2017** dataset, with optional **Karpathy split** for comparable metrics.

## Features

- **Architecture**: EfficientNet B1 (frozen or fine-tunable) → Transformer encoder → GPT-2–style decoder with cross-attention
- **Training**: Cross-entropy on COCO 2017; optional Karpathy train/val split (~113k train + restval)
- **Data augmentation** (when resuming with Karpathy): random resized crop, horizontal flip, mild rotation
- **Evaluation**: BLEU-4 and CIDEr (Karpathy test or COCO val)
- **Inference**: CLI script and FastAPI server; optional React web app for upload-and-caption

## Requirements

- Python 3.8+
- PyTorch, torchvision
- [Hugging Face Transformers](https://github.com/huggingface/transformers) (GPT-2 tokenizer)
- For evaluation: `nltk` (BLEU), optionally `pycocoevalcap` (CIDEr)
- For the API: FastAPI, uvicorn, pydantic (see `requirements.txt`)

## Setup

### 1. Clone or download the project

```bash
cd /path/to/project
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
# or: venv\Scripts\activate  # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For evaluation (BLEU-4):

```bash
pip install nltk
python -c "import nltk; nltk.download('punkt_tab')"
```

For CIDEr:

```bash
pip install pycocoevalcap
```

### 4. COCO 2017 data

Download [COCO 2017](https://cocodataset.org/) and place it so that:

- **Training images**: `train2017/`
- **Validation images**: `val2017/`
- **Annotations**: `annotations/captions_train2017.json`, `annotations/captions_val2017.json`

Default local path assumed by the scripts: `<project_dir>/coco2017/`. Override with `--data_root`.

### 5. Karpathy split (optional, for paper-comparable metrics)

To use the standard Karpathy train/val/test split (~113k train, 5k val, 5k test), you need a Karpathy-style JSON (e.g. `dataset_coco.json` from [coco-caption](https://github.com/tylin/coco-caption) or similar). The file can be under a `root` key with `images` and each image having `split` (`train`/`restval`/`val`/`test`) and `cocoid` (or `id`/`filename`). Place the file where the script can read it and pass `--karpathy_split <path>`.

## Usage

### Training from scratch

**Local (default paths: `./coco2017`, `./checkpoints`):**

```bash
python train_captioning_model.py
```

**With Karpathy split:**

```bash
python train_captioning_model.py --karpathy --karpathy_split /path/to/dataset_coco.json
```

**Options:** `--data_root`, `--save_dir`, `--epochs`, `--lr`, `--batch_size`, `--small` (smaller model), `--dummy` (dummy data). See `python train_captioning_model.py --help`.

### Training on Kaggle

Use GPU and add the COCO 2017 dataset (and optionally a Karpathy split dataset). Then:

```bash
pip install -q transformers
python train_captioning_model.py --kaggle --data_root /kaggle/input/coco-2017-dataset/coco2017
```

With Karpathy split:

```bash
python train_captioning_model.py --kaggle --data_root /kaggle/input/coco-2017-dataset/coco2017 --karpathy --karpathy_split /kaggle/input/karpathy-split/dataset_coco.json
```

See **KAGGLE.md** for more Kaggle-specific notes.

### Resuming training

Resume from a saved checkpoint (e.g. after initial training or to fine-tune the CNN):

```bash
python resume_training.py --checkpoint checkpoints/best_model.pt --data_root ./coco2017
```

**Fine-tune the CNN backbone (unfreeze):**

```bash
python resume_training.py --checkpoint checkpoints/best_model.pt --unfreeze_cnn --lr 3e-5 --dropout 0.2
```

**With Karpathy split and augmentation (recommended when unfreezing):**

```bash
python resume_training.py --checkpoint checkpoints/best_model.pt --data_root ./coco2017 --karpathy --karpathy_split /path/to/dataset_coco.json --unfreeze_cnn --lr 5e-5 --dropout 0.2
```

Resume uses **image augmentation** for the Karpathy training set (random resized crop, horizontal flip, small rotation). Override dropout with `--dropout 0.2` (or other value) if desired.

### Evaluation

**COCO 2017 validation set:**

```bash
python evaluate_captioning.py --checkpoint checkpoints/best_model.pt --data_root ./coco2017
```

**Karpathy test split (for paper comparison):**

```bash
python evaluate_captioning.py --checkpoint checkpoints/best_model.pt --data_root ./coco2017 --karpathy --karpathy_split /path/to/dataset_coco.json
```

Options: `--beam_width`, `--max_length`, `--max_samples`. Requires `nltk` (BLEU-4); install `pycocoevalcap` for CIDEr.

### Inference (single image)

```bash
python inference.py --checkpoint checkpoints/best_model.pt --image path/to/image.jpg
```

### Web API (FastAPI)

Serve the model over HTTP (e.g. for the React caption app):

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API expects a base64-encoded image and returns a generated caption. The app loads the checkpoint from `checkpoints/best_model.pt` by default (edit `CHECKPOINT_PATH` in `app.py` if needed).

### React caption app (optional)

If you have the `caption-app` frontend:

```bash
cd caption-app && npm install && npm start
```

Point the app at the FastAPI backend URL (e.g. `http://localhost:8000`).

## Project structure

| File / folder        | Description |
|----------------------|-------------|
| `train_captioning_model.py` | Main training script (COCO 2017 or Karpathy). |
| `resume_training.py`      | Resume from checkpoint; optional unfreeze CNN, Karpathy, augmentation, dropout override. |
| `evaluate_captioning.py`  | BLEU-4 and CIDEr on val or Karpathy test. |
| `cnn_encoder_decoder_captioning.py` | Model: EfficientNet B1 + Transformer encoder + GPT-style decoder. |
| `karpathy_split.py`       | Load Karpathy split JSON and map to COCO 2017 paths/captions. |
| `inference.py`            | Load model and run caption generation (CLI). |
| `app.py`                  | FastAPI server for caption API. |
| `scst_captioning.py`      | Self-critical sequence training (CIDEr optimization). |
| `KAGGLE.md`               | Kaggle-specific setup and commands. |
| `requirements.txt`        | Python dependencies. |
| `caption-app/`            | Optional React frontend. |
| `checkpoints/`            | Saved checkpoints (e.g. `best_model.pt`). |
| `coco2017/`               | COCO 2017 data (default location). |

## Checkpoints

Training saves:

- **best_model.pt**: Best validation loss (use this for inference and evaluation).
- Optional epoch checkpoints if configured.

Checkpoints contain: `model_state_dict`, `config` (model hyperparameters), `epoch`, `val_loss`, and optionally `optimizer_state_dict`.

## License and references

- COCO: [cocodataset.org](https://cocodataset.org/)
- Karpathy split: commonly used with [coco-caption](https://github.com/tylin/coco-caption); train+restval ≈ 113k images, 5 captions each.
- Metrics: BLEU-4, CIDEr (see coco-caption and related papers).
