# Training on Kaggle

Use a smaller model and Kaggle paths so training fits within Kaggle's GPU limits.

## 1. Create a Kaggle Notebook

- **Settings**: Enable GPU (P100 or T4).
- **Datasets**: Add **COCO 2017** (e.g. search "coco 2017" and add a dataset that has `train2017/`, `val2017/`, `annotations/captions_train2017.json`, `captions_val2017.json`).
- Note the **dataset slug** (e.g. `coco-2017-dataset`). Input path will be `/kaggle/input/<slug>/`.

## 2. Add Your Code

Upload or paste your project files into the notebook (or clone from a repo). You need at least:

- `train_captioning_model.py`
- `cnn_encoder_decoder_captioning.py`

## 3. Install dependencies (first cell)

```bash
!pip install -q transformers
```

## 4. Run training

```bash
!python train_captioning_model.py --kaggle
```

This will:

- Use **data** from `/kaggle/input/coco-2017-dataset/` (override with `--input_dataset YOUR_SLUG` if your dataset name differs).
- Save **checkpoints** to `/kaggle/working/checkpoints/` (best model: `best_model.pt`).
- Use a **smaller model** (embed_dim=256, 3 encoder/decoder layers, frozen CNN backbone, batch_size=24, workers=2).

### Options

| Flag | Effect |
|------|--------|
| `--kaggle` | Use Kaggle paths and small-model defaults |
| `--input_dataset NAME` | Kaggle input folder name (default: `coco-2017-dataset`) |
| `--small` | Use small model (also default when `--kaggle`) |
| `--epochs 10` | Fewer epochs to finish in one session |
| `--batch_size 16` | Smaller batch if OOM |

To use the **full-size model** on Kaggle (more GPU memory), set paths manually and omit `--kaggle`:

```bash
!python train_captioning_model.py --data_root /kaggle/input/coco-2017-dataset --save_dir /kaggle/working/checkpoints --batch_size 16 --epochs 5
```

## 5. After training

- Best checkpoint: `/kaggle/working/checkpoints/best_model.pt`
- Download it from the notebook Output or "Save Version" to keep it.
- Use locally: `python inference.py --checkpoint best_model.pt --image your_image.jpg`
</think>
Adding a note for using the full model on Kaggle and checking for lint issues.
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>
StrReplace