# Bounding-Box Pretraining & Classification

Self-supervised pretraining and supervised fine-tuning pipeline for colorectal histology images that combines bounding-box aware augmentations, attention-overlap pretraining, and ViT-based classifiers (with optional LoRA or block-expansion adaptation). The project was built around SUN datasets exported as CSVs with per-image metadata.

## Repository Layout
- `PreTrainBBox.py` — main entry point. Orchestrates attention-overlap pretraining, supervised training, validation/test reporting, attention visualisation, and experiment logging (W&B + CodeCarbon).
- `data.py` — dataset utilities. Loads SUN CSV splits, extracts bbox crops for the teacher branch, applies Albumentations pipelines, and returns tensors plus metadata (histology class, bbox coords, file path).
- `model_vit.py` — wrapper around Hugging Face ViT/DINO models with PyTorch Lightning utilities, mixup integration, LoRA support (`peft`), and block expansion for layer freezing/duplication.
- `loss.py` — DINOLoss, attention-overlap regulariser, soft-target cross-entropy, and mixup/cutmix helpers.
- `ds_sun_10/` — dataset split used during the paper training.

## Data Expectations
All raw images must live under a `SUNdatabase_complete/` directory organised by diagnosis polarity (Positive/Negative) and case identifiers. Mirror the tree below when ingesting new data so every `image_path` referenced in the CSV files resolves to an on-disk file.

```
SUNdatabase_complete/
└── Positive/
    ├── case1/
    ├── case2/
    └── ...
└── Negative/
    ├── case1/
    ├── case2/
    └── ...
```

The dataset CSVs stored in `ds_sun_10/` (train/val/test) must follow this column schema:
- `image_path` — path relative to the repository root pointing into `SUNdatabase_complete/`.
- `Pathological_diagnosis` — descriptive free text used for logging/metadata.
- `x`, `y`, `width`, `height` — bounding-box coordinates in original pixels.
- `case` — identifier linking back to case-level metadata.
- `histology` — textual label convertible through `utils.histology_to_int_dict_{2|3}classes`.

Case-level CSVs (e.g., `test_cases_sun.csv`) should include:
- `case` — unique identifier (must match the dataset CSV).
- `paris_total` — Paris classification string per case.
- `Pathological_diagnosis` — same description reported in dataset CSVs to keep narratives synced.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124  # pick your CUDA
pip install -r requirements.txt  # create one listing albumentations, transformers, peft, pytorch-lightning, wandb, codecarbon, seaborn, scikit-learn, pandas, tqdm, opencv-python
```
If you do not keep a `requirements.txt`, manually install the packages imported in the scripts.

## Running Experiments
All commands are executed from the repo root.

### CLI arguments (current parser)
| Flag | Default | Purpose |
| --- | --- | --- |
| `--dataset_path` | `ds_sun_10` | Root directory containing the CSV splits. Overridden automatically to `ds_sun_10_pnp` when `--num_classes 2`. |
| `--store_name` | `PROVA` | Output folder for checkpoints, plots, architecture dumps, and `args.json`. |
| `--batch_size` | `32` | Batch size for all dataloaders. |
| `--num_epochs` | `50` | Supervised training epochs. |
| `--pretrain_epochs` | `50` | Attention-overlap pretraining epochs. |
| `--lr` | `1e-6` | Learning rate for AdamW. |
| `--lr_scale` | `0.1` | ReduceLROnPlateau factor (min LR `1e-8`). |
| `--weight_decay` | `5e-5` | AdamW weight decay. |
| `--num_workers` | `16` | Dataloader workers. |
| `--device` | `cuda` | Training device; wraps models with `DataParallel` when multiple GPUs are present. |
| `--variant` | `full` | Backbone mode: `full`, `lora`, or `block` (block expansion). |
| `--rank` | `8` | LoRA rank or block-expansion grouping size. |
| `--split` | `None` | Block expansion split index. |
| `--model_name` | `vit-l32-224-in21k` | Hugging Face ViT/DINO checkpoint id. |
| `--num_classes` | `3` | Number of histology classes (2→`ds_sun_10_pnp`, 3→`ds_sun_10`). |
| `--enable_pretraining` | `False` | Run attention-overlap pretraining before supervised training. |
| `--deactivate_normalization` | `False` | Disable L1 normalization inside the overlap loss. |
| `--drop` | `False` | Drop two cases from the dataset during pretraining. |
| `--short_training` | `False` | Debug flag that subsamples datasets to ~1% (and 100 samples for pretraining). |
| `--resume` | `False` | Resume supervised training from `final_model.pth` in `store_name`. |
| `--only_test` | `False` | Skip training, load `best_model.pth`, and evaluate/test. |
| `--patience` | `15` | Early-stopping patience for both pretraining and supervised phases. |

### Typical workflows

1. **End-to-end training (no pretraining)**
   ```bash
   python PreTrainBBox.py \
     --dataset_path ds_sun_10 \
     --store_name experiment_full \
     --model_name vit-b16-224-in21k \
     --num_classes 3 \
     --batch_size 16 \
     --num_epochs 50
   ```

2. **LoRA + attention-overlap pretraining**
   ```bash
   python PreTrainBBox.py \
     --dataset_path ds_sun_10 \
     --store_name experiment_lora \
     --variant lora \
     --rank 8 \
     --enable_pretraining \
     --pretrain_epochs 50 \
     --model_name vit-b16-224-dino \
     --num_classes 3
   ```
   The script will (a) train a student model to maximise CLS-attention overlap with the bounding boxes, (b) optionally swap in LoRA adapters initialised from the pretrained weights, then (c) run supervised training. Checkpoints: `best_model_pretrain_overlap_only.pth`, `best_model.pth`, `final_model.pth`.

3. **Evaluation only**
   ```bash
   python PreTrainBBox.py --store_name experiment_lora --only_test
   ```
   Assumes `store_name/best_model.pth` exists. Testing logs include balanced accuracy, per-class confusion matrices, and saved attention rollouts.

### Pretraining schema
The pretraining loop optimises the student so that its CLS attention overlaps the provided bounding boxes before supervised fine-tuning.

![Pretraining schema](figures/schema_REBUTTAL.pdf)

## Script Architecture (PreTrainBBox.py)
- **Argument parsing & dataset selection**: argparse sets defaults above; choosing `--num_classes 2` switches to `ds_sun_10_pnp` and imports the matching histology dictionaries from `utils`.
- **Datasets & dataloaders**: `BBoxTeacherStudentDataset_SUN` powers both pretraining (teacher crops + student full image) and supervised splits; `short_training` can slice the datasets for quick runs.
- **Attention-overlap pretraining (optional)**: Builds a `ClassificationModel` (full training mode) and optimises `AttentionOverlapLoss` on CLS attention vs. bbox coordinates using AdamW + ReduceLROnPlateau; best weights are stored at `store_name/best_model_pretrain_overlap_only.pth`.
- **Model adaptation**: After pretraining, the classifier is rebuilt; when `--variant lora`, `load_pretrained_and_apply_lora` attaches adapters (rank `--rank`, `split` for block expansion) optionally initialised from the pretrained checkpoint.
- **Supervised training**: `train_bbox_classifier` runs cross-entropy training with early stopping, logs to W&B, and saves `best_model.pth` and `final_model.pth` in `store_name/`.
- **Evaluation & reporting**: Reloads `best_model.pth`, evaluates on the test split (`test_model`), dumps confusion matrices/classification report/CSV metrics, saves `args.json`, and exports attention overlays via `save_attention_masks_and_images_NEW`.

## Outputs & Logging
- `store_name/` contains checkpoints, attention overlays, Tensor/CSV logs (balanced accuracy, CLS-in/outside metrics), the serialized argument file (`args.json`), and the `model_architecture.txt` dump.
- Weights & Biases logging is enabled by default; configure `WANDB_PROJECT`/`WANDB_ENTITY` env vars.
- CodeCarbon's `@track_emissions` decorator (when active) will emit hardware energy stats alongside training.

## License
Released under the [MIT License](https://opensource.org/license/mit). You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, provided you include the copyright notice and permission text in all copies or substantial portions of the Software.
