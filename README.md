# DINOV2 Bounding-Box Pretraining & Classification

Self-supervised pretraining and supervised fine-tuning pipeline for colorectal histology images that combines bounding-box aware augmentations, DINO-style teacher–student losses, and ViT-based classifiers (with optional LoRA or block-expansion adaptation). The project was built around SUN datasets exported as CSVs with per-image metadata.

## Repository Layout
- `PreTrainBBox.py` – main entry point. Orchestrates DINOLoss-based pretraining, supervised training, validation/test reporting, attention visualisation, and experiment logging (W&B + CodeCarbon).
- `data.py` – dataset utilities. Loads SUN CSV splits, extracts bbox crops for the teacher branch, applies Albumentations pipelines, and returns tensors plus metadata (histology class, bbox coords, file path).
- `model_vit.py` – wrapper around Hugging Face ViT/DINO models with PyTorch Lightning utilities, mixup integration, LoRA support (`peft`), and block expansion for layer freezing/duplication.
- `loss.py` – DINOLoss, attention-overlap regulariser, soft-target cross-entropy, and mixup/cutmix helpers.
- `ds_sun_10/` – dataset split used during the paper training. 

## Data Expectations
All raw images must live under a `SUNdatabase_complete/` directory organised by diagnosis polarity (Positive/Negative) and case identifiers. Mirror the tree below when ingesting new data so every `image_path` referenced in the CSV files resolves to an on-disk file.

```
SUNdatabase_complete/
├── Positive/
│   ├── case1/
│   ├── case2/
│   └── ...
└── Negative/
    ├── case1/
    ├── case2/
    └── ...
```

The dataset CSVs stored in `ds_sun_10/` (train/val/test) must follow this column schema:
- `image_path` – path relative to the repository root pointing into `SUNdatabase_complete/`.
- `Pathological_diagnosis` – descriptive free text used for logging/metadata.
- `x`, `y`, `width`, `height` – bounding-box coordinates in original pixels.
- `case` – identifier linking back to case-level metadata.
- `histology` – textual label convertible through `utils.histology_to_int_dict_{2|3}classes`.

Case-level CSVs (e.g., `test_cases_sun.csv`) should include:
- `case` – unique identifier (must match the dataset CSV).
- `paris_total` – Paris classification string per case.
- `Pathological_diagnosis` – same description reported in dataset CSVs to keep narratives synced.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124  # pick your CUDA
pip install -r requirements.txt  # create one listing albumentations, transformers, peft, pytorch-lightning, wandb, codecarbon, seaborn, scikit-learn, pandas, tqdm, opencv-python
```
If you do not keep a `requirements.txt`, manually install the packages imported in the scripts.

## Running Experiments
All commands are executed from the repo root. Key flags exposed by `PreTrainBBox.py`:

| Argument | Description |
| --- | --- |
| `--dataset_path` | Directory containing the CSV splits (default `ds_sun_10`). Automatically overridden when `--num_classes` is 2 vs 3. |
| `--store_name` | Folder under the repo where checkpoints, plots, and args JSON are saved. |
| `--batch_size`, `--num_epochs`, `--lr`, `--weight_decay`, `--lr_scale` | Optimisation hyperparameters (AdamW + ReduceLROnPlateau by default). |
| `--variant` | `full`, `lora`, or `block`. Controls if the ViT head/backbone are fully trainable, wrapped with LoRA adapters (rank via `--rank`), or expanded layer-wise. |
| `--enable_pretraining` | Enables DINOLoss pretraining between teacher/student crops. Uses `--pretrain_epochs` and saves `best_model_pretrain.pth`. |
| `--model_name` | Hugging Face checkpoint (e.g., `vit-b16-224-dino`, `vit-l32-224-in21k`). |
| `--num_classes` | 2-class or 3-class setup; also selects the correct histology dictionaries. |
| `--only_test` | Skip training and evaluate `best_model.pth`. |
| `--resume` | Resume supervised training from `final_model.pth`. |
| `--penalize_outside`, `--inside_weight`, `--outside_weight` | Tune the attention-overlap loss to encourage CLS attention to stay inside/outside the bbox. |

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

2. **LoRA + DINOLoss pretraining**
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
   The script will (a) train a student model to match the bbox teacher CLS tokens, (b) swap in LoRA adapters initialised from the pretrained weights, then (c) run supervised training. Checkpoints: `best_model_pretrain.pth`, `best_model.pth`, `final_model.pth`.

3. **Evaluation only**
   ```bash
   python PreTrainBBox.py --store_name experiment_lora --only_test
   ```
   Assumes `store_name/best_model.pth` exists. Testing logs include balanced accuracy, per-class confusion matrices, and saved attention rollouts.

### Pretraining schema
The teacher–student loop is illustrated below; the teacher consumes bbox crops while the student sees the full image and is trained to match the teacher CLS tokens before supervised fine-tuning.

![Pretraining schema](figures/schema_REBUTTAL.pdf)

## Outputs & Logging
- `store_name/` contains checkpoints, attention overlays, Tensor/CSV logs (balanced accuracy, CLS-in/outside metrics), and the serialized argument file (`args.json`).
- Weights & Biases logging is enabled by default; configure `WANDB_PROJECT`/`WANDB_ENTITY` env vars.
- CodeCarbon’s `@track_emissions` decorator (when active) will emit hardware energy stats alongside training.

## License
Released under the [MIT License](https://opensource.org/license/mit). You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, provided you include the copyright notice and permission text in all copies or substantial portions of the Software.
