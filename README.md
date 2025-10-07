# KIVA-ICCV: Visual Analogy Learning

A PyTorch implementation of a Siamese Network for solving visual analogies, designed for the ICCV challenge.

## 🚀 Quick Start

### Prerequisites

1. Install [uv](https://docs.astral.sh/uv/)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install dependencies from `pyproject.toml`

```bash
uv sync
```

### Useful commands and basic CLI usage

You can start by copying the `.env.template` file to `.env` and filling in the API token and project name.
```bash
cp .env.template .env
```

The entry point for training and testing the model is:
```
uv run --env-file .env python kiva-iccv/train.py \ 
    --do_train \
    --do_test \
    --epochs 30 \
    --batch_size 64
```
For the full list of options, run `uv run python kiva-iccv/train.py --help`.

Other useful commands:
```bash
make test-train-unit # use the unit test dataset to train and test the model
make test-overfit-validation # train and evaluate on validation set to make sure the model can learn to overfit
make fmt # format the code
```

## 📁 Data

### Set up the data by downloading from different sources and processing them

```bash
make set-up-data
```
After running this, you are ready to go!

#### What the script is doing

1. Load the basic competition data from [official competition repository](https://github.com/ey242/KiVA-challenge):

```bash
uv run python kiva-iccv/utils/download.py
```

This will create the `data/` directory with the following structure:

```
data/
├── train/
├── validation/
├── test/
├── unit/
├── train.json
├── validation.json
└── unit.json
```

**Note:** The `test` dataset has no metadata since the labels have not been released yet.

2. Create the subimages that will be used by the model. This splits the images into 6 subimages for each sample.
The naming convention is:
- `{sample_id}_ex_before.jpg` - Example "before" image
- `{sample_id}_ex_after.jpg` - Example "after" image  
- `{sample_id}_test_before.jpg` - Test "before" image
- `{sample_id}_choice_a.jpg` - Choice A image
- `{sample_id}_choice_b.jpg` - Choice B image
- `{sample_id}_choice_c.jpg` - Choice C image

Run the following commands to create the subimages:
```bash
for dataset in train validation test unit; do
    uv run python kiva-iccv/utils/transform.py --dataset $dataset
done
```

After this, the `data/` directory will have the following structure:
```
data/
├── train/           # Training images
│   ├── sample1.jpg
│   ├── sample2.jpg
│   └── ...
├── validation/      # Validation images
│   ├── sample1.jpg
│   ├── sample2.jpg
│   └── ...
├── test/           # Test images
│   ├── sample1.jpg
│   ├── sample2.jpg
│   └── ...
├── split_train/           # Split training images
│   ├── sample1_ex_before.jpg
│   ├── sample1_ex_after.jpg
│   ├── sample1_test_before.jpg
│   ├── sample1_choice_a.jpg
│   ├── sample1_choice_b.jpg
│   └── sample1_choice_c.jpg
├── split_validation/      # Split validation images
│   └── ...
├── split_test/           # Split test images
│   └── ...
├── train.json            # Training metadata
└── validation.json       # Validation metadata
```

3. Download the untransformed images for the on-the-fly dataset.
We use the same base images as in the KiVA original paper, which are stored in the paper's [repository](https://github.com/ey242/KiVA).

Run the following commands to clone the repository and copy the untransformed images to the `data/KiVA/` directory:
```bash
mkdir -p KiVA
mkdir -p data/KiVA
cd KiVA
git clone --depth 1 --branch main https://github.com/ey242/KiVA.git
cp -r "KiVA/untransformed objects/" ../data/KiVA/
cd ../
rm -rf KiVA
```

## 🧠 Model Architecture

### Overview
The **SiameseAnalogyNetwork** solves visual analogies of the form "A is to B as C is to ?" by learning transformation representations and using cosine similarity to match transformations.

**Input**: 6 images (example before/after, test before, 3 choices)  
**Output**: 4 normalized transformation vectors for similarity comparison

### Supported Encoders

#### **Vision Transformers (Recommended)**
- `vit_small_patch16_224`, `vit_base_patch16_224` - Standard ViT models
- `vit_small_patch16_dinov3` - DINOv3 models with rotary position embeddings (RoPE)

Uses specialized **TransformationEncoder** architectures that process image pairs as unified sequences:
- **Standard ViT**: `[CLS, img1_patches, img2_patches]` with extended positional embeddings and segment embeddings
- **DINOv3**: `[CLS, register_tokens, img1_patches, img2_patches]` with RoPE handling and 4 register tokens
- Output: CLS token embedding representing the transformation

#### **ResNet Models**
- `resnet18`, `resnet50` - Traditional CNN backbones
- Uses subtraction-based transformations (e.g., `after - before`)

### Architecture Components

**Projection Head** (applied to all encoder outputs):
```python
nn.Sequential(
    nn.Linear(encoder_dim, 512),
    nn.ReLU(), Dropout(0.2),
    nn.Linear(512, 512),
    nn.LayerNorm(512)
)
```

**Key Features**:
- Differential learning rates: encoder (0.1×LR), projection (1.0×LR)
- L2 normalization for cosine similarity
- Optional encoder freezing for transfer learning
- Configurable embedding dimensions (default: 512)

### Loss Functions

1. **Standard Triplet Loss** (`standard_triplet`): Anchor-positive-negative triplet learning with margin
2. **Contrastive Analogy Loss** (`contrastive`): Margin-based ranking loss optimizing `max(0, margin - (pos_sim - neg_sim))`
3. **Softmax Analogy Loss** (`softmax`): Temperature-scaled cross-entropy over similarity scores

