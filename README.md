# KIVA-ICCV: Visual Analogy Learning

A PyTorch implementation of a Siamese Network for solving visual analogies, designed for the ICCV challenge.

## 🚀 Quick Start

### Prerequisites

1. Install [uv](https://docs.astral.sh/uv/)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install dependencies

```bash
# Install dependencies using uv
uv sync
```

This will automatically install all dependencies from your `pyproject.toml` file.

### Useful commands and basic CLI usage

You can start by copying the `.env.template` file to `.env` and filling in the API token and project name.
```bash
cp .env.template .env
```

Main commands:
```
uv run --env-file .env python kiva-iccv/train.py --do_train --do_test --epochs 30 --batch_size 64
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
The model implements a **Siamese Analogy Network** designed to solve visual analogies of the form "A is to B as C is to ?". The architecture learns transformation representations that capture the relationship between image pairs and uses cosine similarity to find the best matching transformation.

### Core Architecture Components

#### 1. **SiameseAnalogyNetwork**
The main model class that orchestrates the entire pipeline:
- **Input**: 6 images (example before/after, test before, 3 choices)
- **Output**: 4 normalized transformation vectors for similarity comparison
- **Key Features**:
  - Supports both Vision Transformer (ViT) and ResNet backbones
  - Adaptive architecture selection based on encoder type
  - Configurable embedding dimensions (default: 512)
  - Optional encoder freezing for transfer learning

#### 2. **TransformationEncoder** (ViT-specific)
A novel architecture for ViT models that processes image pairs as unified sequences:

```python
# Key innovations:
- Concatenated patch sequences from both images
- Extended positional embeddings for dual-image input
- Segment embeddings to distinguish between images A and B
- Single CLS token representing the transformation
```

**Architecture Details**:
- Processes two images as a single transformer sequence
- Positional embeddings: `[CLS, img1_patches, img2_patches]`
- Segment embeddings: `{0: CLS, 1: image_A, 2: image_B}`
- Output: CLS token embedding representing the transformation

#### 3. **Projection Head**
A multi-layer projection network applied to all encoder outputs:
```python
nn.Sequential(
    nn.Linear(encoder_dim, embedding_dim),
    nn.ReLU(inplace=True),
    nn.Dropout(0.2),
    nn.Linear(embedding_dim, embedding_dim),
    nn.LayerNorm(embedding_dim)
)
```

### Supported Encoder Backbones

#### **Vision Transformers (Recommended)**
- `vit_small_patch16_224` - Small ViT, 384 features
- `vit_base_patch16_224` - Base ViT, 768 features
- Uses the advanced **TransformationEncoder** architecture
- Processes image pairs as unified sequences

#### **ResNet Models (Legacy)**
- `resnet18` - 512 features
- `resnet50` - 2048 features
- Uses traditional Siamese approach with subtraction-based transformations
- Fallback for comparison with simpler architectures

### Loss Functions

#### 1. **Standard Triplet Loss** (`standard_triplet`)
- **Concept**: Anchor-positive-negative triplet learning
- **Implementation**: Uses `nn.TripletMarginLoss`
- **Parameters**: `margin` (default: 1.0)
- **Use Case**: Strong baseline for metric learning

#### 2. **Contrastive Analogy Loss** (`contrastive`)
- **Concept**: Custom contrastive loss for analogy tasks
- **Implementation**: Margin-based ranking loss
- **Formula**: `max(0, margin - (pos_sim - neg_sim))`
- **Parameters**: `margin` (default: 0.5)
- **Use Case**: Direct optimization for analogy ranking

#### 3. **Softmax Analogy Loss** (`softmax`)
- **Concept**: Cross-entropy over similarity scores
- **Implementation**: Temperature-scaled cosine similarities
- **Formula**: `CrossEntropy(similarities / temperature, correct_idx)`
- **Parameters**: `temperature` (default: 0.07)
- **Use Case**: Probabilistic approach, similar to InfoNCE

### Optimization Setup

#### **Optimizer: AdamW**
- **Differential Learning Rates**:
  - Encoder parameters: `learning_rate × 0.1` (fine-tuning)
  - Projection parameters: `learning_rate` (full learning)
- **Weight Decay**: 1e-4 (default)
- **Rationale**: Slower adaptation of pretrained features, faster learning of task-specific projections

#### **Scheduler: Cosine Annealing**
- **Type**: `CosineAnnealingLR`
- **Schedule**: Smooth decay from initial LR to 0 over training epochs
- **Benefits**: Helps convergence and prevents overfitting in later epochs

### Model Variants & Configuration

#### **Key Hyperparameters**
```python
embedding_dim: int = 512        # Final embedding dimension
freeze_encoder: bool = False    # Whether to freeze backbone
encoder_name: str = "vit_small_patch16_224"  # Backbone architecture
learning_rate: float = 1e-3     # Base learning rate
batch_size: int = 64           # Training batch size
```

#### **Architecture Selection Logic**
```python
if encoder_name.startswith("vit"):
    # Use TransformationEncoder (advanced)
    encoder = TransformationEncoder(encoder_name)
else:
    # Use traditional Siamese approach
    encoder = get_encoder(encoder_name)
```

### Training Strategy

1. **Dual-Phase Learning**: Different learning rates for encoder vs. projection
2. **Cosine Annealing**: Smooth learning rate decay
3. **Normalization**: L2 normalization of final embeddings for cosine similarity
4. **Evaluation**: Argmax over cosine similarities for prediction

### Performance Considerations

- **ViT Models**: More parameters but better transformation modeling
- **ResNet Models**: Faster training, simpler architecture
- **Memory Usage**: ViT processes longer sequences (393 vs 197 tokens)
- **Computational Cost**: TransformationEncoder requires custom positional embeddings

