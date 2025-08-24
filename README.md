# KIVA-ICCV: Visual Analogy Learning

A PyTorch implementation of a Siamese Network for solving visual analogies, designed for the ICCV challenge.

## 🚀 Quick Start

### Prerequisites
```bash
# Install dependencies using uv
uv sync
```

This will automatically install all dependencies from your `pyproject.toml` file.

### Useful commands and basic cli usage

You can start by copying the `.env.example` file to `.env` and filling in the API token and project name.
```bash
cp .env.example .env
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

### Load the data

1. Load the basic competition data running:

```bash
uv run python kiva-iccv/utils/download.py
```

This will create the `data/` directory with the following structure:

**Note:** The `test` directory is not available yet. Will be released on September 1.

```
data/
├── train/
├── validation/
├── test/
├── unit/
├── train.json
├── validation.json
├── test.json
└── unit.json
```

2. Run the following commands to create the subimages that will be used by the model.
```bash
for dataset in train validation test unit; do
    uv run python kiva-iccv/utils/transform.py --dataset $dataset
done
```

3. After this, the `data/` directory will have the following structure:
```
data/
├── train/           # Training images
|   ├── sample1.jpg
|   ├── sample2.jpg
|   └── ...
├── validation/      # Validation images
|   ├── sample1.jpg
|   ├── sample2.jpg
|   └── ...
├── test/           # Test images
|   ├── sample1.jpg
|   ├── sample2.jpg
|   └── ...
├── split_train/           # Split training images
│   ├── sample1_ex_before.jpg
│   ├── sample1_ex_after.jpg
│   ├── sample1_test_before.jpg
│   ├── sample1_choice_a.jpg
│   ├── sample1_choice_b.jpg
│   └── sample1_choice_c.jpg
├── split_validation/      # Split validation images
|   └── ...
├── split_test/           # Split test images
|   └── ...
├── train.json            # Training metadata
├── validation.json       # Validation metadata
└── test.json            # Test metadata
```

### Image Naming Convention
Each sample requires 6 images:
- `{sample_id}_ex_before.jpg` - Example "before" image
- `{sample_id}_ex_after.jpg` - Example "after" image  
- `{sample_id}_test_before.jpg` - Test "before" image
- `{sample_id}_choice_a.jpg` - Choice A image
- `{sample_id}_choice_b.jpg` - Choice B image
- `{sample_id}_choice_c.jpg` - Choice C image
