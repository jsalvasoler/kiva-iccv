import json
import os
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def get_dataset_paths(dataset_keyword: str) -> tuple[str, str]:
    """Returns the data directory and metadata path based on a keyword."""
    base_data_path = "./data"
    mapping = {
        "unit": ("split_unit", "unit.json"),
        "train": ("split_train", "train.json"),
        "validation": ("split_validation", "validation.json"),
        "validation_sample": ("split_validation", "validation.json"),
        "test": ("split_test", "test.json"),
    }
    if dataset_keyword not in mapping:
        raise ValueError(f"Invalid dataset keyword '{dataset_keyword}'.")
    split_dir, meta_file = mapping[dataset_keyword]

    data_dir = os.path.join(base_data_path, split_dir)
    metadata_path = os.path.join(base_data_path, meta_file)

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    return data_dir, metadata_path


class VisualAnalogyDataset(Dataset):
    def __init__(self, data_dir: str, metadata_path: str, transform=None):
        self.root_dir = Path(data_dir)
        self.transform = transform
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        self.sample_ids = list(self.metadata.keys())
        self.label_map = {"(A)": 0, "(B)": 1, "(C)": 2}

        print(f"Loaded {len(self.sample_ids)} samples from {metadata_path}")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def sample_validation_set(self, n: int) -> None:
        self.sample_ids = random.sample(self.sample_ids, n)

    def __getitem__(self, idx: int) -> tuple:
        sample_id = self.sample_ids[idx]
        image_types = [
            "ex_before",
            "ex_after",
            "test_before",
            "choice_a",
            "choice_b",
            "choice_c",
        ]
        images = []
        for img_type in image_types:
            img_path = self.root_dir / f"{sample_id}_{img_type}.jpg"
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            images.append(image)

        correct_choice_str = self.metadata[sample_id]["correct"]
        correct_idx = self.label_map[correct_choice_str]
        return (*images, torch.tensor(correct_idx, dtype=torch.long), sample_id)
