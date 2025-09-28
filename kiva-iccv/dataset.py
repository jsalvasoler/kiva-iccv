import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class VisualAnalogyDataset(Dataset):
    def __init__(self, data_dir: str, metadata_path: str = None, transform=None):
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

        # Handle case where no metadata is provided (test set without labels)
        if metadata_path is None:
            self.metadata = None
            # Infer sample IDs from image files in the directory
            self.sample_ids = self._infer_sample_ids_from_images()
            print(
                f"Loaded {len(self.sample_ids)} samples from {data_dir} (no metadata - test mode)"
            )
        else:
            with open(metadata_path) as f:
                self.metadata = json.load(f)
            self.sample_ids = list(self.metadata.keys())
            print(f"Loaded {len(self.sample_ids)} samples from {metadata_path}")

        self.label_map = {"(A)": 0, "(B)": 1, "(C)": 2}

    def __len__(self) -> int:
        return len(self.sample_ids)

    def labels_available(self) -> bool:
        """Return True if labels/metadata are available, False otherwise."""
        return self.metadata is not None

    def _infer_sample_ids_from_images(self) -> list[str]:
        """Infer sample IDs from image files in the directory when no metadata is available."""
        # Look for files with pattern {sample_id}_ex_before.jpg
        ex_before_files = list(self.root_dir.glob("*_ex_before.jpg"))
        sample_ids = []
        for file_path in ex_before_files:
            # Extract sample_id from filename (remove _ex_before.jpg suffix)
            sample_id = file_path.stem.replace("_ex_before", "")
            sample_ids.append(sample_id)
        return sorted(sample_ids)

    def sample_validation_set(self, n: int) -> None:
        if not self.labels_available():
            raise ValueError("Cannot sample validation set from dataset without labels")
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

        # Handle case where no labels are available
        if self.labels_available():
            correct_choice_str = self.metadata[sample_id]["correct"]
            correct_idx = self.label_map[correct_choice_str]
            return (*images, torch.tensor(correct_idx, dtype=torch.long), sample_id)
        else:
            # Return dummy label (-1) when no labels available
            return (*images, torch.tensor(-1, dtype=torch.long), sample_id)
