# train_analogy.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from pydantic import BaseModel
from pathlib import Path
import json
from PIL import Image
from tqdm import tqdm
import os
import random


# --- 1. Configuration ----------------------------------------------------
class Config(BaseModel):
    """Configuration class for hyperparameters and paths."""

    data_dir: str = "./data/split_unit"
    metadata_path: str = "./data/unit.json"

    # Model & Training
    embedding_dim: int = 512
    margin: float = 1.0
    learning_rate: float = 1e-4
    batch_size: int = 4  # Keep it small for demonstration
    epochs: int = 5
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


config = Config()


# --- 2. Custom Dataset ---------------------------------------------------
class VisualAnalogyDataset(Dataset):
    """
    Custom dataset to load visual analogy problems.
    Each sample consists of 6 images and a label indicating the correct choice.
    """

    def __init__(self, data_dir: str, metadata_path: str, transform=None):
        self.root_dir = Path(data_dir)

        # Set default transform if none provided
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transform

        # Load metadata
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.sample_ids = list(self.metadata.keys())

        # Map choice strings to integer indices
        self.label_map = {"(A)": 0, "(B)": 1, "(C)": 2}

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> tuple:
        sample_id = self.sample_ids[idx]

        # Define the 6 image types for each sample
        image_types = [
            "ex_before",
            "ex_after",
            "test_before",
            "choice_a",
            "choice_b",
            "choice_c",
        ]

        # Load all 6 images
        images = []
        for img_type in image_types:
            img_path = self.root_dir / f"{sample_id}_{img_type}.jpg"
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)  # This should now return a tensor
            images.append(image)

        # Get the correct label index (0, 1, or 2)
        correct_choice_str = self.metadata[sample_id]["correct"]
        correct_idx = self.label_map[correct_choice_str]

        return (*images, torch.tensor(correct_idx, dtype=torch.long))


# --- 3. Model Architecture (from your scaffold) ----------------------------
class SiameseAnalogyNetwork(nn.Module):
    """
    A generic Siamese network that compares two transformations.
    It takes four images (pair1_before, pair1_after, pair2_before, pair2_after)
    and outputs a similarity score between their implied transformations.
    """

    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        # 1. SHARED ENCODER
        resnet = models.resnet18(weights="IMAGENET1K_V1")
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        # 2. PROJECTION HEAD
        self.projection = nn.Linear(resnet.fc.in_features, embedding_dim)
        # 3. SIMILARITY METRIC
        self.similarity = nn.CosineSimilarity(dim=1)

    def _get_transformation_vec(
        self, img_before: torch.Tensor, img_after: torch.Tensor
    ) -> torch.Tensor:
        """Encodes two images and returns their difference vector."""
        f_before = self.projection(self.encoder(img_before).flatten(1))
        f_after = self.projection(self.encoder(img_after).flatten(1))
        return f_after - f_before

    def forward(
        self,
        before1: torch.Tensor,
        after1: torch.Tensor,
        before2: torch.Tensor,
        after2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates the similarity between the transformation (before1 -> after1)
        and the transformation (before2 -> after2).
        """
        # --- Step 1: Calculate the two transformation vectors ---
        transformation1 = self._get_transformation_vec(before1, after1)
        transformation2 = self._get_transformation_vec(before2, after2)

        # --- Step 2: Compute and return the similarity between them ---
        # The output is a tensor of shape (batch_size,) with scores from -1 to 1.
        return self.similarity(transformation1, transformation2)


# --- 4. Loss Function (from your scaffold) ---------------------------------
class ContrastiveAnalogyLoss(nn.Module):
    """
    Calculates a contrastive loss based on similarity scores.
    The goal is to ensure:
      sim(positive_pair) > sim(negative_pair_1) + margin
      sim(positive_pair) > sim(negative_pair_2) + margin
    """

    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        sim_positive: torch.Tensor,
        sim_negative1: torch.Tensor,
        sim_negative2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            sim_positive: Similarity score of the correct (anchor, positive) pair.
            sim_negative1: Similarity score of the first incorrect (anchor, negative) pair.
            sim_negative2: Similarity score of the second incorrect (anchor, negative) pair.
        """
        # Loss for the first negative pair
        loss1 = torch.clamp(self.margin - (sim_positive - sim_negative1), min=0)

        # Loss for the second negative pair
        loss2 = torch.clamp(self.margin - (sim_positive - sim_negative2), min=0)

        # Total loss is the sum (or mean) of individual losses
        total_loss = torch.mean(loss1 + loss2)

        return total_loss


if __name__ == "__main__":
    config = Config()  # Using default values
    device = torch.device(config.device)

    dataset = VisualAnalogyDataset(config.data_dir, config.metadata_path)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    # 2. Model, Loss, and Optimizer
    model = SiameseAnalogyNetwork(embedding_dim=config.embedding_dim).to(device)
    criterion = ContrastiveAnalogyLoss(margin=config.margin)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    print(f"\n🚀 Starting training on {device} with generic Siamese model...")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # 3. Training Loop
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.epochs}")

        for batch in progress_bar:
            ex_before, ex_after, test_before, ch_a, ch_b, ch_c, correct_idx = batch

            # --- Prepare data on the correct device ---
            ex_before, ex_after, test_before = (
                ex_before.to(device),
                ex_after.to(device),
                test_before.to(device),
            )
            choices = torch.stack([ch_a, ch_b, ch_c], dim=1).to(
                device
            )  # Shape: (B, 3, C, H, W)
            correct_idx = correct_idx.to(device)

            # --- Dynamically select positive and negative choices for the batch ---
            batch_size = ex_before.shape[0]

            # Positive choice images (the correct answers)
            # gather expects index to have the same number of dims as the input tensor
            positive_choices = choices.gather(
                1,
                correct_idx.view(batch_size, 1, 1, 1, 1).expand(
                    -1, 1, *choices.shape[2:]
                ),
            ).squeeze(1)

            # Negative choice images
            neg_choices_1, neg_choices_2 = [], []
            for i in range(batch_size):
                # Find the indices of the two negative choices
                neg_indices = [j for j in range(3) if j != correct_idx[i]]
                neg_choices_1.append(choices[i, neg_indices[0]])
                neg_choices_2.append(choices[i, neg_indices[1]])

            negative_choices_1 = torch.stack(neg_choices_1)
            negative_choices_2 = torch.stack(neg_choices_2)

            # --- Forward passes ---
            optimizer.zero_grad()

            # 1. POSITIVE PAIR: Compare (example) with (test -> correct_choice)
            sim_positive = model(ex_before, ex_after, test_before, positive_choices)

            # 2. NEGATIVE PAIR 1: Compare (example) with (test -> incorrect_choice_1)
            sim_negative1 = model(ex_before, ex_after, test_before, negative_choices_1)

            # 3. NEGATIVE PAIR 2: Compare (example) with (test -> incorrect_choice_2)
            sim_negative2 = model(ex_before, ex_after, test_before, negative_choices_2)

            # --- Calculate loss ---
            loss = criterion(sim_positive, sim_negative1, sim_negative2)

            # --- Backward pass and optimization ---
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{config.epochs} - Average Loss: {epoch_loss:.4f}\n")

    print("🏁 Training finished.")
