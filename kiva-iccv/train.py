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
import argparse
import os
import random
from typing import Literal


class Config(BaseModel):
    """Configuration class for hyperparameters."""

    # Paths will be provided by the argparser
    data_dir: str
    metadata_path: str

    # Model & Training
    embedding_dim: int = 512
    margin: float = 0.5  # Adjusted margin for contrastive loss
    learning_rate: float = 1e-4
    batch_size: int = 16  # A larger batch size is usually better
    epochs: int = 10
    num_workers: int = 2
    device: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"
    model_save_path: str = "best_analogy_model.pth"


def get_dataset_paths(dataset_keyword: str) -> tuple[str, str]:
    """Returns the data directory and metadata path based on a keyword."""
    base_data_path = "./data"
    mapping = {
        "unit": ("split_unit", "unit.json"),
        "train": ("split_train", "train.json"),
        "validation": ("split_validation", "validation.json"),
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
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        self.sample_ids = list(self.metadata.keys())
        self.label_map = {"(A)": 0, "(B)": 1, "(C)": 2}

    def __len__(self) -> int:
        return len(self.sample_ids)

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
        return (*images, torch.tensor(correct_idx, dtype=torch.long))


class SiameseAnalogyNetwork(nn.Module):
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        resnet = models.resnet18(weights="IMAGENET1K_V1")
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = nn.Linear(resnet.fc.in_features, embedding_dim)
        self.similarity = nn.CosineSimilarity(dim=1)

    def _get_transformation_vec(
        self, img_before: torch.Tensor, img_after: torch.Tensor
    ) -> torch.Tensor:
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
        t1 = self._get_transformation_vec(before1, after1)
        t2 = self._get_transformation_vec(before2, after2)
        return self.similarity(t1, t2)


class ContrastiveAnalogyLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        sim_positive: torch.Tensor,
        sim_negative1: torch.Tensor,
        sim_negative2: torch.Tensor,
    ) -> torch.Tensor:
        loss1 = torch.clamp(self.margin - (sim_positive - sim_negative1), min=0)
        loss2 = torch.clamp(self.margin - (sim_positive - sim_negative2), min=0)
        return torch.mean(loss1 + loss2)


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """
    Evaluates the model on a given dataset and calculates accuracy.
    """
    model.eval()  # Set the model to evaluation mode
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            ex_before, ex_after, test_before, ch_a, ch_b, ch_c, correct_idx = batch

            ex_before, ex_after, test_before = (
                ex_before.to(device),
                ex_after.to(device),
                test_before.to(device),
            )
            ch_a, ch_b, ch_c = ch_a.to(device), ch_b.to(device), ch_c.to(device)
            correct_idx = correct_idx.to(device)

            sim_a = model(ex_before, ex_after, test_before, ch_a)
            sim_b = model(ex_before, ex_after, test_before, ch_b)
            sim_c = model(ex_before, ex_after, test_before, ch_c)

            # Stack similarity scores: shape becomes (batch_size, 3)
            all_sims = torch.stack([sim_a, sim_b, sim_c], dim=1)

            # Get predictions by finding the index of the max similarity score
            predictions = torch.argmax(all_sims, dim=1)

            # --- Update metrics ---
            total_correct += (predictions == correct_idx).sum().item()
            total_samples += ex_before.size(0)

    accuracy = 100 * total_correct / total_samples
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Siamese Network for Visual Analogies."
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_test", action="store_true", help="Whether to run testing on the test set."
    )
    parser.add_argument(
        "--train_on", type=str, default="train", help="Dataset keyword for training."
    )
    parser.add_argument(
        "--validate_on",
        type=str,
        default="validation",
        help="Dataset keyword for validation.",
    )
    parser.add_argument(
        "--test_on", type=str, default="test", help="Dataset keyword for testing."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Training Phase ---
    if args.do_train:
        # 1. Setup train and validation dataloaders
        train_data_dir, train_meta_path = get_dataset_paths(args.train_on)
        train_config = Config(
            data_dir=train_data_dir, metadata_path=train_meta_path, device=str(device)
        )
        train_dataset = VisualAnalogyDataset(
            train_config.data_dir, train_config.metadata_path
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
            num_workers=train_config.num_workers,
        )

        val_data_dir, val_meta_path = get_dataset_paths(args.validate_on)
        val_dataset = VisualAnalogyDataset(val_data_dir, val_meta_path)
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            num_workers=train_config.num_workers,
        )

        # 2. Initialize model, loss, and optimizer
        model = SiameseAnalogyNetwork(embedding_dim=train_config.embedding_dim).to(
            device
        )
        criterion = ContrastiveAnalogyLoss(margin=train_config.margin)
        optimizer = optim.Adam(model.parameters(), lr=train_config.learning_rate)

        print(
            f"\n🚀 Starting training on '{args.train_on}' dataset, validating on '{args.validate_on}'..."
        )
        best_accuracy = 0.0

        # 3. Training Loop
        for epoch in range(train_config.epochs):
            model.train()
            running_loss = 0.0
            progress_bar = tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{train_config.epochs} [Training]"
            )

            for batch in progress_bar:
                ex_before, ex_after, test_before, ch_a, ch_b, ch_c, correct_idx = batch
                ex_before, ex_after, test_before = (
                    ex_before.to(device),
                    ex_after.to(device),
                    test_before.to(device),
                )
                choices = torch.stack([ch_a, ch_b, ch_c], dim=1).to(device)
                correct_idx = correct_idx.to(device)
                batch_size = ex_before.shape[0]

                positive_choices = choices.gather(
                    1,
                    correct_idx.view(batch_size, 1, 1, 1, 1).expand(
                        -1, 1, *choices.shape[2:]
                    ),
                ).squeeze(1)
                neg_choices_1, neg_choices_2 = [], []
                for i in range(batch_size):
                    neg_indices = [j for j in range(3) if j != correct_idx[i]]
                    neg_choices_1.append(choices[i, neg_indices[0]])
                    neg_choices_2.append(choices[i, neg_indices[1]])
                negative_choices_1 = torch.stack(neg_choices_1)
                negative_choices_2 = torch.stack(neg_choices_2)

                optimizer.zero_grad()
                sim_positive = model(ex_before, ex_after, test_before, positive_choices)
                sim_negative1 = model(
                    ex_before, ex_after, test_before, negative_choices_1
                )
                sim_negative2 = model(
                    ex_before, ex_after, test_before, negative_choices_2
                )
                loss = criterion(sim_positive, sim_negative1, sim_negative2)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            avg_train_loss = running_loss / len(train_loader)

            # --- 💡 Validation at the end of each epoch ---
            val_accuracy = evaluate(model, val_loader, device)
            print(
                f"Epoch {epoch + 1}/{train_config.epochs} - Train Loss: {avg_train_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%"
            )

            # --- 💾 Save the best model ---
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(model.state_dict(), train_config.model_save_path)
                print(f"✨ New best model saved with accuracy: {best_accuracy:.2f}%")

        print("🏁 Training finished.")

    # --- Testing Phase ---
    if args.do_test:
        print(f"\n🧪 Starting testing on '{args.test_on}' dataset...")
        test_data_dir, test_meta_path = get_dataset_paths(args.test_on)
        test_config = Config(
            data_dir=test_data_dir, metadata_path=test_meta_path, device=str(device)
        )

        # 1. Initialize a fresh model instance and load the best saved weights
        model = SiameseAnalogyNetwork(embedding_dim=test_config.embedding_dim).to(
            device
        )
        model.load_state_dict(torch.load(test_config.model_save_path))

        # 2. Setup the test dataloader
        test_dataset = VisualAnalogyDataset(
            test_config.data_dir, test_config.metadata_path
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=test_config.batch_size,
            shuffle=False,
            num_workers=test_config.num_workers,
        )

        # 3. Run final evaluation
        test_accuracy = evaluate(model, test_loader, device)
        print(f"\n✅ Final Test Accuracy: {test_accuracy:.2f}%")
