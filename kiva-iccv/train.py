# train_analogy.py

import json
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from config import Config, create_argument_parser, create_config_from_args
from dataset import VisualAnalogyDataset, get_dataset_paths
from loss import ContrastiveAnalogyLoss, StandardTripletAnalogyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def log_experiment_results(
    config: Config,
    args,
    best_accuracy: float,
    final_train_loss: float,
    final_train_accuracy: float,
    num_parameters: int,
    experiment_type: str = "train",
    test_accuracy: float = None,
) -> None:
    """Log experiment results to results.txt"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    results_file = "results.txt"

    # Create header if file doesn't exist
    if not os.path.exists(results_file):
        with open(results_file, "w") as f:
            f.write("=== EXPERIMENT RESULTS LOG ===\n\n")

    with open(results_file, "a") as f:
        f.write(f"{'=' * 80}\n")
        f.write(f"EXPERIMENT: {experiment_type.upper()}\n")
        f.write(f"TIMESTAMP: {timestamp}\n")
        f.write(f"{'=' * 80}\n\n")

        # Configuration
        f.write("CONFIGURATION:\n")
        f.write(f"  Dataset (train):     {args.train_on}\n")
        f.write(f"  Dataset (validate):  {args.validate_on}\n")
        if experiment_type == "test":
            f.write(f"  Dataset (test):      {args.test_on}\n")
        f.write(f"  Loss type:           {config.loss_type}\n")
        f.write(f"  Margin:              {config.margin}\n")
        f.write(f"  Embedding dim:       {config.embedding_dim}\n")
        f.write(f"  Transformation net:  {config.transformation_net}\n")
        f.write(f"  Freeze encoder:      {config.freeze_encoder}\n")
        f.write(f"  Learning rate:       {config.learning_rate}\n")
        f.write(f"  Batch size:          {config.batch_size}\n")
        f.write(f"  Epochs:              {config.epochs}\n")
        f.write(f"  Num workers:         {config.num_workers}\n")
        f.write(f"  Model parameters:    {num_parameters:,}\n")
        f.write(f"  Device:              {config.device}\n")
        f.write("\n")

        # Results
        f.write("RESULTS:\n")
        if experiment_type == "train":
            f.write(f"  Final train loss:      {final_train_loss:.4f}\n")
            f.write(f"  Final train accuracy:  {final_train_accuracy:.2f}%\n")
            f.write(f"  Best validation acc:   {best_accuracy:.2f}%\n")
        elif experiment_type == "test":
            f.write(f"  Test accuracy:         {test_accuracy:.2f}%\n")
        f.write("\n")

        # Model save path
        f.write(f"MODEL SAVED TO: {config.model_save_path}\n")
        f.write(f"{'=' * 80}\n\n")

    print(f"📝 Experiment results logged to {results_file}")


class SiameseAnalogyNetwork(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512,
        freeze_encoder: bool = False,
        transformation_net: bool = False,
    ):
        super().__init__()
        resnet = models.resnet18(weights="IMAGENET1K_V1")
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.projection = nn.Linear(resnet.fc.in_features, embedding_dim)

        self.transformation_net = (
            nn.Sequential(
                nn.Linear(
                    embedding_dim * 2, embedding_dim
                ),  # Input is a concatenation of two embeddings
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim),
            )
            if transformation_net
            else None
        )

    def forward(
        self,
        ex_before: torch.Tensor,
        ex_after: torch.Tensor,
        test_before: torch.Tensor,
        choice_a: torch.Tensor,
        choice_b: torch.Tensor,
        choice_c: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        # Determine the batch size from one of the tensors
        batch_size = ex_before.size(0)

        # Reshape all tensors into a single batch, preserving the C, H, W dimensions
        # and stacking along a new dimension (dim=0)
        all_tensors = torch.cat(
            [ex_before, ex_after, test_before, choice_a, choice_b, choice_c], dim=0
        )

        # Process the combined batch
        all_embeddings = self.projection(self.encoder(all_tensors).flatten(1))

        # Split the embeddings back based on the batch size
        t_ex_before = all_embeddings[0 * batch_size : 1 * batch_size]
        t_ex_after = all_embeddings[1 * batch_size : 2 * batch_size]
        t_test_before = all_embeddings[2 * batch_size : 3 * batch_size]
        t_choice_a = all_embeddings[3 * batch_size : 4 * batch_size]
        t_choice_b = all_embeddings[4 * batch_size : 5 * batch_size]
        t_choice_c = all_embeddings[5 * batch_size : 6 * batch_size]

        # Calculate the transformation vectors
        if self.transformation_net:
            t_example = self.transformation_net(torch.cat([t_ex_before, t_ex_after], dim=1))
            t_choice_a_vec = self.transformation_net(torch.cat([t_test_before, t_choice_a], dim=1))
            t_choice_b_vec = self.transformation_net(torch.cat([t_test_before, t_choice_b], dim=1))
            t_choice_c_vec = self.transformation_net(torch.cat([t_test_before, t_choice_c], dim=1))
        else:
            t_example = t_ex_after - t_ex_before
            t_choice_a_vec = t_choice_a - t_test_before
            t_choice_b_vec = t_choice_b - t_test_before
            t_choice_c_vec = t_choice_c - t_test_before

        return t_example, t_choice_a_vec, t_choice_b_vec, t_choice_c_vec


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_predictions: bool = False,
    dataset: VisualAnalogyDataset = None,
) -> float:
    """
    Evaluates the model on a given dataset and calculates accuracy.
    Optionally saves predictions to a submission file.
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  -- Evaluating --  "):
            (
                ex_before,
                ex_after,
                test_before,
                ch_a,
                ch_b,
                ch_c,
                correct_idx,
                sample_id,
            ) = batch

            ex_before, ex_after, test_before, ch_a, ch_b, ch_c, correct_idx = (
                ex_before.to(device),
                ex_after.to(device),
                test_before.to(device),
                ch_a.to(device),
                ch_b.to(device),
                ch_c.to(device),
                correct_idx.to(device),
            )

            t_example, t_choice_a, t_choice_b, t_choice_c = model(
                ex_before, ex_after, test_before, ch_a, ch_b, ch_c
            )

            # Compute similarities between example transformation and each choice transformation
            sim_a = torch.cosine_similarity(t_example, t_choice_a, dim=1)
            sim_b = torch.cosine_similarity(t_example, t_choice_b, dim=1)
            sim_c = torch.cosine_similarity(t_example, t_choice_c, dim=1)

            # Stack similarity scores: shape becomes (batch_size, 3)
            all_sims = torch.stack([sim_a, sim_b, sim_c], dim=1)

            # Get predictions by finding the index of the max similarity score
            predictions_batch = torch.argmax(all_sims, dim=1)

            # --- Update metrics ---
            total_correct += (predictions_batch == correct_idx).sum().item()
            total_samples += ex_before.size(0)

            # --- Store predictions for submission if requested ---
            if save_predictions and dataset:
                for i, pred_idx in enumerate(predictions_batch):
                    choice_map = {0: "(A)", 1: "(B)", 2: "(C)"}
                    predicted_choice = choice_map[pred_idx.item()]
                    # Use the sample_id from the batch
                    batch_sample_id = sample_id[i]
                    predictions.append({"id": batch_sample_id, "answer": predicted_choice})

    accuracy = 100 * total_correct / total_samples

    # Save predictions if requested
    if save_predictions and predictions:
        submission_file = "submission.json"
        with open(submission_file, "w") as f:
            json.dump(predictions, f, indent=4)
        print(f"📄 Submission file saved: {submission_file}")

    return accuracy


def train(args, device: torch.device) -> None:
    """Training function that handles the complete training loop."""
    # 1. Setup train and validation dataloaders
    train_data_dir, train_meta_path = get_dataset_paths(args.train_on)
    train_config = create_config_from_args(args, train_data_dir, train_meta_path)
    train_config.device = str(device)
    train_dataset = VisualAnalogyDataset(train_config.data_dir, train_config.metadata_path)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
    )

    val_data_dir, val_meta_path = get_dataset_paths(args.validate_on)
    val_dataset = VisualAnalogyDataset(val_data_dir, val_meta_path)

    if args.validate_on == "validation_sample":
        val_dataset.sample_validation_set(int(len(train_dataset) * 0.1))

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
    )

    # 2. Initialize model, loss, and optimizer
    model = SiameseAnalogyNetwork(
        embedding_dim=train_config.embedding_dim,
        freeze_encoder=train_config.freeze_encoder,
        transformation_net=train_config.transformation_net,
    ).to(device)
    if train_config.loss_type == "standard_triplet":
        criterion = StandardTripletAnalogyLoss(margin=train_config.margin)
    elif train_config.loss_type == "contrastive":
        criterion = ContrastiveAnalogyLoss(margin=train_config.margin)
    optimizer = optim.Adam(model.parameters(), lr=train_config.learning_rate)

    print(
        f"\n🚀 Starting training on '{args.train_on}' dataset,"
        f" validating on '{args.validate_on}'..."
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {total_params}")
    print(f"Number of trainable parameters: {trainable_params}")
    best_accuracy = 0.0

    # 3. Training Loop
    for epoch in range(train_config.epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{train_config.epochs} [Training]"
        )

        for batch in progress_bar:
            ex_before, ex_after, test_before, ch_a, ch_b, ch_c, correct_idx, _ = batch
            ex_before = ex_before.to(device)
            ex_after = ex_after.to(device)
            test_before = test_before.to(device)
            ch_a = ch_a.to(device)
            ch_b = ch_b.to(device)
            ch_c = ch_c.to(device)
            correct_idx = correct_idx.to(device)

            optimizer.zero_grad()
            model_outputs = model(ex_before, ex_after, test_before, ch_a, ch_b, ch_c)
            loss = criterion(model_outputs, correct_idx)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate training accuracy on-the-fly
            with torch.no_grad():
                t_example, t_choice_a, t_choice_b, t_choice_c = model_outputs
                sim_a = torch.cosine_similarity(t_example, t_choice_a, dim=1)
                sim_b = torch.cosine_similarity(t_example, t_choice_b, dim=1)
                sim_c = torch.cosine_similarity(t_example, t_choice_c, dim=1)
                all_sims = torch.stack([sim_a, sim_b, sim_c], dim=1)
                predictions = torch.argmax(all_sims, dim=1)
                train_correct += (predictions == correct_idx).sum().item()
                train_total += ex_before.size(0)

            current_train_acc = 100 * train_correct / train_total if train_total > 0 else 0
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{current_train_acc:.1f}%")

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        # --- 💡 Validation at the end of each epoch ---
        val_accuracy = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch + 1}/{train_config.epochs}:"
            f" Train Loss: {avg_train_loss:.4f}"
            f" | Train Accuracy: {train_accuracy:.2f}%"
            f" | Validation Accuracy: {val_accuracy:.2f}%"
        )

        # --- 💾 Save the best model ---
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            os.makedirs(os.path.dirname(train_config.model_save_path), exist_ok=True)
            torch.save(model.state_dict(), train_config.model_save_path)
            print(f"✨ New best model saved with accuracy: {best_accuracy:.2f}%")

    print("🏁 Training finished.")

    # Log experiment results
    log_experiment_results(
        config=train_config,
        args=args,
        best_accuracy=best_accuracy,
        final_train_loss=avg_train_loss,
        final_train_accuracy=train_accuracy,
        num_parameters=sum(p.numel() for p in model.parameters()),
        experiment_type="train",
    )


def test(args, device: torch.device) -> None:
    """Testing function that evaluates the trained model."""
    print(f"\n🧪 Starting testing on '{args.test_on}' dataset...")
    test_data_dir, test_meta_path = get_dataset_paths(args.test_on)
    test_config = create_config_from_args(args, test_data_dir, test_meta_path)
    test_config.device = str(device)

    # 1. Initialize a fresh model instance and load the best saved weights
    model = SiameseAnalogyNetwork(
        embedding_dim=test_config.embedding_dim,
        transformation_net=test_config.transformation_net,
    ).to(device)
    model.load_state_dict(torch.load(test_config.model_save_path))

    # 2. Setup the test dataloader
    test_dataset = VisualAnalogyDataset(test_config.data_dir, test_config.metadata_path)
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_config.batch_size,
        shuffle=False,
        num_workers=test_config.num_workers,
    )

    # 3. Run final evaluation and generate submission file
    test_accuracy = evaluate(
        model, test_loader, device, save_predictions=True, dataset=test_dataset
    )
    print(f"\n✅ Final Test Accuracy: {test_accuracy:.2f}%")

    # Log test results
    log_experiment_results(
        config=test_config,
        args=args,
        best_accuracy=0.0,  # Not applicable for test
        final_train_loss=0.0,  # Not applicable for test
        final_train_accuracy=0.0,  # Not applicable for test
        num_parameters=sum(p.numel() for p in model.parameters()),
        experiment_type="test",
        test_accuracy=test_accuracy,
    )


# Argument parser setup
parser = create_argument_parser()


if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Training Phase ---
    if args.do_train:
        train(args, device)

    # --- Testing Phase ---
    if args.do_test:
        test(args, device)
