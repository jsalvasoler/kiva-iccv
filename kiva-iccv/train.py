import json
import os
import tempfile
from datetime import datetime

import neptune
import torch
import torch.nn as nn
import torch.optim as optim
from config import Config, create_argument_parser, create_config_from_args
from dataset import VisualAnalogyDataset
from loss import (
    ContrastiveAnalogyLoss,
    SoftmaxAnalogyLoss,
    StandardTripletAnalogyLoss,
)
from model import SiameseAnalogyNetwork
from on_the_fly_dataset import OnTheFlyKiVADataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils.evaluate import run_evaluation_analysis

# Set random seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def setup_output_directory_for_training(args, neptune_run) -> str:
    """Setup output directory structure for Neptune runs.

    Returns:
        str: model_save_path
    """

    if args.output_dir:
        raise ValueError(
            "Cannot specify --output_dir when using --do_train. "
            "Output directory is automatically managed during training."
        )

    if not neptune_run:
        output_dir = tempfile.mkdtemp()
    else:
        # Get Neptune run ID
        run_id = neptune_run["sys/id"].fetch()
        # Create output directory structure
        output_dir = f"./output/{run_id}"

    args.output_dir = output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Create models subdirectory
    os.makedirs(f"{output_dir}/models", exist_ok=True)

    # Set model save path
    model_save_path = f"{output_dir}/models/best_model.pth"

    print(f"📁 Output directory created: {output_dir}")
    return model_save_path


def print_experiment_results(
    config: Config,
    args,
    best_accuracy: float,
    final_train_loss: float,
    final_train_accuracy: float,
    num_parameters: int,
    num_trainable_parameters: int,
    experiment_type: str = "train",
    test_accuracy: float = None,
) -> None:
    """Log experiment results to terminal (no file writing)."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 80)
    print(f"EXPERIMENT: {experiment_type.upper()}")
    print(f"TIMESTAMP: {timestamp}")
    print("=" * 80)
    print("\nCONFIGURATION:")

    # Print config fields and their values
    IGNORED_FIELDS = ["device", "data_dir", "metadata_path", "neptune_api_token", "neptune_project"]
    for field_name, field in config.model_dump().items():
        if field_name not in IGNORED_FIELDS:
            print(f"  {field_name:<20} {field}")

    # Print number of model parameters
    print(f"  {'Model parameters':<20} {num_parameters:,}")
    print(f"  {'Trainable parameters':<20} {num_trainable_parameters:,}")

    # Print dataset info from args if available
    if hasattr(args, "train_on"):
        print(f"  {'Dataset (train)':<20} {args.train_on}")
    if hasattr(args, "validate_on"):
        print(f"  {'Dataset (validate)':<20} {args.validate_on}")
    if experiment_type == "test" and hasattr(args, "test_on"):
        print(f"  {'Dataset (test)':<20} {args.test_on}")
    if hasattr(args, "neptune_url"):
        print(f"  {'Neptune URL':<20} {args.neptune_url}")

    print("\nRESULTS:")
    if experiment_type == "train":
        print(f"  Final train loss:      {final_train_loss:.4f}")
        print(f"  Final train accuracy:  {final_train_accuracy:.2f}%")
        print(f"  Best validation acc:   {best_accuracy:.2f}%")
    elif experiment_type == "test":
        print(f"  Test accuracy:         {test_accuracy:.2f}%")
    print("\n" + ("=" * 80))


def dataset_factory(args, config: Config) -> Dataset:
    distribution = {
        "kiva-Counting": 64,
        "kiva-Reflect": 32,
        "kiva-Resizing": 32,
        "kiva-Rotation": 48,
        "kiva-functions-Counting": 128,
        "kiva-functions-Reflect": 32,
        "kiva-functions-Resizing": 96,
        "kiva-functions-Rotation": 112,
        "kiva-functions-compositionality-Counting,Reflect": 256,
        "kiva-functions-compositionality-Counting,Resizing": 768,
        "kiva-functions-compositionality-Counting,Rotation": 896,
        "kiva-functions-compositionality-Reflect,Resizing": 192,
        "kiva-functions-compositionality-Resizing,Rotation": 96,
    }
    if config.task == "train" and config.use_otf:
        return OnTheFlyKiVADataset(
            objects_dir="./data/KiVA/untransformed objects",
            distribution_config=distribution,
            epoch_length=config.oft_epoch_length,
        )

    return VisualAnalogyDataset(config.data_dir, config.metadata_path)


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
        submission_file = f"{args.output_dir}/submission.json"
        with open(submission_file, "w") as f:
            json.dump(predictions, f, indent=4)
        print(f"📄 Submission file saved: {submission_file}")

    return accuracy


def init_neptune(config: Config, args, experiment_type: str):
    """Initialize Neptune logging."""
    neptune_run = None
    if not config.use_neptune:
        return None

    if not config.neptune_api_token:
        raise ValueError(
            "⚠️  Neptune API token not provided. Please provide it with the --neptune_api_token "
            "flag or set the NEPTUNE_API_TOKEN environment variable."
        )

    neptune_run = neptune.init_run(
        project=config.neptune_project,
        api_token=config.neptune_api_token,
        name=f"kiva-iccv-{experiment_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        tags=[tag for tag, flag in (("train", args.do_train), ("test", args.do_test)) if flag],
    )

    # Log configuration parameters
    neptune_run["parameters"] = {
        "loss_type": config.loss_type,
        "margin": config.margin,
        "embedding_dim": config.embedding_dim,
        "transformation_net": config.transformation_net,
        "freeze_encoder": config.freeze_encoder,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "num_workers": config.num_workers,
        "use_otf": config.use_otf,
    }

    neptune_url = neptune_run.get_url()
    print(f"🔗 Neptune logging enabled: {neptune_url}")
    args.neptune_url = neptune_url

    return neptune_run


def train(args) -> str | None:
    """Training function that handles the complete training loop."""
    # 1. Setup train and validation dataloaders
    train_config = create_config_from_args(args, for_task="train")
    train_dataset = dataset_factory(args, train_config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
    )

    val_config = create_config_from_args(args, for_task="validation")
    val_dataset = dataset_factory(args, val_config)

    if args.validate_on == "validation_sample":
        val_dataset.sample_validation_set(int(len(val_dataset) * 0.1))

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_config.batch_size,
        shuffle=False,
        num_workers=val_config.num_workers,
    )

    # 2. Initialize Neptune logging
    neptune_run = init_neptune(train_config, args, "train")

    # Setup output directory if Neptune is working
    model_save_path = setup_output_directory_for_training(args, neptune_run)

    # Initialize model
    model = SiameseAnalogyNetwork(
        embedding_dim=train_config.embedding_dim,
        freeze_encoder=train_config.freeze_encoder,
        transformation_net=train_config.transformation_net,
    ).to(device)

    # Initialize loss
    criterion = {
        "standard_triplet": StandardTripletAnalogyLoss(margin=train_config.margin),
        "contrastive": ContrastiveAnalogyLoss(margin=train_config.margin),
        "softmax": SoftmaxAnalogyLoss(temperature=train_config.temperature),
    }[train_config.loss_type]

    # Optimizer
    optimizer = optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": train_config.learning_rate * 0.1},
            {
                "params": list(model.projection.parameters())
                + list(model.transformation_net.parameters()),
                "lr": train_config.learning_rate,
            },
        ],
        weight_decay=train_config.weight_decay,
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config.epochs)

    print(
        f"\n🚀 Starting training on '{args.train_on}' dataset,"
        f" validating on '{args.validate_on}'..."
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {total_params}")
    print(f"Number of trainable parameters: {trainable_params}")
    best_accuracy = 0.0

    if train_config.epochs == 0:
        print("Skipping training since epochs is 0.")
        return None

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
            scheduler.step()
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

        # Log to Neptune if enabled
        if neptune_run:
            neptune_run["training/epoch"].append(epoch + 1)
            neptune_run["training/train_loss"].append(avg_train_loss)
            neptune_run["training/train_accuracy"].append(train_accuracy)
            neptune_run["training/val_accuracy"].append(val_accuracy)
            neptune_run["training/learning_rate"].append(train_config.learning_rate)

        # --- 💾 Save the best model (only if Neptune is working) ---
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f"✨ New best model saved with accuracy: {best_accuracy:.2f}%")

    print("🏁 Training finished.")

    # Log final results to Neptune if enabled
    if neptune_run:
        neptune_run["training/final_train_loss"] = avg_train_loss
        neptune_run["training/final_train_accuracy"] = train_accuracy
        neptune_run["training/best_val_accuracy"] = best_accuracy
        neptune_run["training/total_parameters"] = sum(p.numel() for p in model.parameters())
        neptune_run["training/model_save_path"] = model_save_path
        neptune_run_id = neptune_run["sys/id"].fetch()
        neptune_run.stop()
        print("🔗 Neptune run completed and stopped")
    else:
        neptune_run_id = None

    # Log experiment results
    print_experiment_results(
        config=train_config,
        args=args,
        best_accuracy=best_accuracy,
        final_train_loss=avg_train_loss,
        final_train_accuracy=train_accuracy,
        num_parameters=sum(p.numel() for p in model.parameters()),
        num_trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        experiment_type="train",
    )

    return neptune_run_id


def test(args, neptune_run_id: str | None) -> None:
    """Testing function that evaluates the trained model."""
    print(f"\n🧪 Starting testing on '{args.test_on}' dataset...")
    test_config = create_config_from_args(args, for_task="test")

    # Initialize Neptune logging
    if neptune_run_id is None:
        neptune_run = init_neptune(test_config, args, "test")
    else:
        # continue the existing Neptune run
        neptune_run = neptune.init_run(with_id=neptune_run_id)

    # 1. Initialize a fresh model instance and load the best saved weights
    model = SiameseAnalogyNetwork(
        embedding_dim=test_config.embedding_dim,
        transformation_net=test_config.transformation_net,
    ).to(device)

    # Load model from output directory if available
    output_dir = args.output_dir
    model_path = f"{output_dir}/models/best_model.pth"
    if output_dir:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found in output directory: {model_path}")
    else:
        raise ValueError(
            "No output directory available. "
            "If only testing, please specify an output directory with the --output_dir flag. "
            "If also training, this is automatically managed."
        )

    model.load_state_dict(torch.load(model_path))

    # 2. Setup the test dataloader
    test_dataset = dataset_factory(args, test_config)
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_config.batch_size,
        shuffle=False,
        num_workers=test_config.num_workers,
    )

    # 3. Run final evaluation and generate submission file
    test_accuracy = evaluate(
        model, test_loader, device, save_predictions=neptune_run is not None, dataset=test_dataset
    )
    print(f"\n✅ Final Test Accuracy: {test_accuracy:.2f}%")

    # 4. Run evaluation analysis if submission file was generated
    if neptune_run is not None:
        run_evaluation_analysis(args, args.test_on)

    # Log test results to Neptune if enabled
    if neptune_run:
        neptune_run["testing/test_accuracy"] = test_accuracy
        neptune_run["testing/total_parameters"] = sum(p.numel() for p in model.parameters())
        neptune_run.stop()
        print("🔗 Neptune test run completed and stopped")

    # Log test results
    print_experiment_results(
        config=test_config,
        args=args,
        best_accuracy=0.0,  # Not applicable for test
        final_train_loss=0.0,  # Not applicable for test
        final_train_accuracy=0.0,  # Not applicable for test
        num_parameters=sum(p.numel() for p in model.parameters()),
        num_trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
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
        neptune_run_id = train(args)
    else:
        neptune_run_id = None

    # --- Testing Phase ---
    if args.do_test:
        test(args, neptune_run_id)
