import json
import os
import shutil
import tempfile
import warnings
from datetime import datetime

import neptune
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from config import (
    Config,
    create_argument_parser,
)
from dataset import VisualAnalogyDataset
from loss import (
    ContrastiveAnalogyLoss,
    SoftmaxAnalogyLoss,
    StandardTripletAnalogyLoss,
)
from model import SiameseAnalogyNetwork
from on_the_fly_dataset import OnTheFlyKiVADataset
from torch.amp import GradScaler, autocast
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


def extract_neptune_run_id(resume_arg: str) -> str | None:
    """Extract Neptune run ID from resume argument.

    Args:
        resume_arg: Either a Neptune run ID (e.g., 'CLS-123') or output directory path

    Returns:
        Neptune run ID if found, None otherwise
    """
    if not resume_arg:
        return None

    # If it looks like a Neptune run ID (format: ABC-123)
    if len(resume_arg.split("-")) == 2 and resume_arg.replace("-", "").replace("_", "").isalnum():
        return resume_arg

    # If it's a path, try to extract run ID from directory name
    if os.path.isdir(resume_arg):
        # Extract from ./output/CLS-123 format
        dir_name = os.path.basename(resume_arg.rstrip("/"))
        if len(dir_name.split("-")) == 2 and dir_name.replace("-", "").replace("_", "").isalnum():
            return dir_name

    return None


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler | None = None,
) -> tuple[int, float, float, float]:
    """Load checkpoint and restore training state.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into

    Returns:
        Tuple of (start_epoch, best_accuracy, train_accuracy, train_loss)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"🔄 Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Load scaler state if available and scaler is provided
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    # Restore random states for reproducibility
    torch.set_rng_state(checkpoint["random_state"])
    if torch.cuda.is_available() and "cuda_random_state" in checkpoint:
        torch.cuda.set_rng_state(checkpoint["cuda_random_state"])

    start_epoch = checkpoint["epoch"]
    best_accuracy = checkpoint["best_accuracy"]
    train_accuracy = checkpoint.get("train_accuracy", 0.0)
    train_loss = checkpoint.get("train_loss", 0.0)

    print("✅ Checkpoint loaded successfully!")
    print(f"   Resuming from epoch: {start_epoch}")
    print(f"   Best accuracy so far: {best_accuracy:.2f}%")

    return start_epoch, best_accuracy, train_accuracy, train_loss


def load_model_from_checkpoint(checkpoint_path: str, model: torch.nn.Module) -> None:
    """Load only model weights from checkpoint (for testing).

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"🔄 Loading model from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    print("✅ Model loaded from checkpoint")


def setup_output_directory_for_training(args, train_config, neptune_run) -> None:
    """Setup output directory structure for Neptune runs."""

    # Allow output_dir when resuming training
    if args.output_dir and not args.resume:
        raise ValueError(
            "Cannot specify --output_dir when using --do_train (unless resuming). "
            "Output directory is automatically managed during training."
        )

    # Use provided output_dir when resuming, otherwise create new one
    if args.resume and args.output_dir:
        output_dir = args.output_dir
    elif not neptune_run:
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

    # Save config to output directory
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(train_config.model_dump(), f, indent=4)

    print(f"📁 Output directory created: {output_dir}")


def print_experiment_results(
    config: Config,
    args,
    best_accuracy: float,
    final_train_loss: float,
    final_train_accuracy: float,
    num_parameters: int,
    num_trainable_parameters: int,
    experiment_type: str = "train",
    test_accuracy: float | None = None,
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
        x = f"{test_accuracy:.2f}%" if test_accuracy is not None else "N/A"
        print(f"  Test accuracy:         {x}")
    print("\n" + ("=" * 80))


def dataset_factory(args, config: Config) -> Dataset:
    if config.task == "train" and config.use_otf:
        # Load custom distribution config from YAML if provided
        distribution_config = None
        if args.distribution_config:
            if not os.path.exists(args.distribution_config):
                raise FileNotFoundError(
                    f"Distribution config file not found: {args.distribution_config}"
                )
            with open(args.distribution_config) as f:
                distribution_config = yaml.safe_load(f)
                print(f"📋 Loaded custom distribution config from: {args.distribution_config}")
        else:
            print("📋 Using default distribution config")

        return OnTheFlyKiVADataset(
            objects_dir="./data/KiVA/untransformed objects",
            distribution_config=distribution_config,
            epoch_length=config.otf_epoch_length,
        )

    return VisualAnalogyDataset(config.data_dir, config.metadata_path)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_predictions: bool = False,
    dataset: VisualAnalogyDataset = None,
    test_on: str = None,
) -> float | None:
    """
    Evaluates the model on a given dataset and calculates accuracy.
    Optionally saves predictions to a submission file.
    Returns None if no labels are available (test set without ground truth).
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    predictions = []
    has_labels = dataset is not None and dataset.labels_available()

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

            # Use autocast for evaluation if CUDA is available (for consistency)
            if device.type == "cuda":
                with autocast("cuda"):
                    t_example, t_choice_a, t_choice_b, t_choice_c = model(
                        ex_before, ex_after, test_before, ch_a, ch_b, ch_c
                    )
            else:
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

            # --- Update metrics only if labels are available ---
            if has_labels:
                total_correct += (predictions_batch == correct_idx).sum().item()
            total_samples += ex_before.size(0)

            # --- Store predictions for submission if requested ---
            if save_predictions:
                for i, pred_idx in enumerate(predictions_batch):
                    choice_map = {0: "(A)", 1: "(B)", 2: "(C)"}
                    predicted_choice = choice_map[pred_idx.item()]
                    # Use the sample_id from the batch
                    batch_sample_id = sample_id[i]
                    predictions.append({"id": batch_sample_id, "answer": predicted_choice})

    # Calculate accuracy only if labels are available
    accuracy = 100 * total_correct / total_samples if has_labels else None

    # Save predictions if requested
    if save_predictions and predictions:
        # Use test_on-specific filename if provided, otherwise default to submission.json
        if test_on:
            submission_file = f"{args.output_dir}/submission_{test_on}.json"
        else:
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

    # For testing, Neptune is optional - don't raise error if token missing
    if not config.neptune_api_token:
        if experiment_type == "test":
            print(
                "⚠️  Neptune API token not provided. Testing will proceed without Neptune logging."
            )
            return None
        else:
            raise ValueError(
                "⚠️  Neptune API token not provided. Please provide it with the --neptune_api_token "
                "flag or set the NEPTUNE_API_TOKEN environment variable."
            )

    # Resume existing Neptune run if specified
    if args.resume and experiment_type == "train":
        # Try to extract Neptune run ID from resume argument
        resume_id = extract_neptune_run_id(args.resume)
        if resume_id:
            try:
                neptune_run = neptune.init_run(
                    project=config.neptune_project,
                    api_token=config.neptune_api_token,
                    with_id=resume_id,
                )
                print(f"🔄 Resuming Neptune run: {resume_id}")
            except Exception as e:
                print(f"⚠️  Could not resume Neptune run {resume_id}: {e}")
                print("Creating new Neptune run instead...")
                neptune_run = None

    # Create new Neptune run if not resuming or resume failed
    if not neptune_run:
        neptune_run = neptune.init_run(
            project=config.neptune_project,
            api_token=config.neptune_api_token,
            name=f"kiva-iccv-{experiment_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            tags=[tag for tag, flag in (("train", args.do_train), ("test", args.do_test)) if flag]
            + [config.encoder_name, config.loss_type],
        )

    # Log configuration parameters
    neptune_run["parameters"] = config.model_dump()

    neptune_url = neptune_run.get_url()
    print(f"🔗 Neptune logging enabled: {neptune_url}")
    args.neptune_url = neptune_url

    return neptune_run


def train(args) -> str | None:
    """Training function that handles the complete training loop."""
    # 1. Setup train and validation dataloaders
    train_config = Config.from_args(args, for_task="train")
    train_dataset = dataset_factory(args, train_config)

    # Check if training dataset has labels
    if hasattr(train_dataset, "labels_available") and not train_dataset.labels_available():
        raise ValueError(
            "Cannot train on a dataset without labels. "
            "Training requires ground truth labels for supervision. "
            "Please provide a dataset with metadata/labels for training."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=True,
        drop_last=args.use_otf,
        persistent_workers=True,
    )

    val_config = Config.from_args(args, for_task="validation")
    val_dataset = dataset_factory(args, val_config)

    if args.validate_on == "validation_sample":
        val_dataset.sample_validation_set(int(len(val_dataset) * 0.1))

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_config.batch_size,
        shuffle=False,
        num_workers=val_config.num_workers,
    )

    # Initialize Neptune logging
    neptune_run = init_neptune(train_config, args, "train")

    # Setup output directory if Neptune is working
    setup_output_directory_for_training(args, train_config, neptune_run)

    # Initialize model
    model = SiameseAnalogyNetwork(
        embedding_dim=train_config.embedding_dim,
        freeze_encoder=train_config.freeze_encoder,
        encoder_name=train_config.encoder_name,
    ).to(device)

    # Initialize loss
    if train_config.loss_type == "standard_triplet":
        criterion = StandardTripletAnalogyLoss(margin=train_config.margin)
    elif train_config.loss_type == "contrastive":
        criterion = ContrastiveAnalogyLoss(margin=train_config.margin)
    elif train_config.loss_type == "softmax":
        criterion = SoftmaxAnalogyLoss(temperature=train_config.temperature)
    else:
        raise ValueError(f"Unknown loss type: {train_config.loss_type}")

    # Optimizer
    optimizer = optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": train_config.learning_rate * 0.1},
            {"params": model.projection.parameters(), "lr": train_config.learning_rate},
        ],
        weight_decay=train_config.weight_decay,
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config.epochs)

    # Mixed precision scaler
    scaler = (
        GradScaler("cuda") if train_config.use_mixed_precision and device.type == "cuda" else None
    )

    print(
        f"\n🚀 Starting training on '{args.train_on}' dataset,"
        f" validating on '{args.validate_on}'..."
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {total_params}")
    print(f"Number of trainable parameters: {trainable_params}")

    # Print mixed precision status
    if scaler is not None:
        print("⚡ Mixed precision training enabled (torch.cuda.amp)")
    else:
        print("📊 Using standard precision training")

    # Print gradient accumulation info
    effective_batch_size = train_config.batch_size * train_config.gradient_accumulation_steps
    if train_config.gradient_accumulation_steps > 1:
        print(f"🔄 Gradient accumulation enabled: {train_config.gradient_accumulation_steps} steps")
        print(
            f"   Batch size: {train_config.batch_size}, "
            f"Effective batch size: {effective_batch_size}"
        )
    else:
        print(f"   Batch size: {train_config.batch_size}")
    best_accuracy = 0.0
    start_epoch = 0

    # Handle resume training
    if args.resume:
        # Determine output directory for resume
        if not args.output_dir:
            # Try to infer from resume argument
            if os.path.isdir(args.resume):
                args.output_dir = args.resume
            else:
                # Assume it's a Neptune run ID
                args.output_dir = f"./output/{args.resume}"

        # Load checkpoint if it exists (prefer last, fallback to best)
        checkpoint_last_path = f"{args.output_dir}/models/checkpoint_last.pth"
        checkpoint_best_path = f"{args.output_dir}/models/checkpoint_best.pth"

        if os.path.exists(checkpoint_last_path):
            start_epoch, best_accuracy, _, _ = load_checkpoint(
                checkpoint_last_path, model, optimizer, scheduler, scaler
            )
            print("🔄 Resuming from last checkpoint")
        elif os.path.exists(checkpoint_best_path):
            start_epoch, best_accuracy, _, _ = load_checkpoint(
                checkpoint_best_path, model, optimizer, scheduler, scaler
            )
            print("🔄 Resuming from best checkpoint (no last checkpoint found)")
        else:
            print(f"⚠️  No checkpoints found in {args.output_dir}/models/, starting from scratch")

    if train_config.epochs == 0:
        print("Skipping training since epochs is 0.")
        return None

    # Training Loop
    for epoch in range(start_epoch, train_config.epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{train_config.epochs} [Training]"
        )

        optimizer.zero_grad()  # Zero gradients at the start of each epoch

        for batch_idx, batch in enumerate(progress_bar):
            ex_before, ex_after, test_before, ch_a, ch_b, ch_c, correct_idx, _ = batch
            ex_before = ex_before.to(device)
            ex_after = ex_after.to(device)
            test_before = test_before.to(device)
            ch_a = ch_a.to(device)
            ch_b = ch_b.to(device)
            ch_c = ch_c.to(device)
            correct_idx = correct_idx.to(device)

            # Mixed precision forward pass
            if scaler is not None:
                with autocast("cuda"):
                    model_outputs = model(ex_before, ex_after, test_before, ch_a, ch_b, ch_c)
                    loss = criterion(model_outputs, correct_idx)
                    # Normalize loss by accumulation steps
                    loss = loss / train_config.gradient_accumulation_steps

                # Mixed precision backward pass
                scaler.scale(loss).backward()
            else:
                # Standard precision
                model_outputs = model(ex_before, ex_after, test_before, ch_a, ch_b, ch_c)
                loss = criterion(model_outputs, correct_idx)
                # Normalize loss by accumulation steps
                loss = loss / train_config.gradient_accumulation_steps
                loss.backward()

            # Accumulate the loss for logging (multiply back to get original scale)
            running_loss += loss.item() * train_config.gradient_accumulation_steps

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

            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % train_config.gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            current_train_acc = 100 * train_correct / train_total if train_total > 0 else 0
            # Show the original scale loss in progress bar
            progress_bar.set_postfix(
                loss=f"{loss.item() * train_config.gradient_accumulation_steps:.4f}",
                acc=f"{current_train_acc:.1f}%",
            )

        scheduler.step()
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        # --- 💡 Validation at the end of each epoch ---
        val_accuracy = evaluate(model, val_loader, device, dataset=val_dataset)
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
            neptune_run["training/learning_rate"].append(scheduler.get_last_lr()[0])

        # --- 💾 Save checkpoints ---
        # Always save last checkpoint for resuming
        checkpoint_last_path = f"{args.output_dir}/models/checkpoint_last.pth"
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_accuracy": best_accuracy,
            "train_accuracy": train_accuracy,
            "train_loss": avg_train_loss,
            "val_accuracy": val_accuracy,
            "config": train_config.model_dump(),
            "random_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            checkpoint["cuda_random_state"] = torch.cuda.get_rng_state()

        # Save scaler state if using mixed precision
        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()

        torch.save(checkpoint, checkpoint_last_path)

        # Save best checkpoint when validation improves
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            checkpoint_best_path = f"{args.output_dir}/models/checkpoint_best.pth"
            # Update best accuracy in checkpoint before saving
            checkpoint["best_accuracy"] = best_accuracy
            torch.save(checkpoint, checkpoint_best_path)
            print(f"✨ New best model saved with accuracy: {best_accuracy:.2f}%")

    print("🏁 Training finished.")

    # Log final results to Neptune if enabled
    if neptune_run:
        neptune_run["training/final_train_loss"] = avg_train_loss
        neptune_run["training/final_train_accuracy"] = train_accuracy
        neptune_run["training/best_val_accuracy"] = best_accuracy
        neptune_run["training/total_parameters"] = sum(p.numel() for p in model.parameters())
        neptune_run["training/checkpoint_dir"] = f"{args.output_dir}/models/"
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

    with open(f"{args.output_dir}/config.json") as f:
        saved_config = json.load(f)

    # the true config to use is the saved one
    test_config = Config.from_saved_config_for_test(saved_config, test_on=args.test_on)
    # the provided config (through args) compared against the saved config, warning if they differ
    provided_test_config_mdump = Config.from_args(args, for_task="test").model_dump()
    skip_fields = [
        "neptune_url",
        "neptune_project",
        "neptune_api_token",
        "device",
        "data_dir",
        "metadata_path",
        "task",
        "use_neptune",
    ]
    warnings_list = []
    for field, value in test_config.model_dump().items():
        if field in skip_fields:
            continue
        if value != provided_test_config_mdump[field]:
            warnings_list.append(
                f"Config mismatch for {field}: {value} != {provided_test_config_mdump[field]}. "
                f"Using the setting from the saved config, which matches the model checkpoint: "
                f"{field}={value}"
            )
    if warnings_list:
        warnings.warn(
            "Config mismatches detected:\n" + "\n - ".join(warnings_list),
            stacklevel=2,
        )
    # override use_neptune to args.use_neptune
    test_config.use_neptune = args.use_neptune

    # Initialize Neptune logging (optional for testing)
    neptune_run = None
    if test_config.use_neptune:
        if neptune_run_id is None:
            neptune_run = init_neptune(test_config, args, "test")
        else:
            # continue the existing Neptune run
            try:
                neptune_run = neptune.init_run(with_id=neptune_run_id)
            except Exception as e:
                print(f"⚠️  Could not resume Neptune run {neptune_run_id}: {e}")
                print("Testing will proceed without Neptune logging.")
                neptune_run = None
    else:
        print("🔕 Neptune logging disabled for testing")

    # Initialize a fresh model instance and load the best saved weights
    model = SiameseAnalogyNetwork(
        embedding_dim=test_config.embedding_dim,
        encoder_name=test_config.encoder_name,
    ).to(device)

    # Load model from checkpoint in output directory
    output_dir = args.output_dir
    if not output_dir:
        raise ValueError(
            "No output directory available. "
            "If only testing, please specify an output directory with the --output_dir flag. "
            "If also training, this is automatically managed."
        )

    # Try to load from best checkpoint first, then last checkpoint
    checkpoint_best_path = f"{output_dir}/models/checkpoint_best.pth"
    checkpoint_last_path = f"{output_dir}/models/checkpoint_last.pth"

    checkpoint_path = None
    if os.path.exists(checkpoint_best_path):
        checkpoint_path = checkpoint_best_path
        print(f"🔄 Loading best checkpoint for testing: {checkpoint_path}")
    elif os.path.exists(checkpoint_last_path):
        checkpoint_path = checkpoint_last_path
        print(f"🔄 Loading last checkpoint for testing: {checkpoint_path}")
    else:
        raise FileNotFoundError(
            f"No checkpoints found in {output_dir}/models/. "
            f"Expected checkpoint_best.pth or checkpoint_last.pth"
        )

    # Load only the model state from checkpoint
    load_model_from_checkpoint(checkpoint_path, model)
    # check that the saved config is the same as the test config
    with open(f"{output_dir}/config.json") as f:
        saved_config = json.load(f)
        current_config = test_config.model_dump()
        skip_fields = [
            "neptune_url",
            "neptune_project",
            "neptune_api_token",
            "device",
            "data_dir",
            "metadata_path",
            "task",
            "use_neptune",
        ]
        for field, value in saved_config.items():
            if field in skip_fields:
                continue
            if value != current_config[field]:
                raise ValueError(
                    f"Config mismatch for field {field}: {value} != {current_config[field]}"
                )

    # Setup the test dataloader
    test_dataset = dataset_factory(args, test_config)
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_config.batch_size,
        shuffle=False,
        num_workers=test_config.num_workers,
    )

    # Run final evaluation and always generate submission file
    test_accuracy = evaluate(
        model,
        test_loader,
        device,
        save_predictions=True,
        dataset=test_dataset,
        test_on=args.test_on,
    )

    if test_accuracy is not None:
        print(f"\n✅ Final Test Accuracy: {test_accuracy:.2f}%")
        # Run evaluation analysis if submission file was generated and labels are available
        submission_file = f"{args.output_dir}/submission_{args.test_on}.json"
        if os.path.exists(submission_file):
            run_evaluation_analysis(args, args.test_on)
    else:
        print("\n✅ Test predictions generated (no ground truth labels available)")
        # Skip evaluation analysis when no labels are available
        print("⚠️  Skipping evaluation analysis - no ground truth labels available")

    # Log test results to Neptune if enabled
    if neptune_run:
        if test_accuracy is not None:
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

    # --- Cleanup Phase (if train without neptune, this is just a dev run)---
    if not neptune_run_id and not args.do_test:
        shutil.rmtree(args.output_dir)
