import argparse
import os
from typing import Literal

import torch
from pydantic import BaseModel


class Config(BaseModel):
    """Configuration class for hyperparameters."""

    # Paths will be provided by the argparser
    data_dir: str
    metadata_path: str

    # Loss
    loss_type: Literal["standard_triplet", "contrastive"] = "standard_triplet"
    margin: float = 0.5

    # Model & Training
    transformation_net: bool = False
    embedding_dim: int = 512
    freeze_encoder: bool = False
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 64
    epochs: int = 5
    num_workers: int = 4
    device: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"

    # Neptune logging
    use_neptune: bool = False
    neptune_project: str = "jsalvasoler/kiva-iccv"
    neptune_api_token: str = ""


def create_config_from_args(args, data_dir: str, metadata_path: str) -> Config:
    """Create a Config object from parsed arguments, overriding defaults."""
    config_dict = {
        "data_dir": data_dir,
        "metadata_path": metadata_path,
        "loss_type": args.loss_type,
        "margin": args.margin,
        "transformation_net": args.transformation_net,
        "embedding_dim": args.embedding_dim,
        "freeze_encoder": args.freeze_encoder,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "num_workers": args.num_workers,
        "use_neptune": args.use_neptune,
        "neptune_project": args.neptune_project or os.getenv("NEPTUNE_PROJECT", ""),
        "neptune_api_token": args.neptune_api_token or os.getenv("NEPTUNE_API_TOKEN", ""),
    }

    # Convert to Config object
    return Config(**config_dict)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Siamese Network for Visual Analogies."
    )

    # Main action arguments
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument(
        "--do_test", action="store_true", help="Whether to run testing on the test set."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory for model and predictions. Must be specified if only testing.",
    )
    # Dataset arguments
    dataset_options = ["unit", "train", "validation", "test"]
    parser.add_argument(
        "--train_on",
        type=str,
        default="train",
        choices=dataset_options,
        help="Dataset keyword for training.",
    )
    parser.add_argument(
        "--validate_on",
        type=str,
        default="validation_sample",
        choices=dataset_options + ["validation_sample"],
        help="Dataset keyword for validation.",
    )
    parser.add_argument(
        "--test_on",
        type=str,
        default="test",
        choices=dataset_options,
        help="Dataset keyword for testing.",
    )

    # Config override arguments
    parser.add_argument(
        "--loss_type",
        type=str,
        default="contrastive",
        choices=["standard_triplet", "contrastive"],
        help="Loss type",
    )
    parser.add_argument("--margin", type=float, default=0.5, help="Margin for loss functions")
    parser.add_argument(
        "--transformation_net", action="store_true", help="Use transformation network"
    )
    parser.add_argument("--embedding_dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--freeze_encoder", action="store_true", help="Freeze encoder parameters")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )

    parser.add_argument("--use_neptune", action="store_true", help="Use Neptune for logging")
    parser.add_argument("--neptune_project", type=str, default="", help="Neptune project")
    parser.add_argument("--neptune_api_token", type=str, default="", help="Neptune API token")

    return parser
