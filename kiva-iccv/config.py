import argparse
import os
from typing import Literal

import torch
from pydantic import BaseModel, Field


class Config(BaseModel):
    """Configuration class for hyperparameters."""

    # Paths will be provided by the argparser
    data_dir: str = Field(description="Path to the data directory")
    metadata_path: str | None = Field(
        description="Path to the metadata file (None for test datasets without labels)"
    )
    task: Literal["train", "validation", "test"] = Field(description="Task type")
    use_otf: bool = Field(
        default=False,
        description=(
            "Whether to use on-the-fly generation. This only impacts the training dataset, "
            "and not the validation or test datasets."
        ),
    )

    # Loss
    loss_type: Literal["standard_triplet", "contrastive", "softmax"] = Field(
        default="standard_triplet", description="Loss type"
    )
    margin: float = Field(default=0.5, description="Margin for loss functions")
    temperature: float = Field(default=0.07, description="Temperature for loss functions")

    # Model & Training
    embedding_dim: int = Field(default=512, description="Embedding dimension")
    freeze_encoder: bool = Field(default=False, description="Freeze encoder parameters")
    encoder_name: str = Field(
        default="vit_small_patch16_224",
        description=(
            "Encoder name. E.g. resnet18, resnet50, vit_small_patch16_224, vit_base_patch16_224"
        ),
    )
    learning_rate: float = Field(default=1e-3, description="Learning rate")
    weight_decay: float = Field(default=1e-4, description="Weight decay")
    batch_size: int = Field(default=64, description="Batch size")
    gradient_accumulation_steps: int = Field(
        default=1,
        description=(
            "Number of gradient accumulation steps. "
            "Effective batch size = batch_size x gradient_accumulation_steps"
        ),
    )
    epochs: int = Field(default=5, description="Number of epochs")
    num_workers: int = Field(default=8, description="Number of workers for data loading")
    device: Literal["cuda", "cpu"] = Field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        description="Device to use for training",
    )
    otf_epoch_length: int = Field(
        default=8192 * 2,
        description="On-the-fly epoch length - essentially regulates how often we evaluate",
    )
    use_mixed_precision: bool = Field(
        default=True,
        description="Use automatic mixed precision training with torch.cuda.amp",
    )

    # Neptune logging
    use_neptune: bool = Field(default=False, description="Use Neptune for logging")
    neptune_project: str = Field(default="jsalvasoler/kiva-iccv", description="Neptune project")
    neptune_api_token: str = Field(default="", description="Neptune API token")

    @classmethod
    def from_saved_config_for_test(
        cls, saved_config: dict, test_on: Literal["train", "validation", "test"]
    ) -> "Config":
        """Create a Config object from a saved config."""
        data_dir, metadata_path = get_dataset_paths(test_on)
        config_dict = saved_config.copy()
        config_dict["data_dir"] = data_dir
        config_dict["metadata_path"] = metadata_path
        config_dict["task"] = "test"
        return cls(**config_dict)

    @classmethod
    def from_args(cls, args, for_task: Literal["train", "validation", "test"]) -> "Config":
        """Create a Config object from parsed arguments, overriding defaults."""

        dataset_path = {
            "train": args.train_on,
            "validation": args.validate_on,
            "test": args.test_on,
        }
        data_dir, metadata_path = get_dataset_paths(dataset_path[for_task])

        # Convert args to dict and add programmatically set fields
        config_dict = vars(args).copy()
        config_dict.update(
            {
                "task": for_task,
                "data_dir": data_dir,
                "metadata_path": metadata_path,
            }
        )

        # Handle special cases for environment variables
        if not config_dict.get("neptune_project"):
            config_dict["neptune_project"] = os.getenv("NEPTUNE_PROJECT", "")
        if not config_dict.get("neptune_api_token"):
            config_dict["neptune_api_token"] = os.getenv("NEPTUNE_API_TOKEN", "")

        # Convert to Config object
        return cls(**config_dict)


def get_dataset_paths(dataset_keyword: str) -> tuple[str, str | None]:
    """Returns the data directory and metadata path based on a keyword.

    For test datasets without labels, metadata_path will be None.
    """
    base_data_path = "./data"
    mapping = {
        "unit": ("split_unit", "unit.json"),
        "train": ("split_train", "train.json"),
        "validation": ("split_validation", "validation.json"),
        "validation_sample": ("split_validation", "validation.json"),
        "test": ("split_test", None),
    }
    if dataset_keyword not in mapping:
        raise ValueError(f"Invalid dataset keyword '{dataset_keyword}'.")
    split_dir, meta_file = mapping[dataset_keyword]

    data_dir = os.path.join(base_data_path, split_dir)

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Handle case where no metadata file is expected (test without labels)
    if meta_file is None:
        metadata_path = None
    else:
        metadata_path = os.path.join(base_data_path, meta_file)
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    return data_dir, metadata_path


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""

    class CustomFormatter(
        argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter
    ):
        """Show argument defaults and use type names as metavar."""

        pass

    parser = argparse.ArgumentParser(
        description="Train and evaluate a Siamese Network for Visual Analogies.",
        formatter_class=CustomFormatter,
    )

    # Non-config arguments
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Whether to run training on train_on with validation on validate_on.",
    )
    parser.add_argument(
        "--do_test", action="store_true", help="Whether to run testing on the test set."
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Resume training from checkpoint. Provide Neptune run ID or output directory path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory for model and predictions. Must be specified if only testing.",
    )
    parser.add_argument(
        "--distribution_config",
        type=str,
        default="",
        help=(
            "Path to YAML file containing custom distribution config for on-the-fly "
            "dataset generation. Look at distribution_config_example.yaml for an example."
        ),
    )

    # Dataset arguments: unit, train, validation, test
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

    # Add arguments from Config model fields
    for field_name, field_info in Config.model_fields.items():
        if field_name in {"data_dir", "metadata_path", "task"}:
            continue

        # Get field type, default, and description
        field_type = field_info.annotation
        default_value = field_info.default
        description = field_info.description or f"{field_name.replace('_', ' ').title()}"

        # Handle Literal types for choices
        choices = None
        if hasattr(field_type, "__origin__") and field_type.__origin__ is Literal:
            choices = list(field_type.__args__)
            # Use the first choice as the type for argparse
            field_type = type(choices[0])

        # Handle boolean fields with action="store_true"
        if field_type is bool:
            parser.add_argument(
                f"--{field_name}", action="store_true", default=default_value, help=description
            )
        else:
            # For other types, use the field type and default
            parser.add_argument(
                f"--{field_name}",
                type=field_type,
                default=default_value,
                choices=choices,
                help=description,
            )

    return parser
