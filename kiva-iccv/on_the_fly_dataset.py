import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image
from utils.dataset import (
    _kiva_counting,
    _kiva_functions_compositionality_counting_reflect,
    _kiva_functions_compositionality_counting_resizing,
    _kiva_functions_compositionality_counting_rotation,
    _kiva_functions_compositionality_reflect_resizing,
    _kiva_functions_compositionality_resizing_rotation,
    _kiva_functions_counting,
    _kiva_functions_reflect,
    _kiva_functions_resizing,
    _kiva_functions_rotation,
    _kiva_reflect,
    _kiva_resizing,
    _kiva_rotation,
)

# Default distribution config for on-the-fly dataset generation
DEFAULT_DISTRIBUTION_CONFIG = {
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


class OnTheFlyKiVADataset(Dataset):
    def __init__(
        self,
        objects_dir: str,
        distribution_config: dict[str, float] | None = None,
        epoch_length: int = 1000,
        transform=None,
    ):
        self.root_dir = Path(objects_dir)
        self.epoch_length = epoch_length

        # Use default distribution if none provided
        if distribution_config is None:
            distribution_config = DEFAULT_DISTRIBUTION_CONFIG

        # Map transformation types to their specific object directories
        self.dir_map = {
            "Rotation": self.root_dir / "Achiral Objects for Reflect, 2DRotation",
            "Reflect": self.root_dir / "Achiral Objects for Reflect, 2DRotation",
            "Resizing": self.root_dir / "Planar Objects for Resize",
            "Counting": self.root_dir / "Objects for Colour, Counting",
        }
        self.dir_map_composite_to_single = {
            "Counting,Reflect": "Reflect",
            "Counting,Resizing": "Resizing",
            "Counting,Rotation": "Rotation",
            "Reflect,Resizing": "Reflect",
            "Resizing,Rotation": "Rotation",
        }
        for dir_path in self.dir_map.values():
            if not dir_path.exists():
                raise FileNotFoundError(f"Required directory does not exist: {dir_path}")

        # Cache file lists for each directory to avoid repeated os.listdir calls
        self.file_lists = {
            key: [f for f in os.listdir(path) if f.lower().endswith(("png", "jpg", "jpeg"))]
            for key, path in self.dir_map.items()
        }

        self.transform = transform or transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Normalize distribution config
        total_weight = sum(distribution_config.values())
        if total_weight > 0:
            self.distribution_config = {k: v / total_weight for k, v in distribution_config.items()}
        else:
            self.distribution_config = distribution_config

        self.rules = list(self.distribution_config.keys())
        self.weights = list(self.distribution_config.values())

        self.base_canvas_transform = transforms.Resize((600, 600), antialias=True)
        self.canvas_resize = transforms.Resize((300, 300), antialias=True)

        self.param_options = {
            "kiva-functions-compositionality": {
                "Counting": ["+1", "+2", "-1", "-2", "x2", "x3", "d2", "d3"],
                "Reflect": ["X", "Y"],
                "Resizing": ["0.5XY", "2XY", "0.5X", "0.5Y", "2X", "2Y"],
                "Rotation": ["+45", "-45", "+90", "-90", "+135", "-135", "180"],
            },
            "kiva-functions": {
                "Counting": ["+1", "+2", "-1", "-2", "x2", "x3", "d2", "d3"],
                "Reflect": ["X", "Y", "XY"],
                "Resizing": ["0.5XY", "2XY", "0.5X", "0.5Y", "2X", "2Y"],
                "Rotation": ["+45", "-45", "+90", "-90", "+135", "-135", "180"],
            },
            "kiva": {
                "Counting": ["+1", "+2", "-1", "-2"],
                "Reflect": ["X", "Y", ""],
                "Resizing": ["0.5XY", "2XY", "XY"],
                "Rotation": ["+90", "-90", "180"],
            },
        }

        self.start_transformation_options = {
            "kiva-functions": {
                "Reflect": ["X", "Y", ""],
                "Resizing": [
                    "0.8XY",
                    "1.2XY",
                    "0.8X",
                    "1.2X",
                    "0.8Y",
                    "1.2Y",
                    "1X",
                    "1Y",
                ],
                "Rotation": ["+0", "+45", "-45", "+90", "-90", "+135", "-135", "180"],
            },
            "kiva-functions-compositionality": {
                "Reflect": ["X", "Y", ""],
                "Resizing": [
                    "0.8XY",
                    "1.2XY",
                    "0.8X",
                    "1.2X",
                    "0.8Y",
                    "1.2Y",
                    "1X",
                    "1Y",
                ],
                "Rotation": ["+0", "+45", "-45", "+90", "-90", "+135", "-135", "180"],
            },
        }
        self.param_options["kiva-functions-compositionality"] = self.param_options["kiva-functions"]

        print(
            f"Dataset Initialized. An epoch will consist of {self.epoch_length} generated samples."
        )

    def _get_rotation_incorrect_options(self, true_param: str, start_rotation: str) -> list[str]:
        # the result of start_rotation + incorrect_option cannot be the same as the
        # result of start_rotation + true_param
        # Add initial_rotation and angle together, handling + and - signs
        def combine_angles(a1: str, a2: str) -> str:  # Always returns a positive angle
            def s2i(s: str) -> int:
                if s.startswith("+"):
                    return int(s[1:])
                elif s.startswith("-"):
                    return -int(s[1:])
                else:
                    return int(s)

            total = (s2i(a1) + s2i(a2)) % 360  # Wrap into a single turn
            if total in [0, 180]:
                return str(total)
            return f"+{total}"

        final_true = combine_angles(start_rotation, true_param)
        incorrect_options = ["+45", "-45", "+90", "-90", "+135", "-135", "180"]
        for incorrect_option in incorrect_options:
            if combine_angles(start_rotation, incorrect_option) == final_true:
                incorrect_options.remove(incorrect_option)
        return incorrect_options

    def _get_counting_start_options(self, true_param: str) -> list[int]:
        if true_param[0] == "+":
            return [2, 3, 4, 5]
        elif true_param[0] == "-":
            return [7, 6, 5, 4, 3]
        elif true_param == "x2":
            return [2, 3, 4]
        elif true_param == "x3":
            return [1, 2, 3]
        elif true_param == "d2":
            return [8, 6, 4]
        elif true_param == "d3":
            return [9, 6, 3]
        else:
            raise ValueError(f"Invalid true parameter: {true_param}")

    def _get_counting_incorrect_param_options(self, true_param: str, start_count: int) -> list[str]:
        return [
            f"{delta:+}"
            for delta in [-2, -1, 1, 2]
            if 1 <= start_count + delta <= 9 and f"{delta:+}" != true_param
        ]

    def __len__(self) -> int:
        return self.epoch_length

    def _load_random_images(self, n: int, rule_types: list[str]) -> list[torch.Tensor]:
        """Loads n unique random images suitable for the given transformation types."""

        if len(rule_types) == 1:
            primary_rule = rule_types[0]
        else:
            primary_rule = self.dir_map_composite_to_single[",".join(rule_types)]

        dir_path = self.dir_map[primary_rule]
        image_files = self.file_lists[primary_rule]

        if len(image_files) < n:
            raise ValueError(f"Not enough images in {dir_path} to sample {n} unique images.")

        selected_files = random.sample(image_files, n)
        images = []
        for f in selected_files:
            img = read_image(str(dir_path / f))
            # Ensure white background for images with transparency
            if img.shape[0] == 4:  # RGBA image
                # Create white background
                white_bg = torch.full((3, img.shape[1], img.shape[2]), 255, dtype=img.dtype)
                alpha = img[3:4, :, :].float() / 255.0
                rgb = img[:3, :, :].float()
                # Composite over white background
                img = (rgb * alpha + white_bg.float() * (1 - alpha)).byte()
            images.append(self.base_canvas_transform(img))
        return images

    def _generate_kiva_level(self, rule: str) -> tuple:
        # A->B :: C->D. Goal: Identify the same transformation (rule + parameter + values).
        # A and C are different object identities but use the same starting values
        # e.g. Counting+1: A=2->3, C=2->3 (same values, different objects)
        img_A, img_C = self._load_random_images(2, [rule])

        true_param = random.choice(self.param_options["kiva"][rule])
        # Ensure at least 2 other params are available for distractors
        param_choices = [p for p in self.param_options["kiva"][rule] if p != true_param]
        if len(param_choices) < 2:
            raise ValueError(f"Rule {rule} needs at least 3 parameter options for kiva level.")
        incorrect_params = random.sample(param_choices, 2)

        apply_function = {
            "Counting": _kiva_counting,
            "Reflect": _kiva_reflect,
            "Resizing": _kiva_resizing,
            "Rotation": _kiva_rotation,
        }.get(rule, None)

        if apply_function is None:
            raise NotImplementedError(f"Rule {rule} is not implemented.")

        return apply_function(img_A, img_C, true_param, incorrect_params), f"{rule}_{true_param}"

    def _generate_kiva_functions_level(self, rule: str) -> tuple:
        # A->B :: C->D. Goal: Identify the same transformation (rule + parameter + values).
        # A and C are different object identities but use the same starting values
        # e.g. Counting+1: A=2->3, C=2->3 (same values, different objects)
        img_A, img_C = self._load_random_images(2, [rule])

        true_param = random.choice(self.param_options["kiva-functions"][rule])
        options = (
            self.start_transformation_options["kiva-functions"][rule]
            if rule != "Counting"
            else self._get_counting_start_options(true_param)
        )
        start_transformations = random.sample(options, k=2)

        # Ensure at least 2 other params are available for distractors
        if rule == "Counting":
            params_choices = self._get_counting_incorrect_param_options(
                true_param, start_transformations[1]
            )
        elif rule == "Rotation":
            params_choices = self._get_rotation_incorrect_options(
                true_param, start_transformations[1]
            )
        else:
            params_choices = [
                p for p in self.param_options["kiva-functions"][rule] if p != true_param
            ]
        if len(params_choices) < 2:
            raise ValueError(
                f"Rule {rule} needs at least 3 parameter options for kiva-functions level."
            )
        incorrect_params = random.sample(params_choices, k=2)

        apply_function = {
            "Counting": _kiva_functions_counting,
            "Reflect": _kiva_functions_reflect,
            "Resizing": _kiva_functions_resizing,
            "Rotation": _kiva_functions_rotation,
        }.get(rule, None)
        if apply_function is None:
            raise NotImplementedError(f"Rule {rule} is not implemented.")

        return apply_function(
            img_A, img_C, true_param, incorrect_params, start_transformations
        ), f"{rule}_{true_param}"

    def _generate_kiva_functions_compositionality_level(self, rule_composition: str) -> tuple:
        """
        Generates a sample for the compositionality level.
        A -> B :: C -> D.
        The transformation is a sequence of two functions.
        Handles each of the 5 possible compositions separately.
        """
        img_A_base, img_C_base = self._load_random_images(2, rule_composition.split(","))

        if rule_composition == "Counting,Reflect":
            return self._generate_counting_reflect_composition(img_A_base, img_C_base)
        elif rule_composition == "Counting,Resizing":
            return self._generate_counting_resizing_composition(img_A_base, img_C_base)
        elif rule_composition == "Counting,Rotation":
            return self._generate_counting_rotation_composition(img_A_base, img_C_base)
        elif rule_composition == "Reflect,Resizing":
            return self._generate_reflect_resizing_composition(img_A_base, img_C_base)
        elif rule_composition == "Resizing,Rotation":
            return self._generate_resizing_rotation_composition(img_A_base, img_C_base)
        else:
            raise ValueError(f"Unsupported composition: {rule_composition}")

    def _generate_counting_reflect_composition(
        self, img_A_base: torch.Tensor, img_C_base: torch.Tensor
    ) -> tuple:
        """Handle Counting,Reflect composition"""
        true_count_param = random.choice(self.param_options["kiva-functions"]["Counting"])
        true_reflect_param = random.choice(self.param_options["kiva-functions"]["Reflect"])

        incorrect_reflect_param = random.choice(
            [p for p in self.param_options["kiva-functions"]["Reflect"] if p != true_reflect_param]
        )

        start_count_A, start_count_C = random.sample(
            self._get_counting_start_options(true_count_param), k=2
        )
        incorrect_count_param = random.choice(
            self._get_counting_incorrect_param_options(true_count_param, start_count_C)
        )
        start_reflect_A, start_reflect_C = random.sample(
            self.start_transformation_options["kiva-functions"]["Reflect"], k=2
        )

        return _kiva_functions_compositionality_counting_reflect(
            img_A_base,
            img_C_base,
            true_count_param,
            true_reflect_param,
            incorrect_reflect_param,
            incorrect_count_param,
            start_count_A,
            start_count_C,
            start_reflect_A,
            start_reflect_C,
        ), f"Counting{true_count_param}_Reflect{true_reflect_param}"

    def _generate_counting_resizing_composition(
        self, img_A_base: torch.Tensor, img_C_base: torch.Tensor
    ) -> tuple:
        """Handle Counting,Resizing composition"""
        true_count_param = random.choice(self.param_options["kiva-functions"]["Counting"])
        true_resizing_param = random.choice(self.param_options["kiva-functions"]["Resizing"])

        possible_params = [
            p for p in self.param_options["kiva-functions"]["Resizing"] if p != true_resizing_param
        ]
        incorrect_resize_param = random.choice(possible_params)

        start_count_A, start_count_C = random.sample(
            self._get_counting_start_options(true_count_param), k=2
        )
        incorrect_count_param = random.choice(
            self._get_counting_incorrect_param_options(true_count_param, start_count_C)
        )
        start_resize_A, start_resize_C = random.sample(
            self.start_transformation_options["kiva-functions"]["Resizing"], k=2
        )

        return _kiva_functions_compositionality_counting_resizing(
            img_A_base,
            img_C_base,
            true_count_param,
            true_resizing_param,
            incorrect_resize_param,
            incorrect_count_param,
            start_count_A,
            start_count_C,
            start_resize_A,
            start_resize_C,
        ), f"Counting{true_count_param}_Resizing{true_resizing_param}"

    def _generate_counting_rotation_composition(
        self, img_A_base: torch.Tensor, img_C_base: torch.Tensor
    ) -> tuple:
        """Handle Counting,Rotation composition"""
        true_count_param = random.choice(self.param_options["kiva-functions"]["Counting"])
        true_rotation_param = random.choice(self.param_options["kiva-functions"]["Rotation"])

        start_count_A, start_count_C = random.sample(
            self._get_counting_start_options(true_count_param), k=2
        )
        start_rotation_A, start_rotation_C = random.sample(
            self.start_transformation_options["kiva-functions"]["Rotation"], k=2
        )
        incorrect_count_param = random.choice(
            self._get_counting_incorrect_param_options(true_count_param, start_count_C)
        )
        incorrect_rotation_param = random.choice(
            self._get_rotation_incorrect_options(true_rotation_param, start_rotation_C)
        )

        return _kiva_functions_compositionality_counting_rotation(
            img_A_base,
            img_C_base,
            true_count_param,
            true_rotation_param,
            incorrect_rotation_param,
            incorrect_count_param,
            start_count_A,
            start_count_C,
            start_rotation_A,
            start_rotation_C,
        ), f"Counting{true_count_param}_Rotation{true_rotation_param}"

    def _generate_reflect_resizing_composition(
        self, img_A_base: torch.Tensor, img_C_base: torch.Tensor
    ) -> tuple:
        """Handle Reflect,Resizing composition"""
        true_reflect_param = random.choice(self.param_options["kiva-functions"]["Reflect"])
        true_resizing_param = random.choice(self.param_options["kiva-functions"]["Resizing"])

        incorrect_reflect_param = random.choice(
            [p for p in self.param_options["kiva-functions"]["Reflect"] if p != true_reflect_param]
        )
        incorrect_resize_param = random.choice(
            [
                p
                for p in self.param_options["kiva-functions"]["Resizing"]
                if p != true_resizing_param
            ]
        )

        start_reflect_A, start_reflect_C = random.sample(
            self.start_transformation_options["kiva-functions"]["Reflect"],
            k=2,
        )
        start_resize_A, start_resize_C = random.sample(
            self.start_transformation_options["kiva-functions"]["Resizing"],
            k=2,
        )

        return _kiva_functions_compositionality_reflect_resizing(
            img_A_base,
            img_C_base,
            true_reflect_param,
            true_resizing_param,
            incorrect_reflect_param,
            incorrect_resize_param,
            start_reflect_A,
            start_reflect_C,
            start_resize_A,
            start_resize_C,
        ), f"Reflect{true_reflect_param}_Resizing{true_resizing_param}"

    def _generate_resizing_rotation_composition(
        self, img_A_base: torch.Tensor, img_C_base: torch.Tensor
    ) -> tuple:
        """Handle Resizing,Rotation composition"""
        true_resizing_param = random.choice(self.param_options["kiva-functions"]["Resizing"])
        true_rotation_param = random.choice(self.param_options["kiva-functions"]["Rotation"])

        # Get starting states
        start_resize_A, start_resize_C = random.sample(
            self.start_transformation_options["kiva-functions"]["Resizing"], k=2
        )
        start_rotation_A, start_rotation_C = random.sample(
            self.start_transformation_options["kiva-functions"]["Rotation"], k=2
        )

        # Get incorrect parameters (after starting states are defined)
        incorrect_resize_param = random.choice(
            [
                p
                for p in self.param_options["kiva-functions"]["Resizing"]
                if p != true_resizing_param
            ]
        )
        incorrect_rotation_param = random.choice(
            self._get_rotation_incorrect_options(true_rotation_param, start_rotation_C)
        )

        return _kiva_functions_compositionality_resizing_rotation(
            img_A_base,
            img_C_base,
            true_resizing_param,
            true_rotation_param,
            incorrect_resize_param,
            incorrect_rotation_param,
            start_resize_A,
            start_resize_C,
            start_rotation_A,
            start_rotation_C,
        ), f"Resizing{true_resizing_param}_Rotation{true_rotation_param}"

    def __getitem__(self, idx: int) -> tuple:
        if not self.rules:
            raise RuntimeError("Distribution config is empty or all weights are zero.")
        rule_str = random.choices(self.rules, self.weights, k=1)[0]
        level, rule = rule_str.rsplit("-", 1)

        # Dispatch to the correct generation function
        level_function = {
            "kiva": self._generate_kiva_level,
            "kiva-functions": self._generate_kiva_functions_level,
            "kiva-functions-compositionality": self._generate_kiva_functions_compositionality_level,
        }.get(level, None)
        if level_function is None:
            raise NotImplementedError(f"Level '{level}' is not implemented.")

        imgs, sample_id = level_function(rule)
        a, b, c, *choices = imgs

        # Shuffle choices and find the correct index
        correct_choice = choices[0]
        random.shuffle(choices)

        # Find the correct index by comparing tensors element-wise
        correct_idx = 0
        for i, choice in enumerate(choices):
            if torch.equal(choice, correct_choice):
                correct_idx = i
                break

        # Apply final transform to all 6 images
        all_images = [a, b, c] + choices
        final_images = [self.transform(img[:3, :, :]) for img in all_images]

        return (*final_images, torch.tensor(correct_idx, dtype=torch.long), sample_id)


def main(case):
    distribution = {
        "kiva-Counting": 0,
        "kiva-Reflect": 0,
        "kiva-Resizing": 0,
        "kiva-Rotation": 0,
        "kiva-functions-Counting": 0,
        "kiva-functions-Reflect": 0,
        "kiva-functions-Resizing": 0,
        "kiva-functions-Rotation": 0,
        "kiva-functions-compositionality-Counting,Reflect": 0,
        "kiva-functions-compositionality-Counting,Resizing": 0,
        "kiva-functions-compositionality-Counting,Rotation": 0,
        "kiva-functions-compositionality-Reflect,Resizing": 0,
        "kiva-functions-compositionality-Resizing,Rotation": 0,
    }
    distribution[case] = 1

    # --- Instantiate the Dataset and DataLoader ---
    objects_dir = "/home/ubuntu/kiva-iccv/data/KiVA/untransformed objects"
    kiva_dataset = OnTheFlyKiVADataset(
        objects_dir=objects_dir,
        distribution_config=distribution,
        epoch_length=100,  # smaller epoch for quick demo
    )

    kiva_loader = DataLoader(kiva_dataset, batch_size=1, shuffle=True, num_workers=0)

    # --- Fetch and inspect a batch ---
    batch = next(iter(kiva_loader))
    a_batch, b_batch, c_batch, ch1_batch, ch2_batch, ch3_batch, labels, sample_ids = batch

    # --- Optional: Save a visual of the first item in the batch for debugging ---
    save_dir = "debug_batch_images"
    os.makedirs(save_dir, exist_ok=True)

    # Get all 6 images for the first sample
    first_sample_images = [
        a_batch[0],
        b_batch[0],
        c_batch[0],
        ch1_batch[0],
        ch2_batch[0],
        ch3_batch[0],
    ]
    image_names = ["A", "B", "C", "choice1", "choice2", "choice3"]
    import matplotlib.pyplot as plt

    # Denormalize for viewing
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    sample_id_str = sample_ids[0].replace(",", "_")  # Sanitize filename

    # Denormalize and convert all images to numpy arrays for plotting
    denorm_images = []
    for img in first_sample_images:
        denormalized_img = (img * std) + mean
        np_img = denormalized_img.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        denorm_images.append(np_img)

    # Create a grid (2 rows x 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for _, (ax, img, name) in enumerate(zip(axes.flat, denorm_images, image_names, strict=False)):
        ax.imshow(img)
        ax.set_title(name)
        ax.axis("off")

    plt.tight_layout()
    save_dir = os.path.join(save_dir, case)
    os.makedirs(save_dir, exist_ok=True)
    grid_save_path = os.path.join(save_dir, f"{sample_id_str}.png")
    plt.savefig(grid_save_path)
    plt.close(fig)


if __name__ == "__main__":
    distribution = {
        "kiva-Counting": 0,
        "kiva-Reflect": 0,
        "kiva-Resizing": 0,
        "kiva-Rotation": 0,
        "kiva-functions-Counting": 0,
        "kiva-functions-Reflect": 0,
        "kiva-functions-Resizing": 0,
        "kiva-functions-Rotation": 0,
        "kiva-functions-compositionality-Counting,Reflect": 0,
        "kiva-functions-compositionality-Counting,Resizing": 0,
        "kiva-functions-compositionality-Counting,Rotation": 0,
        "kiva-functions-compositionality-Reflect,Resizing": 0,
        "kiva-functions-compositionality-Resizing,Rotation": 0,
    }
    import tqdm

    for case in tqdm.tqdm(distribution):
        main(case)
