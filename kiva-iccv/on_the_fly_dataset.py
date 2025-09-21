import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image


class OnTheFlyKiVADataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        distribution_config: dict[str, float],
        epoch_length: int = 1000,
        transform=None,
    ):
        self.root_dir = Path(data_dir)
        self.epoch_length = epoch_length

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
                "Counting": list(range(1, 10)),
                "Reflect": ["X", "Y", ""],
                # Reasonable continuous starts for resizing and rotation (used by compositionality)
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
                "Counting": list(range(1, 10)),
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

        self.all_rules = ["Counting", "Reflect", "Resizing", "Rotation"]

        self.all_compositionality_rules = [
            "Counting,Reflect",
            "Counting,Resizing",
            "Counting,Rotation",
            "Reflect,Resizing",
            "Resizing,Rotation",
        ]

        print(
            f"Dataset Initialized. An epoch will consist of {self.epoch_length} generated samples."
        )

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
        print(f"DEBUG: Getting image from {dir_path}")

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
        print(f"DEBUG: True param: {true_param}")

        # Ensure at least 2 other params are available for distractors
        param_choices = [p for p in self.param_options["kiva"][rule] if p != true_param]
        if len(param_choices) < 2:
            raise ValueError(f"Rule {rule} needs at least 3 parameter options for kiva level.")
        incorrect_params = random.sample(param_choices, 2)
        print(f"DEBUG: Incorrect params: {incorrect_params}")

        if rule == "Counting":
            from utils.dataset.transformations_kiva import apply_counting

            # For counting, we need to ensure same starting count
            # First generate with img_A to get the starting count
            img_A_initial, img_B_correct, start_count, _ = apply_counting(
                img_A, true_param, type="train"
            )

            # Apply the same transformation to img_C with the same starting count
            img_C_initial, img_D_correct, start_count, _ = apply_counting(
                img_C, true_param, type="train", initial_count=start_count
            )

            # TODO: logic to generate the incorrect options.
            # 1) could be more than 2 incorrect options
            # 2) we want to give more probability to options that are "closer" to the correct option
            # Generate incorrect options with different parameters, but same starting count
            _, img_E_incorrect, _, _ = apply_counting(
                img_C, incorrect_params[0], type="train", initial_count=start_count
            )
            _, img_F_incorrect, _, _ = apply_counting(
                img_C, incorrect_params[1], type="train", initial_count=start_count
            )
        elif rule == "Reflect":
            from utils.dataset.transformations_kiva import apply_reflection

            img_B_correct, _, _ = apply_reflection(img_A, true_param, type="train")
            img_D_correct, _, _ = apply_reflection(img_C, true_param, type="train")
            img_E_incorrect, _, _ = apply_reflection(img_C, incorrect_params[0], type="train")
            img_F_incorrect, _, _ = apply_reflection(img_C, incorrect_params[1], type="train")
            img_A_initial, img_C_initial = img_A, img_C

        elif rule == "Resizing":
            from utils.dataset.transformations_kiva import apply_resizing, paste_on_600

            img_B_correct, _, _ = apply_resizing(img_A, true_param, type="train")
            img_D_correct, _, _ = apply_resizing(img_C, true_param, type="train")
            img_E_incorrect, _, _ = apply_resizing(img_C, incorrect_params[0], type="train")
            img_F_incorrect, _, _ = apply_resizing(img_C, incorrect_params[1], type="train")
            img_A_initial, img_C_initial = img_A, img_C

            # finally paste everything on 600x600 canvas
            img_A_initial = paste_on_600(img_A_initial)
            img_B_correct = paste_on_600(img_B_correct)
            img_C_initial = paste_on_600(img_C_initial)
            img_D_correct = paste_on_600(img_D_correct)
            img_E_incorrect = paste_on_600(img_E_incorrect)
            img_F_incorrect = paste_on_600(img_F_incorrect)

        elif rule == "Rotation":
            from utils.dataset.transformations_kiva import apply_rotation

            img_B_correct, _, _ = apply_rotation(img_A, true_param, type="train")
            img_D_correct, _, _ = apply_rotation(img_C, true_param, type="train")
            img_E_incorrect, _, _ = apply_rotation(img_C, incorrect_params[0], type="train")
            img_F_incorrect, _, _ = apply_rotation(img_C, incorrect_params[1], type="train")
            img_A_initial, img_C_initial = img_A, img_C

        else:
            raise NotImplementedError(f"Rule {rule} is not implemented.")

        a, b, c = img_A_initial, img_B_correct, img_C_initial
        choices = [img_D_correct, img_E_incorrect, img_F_incorrect]

        sample_id = f"{rule}_{true_param}"
        return a, b, c, choices, sample_id

    def _generate_kiva_functions_level(self, rule: str) -> tuple:
        # A->B :: C->D. Goal: Identify the same transformation (rule + parameter + values).
        # A and C are different object identities but use the same starting values
        # e.g. Counting+1: A=2->3, C=2->3 (same values, different objects)
        img_A, img_C = self._load_random_images(2, [rule])

        true_param = random.choice(self.param_options["kiva-functions"][rule])
        print(f"DEBUG: True param: {true_param}")

        # Ensure at least 2 other params are available for distractors
        param_choices = [p for p in self.param_options["kiva-functions"][rule] if p != true_param]
        if len(param_choices) < 2:
            raise ValueError(
                f"Rule {rule} needs at least 3 parameter options for kiva-functions level."
            )
        incorrect_params = random.sample(param_choices, 2)
        print(f"DEBUG: Incorrect params: {incorrect_params}")

        start_transformations = random.choices(
            self.start_transformation_options["kiva-functions"][rule], k=2
        )
        print(f"DEBUG: Start transformations: {start_transformations}")

        if rule == "Counting":
            from utils.dataset.transformations_kiva_adults import apply_counting

            img_A_initial, img_B_correct, _, _ = apply_counting(
                img_A, true_param, type="train", initial_count=start_transformations[0]
            )
            img_C_initial, img_D_correct, _, _ = apply_counting(
                img_C, true_param, type="train", initial_count=start_transformations[1]
            )

            _, img_E_incorrect, _, _ = apply_counting(
                img_C, incorrect_params[0], type="train", initial_count=start_transformations[1]
            )
            _, img_F_incorrect, _, _ = apply_counting(
                img_C, incorrect_params[1], type="train", initial_count=start_transformations[1]
            )

        elif rule == "Reflect":
            from utils.dataset.transformations_kiva_adults import apply_reflection

            img_A_initial, _, _, _ = apply_reflection(img_A, start_transformations[0], type="train")
            img_B_correct, _, _, _ = apply_reflection(img_A_initial, true_param, type="train")

            img_C_initial, _, _, _ = apply_reflection(img_C, start_transformations[1], type="train")
            img_D_correct, _, _, _ = apply_reflection(img_C_initial, true_param, type="train")
            img_E_incorrect, _, _, _ = apply_reflection(
                img_C_initial, incorrect_params[0], type="train"
            )
            img_F_incorrect, _, _, _ = apply_reflection(
                img_C_initial, incorrect_params[1], type="train"
            )

        elif rule == "Resizing":
            from utils.dataset.transformations_kiva_adults import apply_resizing, paste_on_600

            img_A_initial, _, _ = apply_resizing(img_A, start_transformations[0], type="train")
            img_B_correct, _, _ = apply_resizing(img_A_initial, true_param, type="train")

            img_C_initial, _, _ = apply_resizing(img_C, start_transformations[1], type="train")
            img_D_correct, _, _ = apply_resizing(img_C_initial, true_param, type="train")
            img_E_incorrect, _, _ = apply_resizing(img_C_initial, incorrect_params[0], type="train")
            img_F_incorrect, _, _ = apply_resizing(img_C_initial, incorrect_params[1], type="train")

            # finally paste everything on 600x600 canvas
            img_A_initial = paste_on_600(img_A_initial)
            img_B_correct = paste_on_600(img_B_correct)
            img_C_initial = paste_on_600(img_C_initial)
            img_D_correct = paste_on_600(img_D_correct)
            img_E_incorrect = paste_on_600(img_E_incorrect)
            img_F_incorrect = paste_on_600(img_F_incorrect)

        elif rule == "Rotation":
            from utils.dataset.transformations_kiva_adults import apply_rotation

            _, img_A_initial, _, _ = apply_rotation(
                img_A, start_transformations[0], type="train", initial_rotation="+0"
            )
            _, img_B_correct, _, _ = apply_rotation(
                img_A_initial, true_param, type="train", initial_rotation="+0"
            )
            _, img_C_initial, _, _ = apply_rotation(
                img_C, start_transformations[1], type="train", initial_rotation="+0"
            )
            _, img_D_correct, _, _ = apply_rotation(
                img_C_initial, true_param, type="train", initial_rotation="+0"
            )
            _, img_E_incorrect, _, _ = apply_rotation(
                img_C_initial, incorrect_params[0], type="train", initial_rotation="+0"
            )
            _, img_F_incorrect, _, _ = apply_rotation(
                img_C_initial, incorrect_params[1], type="train", initial_rotation="+0"
            )

        else:
            raise NotImplementedError(f"Rule {rule} is not implemented.")

        a, b, c = img_A_initial, img_B_correct, img_C_initial
        choices = [img_D_correct, img_E_incorrect, img_F_incorrect]
        sample_id = f"{rule}_{true_param}"
        return a, b, c, choices, sample_id

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
        from utils.dataset.transformations_kiva_adults import apply_counting, apply_reflection

        # Get parameters
        true_count_param = random.choice(self.param_options["kiva-functions"]["Counting"])
        true_reflect_param = random.choice(self.param_options["kiva-functions"]["Reflect"])

        incorrect_count_param = random.choice(
            [p for p in self.param_options["kiva-functions"]["Counting"] if p != true_count_param]
        )
        incorrect_reflect_param = random.choice(
            [p for p in self.param_options["kiva-functions"]["Reflect"] if p != true_reflect_param]
        )

        # Get starting states
        start_count_A, start_count_C = random.choices(
            self.start_transformation_options["kiva-functions"]["Counting"], k=2
        )
        start_reflect_A, start_reflect_C = random.choices(
            self.start_transformation_options["kiva-functions"]["Reflect"], k=2
        )

        def make_initial(image: torch.Tensor, start_count: int, start_reflect: str) -> torch.Tensor:
            # Apply starting reflection to base object first
            _, reflected, _, _ = apply_reflection(image, start_reflect, type="train")
            # Then create the counted grid with the desired starting count on the reflected object
            initial_grid, _, _, _ = apply_counting(
                reflected, "+1", type="train", initial_count=start_count
            )
            return initial_grid

        def apply_true_chain(
            image: torch.Tensor, start_count: int, true_count: str, true_reflect: str
        ) -> torch.Tensor:
            # Apply the true reflection to the base object first
            _, reflected_correct, _, _ = apply_reflection(image, true_reflect, type="train")
            # Then apply the true counting to the reflected object
            _, out, _, _ = apply_counting(
                reflected_correct, true_count, type="train", initial_count=start_count
            )
            return out

        # Generate A/C initials using starting values
        img_A_initial = make_initial(img_A_base, start_count_A, start_reflect_A)
        img_C_initial = make_initial(img_C_base, start_count_C, start_reflect_C)

        # Generate correct B/D from corresponding starts
        img_B_correct = apply_true_chain(
            img_A_base, start_count_A, true_count_param, true_reflect_param
        )
        img_D_correct = apply_true_chain(
            img_C_base, start_count_C, true_count_param, true_reflect_param
        )

        # E: incorrect first (Reflect) parameter, correct second (Counting)
        img_E_incorrect = apply_true_chain(
            img_C_base, start_count_C, true_count_param, incorrect_reflect_param
        )

        # F: incorrect second (Counting) parameter, correct first (Reflect)
        img_F_incorrect = apply_true_chain(
            img_C_base, start_count_C, incorrect_count_param, true_reflect_param
        )

        a, b, c = img_A_initial, img_B_correct, img_C_initial
        choices = [img_D_correct, img_E_incorrect, img_F_incorrect]
        sample_id = f"Counting{true_count_param}_Reflect{true_reflect_param}"
        return a, b, c, choices, sample_id

    def _generate_counting_resizing_composition(
        self, img_A_base: torch.Tensor, img_C_base: torch.Tensor
    ) -> tuple:
        """Handle Counting,Resizing composition"""
        from utils.dataset.transformations_kiva_adults import (
            apply_counting,
            apply_resizing,
            paste_on_600,
        )

        # Get parameters
        true_param1 = random.choice(self.param_options["kiva-functions"]["Counting"])
        true_param2 = random.choice(self.param_options["kiva-functions"]["Resizing"])

        incorrect_param1 = random.choice(
            [p for p in self.param_options["kiva-functions"]["Counting"] if p != true_param1]
        )
        incorrect_param2 = random.choice(
            [p for p in self.param_options["kiva-functions"]["Resizing"] if p != true_param2]
        )

        # Get starting states
        start_count_A = random.choice(
            self.start_transformation_options["kiva-functions"]["Counting"]
        )
        start_resize_A = random.choice(
            self.start_transformation_options["kiva-functions"]["Resizing"]
        )
        start_count_C = random.choice(
            self.start_transformation_options["kiva-functions"]["Counting"]
        )
        start_resize_C = random.choice(
            self.start_transformation_options["kiva-functions"]["Resizing"]
        )

        # Generate A: apply starting transformations (resize first, then count)
        img_A_start_resized, _, _ = apply_resizing(img_A_base, start_resize_A, type="train")
        img_A_start_resized = paste_on_600(img_A_start_resized)
        img_A_initial, _, _, _ = apply_counting(
            img_A_start_resized, "+1", type="train", initial_count=start_count_A
        )

        # Generate C: apply starting transformations
        img_C_start_resized, _, _ = apply_resizing(img_C_base, start_resize_C, type="train")
        img_C_start_resized = paste_on_600(img_C_start_resized)
        img_C_initial, _, _, _ = apply_counting(
            img_C_start_resized, "+1", type="train", initial_count=start_count_C
        )

        # Generate B: Apply true transformations (resize first, then count)
        img_B_resized, _, _ = apply_resizing(img_A_base, true_param2, type="train")
        img_B_resized = paste_on_600(img_B_resized)
        _, img_B_correct, _, _ = apply_counting(
            img_B_resized, true_param1, type="train", initial_count=start_count_A
        )

        # Generate D (correct): Apply true transformations to C
        img_D_resized, _, _ = apply_resizing(img_C_base, true_param2, type="train")
        img_D_resized = paste_on_600(img_D_resized)
        _, img_D_correct, _, _ = apply_counting(
            img_D_resized, true_param1, type="train", initial_count=start_count_C
        )

        # Generate E (incorrect param1): true count, incorrect resize
        img_E_resized, _, _ = apply_resizing(img_C_base, incorrect_param2, type="train")
        img_E_resized = paste_on_600(img_E_resized)
        _, img_E_incorrect, _, _ = apply_counting(
            img_E_resized, true_param1, type="train", initial_count=start_count_C
        )

        # Generate F (incorrect param2): incorrect count, true resize
        img_F_resized, _, _ = apply_resizing(img_C_base, true_param2, type="train")
        img_F_resized = paste_on_600(img_F_resized)
        _, img_F_incorrect, _, _ = apply_counting(
            img_F_resized, incorrect_param1, type="train", initial_count=start_count_C
        )

        a, b, c = img_A_initial, img_B_correct, img_C_initial
        choices = [img_D_correct, img_E_incorrect, img_F_incorrect]
        sample_id = f"Counting{true_param1}_Resizing{true_param2}"
        return a, b, c, choices, sample_id

    def _generate_counting_rotation_composition(
        self, img_A_base: torch.Tensor, img_C_base: torch.Tensor
    ) -> tuple:
        """Handle Counting,Rotation composition"""

        from utils.dataset.transformations_kiva_adults import apply_counting, apply_rotation

        true_count_param = random.choice(self.param_options["kiva-functions"]["Counting"])
        true_rotation_param = random.choice(self.param_options["kiva-functions"]["Rotation"])

        incorrect_count_param = random.choice(
            [p for p in self.param_options["kiva-functions"]["Counting"] if p != true_count_param]
        )
        incorrect_rotation_param = random.choice(
            [
                p
                for p in self.param_options["kiva-functions"]["Rotation"]
                if p != true_rotation_param
            ]
        )

        # Get starting states
        start_count_A = random.choice(
            self.start_transformation_options["kiva-functions"]["Counting"]
        )
        start_rotation_A = random.choice(
            self.start_transformation_options["kiva-functions"]["Rotation"]
        )
        start_count_C = random.choice(
            self.start_transformation_options["kiva-functions"]["Counting"]
        )
        start_rotation_C = random.choice(
            self.start_transformation_options["kiva-functions"]["Rotation"]
        )

        # Generate A: apply starting transformations (rotate first, then count)
        _, img_A_start_rotated, _, _ = apply_rotation(
            img_A_base, start_rotation_A, type="train", initial_rotation="+0"
        )
        img_A_initial, _, _, _ = apply_counting(
            img_A_start_rotated, "+1", type="train", initial_count=start_count_A
        )

        # Generate C: apply starting transformations
        _, img_C_start_rotated, _, _ = apply_rotation(
            img_C_base, start_rotation_C, type="train", initial_rotation="+0"
        )
        img_C_initial, _, _, _ = apply_counting(
            img_C_start_rotated, "+1", type="train", initial_count=start_count_C
        )

        # Generate B: Apply true transformations (rotate first, then count)
        _, img_B_rotated, _, _ = apply_rotation(
            img_A_base, true_rotation_param, type="train", initial_rotation="+0"
        )
        _, img_B_correct, _, _ = apply_counting(
            img_B_rotated, true_count_param, type="train", initial_count=start_count_A
        )

        # Generate D (correct): Apply true transformations to C
        _, img_D_rotated, _, _ = apply_rotation(
            img_C_base, true_rotation_param, type="train", initial_rotation="+0"
        )
        _, img_D_correct, _, _ = apply_counting(
            img_D_rotated, true_count_param, type="train", initial_count=start_count_C
        )

        # Generate E (incorrect param1 - rotation): true count, incorrect rotation
        _, img_E_rotated, _, _ = apply_rotation(
            img_C_base, incorrect_rotation_param, type="train", initial_rotation="+0"
        )
        _, img_E_incorrect, _, _ = apply_counting(
            img_E_rotated, true_count_param, type="train", initial_count=start_count_C
        )

        # Generate F (incorrect param2 - counting): incorrect count, true rotation
        _, img_F_rotated, _, _ = apply_rotation(
            img_C_base, true_rotation_param, type="train", initial_rotation="+0"
        )
        _, img_F_incorrect, _, _ = apply_counting(
            img_F_rotated, incorrect_count_param, type="train", initial_count=start_count_C
        )

        a, b, c = img_A_initial, img_B_correct, img_C_initial
        choices = [img_D_correct, img_E_incorrect, img_F_incorrect]
        sample_id = f"Counting{true_count_param}_Rotation{true_rotation_param}"
        return a, b, c, choices, sample_id

    def _generate_reflect_resizing_composition(
        self, img_A_base: torch.Tensor, img_C_base: torch.Tensor
    ) -> tuple:
        """Handle Reflect,Resizing composition"""

        from utils.dataset.transformations_kiva_adults import (
            apply_reflection,
            apply_resizing,
            paste_on_600,
        )

        # Get parameters
        true_param1 = random.choice(self.param_options["kiva-functions"]["Reflect"])
        true_param2 = random.choice(self.param_options["kiva-functions"]["Resizing"])

        incorrect_param1 = random.choice(
            [p for p in self.param_options["kiva-functions"]["Reflect"] if p != true_param1]
        )
        incorrect_param2 = random.choice(
            [p for p in self.param_options["kiva-functions"]["Resizing"] if p != true_param2]
        )

        # Get starting states
        start_reflect_A, start_reflect_C = random.choices(
            self.start_transformation_options["kiva-functions"]["Reflect"],
            k=2,
        )
        start_resize_A, start_resize_C = random.choices(
            self.start_transformation_options["kiva-functions"]["Resizing"],
            k=2,
        )

        def apply_reflection_and_resizing(
            image: torch.Tensor, reflect_param: str, resizing_param: str
        ) -> torch.Tensor:
            img_temp, _, _, _ = apply_reflection(image, reflect_param, type="train")
            img_out, _, _ = apply_resizing(img_temp, resizing_param, type="train")
            return paste_on_600(img_out)

        # Generate A: apply starting transformations
        img_A_initial = apply_reflection_and_resizing(img_A_base, start_reflect_A, start_resize_A)

        # Generate C: apply starting transformations
        img_C_initial = apply_reflection_and_resizing(img_C_base, start_reflect_C, start_resize_C)

        # Generate B: Apply true transformations to base A (not compounding starts)
        img_B_correct = apply_reflection_and_resizing(img_A_initial, true_param1, true_param2)

        # Generate D (correct): Apply true transformations to base C
        img_D_correct = apply_reflection_and_resizing(img_C_initial, true_param1, true_param2)

        # Generate E (incorrect param2): true reflect, incorrect resize on base C
        img_E_incorrect = apply_reflection_and_resizing(
            img_C_initial, true_param1, incorrect_param2
        )

        # Generate F (incorrect param1): incorrect reflect, true resize on base C
        img_F_incorrect = apply_reflection_and_resizing(
            img_C_initial, incorrect_param1, true_param2
        )

        a, b, c = img_A_initial, img_B_correct, img_C_initial
        choices = [img_D_correct, img_E_incorrect, img_F_incorrect]
        sample_id = f"Reflect{true_param1}_Resizing{true_param2}"
        return a, b, c, choices, sample_id

    def _generate_resizing_rotation_composition(
        self, img_A_base: torch.Tensor, img_C_base: torch.Tensor
    ) -> tuple:
        """Handle Resizing,Rotation composition"""

        from utils.dataset.transformations_kiva_adults import (
            apply_resizing,
            apply_rotation,
            paste_on_600,
        )

        # Get parameters
        true_param1 = random.choice(self.param_options["kiva-functions"]["Resizing"])
        true_param2 = random.choice(self.param_options["kiva-functions"]["Rotation"])

        incorrect_param1 = random.choice(
            [p for p in self.param_options["kiva-functions"]["Resizing"] if p != true_param1]
        )
        incorrect_param2 = random.choice(
            [p for p in self.param_options["kiva-functions"]["Rotation"] if p != true_param2]
        )

        # Get starting states
        start_resize_A = random.choice(
            self.start_transformation_options["kiva-functions"]["Resizing"]
        )
        start_rotation_A = random.choice(
            self.start_transformation_options["kiva-functions"]["Rotation"]
        )
        start_resize_C = random.choice(
            self.start_transformation_options["kiva-functions"]["Resizing"]
        )
        start_rotation_C = random.choice(
            self.start_transformation_options["kiva-functions"]["Rotation"]
        )

        # Generate A: apply starting transformations
        img_A_temp, _, _ = apply_resizing(img_A_base, start_resize_A, type="train")
        img_A_temp = paste_on_600(img_A_temp)
        _, img_A_initial, _, _ = apply_rotation(
            img_A_temp, start_rotation_A, type="train", initial_rotation="+0"
        )

        # Generate C: apply starting transformations
        img_C_temp, _, _ = apply_resizing(img_C_base, start_resize_C, type="train")
        img_C_temp = paste_on_600(img_C_temp)
        _, img_C_initial, _, _ = apply_rotation(
            img_C_temp, start_rotation_C, type="train", initial_rotation="+0"
        )

        # Generate B: Apply true transformations to base A (not compounding starts)
        img_B_temp, _, _ = apply_resizing(img_A_initial, true_param1, type="train")
        img_B_temp = paste_on_600(img_B_temp)
        _, img_B_correct, _, _ = apply_rotation(
            img_B_temp, true_param2, type="train", initial_rotation="+0"
        )

        # Generate D (correct): Apply true transformations to base C
        img_D_temp, _, _ = apply_resizing(img_C_initial, true_param1, type="train")
        img_D_temp = paste_on_600(img_D_temp)
        _, img_D_correct, _, _ = apply_rotation(
            img_D_temp, true_param2, type="train", initial_rotation="+0"
        )

        # Generate E (incorrect param2): true resize, incorrect rotation on base C
        img_E_temp, _, _ = apply_resizing(img_C_initial, true_param1, type="train")
        img_E_temp = paste_on_600(img_E_temp)
        _, img_E_incorrect, _, _ = apply_rotation(
            img_E_temp, incorrect_param2, type="train", initial_rotation="+0"
        )

        # Generate F (incorrect param1): incorrect resize, true rotation on base C
        img_F_temp, _, _ = apply_resizing(img_C_initial, incorrect_param1, type="train")
        img_F_temp = paste_on_600(img_F_temp)
        _, img_F_incorrect, _, _ = apply_rotation(
            img_F_temp, true_param2, type="train", initial_rotation="+0"
        )

        a, b, c = img_A_initial, img_B_correct, img_C_initial
        choices = [img_D_correct, img_E_incorrect, img_F_incorrect]
        sample_id = f"Resizing{true_param1}_Rotation{true_param2}"
        return a, b, c, choices, sample_id

    def __getitem__(self, idx: int) -> tuple:
        if not self.rules:
            raise RuntimeError("Distribution config is empty or all weights are zero.")
        rule_str = random.choices(self.rules, self.weights, k=1)[0]
        level, rule = rule_str.rsplit("-", 1)

        # Dispatch to the correct generation function
        if level == "kiva":
            a, b, c, choices, sample_id = self._generate_kiva_level(rule)

        elif level == "kiva-functions":
            a, b, c, choices, sample_id = self._generate_kiva_functions_level(rule)

        elif level == "kiva-functions-compositionality":
            a, b, c, choices, sample_id = self._generate_kiva_functions_compositionality_level(rule)

        else:
            raise NotImplementedError(f"Level '{level}' is not implemented.")

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


if __name__ == "__main__":
    # ruff: noqa: ERA001
    # # --- Define the distribution from your table ---
    # distribution = {
    #     "kiva-Counting": 64,
    #     "kiva-Reflect": 32,
    #     "kiva-Resizing": 32,
    #     "kiva-Rotation": 48,
    #     "kiva-functions-Counting": 128,
    #     "kiva-functions-Reflect": 32,
    #     "kiva-functions-Resizing": 96,
    #     "kiva-functions-Rotation": 112,
    #     "kiva-functions-compositionality-Counting,Reflect": 256,
    #     "kiva-functions-compositionality-Counting,Resizing": 768,
    #     "kiva-functions-compositionality-Counting,Rotation": 896,
    #     "kiva-functions-compositionality-Reflect,Resizing": 192,
    #     "kiva-functions-compositionality-Resizing,Rotation": 96,
    # }
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
        "kiva-functions-compositionality-Reflect,Resizing": 1,
        "kiva-functions-compositionality-Resizing,Rotation": 0,
    }

    print("\nUsing the following generation distribution:")
    for rule, prob in distribution.items():
        print(f"  - {rule}: {prob:.2%}")

    # --- Instantiate the Dataset and DataLoader ---
    DATA_DIR = "/home/ubuntu/kiva-iccv/data/KiVA/untransformed objects"
    kiva_dataset = OnTheFlyKiVADataset(
        data_dir=DATA_DIR,
        distribution_config=distribution,
        epoch_length=100,  # smaller epoch for quick demo
    )

    kiva_loader = DataLoader(kiva_dataset, batch_size=1, shuffle=True, num_workers=0)

    # --- Fetch and inspect a batch ---
    print("\nFetching one batch from the DataLoader...")
    batch = next(iter(kiva_loader))
    a_batch, b_batch, c_batch, ch1_batch, ch2_batch, ch3_batch, labels, sample_ids = batch

    print("Batch loaded successfully.")
    print(f"Shape of image 'A' batch: {a_batch.shape}")
    print(f"Labels in the batch: {labels}")
    print(f"Sample IDs in the batch: {sample_ids}")

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
    grid_save_path = os.path.join(save_dir, "debug_grid.png")
    plt.savefig(grid_save_path)
    plt.close(fig)
    print(f"Saved grid image to: {grid_save_path}")

    print(f"Correct choice index for this sample is: {labels[0].item()}")
