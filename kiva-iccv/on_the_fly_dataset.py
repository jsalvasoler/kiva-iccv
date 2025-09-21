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
        true_final_count = start_count + int(true_param[1:])
        possible_final_counts = [x for x in list(range(1, 10)) if x != true_final_count]
        return [
            f"+{c - start_count}" if c - start_count > 0 else f"-{start_count - c}"
            for c in possible_final_counts
            if abs(c - start_count) <= 2 and c != start_count
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
            from utils.dataset.transformations_kiva import apply_resizing

            def resize_and_paste_600(image: torch.Tensor, param: str) -> torch.Tensor:
                img_temp, _, _ = apply_resizing(image, param, type="train")
                print(f"DEBUG: Resized image shape: {img_temp.shape}")

                # Custom paste function that preserves relative sizes within 600x600 canvas
                _, h, w = img_temp.shape
                canvas_size = 600

                # If image is larger than canvas, scale it down but preserve relative sizes
                if max(h, w) > canvas_size:
                    # For 2XY images, we want them to appear larger than others
                    # Scale down less aggressively to show the size difference
                    if param == "2XY":
                        # Scale to fit 80% of canvas to show it's larger
                        scale = (canvas_size * 0.8) / float(max(h, w))
                    else:
                        # Normal scaling for other cases
                        scale = canvas_size / float(max(h, w))

                    new_h, new_w = int(round(h * scale)), int(round(w * scale))
                    img_temp = torch.nn.functional.interpolate(
                        img_temp.unsqueeze(0), size=(new_h, new_w), mode="bilinear", antialias=True
                    ).squeeze(0)
                    _, h, w = img_temp.shape

                # Center the image on the canvas
                pad_left = (canvas_size - w) // 2
                pad_right = canvas_size - w - pad_left
                pad_top = (canvas_size - h) // 2
                pad_bottom = canvas_size - h - pad_top

                return torch.nn.functional.pad(
                    img_temp, (pad_left, pad_top, pad_right, pad_bottom), value=255
                )

            print("DEBUG: Resizing params:")
            print(f" - True param: {true_param}")
            print(f" - Incorrect params: {incorrect_params}")

            img_B_correct = resize_and_paste_600(img_A, true_param)
            img_D_correct = resize_and_paste_600(img_C, true_param)
            img_E_incorrect = resize_and_paste_600(img_C, incorrect_params[0])
            img_F_incorrect = resize_and_paste_600(img_C, incorrect_params[1])
            img_A_initial, img_C_initial = img_A, img_C

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

        options = (
            self.start_transformation_options["kiva-functions"][rule]
            if rule != "Counting"
            else self._get_counting_start_options(true_param)
        )
        start_transformations = random.sample(options, k=2)
        print(f"DEBUG: Start transformations: {start_transformations}")

        # Ensure at least 2 other params are available for distractors
        if rule == "Counting":
            params_choices = self._get_counting_incorrect_param_options(
                true_param, start_transformations[1]
            )
        else:
            params_choices = [
                p for p in self.param_options["kiva-functions"][rule] if p != true_param
            ]
        print(f"DEBUG: Params choices: {params_choices}")
        if len(params_choices) < 2:
            raise ValueError(
                f"Rule {rule} needs at least 3 parameter options for kiva-functions level."
            )
        incorrect_params = random.sample(params_choices, k=2)
        print(f"DEBUG: Incorrect params: {incorrect_params}")

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

        incorrect_reflect_param = random.choice(
            [p for p in self.param_options["kiva-functions"]["Reflect"] if p != true_reflect_param]
        )

        # Get starting states
        start_count_A, start_count_C = random.sample(
            self._get_counting_start_options(true_count_param), k=2
        )
        incorrect_count_param = random.choice(
            self._get_counting_incorrect_param_options(true_count_param, start_count_C)
        )
        start_reflect_A, start_reflect_C = random.sample(
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
        true_count_param = random.choice(self.param_options["kiva-functions"]["Counting"])
        true_resizing_param = random.choice(self.param_options["kiva-functions"]["Resizing"])

        incorrect_options = [
            p for p in self.param_options["kiva-functions"]["Resizing"] if p != true_resizing_param
        ]
        incorrect_resize_param = random.choice(incorrect_options)

        start_count_A, start_count_C = random.sample(
            self._get_counting_start_options(true_count_param), k=2
        )
        incorrect_count_param = random.choice(
            self._get_counting_incorrect_param_options(true_count_param, start_count_C)
        )
        start_resize_A, start_resize_C = random.sample(
            self.start_transformation_options["kiva-functions"]["Resizing"], k=2
        )

        def make_initial(image: torch.Tensor, start_count: int, start_resize: str) -> torch.Tensor:
            img_temp, _, _ = apply_resizing(image, start_resize, type="train")
            img_temp = paste_on_600(img_temp)
            img_out, _, _, _ = apply_counting(
                img_temp, "+1", type="train", initial_count=start_count
            )
            return img_out

        def apply_true_chain(
            image: torch.Tensor, start_count: int, true_count: str, true_resize: str
        ) -> torch.Tensor:
            img_temp, _, _ = apply_resizing(image, true_resize, type="train")
            img_temp = paste_on_600(img_temp)
            _, img_out, _, _ = apply_counting(
                img_temp, true_count, type="train", initial_count=start_count
            )
            return img_out

        # Generate A: apply starting transformations (resize first, then count)
        img_A_initial = make_initial(img_A_base, start_count_A, start_resize_A)
        img_B_correct = apply_true_chain(
            img_A_base, start_count_A, true_count_param, true_resizing_param
        )

        img_C_initial = make_initial(img_C_base, start_count_C, start_resize_C)
        img_D_correct = apply_true_chain(
            img_C_base, start_count_C, true_count_param, true_resizing_param
        )

        img_E_incorrect = apply_true_chain(
            img_C_base, start_count_C, true_count_param, incorrect_resize_param
        )
        img_F_incorrect = apply_true_chain(
            img_C_base, start_count_C, incorrect_count_param, true_resizing_param
        )

        a, b, c = img_A_initial, img_B_correct, img_C_initial
        choices = [img_D_correct, img_E_incorrect, img_F_incorrect]
        sample_id = f"Counting{true_count_param}_Resizing{true_resizing_param}"
        return a, b, c, choices, sample_id

    def _generate_counting_rotation_composition(
        self, img_A_base: torch.Tensor, img_C_base: torch.Tensor
    ) -> tuple:
        """Handle Counting,Rotation composition"""

        from utils.dataset.transformations_kiva_adults import apply_counting, apply_rotation

        true_count_param = random.choice(self.param_options["kiva-functions"]["Counting"])
        true_rotation_param = random.choice(self.param_options["kiva-functions"]["Rotation"])

        incorrect_rotation_param = random.choice(
            [
                p
                for p in self.param_options["kiva-functions"]["Rotation"]
                if p != true_rotation_param
            ]
        )

        # Get starting states
        start_count_A, start_count_C = random.sample(
            self._get_counting_start_options(true_count_param), k=2
        )
        incorrect_count_param = random.choice(
            self._get_counting_incorrect_param_options(true_count_param, start_count_C)
        )
        start_rotation_A, start_rotation_C = random.sample(
            self.start_transformation_options["kiva-functions"]["Rotation"], k=2
        )
        print(f"DEBUG: Start rotations: {start_rotation_A}, {start_rotation_C}")
        print(f"DEBUG: Start counts: {start_count_A}, {start_count_C}")
        print(f"DEBUG: True rotation: {true_rotation_param}")
        print(f"DEBUG: True count: {true_count_param}")
        print(f"DEBUG: Incorrect rotation: {incorrect_rotation_param}")
        print(f"DEBUG: Incorrect count: {incorrect_count_param}")

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
            img_A_start_rotated, true_rotation_param, type="train", initial_rotation="+0"
        )
        _, img_B_correct, _, _ = apply_counting(
            img_B_rotated, true_count_param, type="train", initial_count=start_count_A
        )

        # Generate D (correct): Apply true transformations to C
        _, img_D_rotated, _, _ = apply_rotation(
            img_C_start_rotated, true_rotation_param, type="train", initial_rotation="+0"
        )
        _, img_D_correct, _, _ = apply_counting(
            img_D_rotated, true_count_param, type="train", initial_count=start_count_C
        )

        # Generate E (incorrect param1 - rotation): true count, incorrect rotation
        _, img_E_rotated, _, _ = apply_rotation(
            img_C_start_rotated, incorrect_rotation_param, type="train", initial_rotation="+0"
        )
        _, img_E_incorrect, _, _ = apply_counting(
            img_E_rotated, true_count_param, type="train", initial_count=start_count_C
        )

        # Generate F (incorrect param2 - counting): incorrect count, true rotation
        _, img_F_rotated, _, _ = apply_rotation(
            img_C_start_rotated, true_rotation_param, type="train", initial_rotation="+0"
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

        incorrect_reflect_param = random.choice(
            [p for p in self.param_options["kiva-functions"]["Reflect"] if p != true_param1]
        )
        incorrect_resize_param = random.choice(
            [p for p in self.param_options["kiva-functions"]["Resizing"] if p != true_param2]
        )

        # Get starting states
        start_reflect_A, start_reflect_C = random.sample(
            self.start_transformation_options["kiva-functions"]["Reflect"],
            k=2,
        )
        start_resize_A, start_resize_C = random.sample(
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
            img_C_initial, true_param1, incorrect_resize_param
        )

        # Generate F (incorrect param1): incorrect reflect, true resize on base C
        img_F_incorrect = apply_reflection_and_resizing(
            img_C_initial, incorrect_reflect_param, true_param2
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

        incorrect_resize_param = random.choice(
            [p for p in self.param_options["kiva-functions"]["Resizing"] if p != true_param1]
        )
        incorrect_rotate_param = random.choice(
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
            img_E_temp, incorrect_rotate_param, type="train", initial_rotation="+0"
        )

        # Generate F (incorrect param1): incorrect resize, true rotation on base C
        img_F_temp, _, _ = apply_resizing(img_C_initial, incorrect_resize_param, type="train")
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


def main(case):
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
        "kiva-functions-compositionality-Reflect,Resizing": 0,
        "kiva-functions-compositionality-Resizing,Rotation": 0,
    }
    distribution[case] = 1

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
    save_dir = os.path.join(save_dir, case)
    os.makedirs(save_dir, exist_ok=True)
    grid_save_path = os.path.join(save_dir, f"{sample_id_str}.png")
    plt.savefig(grid_save_path)
    plt.close(fig)
    print(f"Saved grid image to: {grid_save_path}")

    print(f"Correct choice index for this sample is: {labels[0].item()}")


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
        print(f"Finished {case}")
