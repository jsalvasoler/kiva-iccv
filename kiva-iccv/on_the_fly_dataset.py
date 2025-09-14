import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms import functional as F


def _generate_grid_image(image: torch.Tensor, count: int, canvas_size: int = 600) -> torch.Tensor:
    """Generates an image with a grid of `count` copies of the input image."""
    # Ensure the input is a 4-channel tensor (RGBA) for transparency
    if image.shape[0] == 3:
        alpha_channel = torch.full((1, image.shape[1], image.shape[2]), 255, dtype=image.dtype)
        image = torch.cat((image, alpha_channel), dim=0)

    max_items_per_row = int((10**0.5) + 1)
    item_size = min(canvas_size, canvas_size) // max_items_per_row
    shrunken_image = F.resize(image, (item_size, item_size), antialias=True)

    canvas = torch.zeros((4, canvas_size, canvas_size), dtype=image.dtype)

    for i in range(count):
        x = (i % max_items_per_row) * item_size
        y = (i // max_items_per_row) * item_size
        if y + item_size <= canvas_size and x + item_size <= canvas_size:
            canvas[:, y : y + item_size, x : x + item_size] = shrunken_image

    return canvas


def _apply_counting(
    image: torch.Tensor, param: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Applies a counting transformation, returning initial, correct, and incorrect versions."""
    if param not in {"+1", "+2", "-1", "-2", "x2", "x3", "d2", "d3"}:
        raise ValueError("Invalid counting operation.")

    starting_options_map = {
        "+": [2, 3, 4, 5],
        "-": [7, 6, 5, 4, 3],
        "x": {2: [2, 3, 4], 3: [1, 2, 3]},
        "d": {2: [8, 6, 4], 3: [9, 6, 3]},
    }
    operation, param_num = param[0], int(param[1:])
    options = (
        starting_options_map[operation]
        if operation in ["+", "-"]
        else starting_options_map[operation][param_num]
    )
    start_count = random.choice(options)

    if operation == "+":
        correct_count, incorrect_count = start_count + param_num, start_count - 1
    elif operation == "-":
        correct_count, incorrect_count = start_count - param_num, start_count + 1
    elif operation == "x":
        correct_count, incorrect_count = start_count * param_num, start_count + 1
    elif operation == "d":
        correct_count, incorrect_count = start_count // param_num, start_count - 1

    # Ensure counts are within a reasonable range
    correct_count = max(1, correct_count)
    incorrect_count = max(1, incorrect_count)
    if incorrect_count == correct_count:
        incorrect_count += 1

    initial_image = _generate_grid_image(image, start_count)
    correct_image = _generate_grid_image(image, correct_count)
    incorrect_image = _generate_grid_image(image, incorrect_count)

    return initial_image, correct_image, incorrect_image


def _apply_reflection(
    image: torch.Tensor, param: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Applies a reflection transformation, returning initial, correct, and incorrect versions."""
    initial_image = image.clone()
    if param == "X":
        correct_image = F.vflip(image)
        incorrect_image = F.hflip(image)
    elif param == "Y":
        correct_image = F.hflip(image)
        incorrect_image = F.vflip(image)
    elif param == "XY":
        correct_image = F.hflip(F.vflip(image))
        incorrect_image = F.vflip(image) if random.random() > 0.5 else F.hflip(image)
    else:
        raise ValueError("Invalid reflection factor. Choose from 'X', 'Y', or 'XY'.")
    return initial_image, correct_image, incorrect_image


def _apply_resizing(
    image: torch.Tensor, param: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Applies a resizing transformation, returning initial, correct, and incorrect versions."""
    canvas_size = 600

    # Base image should be smaller to allow for enlargement
    base_img = F.resize(image, (canvas_size // 2, canvas_size // 2), antialias=True)

    factors = {
        "0.5XY": (0.5, "2XY"),
        "2XY": (2.0, "0.5XY"),
        "0.5X": (0.5, "0.5Y"),
        "0.5Y": (0.5, "0.5X"),
        "2X": (2.0, "2Y"),
        "2Y": (2.0, "2X"),
    }
    correct_factor, incorrect_param = factors[param]

    w, h = base_img.shape[2], base_img.shape[1]

    # Calculate correct dimensions
    scale_x = correct_factor if "X" in param else 1.0
    scale_y = correct_factor if "Y" in param else 1.0
    correct_dims = (int(h * scale_y), int(w * scale_x))

    # Calculate incorrect dimensions
    inc_factor = 0.5 if "2" in incorrect_param else 2.0
    inc_scale_x = inc_factor if "X" in incorrect_param else 1.0
    inc_scale_y = inc_factor if "Y" in incorrect_param else 1.0
    incorrect_dims = (int(h * inc_scale_y), int(w * inc_scale_x))

    correct_image = F.resize(base_img, correct_dims, antialias=True)
    incorrect_image = F.resize(base_img, incorrect_dims, antialias=True)

    def pad_to_canvas(img):
        pad_left = (canvas_size - img.shape[2]) // 2
        pad_top = (canvas_size - img.shape[1]) // 2
        pad_right = canvas_size - img.shape[2] - pad_left
        pad_bottom = canvas_size - img.shape[1] - pad_top
        return F.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=0)

    return pad_to_canvas(base_img), pad_to_canvas(correct_image), pad_to_canvas(incorrect_image)


def _apply_rotation(
    image: torch.Tensor, param: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Applies a rotation transformation, returning initial, correct, and incorrect versions."""
    angle_map = {
        "+45": -45,
        "-45": 45,
        "+90": -90,
        "-90": 90,
        "+135": -135,
        "-135": 135,
        "180": 180,
    }
    if param not in angle_map:
        raise ValueError("Invalid rotation angle.")

    correct_angle = angle_map[param]
    # Pick a different, non-trivial rotation for the incorrect choice
    incorrect_angle = random.choice([a for a in [-90, 90, 180] if a != correct_angle])

    initial_image = image.clone()
    correct_image = F.rotate(image, correct_angle)
    incorrect_image = F.rotate(image, incorrect_angle)

    return initial_image, correct_image, incorrect_image


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

        self.transformation_functions = {
            "Counting": _apply_counting,
            "Reflect": _apply_reflection,
            "Resizing": _apply_resizing,
            "Rotation": _apply_rotation,
        }

        # normalize distribution config
        distribution_config = {
            k: v / sum(distribution_config.values()) for k, v in distribution_config.items()
        }

        self.rules = list(distribution_config.keys())
        self.weights = list(distribution_config.values())
        self.base_canvas_transform = transforms.Resize((600, 600), antialias=True)

        self.param_options = {
            "Counting": ["+1", "+2", "-1", "-2", "x2", "x3", "d2", "d3"],
            "Reflect": ["X", "Y", "XY"],
            "Resizing": ["0.5XY", "2XY", "0.5X", "0.5Y", "2X", "2Y"],
            "Rotation": ["+45", "-45", "+90", "-90", "+135", "-135", "180"],
        }
        self.all_rules = list(self.param_options.keys())

        print(
            f"Dataset Initialized. An epoch will consist of {self.epoch_length} generated samples."
        )

    def __len__(self) -> int:
        return self.epoch_length

    def _load_random_images(self, n: int, rule_types: list[str]) -> list[torch.Tensor]:
        """Loads n unique random images suitable for the given transformation types."""
        # For composite transformations, find a common directory if possible, else default
        primary_rule = rule_types[0]
        dir_path = self.dir_map[primary_rule]

        image_files = self.file_lists[primary_rule]
        if len(image_files) < n:
            raise ValueError(f"Not enough images in {dir_path} to sample {n} unique images.")

        selected_files = random.sample(image_files, n)
        return [self.base_canvas_transform(read_image(str(dir_path / f))) for f in selected_files]

    def _apply_and_get_images(self, func, image, param):
        """Helper to get initial and correct images from a transform function."""
        initial, correct, _ = func(image, param)
        return initial, correct

    def __getitem__(self, idx: int) -> tuple:
        rule_str = random.choices(self.rules, self.weights, k=1)[0]
        level, rule = rule_str.rsplit("-", 1)

        if level == "kiva":
            # A->B :: C->D. Goal: Identify the same transformation (rule + parameter).
            # A and C are the same object identity.
            img_A = self._load_random_images(1, [rule])[0]
            img_C = img_A.clone()  # Same object identity

            true_param = random.choice(self.param_options[rule])

            # Ensure at least 2 other params are available for distractors
            param_choices = [p for p in self.param_options[rule] if p != true_param]
            if len(param_choices) < 2:
                raise ValueError(f"Rule {rule} needs at least 3 parameter options for kiva level.")
            incorrect_params = random.sample(param_choices, 2)

            transform_func = self.transformation_functions[rule]

            img_A_initial, img_B_correct = self._apply_and_get_images(
                transform_func, img_A, true_param
            )
            img_C_initial, img_D_correct = self._apply_and_get_images(
                transform_func, img_C, true_param
            )

            _, img_E_incorrect = self._apply_and_get_images(
                transform_func, img_C, incorrect_params[0]
            )
            _, img_F_incorrect = self._apply_and_get_images(
                transform_func, img_C, incorrect_params[1]
            )

            a, b, c = img_A_initial, img_B_correct, img_C_initial
            choices = [img_D_correct, img_E_incorrect, img_F_incorrect]
            sample_id = f"{rule_str}_{true_param}"

        elif level == "kiva-functions":
            # A->B :: C->D. Goal: Identify same rule, can be different params.
            # Object identity changes.
            img_A, img_C = self._load_random_images(2, [rule])

            true_param_A = random.choice(self.param_options[rule])
            true_param_C = random.choice(self.param_options[rule])  # Params can differ

            transform_func = self.transformation_functions[rule]

            # Create distractors from different transformation RULES
            distractor_rules = random.sample([r for r in self.all_rules if r != rule], 2)

            img_A_initial, img_B_correct = self._apply_and_get_images(
                transform_func, img_A, true_param_A
            )
            img_C_initial, img_D_correct = self._apply_and_get_images(
                transform_func, img_C, true_param_C
            )

            # Distractor 1
            distractor_func_1 = self.transformation_functions[distractor_rules[0]]
            distractor_param_1 = random.choice(self.param_options[distractor_rules[0]])
            _, img_E_incorrect = self._apply_and_get_images(
                distractor_func_1, img_C, distractor_param_1
            )

            # Distractor 2
            distractor_func_2 = self.transformation_functions[distractor_rules[1]]
            distractor_param_2 = random.choice(self.param_options[distractor_rules[1]])
            _, img_F_incorrect = self._apply_and_get_images(
                distractor_func_2, img_C, distractor_param_2
            )

            a, b, c = img_A_initial, img_B_correct, img_C_initial
            choices = [img_D_correct, img_E_incorrect, img_F_incorrect]
            sample_id = f"{rule_str}"

        elif level == "kiva-functions-compositionality":
            # A->B :: C->D. Goal: Identify two rules applied in sequence. Object identity changes.
            rule1, rule2 = rule.split(",")
            img_A, img_C = self._load_random_images(2, [rule1, rule2])

            func1 = self.transformation_functions[rule1]
            param1 = random.choice(self.param_options[rule1])
            func2 = self.transformation_functions[rule2]
            param2 = random.choice(self.param_options[rule2])

            # Generate A -> B
            _, img_A_intermediate = self._apply_and_get_images(func1, img_A, param1)
            _, img_B_final = self._apply_and_get_images(func2, img_A_intermediate, param2)

            # Generate C -> D (Correct Choice)
            _, img_C_intermediate = self._apply_and_get_images(func1, img_C, param1)
            _, img_D_final = self._apply_and_get_images(func2, img_C_intermediate, param2)

            # Generate Distractors (apply only one of the two transformations)
            _, img_E_incorrect = self._apply_and_get_images(func1, img_C, param1)  # Only rule 1
            _, img_F_incorrect = self._apply_and_get_images(func2, img_C, param2)  # Only rule 2

            a, b, c = img_A, img_B_final, img_C
            choices = [img_D_final, img_E_incorrect, img_F_incorrect]
            sample_id = f"{rule_str}"

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
        "kiva-Counting": 1,
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

    print("\nUsing the following generation distribution:")
    for rule, prob in distribution.items():
        print(f"  - {rule}: {prob:.2%}")

    # --- Instantiate the Dataset and DataLoader ---
    DATA_DIR = "/home/ubuntu/kiva-iccv/KiVA/untransformed objects"
    kiva_dataset = OnTheFlyKiVADataset(
        data_dir=DATA_DIR,
        distribution_config=distribution,
        epoch_length=100,  # smaller epoch for quick demo
    )

    kiva_loader = DataLoader(kiva_dataset, batch_size=4, shuffle=True, num_workers=2)

    # --- Fetch and inspect a batch ---
    print("\nFetching one batch from the DataLoader...")
    try:
        batch = next(iter(kiva_loader))
        a_batch, b_batch, c_batch, ch1_batch, ch2_batch, ch3_batch, labels, sample_ids = batch

        print("Batch loaded successfully.")
        print(f"Shape of image 'A' batch: {a_batch.shape}")
        print(f"Labels in the batch: {labels}")
        print(f"Sample IDs in the batch: {sample_ids}")

        # --- Optional: Save a visual of the first item in the batch for debugging ---
        save_dir = "debug_batch_images_v2"
        os.makedirs(save_dir, exist_ok=True)

        # Create a grid of all 6 images for the first sample
        first_sample_images = torch.stack(
            [a_batch[0], b_batch[0], c_batch[0], ch1_batch[0], ch2_batch[0], ch3_batch[0]]
        )

        # Denormalize for viewing
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        first_sample_images = (first_sample_images * std) + mean

        grid = transforms.ToPILImage()(torch.cat(list(first_sample_images), dim=2))

        sample_id_str = sample_ids[0].replace(",", "_")  # Sanitize filename
        save_path = os.path.join(save_dir, f"{sample_id_str}.png")
        grid.save(save_path)
        print(f"Saved visualization of first sample to: {save_path}")
        print(f"Correct choice index for this sample is: {labels[0].item()}")

    except (ValueError, FileNotFoundError) as e:
        print("\n--- ERROR ---")
        print(e)
        print("Please check your configuration and data paths.")
