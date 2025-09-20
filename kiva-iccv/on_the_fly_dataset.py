import os
import random
import shutil
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

        # normalize distribution config
        distribution_config = {
            k: v / sum(distribution_config.values()) for k, v in distribution_config.items()
        }

        self.rules = list(distribution_config.keys())
        self.weights = list(distribution_config.values())
        self.base_canvas_transform = transforms.Resize((600, 600), antialias=True)

        self.param_options = {
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
                "Resizing": ["0.8XY", "0.8X", "1.2XY", "1.2X", "0.8Y", "1.2Y", "1XY"],
                "Rotation": ["+0", "+45", "-45", "+90", "-90", "+135", "-135"],
            },
        }

        self.param_options["kiva-functions-compositionality"] = self.param_options["kiva-functions"]

        self.all_rules = ["Counting", "Reflect", "Resizing", "Rotation"]

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

    def _apply_and_get_images(self, func, image, param, start_count=None):
        """Helper to get initial and correct images from a transform function."""
        if func == self.transformation_functions["Counting"]:
            if start_count is not None:
                initial, correct, _, _, _, _ = func(
                    image, param, type="test", initial_count=start_count
                )
            else:
                initial, correct, _, _, _, _ = func(image, param, type="test")
        elif func == self.transformation_functions["Reflect"]:
            initial, correct, _, _ = func(image, param, type="test")
        elif func == self.transformation_functions["Resizing"]:
            initial, correct, _, _ = func(image, param, type="test")
        elif func == self.transformation_functions["Rotation"]:
            initial, correct, _, _, _, _ = func(image, param, type="test")
        return initial, correct

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
        start_transformations = ["+0", "+0"]
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

            true_param = "180"
            incorrect_params[0] = "+135"
            incorrect_params[1] = "-90"
            start_transformations[0] = "+0"
            start_transformations[1] = "+0"

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

    def __getitem__(self, idx: int) -> tuple:
        rule_str = random.choices(self.rules, self.weights, k=1)[0]
        level, rule = rule_str.rsplit("-", 1)

        if level == "kiva":
            a, b, c, choices, sample_id = self._generate_kiva_level(rule)

        elif level == "kiva-functions":
            a, b, c, choices, sample_id = self._generate_kiva_functions_level(rule)

        elif level == "kiva-functions-compositionality":
            raise NotImplementedError(f"Level '{level}' is not implemented.")
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
        "kiva-functions-Rotation": 1,
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
    # wipe it first
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
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
