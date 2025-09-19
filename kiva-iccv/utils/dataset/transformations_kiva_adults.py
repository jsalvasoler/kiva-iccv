import random

import torch
from torchvision import transforms
from torchvision.transforms import functional as F

# -----------------------------------------------------
# Transformation functions


def apply_color(image, target_color, type="train", train_color=None):
    color_map = {
        "Red": torch.tensor([255, 0, 0], dtype=torch.float32),
        "Green": torch.tensor([0, 128, 0], dtype=torch.float32),
        "Blue": torch.tensor([0, 0, 255], dtype=torch.float32),
        "Yellow": torch.tensor([255, 255, 50], dtype=torch.float32),  # (R + G)
        "Grey": None,  # Special case for greyscale
    }

    if target_color not in color_map:
        raise ValueError("Invalid color. Choose from 'Red', 'Yellow', 'Green', 'Blue', or 'Grey'.")

    available_colors = [
        color for color in color_map if color != target_color and color != train_color
    ]

    initial_color, incorrect_color = random.sample(available_colors, 2)  # Sample without overlap

    def color_overlap(img, color):
        has_alpha = img.shape[0] == 4
        alpha_channel = None

        if has_alpha:  # Separate alpha channel if present
            alpha_channel = img[3, :, :].clone()
            img = img[:3, :, :]

        img = img.float() / 255.0  # Normalize to [0, 1]
        height, width = img.shape[1], img.shape[2]

        if color == "Grey":  # Convert to grayscale
            grayscale = img.mean(dim=0, keepdim=True)  # Average over RGB channels
            blended_img = grayscale.expand_as(img)
        else:
            target_color = color_map[color].view(3, 1, 1).repeat(1, height, width) / 255.0
            blended_img = (img + target_color) / 2  # Blend the original image with the target color

        blended_img = torch.clamp(blended_img, 0, 1)
        blended_img = (blended_img * 255).byte()

        if has_alpha:  # Reattach alpha channel if it exists
            blended_img = torch.cat((blended_img, alpha_channel.unsqueeze(0)), dim=0)

        return blended_img

    initial_image = color_overlap(image, initial_color)
    correct_image = color_overlap(image, target_color)
    incorrect_image = color_overlap(image, incorrect_color)

    if type == "train":
        return initial_image, correct_image, initial_color, target_color
    elif type == "test":
        return (
            initial_image,
            correct_image,
            incorrect_image,
            initial_color,
            target_color,
            incorrect_color,
        )


def generate_grid_image(image, count):
    _, img_height, img_width = image.shape

    # Resize the base object image to fit in a grid
    max_items_per_row = int((10**0.5) + 1)
    item_size = min(img_width, img_height) // max_items_per_row
    shrunken_image = F.resize(image, (item_size, item_size), antialias=True)

    if (
        shrunken_image.shape[0] == 3
    ):  # If no alpha channel, add alpha channel with 255 (opaque) for objects
        alpha_channel = torch.ones((1, item_size, item_size), dtype=image.dtype) * 255
        shrunken_image = torch.cat((shrunken_image, alpha_channel), dim=0)

    canvas = torch.full((4, img_height, img_width), 255, dtype=image.dtype)
    canvas[3, :, :] = 0  # Set alpha channel to 0 (transparent)

    # Arrange objects on canvas in grid layout
    for i in range(count):
        x = (i % max_items_per_row) * item_size
        y = (i // max_items_per_row) * item_size
        canvas[:, y : y + item_size, x : x + item_size] = (
            shrunken_image  # Paste shrunken image onto canvas
        )

    return canvas


def apply_counting(image, param, type="train", initial_count=None):
    if param not in {"+1", "+2", "-1", "-2", "x2", "x3", "d2", "d3"}:
        raise ValueError(
            "Invalid counting operation. "
            "Choose from '+1', '+2', '-1', '-2', 'x2', 'x3', 'd2', or 'd3'."
        )

    starting_options_map = {
        "+": [2, 3, 4, 5],
        "-": [7, 6, 5, 4, 3],
        "x": {2: [2, 3, 4], 3: [1, 2, 3]},
        "d": {2: [8, 6, 4], 3: [9, 6, 3]},
    }

    def calculate_counts(operation, starting_count, param_num):
        if operation == "+":
            return starting_count + param_num, starting_count - 1
        elif operation == "-":
            return starting_count - param_num, starting_count + 1
        elif operation == "x":
            return starting_count * param_num, starting_count + 1
        elif operation == "d":
            return starting_count // param_num, starting_count - 1

    operation = param[0]  # +, -, x, d
    param_num = int(param[1:])  # 1, 2, or 3

    starting_options = (
        starting_options_map[operation]
        if operation in ["+", "-"]
        else starting_options_map[operation][param_num]
    )
    starting_count = random.choice(
        [num for num in starting_options if num != initial_count] or starting_options
    )
    correct_count, incorrect_count = calculate_counts(operation, starting_count, param_num)

    initial_image = generate_grid_image(image, starting_count)
    correct_image = generate_grid_image(image, correct_count)
    incorrect_image = generate_grid_image(image, incorrect_count)

    if type == "train":
        return initial_image, correct_image, starting_count, correct_count
    elif type == "test":
        return (
            initial_image,
            correct_image,
            incorrect_image,
            starting_count,
            correct_count,
            incorrect_count,
        )


def apply_reflection(image, parameter, type="train"):
    if parameter == "X":
        correct_image = F.vflip(image)
        incorrect_image = F.hflip(image)  # Flip along the Y-axis for incorrect
        incorrect_option = "Y"
    elif parameter == "Y":
        correct_image = F.hflip(image)
        incorrect_image = F.vflip(image)  # Flip along the X-axis for incorrect
        incorrect_option = "X"
    elif parameter == "XY":
        correct_image = F.hflip(F.vflip(image))  # Reflect across both X- and Y-axes
        incorrect_option = random.choice(["X", "Y"])
        incorrect_image = F.hflip(image) if incorrect_option == "X" else F.vflip(image)
    elif parameter == "":
        correct_image = image
        incorrect_image = F.vflip(image)
        incorrect_option = random.choice(["X", "Y", "XY"])
    else:
        raise ValueError("Invalid reflect factor. Choose from 'X', 'Y', 'XY', or ''.")

    if type == "train":
        return correct_image, image, 0, parameter
    elif type == "test":
        return correct_image, incorrect_image, 0, parameter, incorrect_option


def paste_on_600(img: torch.Tensor, canvas_size: int = 600) -> torch.Tensor:
    _, h, w = img.shape

    # Down-scale very large inputs so the larger edge is 600
    if max(h, w) > canvas_size:
        scale = canvas_size / float(max(h, w))
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        img = F.resize(img, (new_h, new_w), antialias=True)
        _, h, w = img.shape

    pad_left = (canvas_size - w) // 2
    pad_right = canvas_size - w - pad_left
    pad_top = (canvas_size - h) // 2
    pad_bottom = canvas_size - h - pad_top

    return F.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=255)


def apply_resizing(image, factor: str, type="train"):
    """
    Applies a specified resizing transformation to an image.

    This generalized version can handle factors like '0.8X', '1.2Y', '1.5XY', etc.
    It dynamically parses the factor string to determine the scale and axis.

    Args:
        image (torch.Tensor): The input image tensor (C, H, W).
        factor (str): The resizing factor string, e.g., "0.8X", "1.2Y", "1.5XY".
                      It consists of a float followed by 'X', 'Y', or 'XY'.
        type (str, optional): The mode of operation.
                               - "train": Returns the correctly resized image and its label.
                               - "test": Returns the correct image, an incorrect alternative,
                                         and their respective labels. Defaults to "train".

    Returns:
        - if type is "train": (correct_image, 0, factor)
        - if type is "test": (correct_image, incorrect_image, 0, factor, incorrect_option)
    """
    print(f"DEBUG: factor: {factor}")
    # --- 1. Parse the factor string to get scale and axis ---
    try:
        if factor.endswith("XY"):
            scale = float(factor[:-2])
            axis = "XY"
        elif factor.endswith("X"):
            scale = float(factor[:-1])
            axis = "X"
        elif factor.endswith("Y"):
            scale = float(factor[:-1])
            axis = "Y"
        else:
            # This case will be caught by the ValueError below if no match
            raise ValueError
    except (ValueError, IndexError) as e:
        raise ValueError(
            "Invalid resize factor format. "
            "Expected a float followed by 'X', 'Y', or 'XY'. "
            "Examples: '0.8X', '1.2Y', '1.5XY'."
        ) from e

    print(f"DEBUG: scale: {scale}, axis: {axis}")

    # --- 2. Handle the pre-enlarging step for downscaling ---
    # This logic is kept from the original to potentially improve downscaling quality.
    base_img = image
    # if scale < 1.0:
    #     H, W = image.shape[1:]
    #     base_img = F.resize(image, (int(H / scale), int(W / scale)), antialias=True)

    # --- 3. Determine correct and incorrect transformation parameters ---
    if axis == "XY":
        correct_resize_factors = (scale, scale)
        # For symmetric scaling, the incorrect option is the reciprocal
        incorrect_scale = 1.0 / scale
        incorrect_resize_factors = (incorrect_scale, incorrect_scale)
        # Format to 2 decimal places for a clean label
        incorrect_option = f"{incorrect_scale:.2f}XY"
    elif axis == "X":
        correct_resize_factors = (scale, 1.0)
        # For asymmetric scaling, the incorrect option is scaling the other axis
        incorrect_resize_factors = (1.0, scale)
        incorrect_option = f"{scale}Y"
    elif axis == "Y":
        correct_resize_factors = (1.0, scale)
        # For asymmetric scaling, the incorrect option is scaling the other axis
        incorrect_resize_factors = (scale, 1.0)
        incorrect_option = f"{scale}X"

    # --- 4. Calculate new dimensions and apply transformations ---
    new_width, new_height = base_img.shape[2], base_img.shape[1]

    # Remember: transforms.Resize expects (height, width)
    correct_new_height = int(new_height * correct_resize_factors[1])
    correct_new_width = int(new_width * correct_resize_factors[0])

    print(f"DEBUG: new_height: {new_height}, new_width: {new_width}")
    print(f"DEBUG: correct_resize_factors: {correct_resize_factors}")
    print(f"DEBUG: correct_new_height: {correct_new_height}, correct_new_width: {correct_new_width}")

    incorrect_new_height = int(new_height * incorrect_resize_factors[1])
    incorrect_new_width = int(new_width * incorrect_resize_factors[0])

    correct_image = transforms.Resize((correct_new_height, correct_new_width), antialias=True)(
        base_img
    )

    # --- 5. Return based on the specified type ---
    if type == "train":
        return correct_image, 0, factor

    elif type == "test":
        incorrect_image = transforms.Resize(
            (incorrect_new_height, incorrect_new_width), antialias=True
        )(base_img)
        return correct_image, incorrect_image, 0, factor, incorrect_option

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/ubuntu/kiva-iccv/kiva-iccv")
    from utils.dataset.transformations_kiva_adults import apply_resizing_original, apply_resizing
    from on_the_fly_dataset import OnTheFlyKiVADataset

    DATA_DIR = "/home/ubuntu/kiva-iccv/data/KiVA/untransformed objects"
    kiva_dataset = OnTheFlyKiVADataset(
        data_dir=DATA_DIR,
        distribution_config={"kiva-functions-Resizing": 1},
        epoch_length=100,  # smaller epoch for quick demo
    )


    img = kiva_dataset._load_random_images(1, ["Resizing"])[0]

    import torch
    import matplotlib.pyplot as plt
    import os
    from torchvision.utils import make_grid
    from torchvision.transforms.functional import to_pil_image

    # Create debug directory if it doesn't exist
    debug_dir = "debug_batch_images"
    os.makedirs(debug_dir, exist_ok=True)

    # Ensure img is in [0, 1] range for visualization
    def normalize_img(img: torch.Tensor) -> torch.Tensor:
        # If img is float and max > 1, scale to [0,1]
        if img.dtype == torch.float32 or img.dtype == torch.float64:
            if img.max() > 1.0:
                return img / 255.0
            else:
                return img
        # If img is uint8, convert to float and scale
        elif img.dtype == torch.uint8:
            return img.float() / 255.0
        else:
            return img

    img_vis = normalize_img(img)
    out, _, _ = apply_resizing(img, "0.5XY", "train")
    out_vis = normalize_img(out)

    # Compute difference image
    diff = (out_vis - img_vis).abs()

    # Normalize diff for visualization
    diff_vis = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)

    # Stack images for grid: original, resized, diff
    images = torch.stack([img_vis, out_vis, diff_vis])

    # Make grid (3 images in a row)
    grid = make_grid(images, nrow=3, normalize=False)

    # Save grid to file
    grid_img = to_pil_image(grid)
    grid_path = os.path.join(debug_dir, "resizing_debug_grid.png")
    grid_img.save(grid_path)
    print(f"Saved debug grid image to {grid_path}")

def apply_resizing_original(image, factor, type="train"):
    enlarge_first = factor.startswith("0.5")
    base_img = image
    if enlarge_first:
        H, W = image.shape[1:]
        base_img = F.resize(image, (H * 2, W * 2), antialias=True)
    image = base_img

    resize_factors = {
        "0.5XY": ((0.5, 0.5), (2.0, 2.0), "2XY"),
        "2XY": ((2.0, 2.0), (0.5, 0.5), "0.5XY"),
        "0.5X": ((0.5, 1.0), (1.0, 0.5), "0.5Y"),
        "0.5Y": ((1.0, 0.5), (0.5, 1.0), "0.5X"),
        "2X": ((2.0, 1.0), (1.0, 2.0), "2Y"),
        "2Y": ((1.0, 2.0), (2.0, 1.0), "2X"),
    }

    try:
        correct_resize_factors, incorrect_resize_factors, incorrect_option = resize_factors[factor]
    except KeyError as e:
        raise ValueError(
            "Invalid resize factor. Choose from '0.5XY', '2XY', '0.5X', '0.5Y', '2X', or '2Y'."
        ) from e

    new_width, new_height = image.shape[2], image.shape[1]

    correct_new_width = int(new_width * correct_resize_factors[0])
    correct_new_height = int(new_height * correct_resize_factors[1])

    incorrect_new_width = int(new_width * incorrect_resize_factors[0])
    incorrect_new_height = int(new_height * incorrect_resize_factors[1])

    correct_image = transforms.Resize((correct_new_height, correct_new_width), antialias=True)(
        image
    )
    incorrect_image = transforms.Resize(
        (incorrect_new_height, incorrect_new_width), antialias=True
    )(image)

    if type == "train":
        return correct_image, 0, factor
    elif type == "test":
        return correct_image, incorrect_image, 0, factor, incorrect_option


def apply_rotation(image, angle, type="train", train_angle=None):
    matches = {
        "+45": ["+135", "180"],
        "-45": ["-135", "180"],
        "+90": ["180"],
        "-90": ["180"],
        "+135": ["+45", "180"],
        "-135": ["-45", "180"],
        "180": ["+45", "-45", "+90", "-90"],
    }

    if angle in matches and matches[angle]:
        incorrect_angle = random.choice(matches[angle])
    else:
        raise ValueError(
            "Invalid rotation angle. Choose from '+45', '-45', '+90', '-90', '+135', or '-135'."
        )

    initial_rotation = random.choice(
        [angle for angle in ["+45", "-45", "+90", "-90", "+135", "-135"] if angle != train_angle]
        or ["+45", "-45", "+90", "-90", "+135", "-135"]
    )

    def parse_angle(angle):
        if angle[:1] == "+":
            return -int(angle[1:])
        elif angle[:1] == "-":
            return int(angle[1:])
        elif angle == "180":
            return 180

    def combine_angles(a1: str, a2: str) -> str:  # Always returns a positive angle
        def s2i(s: str) -> int:
            if s.startswith("+"):
                return int(s[1:])
            elif s.startswith("-"):
                return -int(s[1:])
            else:
                return int(s)

        total = (s2i(a1) + s2i(a2)) % 360  # Wrap into a single turn

        if total == 0:
            return "0"
        if total == 180:
            return "180"
        return f"{total}"

    final_start_angle = combine_angles(initial_rotation, "0")
    final_correct_angle = combine_angles(initial_rotation, angle)
    final_incorrect_angle = combine_angles(combine_angles(initial_rotation, angle), incorrect_angle)

    original_image = F.rotate(image, parse_angle(initial_rotation))
    correct_image = F.rotate(original_image, parse_angle(angle))
    incorrect_image = F.rotate(original_image, parse_angle(incorrect_angle))

    if type == "train":
        return original_image, correct_image, final_start_angle, final_correct_angle
    elif type == "test":
        return (
            original_image,
            correct_image,
            incorrect_image,
            final_start_angle,
            final_correct_angle,
            final_incorrect_angle,
        )
