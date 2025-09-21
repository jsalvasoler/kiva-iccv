import random

import torch
from torchvision import transforms
from torchvision.transforms import functional as F

# -----------------------------------------------------
# Transformation functions


def apply_color(image, target_color, type, initial_color=None):
    color_map = {
        "Red": torch.tensor([255, 0, 0], dtype=torch.float32),  # Red RGB
        "Green": torch.tensor([0, 128, 0], dtype=torch.float32),  # Green RGB
        "Blue": torch.tensor([0, 0, 255], dtype=torch.float32),  # Blue RGB
    }
    colors = ["Red", "Green", "Blue"]
    if target_color in colors:
        colors.remove(target_color)
    else:
        raise ValueError("Invalid color. Choose from 'Red', 'Green', or 'Blue'.")

    initial_color = (
        random.choice(colors) if initial_color is None else initial_color
    )  # Randomly select initial color if not provided
    incorrect_color = (
        colors[0] if initial_color == colors[1] else colors[1]
    )  # Remaining color for incorrect test option

    def color_overlap(img, color):
        has_alpha = img.shape[0] == 4
        alpha_channel = None

        if has_alpha:  # Separate alpha channel if present
            alpha_channel = img[3, :, :].clone()
            img = img[:3, :, :]

        img = img.float() / 255.0  # (3, H, W)
        height, width = img.shape[1], img.shape[2]

        target_color = color_map[color].view(3, 1, 1).repeat(1, height, width) / 255.0

        blended_img = (img + target_color) / 2  # Blend the original image with target color
        # To create a more distinctive change, use the option below; increasing the color
        # distinction
        # will decrease differentiation in object details
        # ruff: noqa: ERA001
        # blended_img = img * (1 - 0.7) + target_color * 0.7
        # # Blend the original image (0.3) with target color (0.7)

        blended_img = torch.clamp(blended_img, 0, 1)
        blended_img = (blended_img * 255).byte()

        if has_alpha:  # Reattach the alpha channel if it was present
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

    # Create canvas with white RGB channels and transparent alpha
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


def apply_counting(image, param, type, initial_count=None):
    if param not in {"+1", "+2", "-1", "-2"}:
        raise ValueError("Invalid counting operation. Choose from '+1', '+2', '-1', or '-2'.")

    starting_options_map = {
        "+": [2, 3, 4, 5],
        "-": [7, 6, 5, 4, 3],
    }

    def calculate_counts(operation, starting_count, param_num):
        if operation == "+":
            return starting_count + param_num, starting_count - 1
        elif operation == "-":
            return starting_count - param_num, starting_count + 1

    operation = param[0]  # +, -
    param_num = int(param[1:])  # 1, 2

    starting_count = (
        initial_count
        if initial_count is not None
        else random.choice(starting_options_map[operation])
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


def apply_reflection(image, parameter, type):
    if parameter == "X":
        correct_image = F.vflip(image)
        incorrect_image = F.hflip(image)  # Flip along the Y-axis for incorrect
        incorrect_option = "Y"
    elif parameter == "Y":
        correct_image = F.hflip(image)
        incorrect_image = F.vflip(image)  # Flip along the X-axis for incorrect
        incorrect_option = "X"
    elif parameter == "":
        correct_image = image
        incorrect_image = F.vflip(image)
        incorrect_option = "X"
    else:
        raise ValueError("Invalid reflect factor. Choose from 'X' or 'Y'.")

    if type == "train":
        return correct_image, 0, parameter
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


def apply_resizing(image, factor, type):
    # Resizing transformation
    image = transforms.Resize((300, 300), antialias=True)(image)

    # Pre-enlarging step for downscaling
    enlarge_first = factor.startswith("0.5")
    base_img = image
    if enlarge_first:
        H, W = image.shape[1:]
        base_img = F.resize(image, (H * 2, W * 2), antialias=True)
    image = base_img

    if factor == "0.5XY":
        correct_resize_factor = 0.5
        incorrect_resize_factor = 2.0
        incorrect_option = "2XY"
    elif factor == "2XY":
        correct_resize_factor = 2.0
        incorrect_resize_factor = 0.5
        incorrect_option = "0.5XY"
    elif factor == "XY":
        correct_resize_factor = 1.0
        incorrect_resize_factor = 1.0
        incorrect_option = "2XY"
    else:
        raise ValueError(f"Invalid resize factor: {factor}. Choose from '0.5XY' or '2XY'.")

    # Use original image dimensions for calculation, not the pre-enlarged base_img
    new_width, new_height = image.shape[2], image.shape[1]  # Original dimensions

    correct_new_width = int(new_width * correct_resize_factor)
    correct_new_height = int(new_height * correct_resize_factor)

    incorrect_new_width = int(new_width * incorrect_resize_factor)
    incorrect_new_height = int(new_height * incorrect_resize_factor)

    # Apply transformations to the appropriate base image (original or pre-enlarged)
    correct_image = transforms.Resize((correct_new_height, correct_new_width), antialias=True)(
        base_img
    )
    incorrect_image = transforms.Resize(
        (incorrect_new_height, incorrect_new_width), antialias=True
    )(base_img)

    if type == "train":
        return paste_on_600(correct_image), 0, factor
    elif type == "test":
        return (
            paste_on_600(correct_image),
            paste_on_600(incorrect_image),
            0,
            factor,
            incorrect_option,
        )


def apply_rotation(image, angle, type):
    if angle == "+90":
        correct_angle = -90
        incorrect_angle = 180
    elif angle == "-90":
        correct_angle = 90
        incorrect_angle = 180
    elif angle == "180":
        correct_angle = 180
        incorrect_angle = random.choice([90, -90])
    else:
        raise ValueError("Invalid rotation angle. Choose from '+90', '-90', or '180'.")

    def combine_angles(a1: str, a2: str) -> str:  # Always returns a positive angle
        def to_int(a):
            # int → int
            if isinstance(a, int):
                return a
            # str → int
            if a.startswith("+"):
                return int(a[1:])
            if a.startswith("-"):
                return -int(a[1:])
            return int(a)

        total = (to_int(a1) + to_int(a2)) % 360  # Wrap once around
        return str(total)

    final_correct_angle = combine_angles("0", angle)
    final_incorrect_angle = combine_angles(combine_angles("0", angle), incorrect_angle)

    correct_image = F.rotate(image, correct_angle)
    incorrect_image = F.rotate(image, incorrect_angle)

    if type == "train":
        return correct_image, 0, final_correct_angle
    elif type == "test":
        return correct_image, incorrect_image, 0, final_correct_angle, final_incorrect_angle
