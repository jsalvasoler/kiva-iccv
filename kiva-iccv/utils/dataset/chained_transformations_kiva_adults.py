import random

import torch
from torchvision import transforms
from torchvision.transforms import functional as F

"""
Only 2 transformation pairs are supported.

Test Options:
1. Correct
2. Correct in Domain A, Incorrect in Domain B
3. Incorrect in Domain A, Correct in Domain B
4. No change
5. Doesn't apply
"""


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

    canvas = torch.zeros(
        (4, img_height, img_width), dtype=image.dtype
    )  # 4 channels (RGBA), 0 alpha for transparency

    # Arrange objects on canvas in grid layout
    for i in range(count):
        x = (i % max_items_per_row) * item_size
        y = (i // max_items_per_row) * item_size
        canvas[:, y : y + item_size, x : x + item_size] = (
            shrunken_image  # Paste shrunken image onto canvas
        )

    return canvas


def apply_counting(image, param, type, keep_initial_value=None, avoid_initial_value=None):
    if param not in {"+1", "+2", "-1", "-2", "x2", "x3", "d2", "d3"}:
        raise ValueError(
            "Invalid counting operation. "
            "Choose from '+1', '+2', '-1', '-2', 'x2', 'x3', 'd2', or 'd3'."
        )

    starting_options_map = {
        "+": [2, 3, 4, 5],
        "-": [7, 6, 5, 4],  # removed 3 for -2
        "x": {2: [2, 3, 4], 3: [1, 2, 3]},
        "d": {2: [8, 6, 4], 3: [9, 6]},  # removed 3 for d3
    }

    def calculate_counts(operation, starting_count, param_num):
        if operation == "+":
            correct_count = starting_count + param_num
            return correct_count, random.choice([correct_count - 1, correct_count + 1])
        elif operation == "-":
            correct_count = starting_count - param_num
            return correct_count, random.choice([correct_count - 1, correct_count + 1])
        elif operation == "x":
            correct_count = starting_count * param_num
            return correct_count, random.choice([correct_count - 1, correct_count + 1])
        elif operation == "d":
            correct_count = starting_count // param_num
            return correct_count, random.choice([correct_count - 1, correct_count + 1])

    operation = param[0]  # +, -, x, d
    param_num = int(param[1:])  # 1, 2, or 3

    if keep_initial_value is not None:
        starting_count = int(keep_initial_value)
    else:
        starting_options = (
            starting_options_map[operation]
            if operation in ["+", "-"]
            else starting_options_map[operation][param_num]
        )
        starting_count = random.choice(
            [num for num in starting_options if num != avoid_initial_value] or starting_options
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
            [correct_count, incorrect_count],
        )


def apply_reflection(image, parameter, type):
    if parameter == "X":
        correct_image = F.vflip(image)
        incorrect_image = random.choice(
            [F.hflip(image), image]
        )  # Flip along the Y-axis for incorrect
        incorrect_option = "Y"
    elif parameter == "Y":
        correct_image = F.hflip(image)
        incorrect_image = random.choice(
            [F.vflip(image), image]
        )  # Flip along the X-axis for incorrect
        incorrect_option = "X"
    elif parameter == "XY":
        correct_image = F.hflip(F.vflip(image))  # Reflect across both X- and Y-axes
        incorrect_option = random.choice(["X", "Y"])
        incorrect_image = F.hflip(image) if incorrect_option == "X" else F.vflip(image)
    else:
        raise ValueError("Invalid reflect factor. Choose from 'X', 'Y', or 'XY'.")

    if type == "train":
        return correct_image, image, parameter, parameter
    elif type == "test":
        return image.clone(), correct_image, incorrect_image, 0, incorrect_option


def paste_on_600(img: torch.Tensor, canvas_size: int = 600) -> torch.Tensor:
    """
    Returns a 600 × 600 tensor.  The input *content* is left untouched:
    - if smaller than 600 it is centred and padded with zeros / transparent alpha
    - if one edge happens to be >600 we first down-scale so the larger edge becomes 600
      (this should not happen in your current pipeline, but keeps things safe)
    """
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

    return F.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)


def apply_resizing(image, factor, type):
    enlarge_first = factor.startswith("0.5")
    base_img = image
    if enlarge_first:
        H, W = image.shape[1:]
        base_img = F.resize(image, (H * 2, W * 2), antialias=True)
    image = base_img

    def get_opts(factor):
        # parse scale and dim without regex
        if factor.startswith("0.5"):
            scale = 0.5
            dim = factor[3:]  # Skip "0.5"
        else:
            scale = 2.0
            dim = factor[1:]  # Skip "2"

        # Define correct (w,h)
        correct = (scale if "X" in dim else 1, scale if "Y" in dim else 1)

        # Pick a random other dim
        dims = ["X", "Y", "XY"]
        other = random.choice([d for d in dims if d != dim])

        # Define incorrect (w,h) and its key
        incorrect = (scale if "X" in other else 1, scale if "Y" in other else 1)
        incorrect_key = f"{scale}{other}"

        return correct, incorrect, incorrect_key

    try:
        correct_resize_factors, incorrect_resize_factors, incorrect_option = get_opts(factor)
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
        return paste_on_600(image.clone()), paste_on_600(correct_image), 0, factor
    elif type == "test":
        return (
            paste_on_600(image.clone()),
            paste_on_600(correct_image),
            paste_on_600(incorrect_image),
            0,
            incorrect_option,
        )


def apply_rotation(image, angle, type, keep_initial_value=None, avoid_initial_value=None):
    if keep_initial_value is not None:
        initial_rotation = keep_initial_value
    else:
        initial_rotation = random.choice(
            [
                angle
                for angle in ["+45", "-45", "+90", "-90", "+135", "-135"]
                if angle != avoid_initial_value
            ]
            or ["+45", "-45", "+90", "-90", "+135", "-135"]
        )

    def parse_angle(angle):
        if angle[:1] == "+":
            return -int(angle[1:])
        elif angle[:1] == "-":
            return int(angle[1:])
        elif angle == "180":
            return 180

    random_incorrect_angle = random.choice(["-90", "+90"])

    original_image = F.rotate(image, parse_angle(initial_rotation))
    correct_image = F.rotate(original_image, parse_angle(angle))
    incorrect_image = F.rotate(correct_image, parse_angle(random_incorrect_angle))

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

        if total == 0:
            return "0"
        if total == 180:
            return "180"
        return f"{total}"

    final_initial_angle = combine_angles(
        initial_rotation, "0"
    )  # Initial rotation is always the first angle
    final_correct_angle = combine_angles(initial_rotation, angle)
    final_incorrect_angle = combine_angles(
        combine_angles(initial_rotation, angle), random_incorrect_angle
    )

    if type == "train":
        return (
            original_image,
            correct_image,
            final_initial_angle,
            combine_angles(initial_rotation, angle),
        )
    elif type == "test":
        return (
            original_image,
            correct_image,
            incorrect_image,
            final_initial_angle,
            [final_correct_angle, final_incorrect_angle],
        )


def apply_transformation_chain(
    image, transformation, parameter, type, keep_initial_value=None, avoid_initial_value=None
):
    if transformation == "Counting":
        if type == "train":
            initial_image, correct_image, input_val, output_val = apply_counting(
                image, parameter, type="train", keep_initial_value=keep_initial_value
            )
            return initial_image, correct_image, input_val, output_val
        else:
            initial_image, correct_image, incorrect_image, input_val, incorrect_val = (
                apply_counting(
                    image,
                    parameter,
                    type="test",
                    keep_initial_value=keep_initial_value
                    if keep_initial_value is not None
                    else None,
                    avoid_initial_value=avoid_initial_value,
                )
            )
            return initial_image, correct_image, incorrect_image, input_val, incorrect_val
    elif transformation == "Reflect":
        if type == "train":
            original, correct_image, input_val, output_val = apply_reflection(
                image, parameter, type="train"
            )
            return original, correct_image, input_val, output_val
        else:
            original, correct_image, incorrect_image, input_val, incorrect_val = apply_reflection(
                image, parameter, type="test"
            )
            return original, correct_image, incorrect_image, input_val, incorrect_val
    elif transformation == "Resize":
        if type == "train":
            original, correct_image, input_val, output_val = apply_resizing(
                image, parameter, type="train"
            )
            return original, correct_image, input_val, output_val
        else:
            original, correct_image, incorrect_image, input_val, incorrect_val = apply_resizing(
                image, parameter, type="test"
            )
            return original, correct_image, incorrect_image, input_val, incorrect_val
    elif transformation == "2DRotation":
        if type == "train":
            original, correct_image, input_val, output_val = apply_rotation(
                image, parameter, type="train", keep_initial_value=keep_initial_value
            )
            return original, correct_image, input_val, output_val
        else:
            original, correct_image, incorrect_image, input_val, incorrect_val = apply_rotation(
                image,
                parameter,
                type="test",
                keep_initial_value=keep_initial_value if keep_initial_value is not None else None,
                avoid_initial_value=avoid_initial_value,
            )
            return original, correct_image, incorrect_image, input_val, incorrect_val
