"""
Break a sample of the KiVA dataset into 6 parts:
-
1. ex_before
2. ex_after
3. test_before
4. choice_a
5. choice_b
6. choice_c

"""

import argparse
import json
import os
from multiprocessing import Pool

import tqdm
from PIL import Image, ImageDraw


def investigate_sizes(dataset_json_path: str) -> None:
    """Investigate image sizes in the dataset."""

    image_dir = dataset_json_path.replace(".json", "")
    images = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    if not dataset_json_path.endswith("test.json"):
        with open(dataset_json_path) as f:
            dataset = json.load(f)
            assert len(images) == len(dataset), "Number of images and dataset must match"

    sizes = []

    for image in images:
        img_path = f"{image_dir}/{image}"
        img = Image.open(img_path)
        sizes.append(img.size)

    assert len(set(sizes)) == 1, "All images must have the same size"


def split_kiva_image(img_path: str) -> dict[str, Image.Image]:
    """
    Splits a composite image from the KiVA dataset into its 6 logical parts.

    The function operates based on the consistent structure of KiVA images:
    - A top section (the first 1/3 of the height) contains the training example.
    - A bottom section (the remaining 2/3 of the height) contains the test choices.
    - Both sections are divided by vertical lines into 'before' and 'after' states.

    Args:
        img_path: Path to the composite KiVA image.

    Returns:
        A dictionary containing the 6 cropped PIL Image objects:
        - 'ex_before': The "before" state from the top training example.
        - 'ex_after': The "after" state from the top training example.
        - 'test_before': The common "before" state from the bottom test section.
        - 'choice_a': The "after" state for choice A.
        - 'choice_b': The "after" state for choice B.
        - 'choice_c': The "after" state for choice C.
    """
    # Load the composite image and get its dimensions
    img = Image.open(img_path)

    # get the width and height of the image
    w, h = img.size
    mid_w = w // 2

    s = 600
    v_top = 50
    v_bottom = 830
    ex_key = 1294
    test_before_key = 52
    a_key = 653
    b_key = 1898
    c_key = 3143

    CROP_BOXES = {
        "ex_before": (ex_key, v_top, ex_key + s, v_top + s),
        "ex_after": (2 * mid_w - ex_key - s, v_top, 2 * mid_w - ex_key, v_top + s),
        "test_before": (test_before_key, v_bottom, test_before_key + s, v_bottom + s),
        "choice_a": (a_key, v_bottom, a_key + s, v_bottom + s),
        "choice_b": (b_key, v_bottom, b_key + s, v_bottom + s),
        "choice_c": (c_key, v_bottom, c_key + s, v_bottom + s),
    }

    def add_white_border(img, border_size=5):
        # Create a drawing context
        draw = ImageDraw.Draw(img)
        w, h = img.size

        draw.rectangle([(0, 0), (w, border_size)], fill="white")
        draw.rectangle([(0, h - border_size), (w, h)], fill="white")
        draw.rectangle([(0, 0), (border_size, h)], fill="white")
        draw.rectangle([(w - border_size, 0), (w, h)], fill="white")

        return img

    parts = {}
    for part_name, box in CROP_BOXES.items():
        parts[part_name] = add_white_border(img.crop(box), border_size=4)
        assert parts[part_name].size == (600, 600), f"Part {part_name} has wrong size"

    return parts


def save_split_parts(img_path: str, output_dir: str) -> None:
    """
    Split a KiVA image and save the parts to separate files.

    Args:
        img_path: Path to the composite KiVA image
        output_dir: Directory to save the split parts
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get the base filename without extension
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    # Split the image using the corrected function
    parts = split_kiva_image(img_path)

    # Save each part as JPG
    for part_name, part_img in parts.items():
        output_path = os.path.join(output_dir, f"{base_name}_{part_name}.jpg")
        part_img.save(output_path)


def process_single_image(args: tuple[str, str, str]) -> None:
    """
    Process a single image - used by multiprocessing pool.

    Args:
        args: Tuple of (trial_id, dataset_dir, output_dir)
    """
    image, dataset_dir, output_dir = args
    img_path = os.path.join(dataset_dir, image)
    save_split_parts(img_path, output_dir)


def transform_dataset(dataset_json_path: str, output_dir: str) -> None:
    """
    Transform the dataset into the new format using multiprocessing.
    """
    image_dir = dataset_json_path.replace(".json", "")
    which = image_dir.split("/")[-1]
    images = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

    # Get the dataset directory from the json path
    dataset_dir = dataset_json_path.replace(".json", "")

    # Prepare arguments for multiprocessing
    args_list = [(image, dataset_dir, output_dir) for image in images]

    # Use multiprocessing pool to process images in parallel
    with Pool(processes=16) as pool:
        list(
            tqdm.tqdm(
                pool.imap(process_single_image, args_list),
                total=len(args_list),
                desc=f"Processing {which} dataset",
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform KIVA dataset images into split subimages."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="unit",
        choices=["unit", "train", "validation", "test"],
        help="Which dataset to process (unit, train, validation, test).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    which_dataset = args.dataset
    investigate_sizes(f"./data/{which_dataset}.json")

    os.makedirs(f"./data/split_{which_dataset}", exist_ok=True)
    transform_dataset(f"./data/{which_dataset}.json", f"./data/split_{which_dataset}")
