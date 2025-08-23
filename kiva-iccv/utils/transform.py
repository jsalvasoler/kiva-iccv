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
from PIL import Image


def investigate_sizes(dataset_json_path: str) -> None:
    """Investigate image sizes in the dataset."""
    with open(dataset_json_path) as f:
        dataset = json.load(f)

    sizes = []

    for trial_id, _ in dataset.items():
        img_path = dataset_json_path.replace(".json", f"/{trial_id}.jpg")
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

    s = 560
    v_top = 50
    v_bottom = 860
    mid_w = 3795 // 2
    ex_key = 1299

    CROP_BOXES = {
        "ex_before": (ex_key, v_top, ex_key + s, v_top + s),
        "ex_after": (2 * mid_w - ex_key - s, v_top, 2 * mid_w - ex_key, v_top + s),
        "test_before": (60, v_bottom, 60 + s, v_bottom + s),
        "choice_a": (675, v_bottom, 675 + s, v_bottom + s),
        "choice_b": (1915, v_bottom, 1915 + s, v_bottom + s),
        "choice_c": (3175, v_bottom, 3175 + s, v_bottom + s),
    }

    parts = {}
    for part_name, box in CROP_BOXES.items():
        parts[part_name] = img.crop(box)

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
    trial_id, dataset_dir, output_dir = args
    img_path = os.path.join(dataset_dir, f"{trial_id}.jpg")
    save_split_parts(img_path, output_dir)


def transform_dataset(dataset_json_path: str, output_dir: str) -> None:
    """
    Transform the dataset into the new format using multiprocessing.
    """
    with open(dataset_json_path) as f:
        dataset = json.load(f)

    # Get the dataset directory from the json path
    dataset_dir = dataset_json_path.replace(".json", "")

    # Prepare arguments for multiprocessing
    args_list = [(trial_id, dataset_dir, output_dir) for trial_id in dataset.keys()]

    # Use multiprocessing pool to process images in parallel
    with Pool(processes=16) as pool:
        list(
            tqdm.tqdm(
                pool.imap(process_single_image, args_list),
                total=len(args_list),
                desc="Processing images",
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
