import json
import os
import shutil
import zipfile

import requests
from helper import get_content_path, setup_kiva_data_set


def download_data(data_path: str):
    # Initial Setup: fill in specifications ---

    # Create necessary directories
    os.makedirs(data_path, exist_ok=True)  # main data directory

    # Load helper.py
    print("Setting up helper.py...")

    content_path = get_content_path()
    os.makedirs(os.path.join(content_path, "output"), exist_ok=True)  # helper directory

    # Set up Training Data
    train_trials, train_stimuli = setup_kiva_data_set("train", data_path)

    # Set up Validation Data
    val_trials, val_stimuli = setup_kiva_data_set("validation", data_path)

    # remove content_path / output
    shutil.rmtree(os.path.join(content_path, "output"))

    # create a unit dataset, which contains 16 samples from the train set

    # 1. open the train.json file
    with open(os.path.join(data_path, "train.json")) as f:
        train_data = json.load(f)

    # 2. take first 16 trial IDs and their data, and save them to a new json file
    first_16_trial_ids = list(train_data.keys())[:16]
    unit_trials_dict = {trial_id: train_data[trial_id] for trial_id in first_16_trial_ids}

    # Ensure the unit directory exists
    os.makedirs(os.path.join(data_path, "unit"), exist_ok=True)

    with open(os.path.join(data_path, "unit.json"), "w") as f:
        json.dump(unit_trials_dict, f, indent=2)

    # 3. copy the first 16 images from the train set to the unit set
    for trial_id in first_16_trial_ids:
        src_img = os.path.join(data_path, "train", f"{trial_id}.jpg")
        dst_img = os.path.join(data_path, "unit", f"{trial_id}.jpg")
        shutil.copy(src_img, dst_img)

    print(f"\n--- Setting up unit data within {data_path} ---")
    print(f"Copied {len(first_16_trial_ids)} images to create the unit dataset")

    # 4. remove the content_path directory
    shutil.rmtree(content_path)
    print("\n--- KiVA Data Setup Complete! ---")


def download_test_data():
    # URL and target paths
    url = "https://storage.googleapis.com/kiva-challenge-bucket/test.zip"
    target_dir = "./data"
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, "test.zip")

    # Download the zip file
    print("Downloading test data...")
    response = requests.get(url, stream=True)
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Download complete.")

    # Unzip the contents
    print("Unzipping test data...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)
    print("Unzipping complete.")

    # (Optional) remove the zip file after extraction and the __MACOSX directory
    macosx_dir = os.path.join(target_dir, "__MACOSX")
    if os.path.exists(macosx_dir):
        shutil.rmtree(macosx_dir)
    os.remove(zip_path)
    print("Done! Test data is in:", target_dir)


if __name__ == "__main__":
    data_path = "./data/"  # all KiVA data (images and annotations) will be placed here
    download_data(data_path)
    download_test_data()
