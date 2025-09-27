import json
import os
import shutil
import zipfile

import requests


def download_data(data_path: str) -> None:
    # Create necessary directories
    os.makedirs(data_path, exist_ok=True)  # main data directory

    # Download and extract train, validation, and test datasets
    datasets = ["train", "validation", "test"]

    for dataset in datasets:
        print(f"Downloading {dataset} data...")

        # Download zip file
        zip_url = f"https://storage.googleapis.com/kiva-challenge-bucket/{dataset}.zip"
        zip_path = os.path.join(data_path, f"{dataset}.zip")

        response = requests.get(zip_url, stream=True)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        # Extract zip file
        print(f"Extracting {dataset} data...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_path)

        # Remove zip file and __MACOSX directory if it exists
        os.remove(zip_path)
        macosx_dir = os.path.join(data_path, "__MACOSX")
        if os.path.exists(macosx_dir):
            shutil.rmtree(macosx_dir)

        # Download JSON metadata for train and validation (not for test)
        if dataset != "test":
            print(f"Downloading {dataset} metadata...")
            json_url = f"https://storage.googleapis.com/kiva-key-bucket/{dataset}.json"
            json_path = os.path.join(data_path, f"{dataset}.json")

            response = requests.get(json_url, stream=True)
            response.raise_for_status()
            with open(json_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

    # Create a unit dataset, which contains 16 samples from the train set
    print("Creating unit dataset...")

    # 1. Open the train.json file
    with open(os.path.join(data_path, "train.json")) as f:
        train_data = json.load(f)

    # 2. Take first 16 trial IDs and their data, and save them to a new json file
    first_16_trial_ids = list(train_data.keys())[:16]
    unit_trials_dict = {trial_id: train_data[trial_id] for trial_id in first_16_trial_ids}

    # Ensure the unit directory exists
    os.makedirs(os.path.join(data_path, "unit"), exist_ok=True)

    with open(os.path.join(data_path, "unit.json"), "w") as f:
        json.dump(unit_trials_dict, f, indent=2)

    # 3. Copy the first 16 images from the train set to the unit set
    for trial_id in first_16_trial_ids:
        src_img = os.path.join(data_path, "train", f"{trial_id}.jpg")
        dst_img = os.path.join(data_path, "unit", f"{trial_id}.jpg")
        shutil.copy(src_img, dst_img)

    print(f"Copied {len(first_16_trial_ids)} images to create the unit dataset")
    print("--- KiVA Data Setup Complete! ---")


if __name__ == "__main__":
    data_path = "./data/"  # all KiVA data (images and annotations) will be placed here
    download_data(data_path)
