"""
AI usage statement:

AI was used to assist with researching and debugging, as well as helping
with creating docstrings. All code was writte, reviewed and/or modified by a human.
"""

import os
import json
import subprocess
import zipfile
from pathlib import Path


def download_kaggle_dataset(dataset_name: str, output_path: str) -> str | None:
    # Check if dataset already exists
    if os.path.exists(output_path):
        image_files = list(Path(output_path).glob("**/*.jpg"))
        if image_files:
            print(
                f"Found existing dataset at {output_path} with {len(image_files)} images"
            )
            return output_path

    os.makedirs(output_path, exist_ok=True)
    print(f"Downloading {dataset_name} using curl...")

    # Find and load kaggle.json credentials
    if os.path.exists("kaggle/kaggle.json"):
        with open("kaggle/kaggle.json", "r") as f:
            credentials = json.load(f)
    else:
        raise FileNotFoundError("Kaggle credentials not found")

    zip_path = os.path.join(output_path, "dataset.zip")

    # Build curl command
    curl_cmd = [
        "curl",
        "-L",
        f"https://www.kaggle.com/api/v1/datasets/download/{dataset_name}",
        "-o",
        zip_path,
        "--header",
        f"Authorization: Basic {credentials['username']}:{credentials['key']}",
    ]

    try:
        # Download
        subprocess.run(curl_cmd, check=True)
        print(f"Dataset downloaded to {zip_path}")

        # Extract
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_path)
        print(f"Dataset extracted to {output_path}")

        # Remove zip file
        os.remove(zip_path)
        return output_path

    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        return None


def verify_kaggle_dataset(path: str) -> bool:
    # Check if dataset already exists
    if os.path.exists(path):
        image_files = list(Path(path).glob("**/*.jpg"))
        if image_files:
            print(f"Found existing dataset at {path} with {len(image_files)} images")
            return True

    print(f"Dataset not found at {path}")
    return False
