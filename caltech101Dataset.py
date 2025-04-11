import os
import json
import subprocess
import zipfile
from pathlib import Path
import base64
import torch
from PIL import Image
from torchvision import transforms


def download_caltech101_dataset(output_path="caltech_101"):
    """
    Downloads the CalTech101 dataset from Kaggle and extracts it to the specified output path.

    Args:
        output_path (str): Directory where the dataset will be downloaded and extracted.
                          Default is "caltech_101".

    Returns:
        str: Path to the extracted dataset directory, or None if download failed.
    """

    dataset_name = "imbikramsaha/caltech-101"

    # Check if dataset already exists
    if os.path.exists(output_path):
        image_files = list(Path(output_path).glob("**/*.jpg"))
        if image_files:
            print(f"Found existing dataset at {output_path} with {len(image_files)} images")
            return output_path

    # Create a temporary directory for the zip file
    temp_dir = os.path.join(os.path.dirname(output_path), "temp_download")
    os.makedirs(temp_dir, exist_ok=True)

    print(f"Downloading CalTech101 dataset using curl...")

    # Find and load kaggle.json credentials
    kaggle_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_path):
        if os.path.exists("kaggle/kaggle.json"):
            os.makedirs(os.path.dirname(kaggle_path), exist_ok=True)
            with open("kaggle/kaggle.json", "r") as f:
                credentials = json.load(f)
        else:
            raise FileNotFoundError("Kaggle credentials not found")
    else:
        with open(kaggle_path, "r") as f:
            credentials = json.load(f)

    zip_path = os.path.join(temp_dir, "caltech101.zip")

    # Properly encode credentials for Authorization header
    auth_token = base64.b64encode(f"{credentials['username']}:{credentials['key']}".encode()).decode()

    # Build curl command
    curl_cmd = [
        "curl", "-L",
        f"https://www.kaggle.com/api/v1/datasets/download/{dataset_name}",
        "-o", zip_path,
        "--header", f"Authorization: Basic {auth_token}"
    ]

    try:
        # Download
        subprocess.run(curl_cmd, check=True)
        print(f"Dataset downloaded to {zip_path}")

        # Extract directly to the target directory
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # First, get the list of top-level directories in the zip
            top_dirs = {item.split('/')[0] for item in zip_ref.namelist() if '/' in item}

            # Extract and rename if needed
            for item in zip_ref.namelist():
                # Skip directories
                if item.endswith('/'):
                    continue

                # For files, extract by modifying the path
                parts = item.split('/')
                if len(parts) > 1 and parts[0] in top_dirs:
                    # Remove the top-level directory (caltech-101) from the path
                    new_path = '/'.join(parts[1:])
                    # Extract to the output directory
                    source = zip_ref.open(item)
                    target_path = os.path.join(output_path, new_path)

                    # Ensure directory exists
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)

                    # Extract the file
                    with open(target_path, 'wb') as target:
                        target.write(source.read())
                    source.close()

        print(f"Dataset extracted to {output_path}")

        # Remove temporary directory and zip file
        os.remove(zip_path)
        os.rmdir(temp_dir)

        return output_path

    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        return None


class Caltech101Dataset(torch.utils.data.Dataset):
    """
    Dataset class for the CalTech101 dataset with support for class selection
    and train/val/test splitting.
    """

    def __init__(self,
                 root_dir,
                 split='train',
                 selected_classes=None,
                 transform=None,
                 download=False,
                 split_ratios=(0.7, 0.15, 0.15),
                 random_seed=42):
        """
        Args:
            root_dir: Directory containing the dataset or where it will be downloaded
            split: One of 'train', 'val', 'test'
            selected_classes: List of class names to include (None for all classes)
            transform: Optional transform to be applied on images
            download: Whether to download the dataset if not found
            split_ratios: (train_ratio, val_ratio, test_ratio)
            random_seed: Random seed for reproducible splits
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.random_seed = random_seed

        # Download the dataset if needed
        if download and not self._check_exists():
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. Set download=True to download it.")

        # Get all available classes
        all_classes = self._get_all_classes()

        # Filter classes if specified
        if selected_classes:
            invalid_classes = [c for c in selected_classes if c not in all_classes]
            if invalid_classes:
                raise ValueError(f"Invalid class names: {invalid_classes}")
            self.classes = [c for c in all_classes if c in selected_classes]
        else:
            self.classes = all_classes

        # Create class to index mapping
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Load all samples and apply split
        self.samples = self._load_samples(split_ratios)

    def _check_exists(self):
        """Check if the dataset exists in the root directory."""
        return self.root_dir.exists() and any(self.root_dir.glob("**/*.jpg"))

    def _download(self):
        """Download the CalTech101 dataset."""
        download_caltech101_dataset(self.root_dir)

    def _get_all_classes(self):
        """Get all class names from the dataset directory."""
        class_dirs = [d for d in self.root_dir.iterdir()
                      if d.is_dir() and not d.name.startswith('.')]
        return sorted([d.name for d in class_dirs])

    def _load_samples(self, split_ratios):
        """Load samples and apply train/val/test split for each class."""
        import random
        random.seed(self.random_seed)

        train_ratio, val_ratio, test_ratio = split_ratios
        samples = []

        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue

            # Find all images for this class
            images = list(class_dir.glob("*.jpg"))
            if not images:
                # Try nested directories
                images = list(class_dir.glob("**/*.jpg"))

            if not images:
                print(f"Warning: No images found for class '{class_name}'")
                continue

            # Create split for this class
            indices = list(range(len(images)))
            random.shuffle(indices)

            val_size = max(1, int(val_ratio * len(indices)))
            test_size = max(1, int(test_ratio * len(indices)))

            val_indices = indices[:val_size]
            test_indices = indices[val_size:val_size + test_size]
            train_indices = indices[val_size + test_size:]

            # Select indices based on requested split
            if self.split == 'train':
                split_indices = train_indices
            elif self.split == 'val':
                split_indices = val_indices
            elif self.split == 'test':
                split_indices = test_indices
            else:
                raise ValueError(f"Invalid split: {self.split}")

            # Add selected samples for this class
            class_idx = self.class_to_idx[class_name]
            for idx in split_indices:
                samples.append((images[idx], class_idx))

        return samples

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transforms if any
        if self.transform:
            image = self.transform(image)

        return image, label



if __name__ == "__main__":
    train_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets with selected classes
    selected_classes = ['airplanes', 'chair', 'elephant']
    train_dataset = Caltech101Dataset(
        root_dir='caltech_101',
        split='train',
        selected_classes=selected_classes,
        transform=train_transform
    )

    val_dataset = Caltech101Dataset(
        root_dir='caltech_101',
        split='val',
        selected_classes=selected_classes,
        transform=train_transform
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )


    def test_dataloader(loader, n_samples=5):
        # Get a batch
        images, labels = next(iter(loader))

        # Print shapes and information
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Label values: {labels[:n_samples]}")

        # Map labels back to class names
        reverse_mapping = {idx: name for name, idx in train_dataset.class_to_idx.items()}
        class_names = [reverse_mapping[label.item()] for label in labels[:n_samples]]
        print(f"Classes: {class_names}")

        # Import if needed (add at top of file)
        import matplotlib.pyplot as plt
        import numpy as np

        # Display images
        fig, axes = plt.subplots(1, n_samples, figsize=(15, 3))
        for i in range(n_samples):
            img = images[i].permute(1, 2, 0).numpy()  # Convert to H×W×C for matplotlib

            # Denormalize image
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)  # Clip to valid range for display

            axes[i].imshow(img)
            axes[i].set_title(f"{class_names[i]}")
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    # Test the loader
    test_dataloader(train_loader)