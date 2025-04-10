from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import os
import json
import subprocess
import zipfile
import torch
import random
from torchvision import transforms
from data import OxfordPetDataset, create_dataloaders, SegmentationToTensor
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import Caltech101
import urllib.request
import requests
import tarfile
from pathlib import Path


RANDOM_SEED = 27

# note if autodownload fails, please download dataset from the following URL:
# https://www.kaggle.com/datasets/arnaud58/landscape-pictures
# and place the images under a directory called landscape_data

def download_kaggle_dataset(dataset_name, output_path="landscape_data"):

    # Check if dataset already exists
    if os.path.exists(output_path):
        image_files = list(Path(output_path).glob("**/*.jpg"))
        if image_files:
            print(f"Found existing dataset at {output_path} with {len(image_files)} images")
            return output_path

    os.makedirs(output_path, exist_ok=True)
    print(f"Downloading {dataset_name} using curl...")

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

    zip_path = os.path.join(output_path, "dataset.zip")

    # Build curl command
    curl_cmd = [
        "curl", "-L",
        f"https://www.kaggle.com/api/v1/datasets/download/{dataset_name}",
        "-o", zip_path,
        "--header", f"Authorization: Basic {credentials['username']}:{credentials['key']}"
    ]

    try:
        # Download
        subprocess.run(curl_cmd, check=True)
        print(f"Dataset downloaded to {zip_path}")

        # Extract
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_path)
        print(f"Dataset extracted to {output_path}")

        # Remove zip file
        os.remove(zip_path)
        return output_path

    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        return None


def download_caltech101_manually(root="./caltech101_data"):
    """Download Caltech101 dataset using multiple fallback sources."""
    import os
    import tarfile
    import requests
    from pathlib import Path

    os.makedirs(root, exist_ok=True)

    # Define multiple potential download URLs
    download_urls = [
        "https://data.vision.ee.ethz.ch/cvl/wp-content/uploads/2022/04/caltech-101.tar.gz",  # ETH mirror
        "http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz",  # Original source
        # Add more mirror URLs here if needed
    ]

    dataset_path = Path(root) / "caltech-101.tar.gz"
    extracted_dir = Path(root) / "101_ObjectCategories"

    # If already extracted, we're done
    if extracted_dir.exists():
        print("Dataset already extracted.")
        return root

    # If not downloaded yet, try each URL
    if not dataset_path.exists():
        print(f"Downloading Caltech101 dataset to {dataset_path}...")

        for url in download_urls:
            try:
                print(f"Attempting download from: {url}")
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()

                with open(dataset_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                print("Download complete!")
                break

            except Exception as e:
                print(f"Download failed: {e}")
                continue
        else:
            # All download attempts failed
            print("\nAll download attempts failed. Please try installing gdown and using torchvision:")
            print("pip install gdown")
            print("\nOr download manually from:")
            print("http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz")
            print(f"And extract to: {root}")
            raise RuntimeError("Could not download Caltech101 dataset")
    else:
        print("Caltech101 archive already downloaded.")

    # Extract the dataset if not already extracted
    if not extracted_dir.exists():
        print("Extracting dataset...")
        with tarfile.open(dataset_path, 'r:gz') as tar:
            tar.extractall(path=root)
        print("Extraction complete.")

    return root

def create_virtual_splits_for_landscape(bg_base_dir, train_ratio=0.7, val_ratio=0.15,
                                       test_ratio=0.15, random_seed=RANDOM_SEED):
    """Create virtual splits for any directory containing images.

    Args:
        bg_base_dir: Path to the directory containing background images
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Random seed

    Returns:
        tuple: (train_indices, val_indices, test_indices) for the dataset
    """
    random.seed(random_seed)
    bg_path = Path(bg_base_dir)

    if not bg_path.exists():
        raise ValueError(f"Background folder {bg_path} does not exist.")

    # Find all image files recursively
    all_images = list(bg_path.glob("**/*.jpg"))

    if not all_images:
        raise ValueError(f"No images found in directory {bg_path}")

    print(f"Found {len(all_images)} images in {bg_path}.")

    # create indices
    all_indices = list(range(len(all_images)))

    random.shuffle(all_indices)

    # calculate splits
    total_images = len(all_images)
    val_size = int(val_ratio * total_images)
    test_size = int(test_ratio * total_images)

    # create splits
    val_indices = all_indices[:val_size]
    test_indices = all_indices[val_size:val_size + test_size]
    train_indices = all_indices[val_size + test_size:]

    print(f"Created virtual splits: training ({len(train_indices)}, {len(train_indices) / total_images:.1%}), "
          f"validation ({len(val_indices)}, {len(val_indices) / total_images:.1%}), "
          f"testing ({len(test_indices)}, {len(test_indices) / total_images:.1%})")

    return train_indices, val_indices, test_indices


class BackgroundDataset(Dataset):
    """Dataset for background images (without pets)."""

    def __init__(self, bg_dir, indices=None, transform=None, target_label=-1, target_type=["class"]):
        """Initialize the background dataset.

        Args:
            bg_dir: Directory containing background images
            indices: Optional indices to use for this dataset (for virtual splits)
            transform: Transforms to apply to images
            target_label: Label to use for background images
            target_type: Types of targets to generate
        """
        self.bg_dir = Path(bg_dir)
        self.transform = transform
        self.target_label = target_label
        self.target_type = target_type if isinstance(target_type, list) else [target_type]

        # Find all images
        self.bg_files = list(self.bg_dir.glob("**/*.jpg"))

        if not self.bg_files:
            raise ValueError(f"No background images found in {self.bg_dir}")

        # Use indices if provided
        if indices is not None:
            self.bg_files = [self.bg_files[i] for i in indices]

        print(f"Using {len(self.bg_files)} background images from {self.bg_dir}")

    def __len__(self):
        return len(self.bg_files)

    def __getitem__(self, idx):
        img_path = self.bg_files[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        original_width, original_height = image.size

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Create targets based on target_type
        if len(self.target_type) == 1:
            target = self._get_target(self.target_type[0], original_width, original_height)
            return image, target
        else:
            targets = {}
            for t_type in self.target_type:
                targets[t_type] = self._get_target(t_type, original_width, original_height)
            return image, targets

    def _get_target(self, target_type, original_width, original_height):
        """Generate appropriate targets for background images."""
        if target_type in ["class", "breed"]:
            return self.target_label
        # No species for background images
        elif target_type == "species":
            return -1
            # Return empty/zero bounding box
        elif target_type == "bbox":
            return torch.zeros(4, dtype=torch.float32)
        # Return blank mask (1-channel, 256x256)
        elif target_type == "segmentation":
            return torch.zeros((64, 64), dtype=torch.float32)
        else:
            raise ValueError(f"Unknown target_type: {target_type}")


class Caltech101FilteredDataset(Dataset):
    def __init__(self, root="./caltech101_data", transform=None,
                 target_type=["class"], download=True, class_offset=37,
                 selected_categories=None):

        self.transform = transform
        self.target_type = target_type if isinstance(target_type, list) else [target_type]
        self.class_offset = class_offset

        # Default categories - select interesting non-animal categories
        default_categories = ['airplanes', 'car_side', 'ferry', 'laptop', 'motorbikes']
        self.selected_categories = selected_categories or default_categories

        if download:
            download_caltech101_manually(root)

        # Initialize with the manually downloaded dataset
        self._init_dataset(root)

        # Filter for selected categories
        self.indices = []
        for i, (_, category_idx) in enumerate(self.samples):
            category = self.categories[category_idx]
            if category in self.selected_categories:
                self.indices.append(i)

        print(f"Caltech101 dataset: Selected {len(self.indices)} images from categories "
              f"{self.selected_categories}")

    def _init_dataset(self, root):
        """Initialize dataset from files on disk rather than using torchvision."""
        from PIL import Image
        import os

        root_dir = os.path.join(root, "101_ObjectCategories")
        categories = sorted([d for d in os.listdir(root_dir)
                             if os.path.isdir(os.path.join(root_dir, d))])

        self.categories = categories
        self.category_to_idx = {cat: i for i, cat in enumerate(categories)}
        self.class_names = {}
        for i, category in enumerate(self.selected_categories):
            self.class_names[self.class_offset + i] = category

        # Collect all samples
        self.samples = []
        for category in categories:
            category_dir = os.path.join(root_dir, category)
            for img_name in os.listdir(category_dir):
                if img_name.endswith('.jpg'):
                    img_path = os.path.join(category_dir, img_name)
                    category_idx = self.category_to_idx[category]
                    self.samples.append((img_path, category_idx))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        orig_idx = self.indices[idx]

        image, category_idx = self.dataset[orig_idx]
        category = self.dataset.categories[category_idx]

        # Map to our consecutive numbering
        new_label = self.selected_categories.index(category) + self.class_offset

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Create targets
        if len(self.target_type) == 1:
            target = self._get_target(self.target_type[0], new_label, image.shape)
            return image, target
        else:
            targets = {}
            for t_type in self.target_type:
                targets[t_type] = self._get_target(t_type, new_label, image.shape)
            return image, targets

    def _get_target(self, target_type, label, img_shape):
        if target_type == "class":
            return label
        elif target_type == "species":
            # Map categories to non-pet species numbers (2+)
            category = self.selected_categories[label - self.class_offset]
            if category in ['airplanes', 'ferry']:
                return 2  # air/water vehicles
            elif category in ['car_side', 'motorbikes']:
                return 3  # road vehicles
            else:
                return 4  # other objects
        elif target_type == "bbox":
            # Use whole image as bbox
            return torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
        elif target_type == "segmentation":
            h, w = img_shape[1:] if len(img_shape) > 2 else (64, 64)
            return torch.zeros((h, w), dtype=torch.float32)
        else:
            raise ValueError(f"Unknown target_type: {target_type}")


class MixedDataset(Dataset):
    """
    Dataset that combines the OxfordPetDataset with a secondary dataset
    at specified intervals.
    """

    def __init__(self, main_dataset, secondary_dataset, mixing_ratio=5):
        """
        Initialize a mixed dataset with a primary and secondary dataset.

        Args:
            main_dataset: OxfordPetDataset
            secondary_dataset: BackgroundDataset or STL10FilteredDataset
            mixing_ratio: Insert a secondary image in line with 'mixing_ratio'
        """
        self.main_dataset = main_dataset
        self.secondary_dataset = secondary_dataset
        self.mixing_ratio = mixing_ratio
        self._create_index_map()

    def _create_index_map(self):
        self.index_map = []
        secondary_indices = list(range(len(self.secondary_dataset)))
        random.shuffle(secondary_indices)

        main_size = len(self.main_dataset)
        secondary_insertions = main_size // self.mixing_ratio

        # Make sure there are enough secondary images
        if secondary_insertions > len(secondary_indices):
            # Repeat secondary indices if required
            multiplier = (secondary_insertions // len(secondary_indices)) + 1
            secondary_indices = secondary_indices * multiplier

        secondary_indices = secondary_indices[:secondary_insertions]
        secondary_counter = 0

        for i in range(main_size):
            self.index_map.append(('main', i))

            # Insert secondary images
            if (i + 1) % self.mixing_ratio == 0 and secondary_counter < len(secondary_indices):
                self.index_map.append(('secondary', secondary_indices[secondary_counter]))
                secondary_counter += 1

        print(f"Mixed dataset created with {main_size} main images and {secondary_counter} secondary images")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        dataset_type, dataset_idx = self.index_map[idx]

        if dataset_type == 'main':
            return self.main_dataset[dataset_idx]
        else:
            return self.secondary_dataset[dataset_idx]

def create_pet_datasets(data_directory, train_transform, val_test_transform, target_type,
                       normalize_bbox, target_transform, lazy_loading, train_ratio,
                       val_ratio, test_ratio, random_seed):
    """Create train, validation and test Oxford Pet datasets."""
    train_dataset = OxfordPetDataset(
        root_dir=data_directory,
        transform=train_transform,
        target_type=target_type,
        normalize_bbox=normalize_bbox,
        target_transform=target_transform,
        cache_in_memory=not lazy_loading,
        split="train",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
        resize_size=64  # Match the transforms size
    )

    val_dataset = OxfordPetDataset(
        root_dir=data_directory,
        transform=val_test_transform,
        target_type=target_type,
        normalize_bbox=normalize_bbox,
        target_transform=target_transform,
        cache_in_memory=not lazy_loading,
        split="val",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
        resize_size=64
    )

    test_dataset = OxfordPetDataset(
        root_dir=data_directory,
        transform=val_test_transform,
        target_type=target_type,
        normalize_bbox=normalize_bbox,
        target_transform=target_transform,
        cache_in_memory=not lazy_loading,
        split="test",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
        resize_size=64
    )

    return train_dataset, val_dataset, test_dataset


def create_background_datasets(bg_directory, train_transform, val_test_transform, target_type, bg_label):
    """Create train, validation and test background datasets."""
    # Create virtual splits for background images
    train_indices, val_indices, test_indices = create_virtual_splits_for_landscape(bg_directory)

    train_dataset = BackgroundDataset(
        bg_directory,
        indices=train_indices,
        transform=train_transform,
        target_label=bg_label,
        target_type=target_type
    )

    val_dataset = BackgroundDataset(
        bg_directory,
        indices=val_indices,
        transform=val_test_transform,
        target_label=bg_label,
        target_type=target_type
    )

    test_dataset = BackgroundDataset(
        bg_directory,
        indices=test_indices,
        transform=val_test_transform,
        target_label=bg_label,
        target_type=target_type
    )

    return train_dataset, val_dataset, test_dataset


def create_caltech101_datasets(caltech101_root, train_transform, val_test_transform,
                               target_type, caltech101_class_offset, selected_categories=None):
    """Create train, validation and test Caltech101 datasets."""

    try:
        # Try manual download first
        download_caltech101_manually(caltech101_root)
    except Exception as e:
        print(f"Manual download failed: {e}")
        print("Falling back to torchvision downloader (requires gdown)")
        try:
            # Try the original torchvision version (requires gdown)
            import importlib.util
            if importlib.util.find_spec("gdown") is None:
                print("Please install gdown: pip install gdown")
                raise ImportError("gdown is required for Caltech101 dataset")

            dummy = Caltech101(root=caltech101_root, download=True)

        except Exception as e2:
            print(f"All download methods failed: {e2}")
            raise ValueError("Could not download Caltech101 dataset. Please download manually.")

    # Create the full dataset
    full_dataset = Caltech101FilteredDataset(
        root=caltech101_root,
        transform=None,
        target_type=target_type,
        download=False,  # Already downloaded manually
        class_offset=caltech101_class_offset,
        selected_categories=selected_categories
    )

    # Create splits
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))

    # Shuffle with fixed seed
    random_state = random.Random(RANDOM_SEED)
    random_state.shuffle(indices)

    # 70/15/15 split
    train_split = int(0.7 * dataset_size)
    val_split = int(0.85 * dataset_size)

    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]

    # Create datasets with appropriate transforms
    train_dataset = Caltech101FilteredDataset(
        root=caltech101_root,
        transform=train_transform,
        target_type=target_type,
        download=False,
        class_offset=caltech101_class_offset,
        selected_categories=selected_categories
    )
    train_dataset.indices = [full_dataset.indices[i] for i in train_indices]

    val_dataset = Caltech101FilteredDataset(
        root=caltech101_root,
        transform=val_test_transform,
        target_type=target_type,
        download=False,
        class_offset=caltech101_class_offset,
        selected_categories=selected_categories
    )
    val_dataset.indices = [full_dataset.indices[i] for i in val_indices]

    test_dataset = Caltech101FilteredDataset(
        root=caltech101_root,
        transform=val_test_transform,
        target_type=target_type,
        download=False,
        class_offset=caltech101_class_offset,
        selected_categories=selected_categories
    )
    test_dataset.indices = [full_dataset.indices[i] for i in test_indices]

    return train_dataset, val_dataset, test_dataset


def create_mixed_dataset_loaders(train_pet_dataset, val_pet_dataset, test_pet_dataset,
                                 train_secondary, val_secondary, test_secondary,
                                 batch_size, mixing_ratio):
    """Create mixed datasets and dataloaders."""
    # Create mixed datasets
    train_mixed = MixedDataset(train_pet_dataset, train_secondary, mixing_ratio)
    val_mixed = MixedDataset(val_pet_dataset, val_secondary, mixing_ratio)
    test_mixed = MixedDataset(test_pet_dataset, test_secondary, mixing_ratio)

    # Create dataloaders
    train_loader = DataLoader(
        train_mixed,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_mixed,
        batch_size=batch_size
    )

    test_loader = DataLoader(
        test_mixed,
        batch_size=batch_size
    )

    return train_loader, val_loader, test_loader


def create_mixed_dataloaders(batch_size=32, train_ratio=0.7, val_ratio=0.15,
                             test_ratio=0.15, random_seed=RANDOM_SEED, target_type=["class"],
                             normalize_bbox=True, data_directory="oxford_pet_data",
                             secondary_dataset_type="background",
                             bg_directory=None, caltech101_root="./caltech101_data",
                             caltech101_selected_categories=None, caltech101_class_offset=37,
                             mixing_ratio=5, bg_label=-1,
                             use_augmentation=False, lazy_loading=True):
    """Create PyTorch DataLoaders with secondary dataset integration."""

    # Setup common transforms
    train_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,
                               hue=0.1) if use_augmentation else transforms.Lambda(lambda x: x),
        transforms.RandomGrayscale(p=0.2) if use_augmentation else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Setup segmentation transform if needed
    target_transform = None
    if "segmentation" in target_type if isinstance(target_type, list) else target_type == "segmentation":
        target_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            SegmentationToTensor(),
        ])

    # Create Oxford Pet datasets
    train_pet_dataset, val_pet_dataset, test_pet_dataset = create_pet_datasets(
        data_directory, train_transform, val_test_transform, target_type,
        normalize_bbox, target_transform, lazy_loading, train_ratio,
        val_ratio, test_ratio, random_seed
    )

    # Create secondary datasets based on type
    try:
        if secondary_dataset_type.lower() == "background":
            # Validate background directory
            if bg_directory is None or not os.path.exists(bg_directory):
                raise ValueError("Background directory not found or not specified")

            train_secondary, val_secondary, test_secondary = create_background_datasets(
                bg_directory, train_transform, val_test_transform,
                target_type, bg_label
            )

        elif secondary_dataset_type.lower() == "caltech101":
            train_secondary, val_secondary, test_secondary = create_caltech101_datasets(
                caltech101_root, train_transform, val_test_transform,
                target_type, caltech101_class_offset, caltech101_selected_categories
            )

        else:
            raise ValueError(f"Unknown secondary dataset type: {secondary_dataset_type}")

    except ValueError as e:
        print(f"Error creating secondary dataset: {e}")
        print("Using standard dataloaders without mixing.")
        return create_dataloaders(
            batch_size, train_ratio, val_ratio, test_ratio, random_seed,
            target_type, normalize_bbox, data_directory, use_augmentation, lazy_loading
        )

    # Create mixed datasets and dataloaders
    return create_mixed_dataset_loaders(
        train_pet_dataset, val_pet_dataset, test_pet_dataset,
        train_secondary, val_secondary, test_secondary,
        batch_size, mixing_ratio
    )


if __name__ == "__main__":

    # Download the landscape pictures dataset
    bg_directory = download_kaggle_dataset("arnaud58/landscape-pictures")

    if bg_directory:

        # Check if images were downloaded properly
        image_paths = list(Path(bg_directory).glob("**/*.jpg"))
        image_count = len(image_paths)

        if image_count > 0:
            # Create the mixed dataloaders with the landscape images
            train_loader, val_loader, test_loader = create_mixed_dataloaders(
                batch_size=32,
                data_directory="oxford_pet_data",
                bg_directory=bg_directory,
                target_type=["species", "class", "bbox", "segmentation"],
                mixing_ratio=5,
                use_augmentation=True,
                lazy_loading=True
            )

            print(f"\nTraining images: {len(train_loader.dataset)}")
            print(f"Validation images: {len(val_loader.dataset)}")
            print(f"Testing images: {len(test_loader.dataset)}")


    def visualize_batch(dataloader, num_samples=5):
        # Get a batch
        images, targets = next(iter(dataloader))

        print(targets)

        mask = targets["segmentation"][0]  # Get the first segmentation mask from the batch

        print(f"Mask tensor shape: {mask.shape}")
        print(f"Mask tensor dtype: {mask.dtype}")
        print(f"Unique values in mask: {torch.unique(mask)}")

        print("Mask tensor values:")
        print(mask)

        # Create a figure
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))

        for i in range(num_samples):

            img = images[i].permute(1, 2, 0).numpy()

            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)

            axes[i].imshow(img)

            # Get target info
            if isinstance(targets, dict):
                # Get class and bbox if available
                class_label = targets['class'][i].item() if 'class' in targets else None
                bbox = targets['bbox'][i] if 'bbox' in targets else None

                title = f"Class: {class_label}"
                axes[i].set_title(title)

                # Draw bbox if available
                if bbox is not None and not (bbox == 0).all():
                    # Get bbox as (xmin, ymin, xmax, ymax)
                    xmin, ymin, xmax, ymax = bbox.tolist()
                    # Compute width and height
                    width = xmax - xmin
                    height = ymax - ymin

                    import matplotlib.patches as patches
                    rect = patches.Rectangle(
                        (xmin * 64, ymin * 64), width * 64, height * 64,
                        linewidth=2, edgecolor='r', facecolor='none'
                    )
                    axes[i].add_patch(rect)
            else:
                # Simple target (just class)
                axes[i].set_title(f"Class: {targets[i].item()}")

            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    print("\nVisualising samples from the training set:")

    visualize_batch(train_loader)

    print("\nTesting iteration through the training dataset:")
    for batch_idx, (images, targets) in enumerate(train_loader):
        if batch_idx == 0:
            print(f"Batch shape: {images.shape}")
            if isinstance(targets, dict):
                for k, v in targets.items():
                    print(f"Target {k} shape: {v.shape}")
            else:
                print(f"Target shape: {targets.shape}")

        # Only process a few batches for testing
        if batch_idx >= 2:
            break

    train_loader, val_loader, test_loader = create_mixed_dataloaders(
        batch_size=32,
        data_directory="oxford_pet_data",
        secondary_dataset_type="caltech101",
        caltech101_root="./caltech101_data",
        caltech101_selected_categories=['airplanes', 'car_side', 'ferry', 'laptop', 'motorbikes'],
        caltech101_class_offset=37,  # Offset for class indices (pets are 0-36)
        target_type=["species", "class", "bbox", "segmentation"],
        mixing_ratio=5,
        use_augmentation=True,
        lazy_loading=True,
    )

    print(f"\nCaltech101 Mixed Dataset:")
    print(f"Training images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")
    print(f"Testing images: {len(test_loader.dataset)}")

    # Test visualization with the existing visualize_batch function
    visualize_batch(train_loader)

    # Test visualization with the existing visualize_batch function
    print("\nVisualizing samples from the training set:")
    visualize_batch(train_loader)

    print("\nDataset testing completed successfully!")