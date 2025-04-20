from pathlib import Path
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from custom_types import DatasetGroup
from new_data_utils import download_kaggle_dataset


class BackgroundDataset(Dataset):
    """Dataset for background images (without pets)."""

    def __init__(
        self,
        image_size: tuple[int, int],
        transform: transforms.Compose,
        target_label: int,
        target_type: list[str],
        train_ratio: float,
        val_ratio: float,
        split: DatasetGroup,
    ):
        """Initialize the background dataset.

        Args:
            bg_dir: Directory containing background images
            indices: Optional indices to use for this dataset (for virtual splits)
            transform: Transforms to apply to images
            target_label: Label to use for background images
            target_type: Types of targets to generate
        """
        self.bg_dir = Path(
            download_kaggle_dataset("arnaud58/landscape-pictures", "data/background_images")
        )
        self.transform = transform
        self.target_label = target_label
        self.target_type = target_type
        self.image_size = image_size

        # Find all images
        self.bg_files = list(self.bg_dir.glob("**/*.jpg"))

        if not self.bg_files:
            raise ValueError(f"No background images found in {self.bg_dir}")

        # Use indices if provided
        train_data, val_data, test_data = self._split_dataset(
            self.bg_files, train_ratio, val_ratio
        )

        if split == "train":
            self.bg_files = train_data
        elif split == "val":
            self.bg_files = val_data
        elif split == "test":
            self.bg_files = test_data

        print(f"Using {len(self.bg_files)} background images from {self.bg_dir}")

    def __len__(self):
        return len(self.bg_files)

    def __getitem__(self, idx):
        img_path = self.bg_files[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Create targets based on target_type
        if len(self.target_type) == 1:
            target = self._get_target(
                self.target_type[0]
            )
            return image, target
        else:
            targets = {}
            for t_type in self.target_type:
                targets[t_type] = self._get_target(
                    t_type
                )
            return image, targets

    def _get_target(self, target_type):
        """Generate appropriate targets for background images."""
        if target_type == "is_animal":
            return 0  # Background images have no animals
        elif target_type in ["class", "breed"]:
            return self.target_label
        # No species for background images
        elif target_type == "species":
            return -1
            # Return empty/zero bounding box
        elif target_type == "bbox":
            return torch.zeros(4, dtype=torch.float32)
        # Return blank mask (1-channel, 256x256)
        elif target_type == "segmentation":
            return torch.zeros(self.image_size, dtype=torch.long)
        else:
            raise ValueError(f"Unknown target_type: {target_type}")

    def _split_dataset(
        self,
        dataset,
        train_ratio,
        val_ratio,
    ):
        """Split the dataset into training, validation, and test sets.

        Args:
            dataset: List of data items
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set

        Returns:
            tuple: (train_data, val_data, test_data) lists
        """

        # Shuffle the dataset
        shuffled_data = dataset.copy()
        random.shuffle(shuffled_data)

        # Calculate split sizes
        dataset_size = len(shuffled_data)
        train_size = int(train_ratio * dataset_size)
        val_size = int(val_ratio * dataset_size)

        # Split the dataset
        train_data = shuffled_data[:train_size]
        val_data = shuffled_data[train_size : train_size + val_size]
        test_data = shuffled_data[train_size + val_size :]

        print(
            f"Dataset split complete: training ({len(train_data) / dataset_size:.1%}), "
            f"validation ({len(val_data) / dataset_size:.1%}), testing ({len(test_data) / dataset_size:.1%})"
        )

        return train_data, val_data, test_data
