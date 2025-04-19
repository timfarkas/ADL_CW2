import random
from PIL import Image

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset

from custom_types import ModelTarget
from datasets.background_dataset import BackgroundDataset
from datasets.oxford_pet_dataset import OxfordPetDataset
from datasets.mixed_dataset import MixedDataset

dataset_groups = ["test", "train", "val"]


class SegmentationToTensor:
    def __call__(self, pic):
        tensor = torch.as_tensor(np.array(pic), dtype=torch.long)
        return tensor


class DatasetManager:
    pet_datasets: tuple[OxfordPetDataset, OxfordPetDataset, OxfordPetDataset]
    bg_datasets: tuple[BackgroundDataset, BackgroundDataset, BackgroundDataset] | None
    mixed_datasets: tuple[MixedDataset, MixedDataset, MixedDataset] | None

    augmentations: list = [
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
    ]

    def __init__(
        self,
        image_size: tuple = (224, 224),
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        target_type: list[ModelTarget] = ["species", "breed", "bbox", "segmentation"],
        normalize_bbox: bool = True,
        bg_label: int = -1,
        use_augmentation: bool = False,
        lazy_loading: bool = True,
        mixed: bool = True,
        mixing_ratio: int = 5,
    ):
        if train_ratio + val_ratio >= 1.0:
            raise ValueError("The sum of train and val ratios must less than 1.0")

        self.image_size = image_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.target_type = target_type
        self.normalize_bbox = normalize_bbox
        self.mixing_ratio = mixing_ratio
        self.bg_label = bg_label
        self.lazy_loading = lazy_loading

        transforms_list = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        self.test_inputs_transform = transforms.Compose(transforms_list)
        self.segmentation_transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=Image.NEAREST),
                SegmentationToTensor(),
            ]
        )

        if use_augmentation:
            self.train_inputs_transform = transforms.Compose(
                transforms_list[:1] + self.augmentations + transforms_list[1:]
            )
        else:
            self.train_inputs_transform = transforms.Compose(transforms_list)

        self.generate_oxford_dataset()
        if mixed:
            self.generate_background_dataset()
            self.generate_mixed_dataset()

        self.datasets = self.pet_datasets if not mixed else self.mixed_datasets

    def generate_oxford_dataset(self):
        """Load the Oxford Pets dataset."""
        self.pet_datasets = tuple(
            OxfordPetDataset(
                image_size=self.image_size,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                target_type=self.target_type,
                normalize_bbox=self.normalize_bbox,
                transform=self.train_inputs_transform,
                target_transform=self.segmentation_transform,
                cache_in_memory=not self.lazy_loading,
                split=dataset_group,
            )
            for dataset_group in dataset_groups
        )

    def generate_background_dataset(self):
        """Load the background dataset."""
        self.bg_datasets = tuple(
            BackgroundDataset(
                image_size=self.image_size,
                transform=self.train_inputs_transform,
                target_label=self.bg_label,
                target_type=self.target_type,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                split=dataset_group,
            )
            for dataset_group in dataset_groups
        )

    def generate_mixed_dataset(self):
        """Generate a mixed dataset with background images."""
        self.mixed_datasets = tuple(
            MixedDataset(
                main_dataset=self.pet_datasets[i],
                bg_dataset=self.bg_datasets[i],
                mixing_ratio=self.mixing_ratio,
            )
            for i in range(len(dataset_groups))
        )


class MixedDataset(Dataset):
    """
    Dataset that combines the OxfordPetDataset with background images
    at specified intervals.
    """

    def __init__(self, main_dataset, bg_dataset, mixing_ratio):
        """
        Initialize a mixed dataset with Oxford Pets and background images.

        Args:
            main_dataset: The primary dataset (OxfordPetDataset)
            bg_dataset: The background dataset
            mixing_ratio: Insert a background image every 'mixing_ratio' images
        """
        self.main_dataset = main_dataset
        self.bg_dataset = bg_dataset
        self.mixing_ratio = mixing_ratio
        self._create_index_map()

    def _create_index_map(self):
        """Create a mapping from indices to (dataset, idx) pairs."""
        self.index_map = []
        bg_indices = list(range(len(self.bg_dataset)))
        random.shuffle(bg_indices)

        main_size = len(self.main_dataset)
        bg_insertions = main_size // self.mixing_ratio

        # Make sure there are enough background images
        if bg_insertions > len(bg_indices):
            # Repeat background indices if required
            multiplier = (bg_insertions // len(bg_indices)) + 1
            bg_indices = bg_indices * multiplier

        bg_indices = bg_indices[:bg_insertions]
        bg_counter = 0

        # Create the mixed index map
        for i in range(main_size):
            self.index_map.append(("main", i))

            # Insert background images
            if (i + 1) % self.mixing_ratio == 0 and bg_counter < len(bg_indices):
                self.index_map.append(("bg", bg_indices[bg_counter]))
                bg_counter += 1

        print(
            f"Mixed dataset created with {main_size} main images and {bg_counter} background images"
        )

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        dataset_type, dataset_idx = self.index_map[idx]

        if dataset_type == "main":
            return self.main_dataset[dataset_idx]
        else:  # 'bg'
            return self.bg_dataset[dataset_idx]
