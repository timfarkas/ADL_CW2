import os
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
from torchvision import transforms
from data import OxfordPetDataset, create_dataloaders, SegmentationToTensor
import matplotlib.pyplot as plt
import numpy as np
from caltech101Dataset import Caltech101Dataset


class Caltech101Adapter(Dataset):
    """
    Adapter class to make Caltech101Dataset compatible with OxfordPetDataset targets
    """

    def __init__(self, caltech_dataset, target_type=["class"], target_label=-1):
        """
        Initialize the adapter for Caltech101Dataset.

        Args:
            caltech_dataset: The Caltech101Dataset instance
            target_type: Target types to generate ("class", "species", "bbox", "segmentation")
            target_label: Label to use for class targets (typically -1 or a specific class ID)
        """
        self.caltech_dataset = caltech_dataset
        self.target_type = target_type if isinstance(target_type, list) else [target_type]
        self.target_label = target_label

    def __len__(self):
        return len(self.caltech_dataset)

    def __getitem__(self, idx):
        image, original_label = self.caltech_dataset[idx]

        # Create targets in the same format as OxfordPetDataset
        if len(self.target_type) == 1:
            target = self._get_target(self.target_type[0], original_label)
            return image, target
        else:
            targets = {}
            for t_type in self.target_type:
                targets[t_type] = self._get_target(t_type, original_label)
            return image, targets

    def _get_target(self, target_type, original_label):
        """Generate appropriate targets compatible with OxfordPetDataset."""
        if target_type in ["class", "breed"]:
            # make CalTech have negative labels`
            return -1 - original_label
        elif target_type == "species":
            # Use the same strategy for species as class
            return -1 - original_label
        elif target_type == "bbox":
            # Return zero's bounding box
            return torch.zeros(4, dtype=torch.float32)
        elif target_type == "segmentation":
            # Return blank mask (1-channel, 64x64)
            return torch.zeros((64, 64), dtype=torch.float32)
        else:
            raise ValueError(f"Unknown target_type: {target_type}")


class MixedCaltechPetDataset(Dataset):
    """
    Dataset that combines the OxfordPetDataset with Caltech101 images
    at specified intervals.
    """

    def __init__(self, pet_dataset, caltech_dataset, mixing_ratio=5, use_original_labels=False):
        """
        Initialize a mixed dataset with Oxford Pets and Caltech101 images.

        Args:
            pet_dataset: The primary dataset (OxfordPetDataset)
            caltech_dataset: The Caltech101 dataset adapter
            mixing_ratio: Insert a Caltech image every 'mixing_ratio' images
            use_original_labels: Whether to keep original Caltech labels or use target_label
        """
        self.pet_dataset = pet_dataset
        self.caltech_dataset = caltech_dataset
        self.mixing_ratio = mixing_ratio
        self.use_original_labels = use_original_labels
        self._create_index_map()

    def _create_index_map(self):
        self.index_map = []
        caltech_indices = list(range(len(self.caltech_dataset)))
        random.shuffle(caltech_indices)

        pet_size = len(self.pet_dataset)
        caltech_insertions = pet_size // self.mixing_ratio

        # Make sure there are enough Caltech images
        if caltech_insertions > len(caltech_indices):
            # If required repeat Caltech indices
            multiplier = (caltech_insertions // len(caltech_indices)) + 1
            caltech_indices = caltech_indices * multiplier

        caltech_indices = caltech_indices[:caltech_insertions]
        caltech_counter = 0

        for i in range(pet_size):
            self.index_map.append(('pet', i))

            # Insert images
            if (i + 1) % self.mixing_ratio == 0 and caltech_counter < len(caltech_indices):
                self.index_map.append(('caltech', caltech_indices[caltech_counter]))
                caltech_counter += 1

        print(f"Mixed dataset created with {pet_size} pet images and {caltech_counter} Caltech101 images")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        dataset_type, dataset_idx = self.index_map[idx]

        if dataset_type == 'pet':
            return self.pet_dataset[dataset_idx]
        else:  # 'caltech'
            return self.caltech_dataset[dataset_idx]


def create_mixed_caltech_pet_dataloaders(
        batch_size=32, train_ratio=0.7, val_ratio=0.15,
        test_ratio=0.15, random_seed=42, target_type=["class"],
        normalize_bbox=True, pet_data_directory="oxford_pet_data",
        caltech_data_directory="caltech_101", selected_caltech_classes=None,
        mixing_ratio=5, use_augmentation=False, lazy_loading=True):
    """Create PyTorch DataLoaders with Caltech101 and OxfordPet integration.

    Args:
        batch_size: Batch size for DataLoaders
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Seed for reproducible splitting
        target_type: Target type for the model ("class", "species", "bbox", or "segmentation")
        normalize_bbox: Whether to normalize bounding box coordinates
        pet_data_directory: Directory containing the OxfordPet dataset files
        caltech_data_directory: Directory containing Caltech101 dataset
        selected_caltech_classes: List of Caltech101 classes to include (None for all)
        mixing_ratio: Insert a Caltech image every n images (e.g., 5 means every 5th image)
        use_augmentation: Whether to use data augmentation for training
        lazy_loading: Whether to load images on-demand (True) or preload into memory (False)

    Returns:
        tuple: (train_loader, val_loader, test_loader) DataLoader instances
    """
    if not os.path.exists(caltech_data_directory):
        print("Warning: Caltech101 directory not found. Using standard OxfordPet dataloaders.")
        return create_dataloaders(
            batch_size, train_ratio, val_ratio, test_ratio, random_seed,
            target_type, normalize_bbox, pet_data_directory, use_augmentation, lazy_loading
        )

    # Define image transforms
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    val_test_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define target transform if segmentation is required
    target_transform = None
    if "segmentation" in target_type if isinstance(target_type, list) else target_type == "segmentation":
        target_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            SegmentationToTensor(),
        ])

    # Create Oxford Pet datasets
    train_pet_dataset = OxfordPetDataset(
        root_dir=pet_data_directory,
        transform=train_transform,
        target_type=target_type,
        normalize_bbox=normalize_bbox,
        target_transform=target_transform,
        cache_in_memory=not lazy_loading,
        split="train",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed
    )

    val_pet_dataset = OxfordPetDataset(
        root_dir=pet_data_directory,
        transform=val_test_transform,
        target_type=target_type,
        normalize_bbox=normalize_bbox,
        target_transform=target_transform,
        cache_in_memory=not lazy_loading,
        split="val",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed
    )

    test_pet_dataset = OxfordPetDataset(
        root_dir=pet_data_directory,
        transform=val_test_transform,
        target_type=target_type,
        normalize_bbox=normalize_bbox,
        target_transform=target_transform,
        cache_in_memory=not lazy_loading,
        split="test",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed
    )

    try:
        # Create Caltech101 datasets for each split
        train_caltech_dataset = Caltech101Dataset(
            root_dir=caltech_data_directory,
            split='train',
            selected_classes=selected_caltech_classes,
            transform=train_transform,
            download=True,
            split_ratios=(train_ratio, val_ratio, test_ratio),
            random_seed=random_seed
        )

        val_caltech_dataset = Caltech101Dataset(
            root_dir=caltech_data_directory,
            split='val',
            selected_classes=selected_caltech_classes,
            transform=val_test_transform,
            download=False,
            split_ratios=(train_ratio, val_ratio, test_ratio),
            random_seed=random_seed
        )

        test_caltech_dataset = Caltech101Dataset(
            root_dir=caltech_data_directory,
            split='test',
            selected_classes=selected_caltech_classes,
            transform=val_test_transform,
            download=False,
            split_ratios=(train_ratio, val_ratio, test_ratio),
            random_seed=random_seed
        )

        # Create adapter instances for Caltech datasets
        train_caltech_adapter = Caltech101Adapter(
            train_caltech_dataset,
            target_type=target_type,
        )

        val_caltech_adapter = Caltech101Adapter(
            val_caltech_dataset,
            target_type=target_type,
        )

        test_caltech_adapter = Caltech101Adapter(
            test_caltech_dataset,
            target_type=target_type,
        )

    except Exception as e:
        print(f"Error creating Caltech101 dataset: {e}")
        print("Using standard OxfordPet dataloaders without Caltech101 mixing.")
        return create_dataloaders(
            batch_size, train_ratio, val_ratio, test_ratio, random_seed,
            target_type, normalize_bbox, pet_data_directory, use_augmentation, lazy_loading
        )

    # Create mixed datasets
    train_mixed_dataset = MixedCaltechPetDataset(
        train_pet_dataset,
        train_caltech_adapter,
        mixing_ratio,
    )

    val_mixed_dataset = MixedCaltechPetDataset(
        val_pet_dataset,
        val_caltech_adapter,
        mixing_ratio,
    )

    test_mixed_dataset = MixedCaltechPetDataset(
        test_pet_dataset,
        test_caltech_adapter,
        mixing_ratio,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_mixed_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_mixed_dataset,
        batch_size=batch_size
    )

    test_loader = DataLoader(
        test_mixed_dataset,
        batch_size=batch_size
    )

    return train_loader, val_loader, test_loader


def visualize_mixed_batch(dataloader, num_samples=5):
    """
    Visualize a batch of mixed data from the dataloader.

    Args:
        dataloader: The dataloader to visualize
        num_samples: Number of samples to visualize
    """
    # Get a batch
    images, targets = next(iter(dataloader))

    # Create a figure
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))

    # Print target information
    print(f"Target type: {type(targets)}")
    if isinstance(targets, dict):
        for key, value in targets.items():
            print(f"Target '{key}' shape: {value.shape}, dtype: {value.dtype}")
            if key == 'class':
                print(f"Classes in batch: {value[:num_samples].tolist()}")
    else:
        print(f"Target shape: {targets.shape}, dtype: {targets.dtype}")
        print(f"Classes in batch: {targets[:num_samples].tolist()}")

    for i in range(num_samples):
        img = images[i].permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        axes[i].imshow(img)

        # Display target information in title
        if isinstance(targets, dict):
            class_label = targets['class'][i].item() if 'class' in targets else None
            species = targets['species'][i].item() if 'species' in targets else None

            title = f"Class: {class_label}"
            if species is not None:
                title += f"\nSpecies: {species}"
            axes[i].set_title(title)

            # Draw bbox if available
            if 'bbox' in targets and not (targets['bbox'][i] == 0).all():
                xmin, ymin, xmax, ymax = targets['bbox'][i].tolist()
                width = xmax - xmin
                height = ymax - ymin

                import matplotlib.patches as patches
                rect = patches.Rectangle(
                    (xmin * 64, ymin * 64), width * 64, height * 64,
                    linewidth=2, edgecolor='r', facecolor='none'
                )
                axes[i].add_patch(rect)
        else:
            axes[i].set_title(f"Class: {targets[i].item()}")

        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Selected classes from Caltech101
    selected_classes = ['umbrella', 'chair', 'elephant', 'anchor', 'camera', 'cup', 'ferry', 'lamp', 'sunflower']

    # Create mixed dataloaders with Caltech101 and OxfordPet images
    train_loader, val_loader, test_loader = create_mixed_caltech_pet_dataloaders(
        batch_size=32,
        pet_data_directory="oxford_pet_data",
        caltech_data_directory="caltech_101",
        selected_caltech_classes=selected_classes,
        target_type=["species", "class", "bbox", "segmentation"],
        mixing_ratio=5,
        use_augmentation=True,
        lazy_loading=True
    )
    train_loader_iter = iter(train_loader)

    images1, targets1 = next(train_loader_iter)
    print("First batch targets:")
    print(targets1)

    images2, targets2 = next(train_loader_iter)
    print("\nSecond batch targets:")
    print(targets2)

    print(f"\nTraining images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")
    print(f"Testing images: {len(test_loader.dataset)}")

    print("\nVisualizing samples from the training set:")
    visualize_mixed_batch(train_loader)

    # Test iteration through the dataset
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

    print("\nDataset testing completed successfully!")