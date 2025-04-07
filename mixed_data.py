from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import os
import torch
import random
from torchvision import transforms
from data import OxfordPetDataset, create_dataloaders, SegmentationToTensor
import matplotlib.pyplot as plt
import numpy as np

# Instructions - before running the code below:
# 1) Please download the dataset from https://drive.google.com/drive/folders/1ZBaMJxZtUNHIuGj8D8v3B9Adn8dbHwSS?usp=sharing
# 2) Place BG-20k folder inside of the ADL_CW2 directory

RANDOM_SEED = 27


def create_virtual_splits(bg_base_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=RANDOM_SEED):
    """Create virtual splits of images using indices instead of physical copying.

    Args:
        bg_base_dir: Path to the base BG-20k directory
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility

    Returns:
        tuple: (train_indices, val_indices, test_indices) for the dataset
    """

    random.seed(random_seed)

    bg_path = Path(bg_base_dir)

    if not bg_path.exists():
        raise ValueError(f"Background folder {bg_path} does not exist.")

    # Path to train directory
    # Splitting the approx. 3000 images in the train directory
    train_dir = bg_path / "train"

    if not train_dir.exists():
        raise ValueError(f"Train directory {train_dir} does not exist.")

    # Find all image files
    all_images = list(train_dir.glob("*.jpg"))

    if not all_images:
        raise ValueError(f"No images found in directory {train_dir}")

    print(f"Found {len(all_images)} images in directory.")

    # Create indices for all images
    all_indices = list(range(len(all_images)))

    # Shuffle the indices
    random.shuffle(all_indices)

    # Calculate split sizes
    total_images = len(all_images)
    val_size = int(val_ratio * total_images)
    test_size = int(test_ratio * total_images)

    # Create the splits
    val_indices = all_indices[:val_size]
    test_indices = all_indices[val_size:val_size + test_size]
    train_indices = all_indices[val_size + test_size:]

    print(f"Created virtual splits: training ({len(train_indices)}, {len(train_indices) / total_images:.1%}), "
          f"validation ({len(val_indices)}, {len(val_indices) / total_images:.1%}), "
          f"testing ({len(test_indices)}, {len(test_indices) / total_images:.1%})")

    return train_indices, val_indices, test_indices


class BackgroundDataset(Dataset):
    """Dataset for BG-20K background images (without pets)."""

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
        self.bg_files = list(self.bg_dir.glob("*.jpg"))

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


class MixedDataset(Dataset):
    """
    Dataset that combines the OxfordPetDataset with background images
    at specified intervals.
    """

    def __init__(self, main_dataset, bg_dataset, mixing_ratio=5):
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
            self.index_map.append(('main', i))

            # Insert background images
            if (i + 1) % self.mixing_ratio == 0 and bg_counter < len(bg_indices):
                self.index_map.append(('bg', bg_indices[bg_counter]))
                bg_counter += 1

        print(f"Mixed dataset created with {main_size} main images and {bg_counter} background images")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        dataset_type, dataset_idx = self.index_map[idx]

        if dataset_type == 'main':
            return self.main_dataset[dataset_idx]
        else:  # 'bg'
            return self.bg_dataset[dataset_idx]


def create_mixed_dataloaders(batch_size=32, train_ratio=0.7, val_ratio=0.15,
                             test_ratio=0.15, random_seed=RANDOM_SEED, target_type=["class"],
                             normalize_bbox=True, data_directory="oxford_pet_data",
                             bg_directory=None, mixing_ratio=5, bg_label=-1,
                             use_augmentation=False, lazy_loading=True):
    """Create PyTorch DataLoaders with background image integration.

    Args:
        batch_size: Batch size for DataLoaders
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Seed for reproducible splitting
        target_type: Target type for the model ("class", "species", "bbox", or "segmentation")
        normalize_bbox: Whether to normalize bounding box coordinates
        data_directory: Directory containing the dataset files
        bg_directory: Directory containing background images
        mixing_ratio: Insert a background image every n images (e.g., 5 means every 5th image)
        bg_label: Class label to use for background images (usually -1 for "no object")
        use_augmentation: Whether to use data augmentation for training
        lazy_loading: Whether to load images on-demand (True) or preload into memory (False)

    Returns:
        tuple: (train_loader, val_loader, test_loader) DataLoader instances
    """
    if bg_directory is None or not os.path.exists(bg_directory):
        print("Warning: Background directory not found or not specified. Using standard dataloaders.")
        return create_dataloaders(
            batch_size, train_ratio, val_ratio, test_ratio, random_seed,
            target_type, normalize_bbox, data_directory, use_augmentation, lazy_loading
        )

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

    target_transform = None
    if "segmentation" in target_type if isinstance(target_type, list) else target_type == "segmentation":
        target_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            SegmentationToTensor(),
        ])

    # Create Oxford Pet datasets
    train_pet_dataset = OxfordPetDataset(
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
        random_seed=random_seed
    )

    val_pet_dataset = OxfordPetDataset(
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
        random_seed=random_seed
    )

    test_pet_dataset = OxfordPetDataset(
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
        random_seed=random_seed
    )

    # Create background datasets
    try:
        train_bg_dataset = BackgroundDataset(
            bg_dir=bg_directory,
            transform=train_transform,
            target_label=bg_label,
            target_type=target_type
        )

        val_bg_dataset = BackgroundDataset(
            bg_dir=bg_directory,
            transform=val_test_transform,
            target_label=bg_label,
            target_type=target_type
        )

        test_bg_dataset = BackgroundDataset(
            bg_dir=bg_directory,
            transform=val_test_transform,
            target_label=bg_label,
            target_type=target_type
        )
    except ValueError as e:
        print(f"Error creating background dataset: {e}")
        print("Using standard dataloaders without background mixing.")
        return create_dataloaders(
            batch_size, train_ratio, val_ratio, test_ratio, random_seed,
            target_type, normalize_bbox, data_directory, use_augmentation, lazy_loading
        )

    # Create mixed datasets
    train_mixed_dataset = MixedDataset(train_pet_dataset, train_bg_dataset, mixing_ratio)
    val_mixed_dataset = MixedDataset(val_pet_dataset, val_bg_dataset, mixing_ratio)
    test_mixed_dataset = MixedDataset(test_pet_dataset, test_bg_dataset, mixing_ratio)

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


if __name__ == "__main__":

    # test script

    train_loader, val_loader, test_loader = create_mixed_dataloaders(
        batch_size=32,
        data_directory="oxford_pet_data",
        bg_directory="bg-20k/train",
        target_type=["class", "bbox", "segmentation"],
        mixing_ratio=5,
        use_augmentation=True,
        lazy_loading=True
    )

    print(f"Training images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")
    print(f"Testing images: {len(test_loader.dataset)}")


    def visualize_batch(dataloader, num_samples=5):
        # Get a batch
        images, targets = next(iter(dataloader))

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


    print("\nVisualizing samples from the training set:")
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

    print("\nDataset testing completed successfully!")