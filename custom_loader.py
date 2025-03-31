# AI Usage Statement: AI assistance was used to help
# assist docstrings for this code.

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from custom_data import OxfordPetDataset
import random

RANDOM_SEED = 27

class OxfordPetTorchAdapter(Dataset):
    """PyTorch Dataset adapter for the Oxford-IIIT Pet Dataset.

    Converts the basic dataset into PyTorch-compatible format with various
    target options.
    """
    def __init__(self, data_items, oxford_dataset, transform=None, target_type="class", normalize_bbox=True):
        """Initialize the PyTorch dataset adapter.

        Args:
            data_items: List of data items (img_path, class_idx, species_idx, bbox)
            oxford_dataset: OxfordPetDataset instance for image loading
            transform: Optional torchvision transforms to apply to images
            target_type: Type of target ("class", "species", or "bbox")
            normalize_bbox: Whether to normalize bounding box coordinates to [0,1]
        """
        self.data_items = data_items
        self.oxford_dataset = oxford_dataset
        self.transform = transform
        self.target_type = target_type
        self.normalize_bbox = normalize_bbox

    def __len__(self):
        """Return the number of items in the dataset.

        Returns:
            int: Number of items
        """
        return len(self.data_items)

    def __getitem__(self, idx):
        """Get a dataset item by index.

        Args:
            idx: Index of the item to retrieve

        Returns:
            tuple: (image, target) where target depends on target_type

        Raises:
            ValueError: If target_type is not recognized
        """
        img_path, class_idx, species_idx, bbox = self.data_items[idx]

        # Load image
        image = self.oxford_dataset.load_image(img_path)
        original_width, original_height = image.size

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # Return appropriate target
        if self.target_type in ["class", "breed"]:
            return image, class_idx
        elif self.target_type == "species":
            return image, species_idx
        elif self.target_type == "bbox":
            if bbox is None:
                # If no bbox available, return zeros or default values
                bbox_tensor = torch.zeros(4, dtype=torch.float32)
            else:
                xmin, ymin, xmax, ymax = bbox

                if self.normalize_bbox:
                    xmin = xmin / original_width
                    xmax = xmax / original_width
                    ymin = ymin / original_height
                    ymax = ymax / original_height

                bbox_tensor = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32)
            return image, bbox_tensor

        else:
            raise ValueError(f"Unknown target_type: {self.target_type}")

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=RANDOM_SEED):
    """Split the dataset into training, validation, and test sets.

    Args:
        dataset: OxfordPetDataset instance
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Seed for reproducible splitting

    Returns:
        tuple: (train_data, val_data, test_data) lists
    """

    random.seed(random_seed)

    # retrieve and shuffle all data
    all_data = dataset.get_all_data()
    random.shuffle(all_data)

    # calc split sizes and apportion data
    dataset_size = len(all_data)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)

    train_data = all_data[:train_size]
    val_data = all_data[train_size:train_size + val_size]
    test_data = all_data[train_size + val_size:]

    print(f"Dataset split complete: training ({len(train_data) / dataset_size:.1%}), "
          f"validation ({len(val_data) / dataset_size:.1%}), testing ({len(test_data) / dataset_size:.1%})")

    return train_data, val_data, test_data


def create_dataloaders(dataset, batch_size=32, train_ratio=0.7, val_ratio=0.15,
                       test_ratio=0.15, random_seed=RANDOM_SEED, target_type="class", normalize_bbox=True):
    """Create PyTorch DataLoaders for training, validation, and testing.

    Args:
        dataset: OxfordPetDataset instance
        batch_size: Batch size for DataLoaders
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Seed for reproducible splitting
        target_type: Target type for the model ("class", "species", or "bbox")
        normalize_bbox: Whether to normalize bounding box coordinates

    Returns:
        tuple: (train_loader, val_loader, test_loader) DataLoader instances
    """

    # define transformations - currently commented out until instructed that we need them
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(10),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # split dataset
    train_data, val_data, test_data = split_dataset(
        dataset, train_ratio, val_ratio, test_ratio, random_seed
    )

    # Create PyTorch datasets and dataloaders
    train_dataset = OxfordPetTorchAdapter(
        train_data, dataset, transform=train_transform, target_type=target_type, normalize_bbox=normalize_bbox
    )
    val_dataset = OxfordPetTorchAdapter(
        val_data, dataset, transform=val_test_transform, target_type=target_type, normalize_bbox=normalize_bbox
    )
    test_dataset = OxfordPetTorchAdapter(
        test_data, dataset, transform=val_test_transform, target_type=target_type, normalize_bbox=normalize_bbox
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader


# Example usage:
if __name__ == "__main__":

    # Prepare dataset
    dataset = OxfordPetDataset().prepare_dataset()

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(dataset, batch_size=32)

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Testing batches: {len(test_loader)}")