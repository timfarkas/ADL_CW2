# AI Usage Statement: AI assistance was used to help
# assist docstrings for this code.

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from custom_data import OxfordPetDataset
import random
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


RANDOM_SEED = 27

class OxfordPetTorchAdapter(Dataset):
    """PyTorch Dataset adapter for the Oxford-IIIT Pet Dataset.

    Converts the basic dataset into PyTorch-compatible format with various
    target options.
    """
    def __init__(self, data_items, oxford_dataset, transform=None, target_type=["class"],
                 normalize_bbox=True, target_transform=None, data_directory="oxford_pet_data"):
        """Initialize the PyTorch dataset adapter.

        Args:
            data_items: List of data items (img_path, class_idx, species_idx, bbox)
            oxford_dataset: OxfordPetDataset instance for image loading
            transform: Optional torchvision transforms to apply to images
            target_type: Type of target ("class", "species", "bbox", or "segmentation")
            normalize_bbox: Whether to normalize bounding box coordinates to [0,1]
            target_transform: Optional transforms to apply to targets (for segmentation masks)
            data_directory: Directory containing the dataset files (default: "oxford_pet_data")
        """
        self.data_items = data_items
        self.oxford_dataset = oxford_dataset
        self.transform = transform

        if isinstance(target_type, str):
            self.target_type = [target_type]
        else:
            self.target_type = target_type

        self.normalize_bbox = normalize_bbox
        self.target_transform = target_transform
        self.data_directory = data_directory

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

        if self.transform:
            image = self.transform(image)

        if len(self.target_type) == 1:
            return image, self._get_target(self.target_type[0], img_path, class_idx, species_idx, bbox, original_width, original_height)

        targets = {}
        for target_type in self.target_type:
            targets[target_type] = self._get_target(target_type, img_path, class_idx, species_idx, bbox, original_width, original_height)

        return image, targets


    def _get_target(self, target_type, img_path, class_idx, species_idx, bbox, original_width, original_height):
        # Apply transformations if provided


        # Return appropriate target
        if target_type in ["class", "breed"]:
            return class_idx
        elif target_type == "species":
            return species_idx
        elif target_type == "bbox":
            if bbox is None:
                # If no bbox available, return zeros or default values
                bbox_tensor = torch.zeros(4, dtype=torch.float32)
            else:
                xmin, ymin, xmax, ymax = bbox

                new_xmin, new_ymin, new_xmax, new_ymax = adjust_bbox_for_center_crop(
                    xmin, ymin, xmax, ymax,
                    orig_w=original_width,
                    orig_h=original_height,
                    final_size=256
                )

                if self.normalize_bbox:
                    new_xmin /= 256.0
                    new_ymin /= 256.0
                    new_xmax /= 256.0
                    new_ymax /= 256.0

                bbox_tensor = torch.tensor([new_xmin, new_ymin, new_xmax, new_ymax], dtype=torch.float32)
            return bbox_tensor
        elif target_type == "segmentation":
            # Load segmentation mask from trimaps directory
            base_name = os.path.splitext(os.path.basename(img_path))[0].lower()
            seg_path = os.path.join(self.data_directory, 'annotations/trimaps', base_name + '.png')
            
            try:
                mask = Image.open(seg_path)
                
                # Apply target transform if provided
                if self.target_transform:
                    mask = self.target_transform(mask)
                else:
                    # Convert to tensor by default if no transform provided
                    mask = transforms.ToTensor()(mask)
                
                return mask
            except FileNotFoundError:
                # If segmentation mask is not found, return a blank mask
                print(f"Warning: Segmentation mask not found for {base_name}")
                blank_mask = torch.zeros((1, original_height, original_width), dtype=torch.float32)
                if self.target_transform:
                    # Resize blank mask to match target transform expectations
                    blank_mask = torch.zeros((1, 256, 256), dtype=torch.float32)
                return blank_mask
        else:
            raise ValueError(f"Unknown target_type: {target_type}")

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

def adjust_bbox_for_center_crop(xmin, ymin, xmax, ymax, orig_w, orig_h, final_size=256):

    # determine the scale factor used by transform.Resize(256)
    shorter_side = min(orig_w, orig_h)
    scale = final_size / float(shorter_side)

    # compute scaled width and height
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    # scale the bbox coordinates
    scaled_xmin = xmin * scale
    scaled_xmax = xmax * scale
    scaled_ymin = ymin * scale
    scaled_ymax = ymax * scale

    # determine amount cropped from left or top side
    crop_x = max(0, (new_w - final_size) / 2.0)
    crop_y = max(0, (new_h - final_size) / 2.0)

    # shift the scaled bbox by the crop offset
    new_xmin = scaled_xmin - crop_x
    new_xmax = scaled_xmax - crop_x
    new_ymin = scaled_ymin - crop_y
    new_ymax = scaled_ymax - crop_y

    # clamp to avoid negative coords or overshoot
    new_xmin = max(0, min(final_size, new_xmin))
    new_xmax = max(0, min(final_size, new_xmax))
    new_ymin = max(0, min(final_size, new_ymin))
    new_ymax = max(0, min(final_size, new_ymax))

    return new_xmin, new_ymin, new_xmax, new_ymax

def create_dataloaders(dataset, batch_size=32, train_ratio=0.7, val_ratio=0.15,
                       test_ratio=0.15, random_seed=RANDOM_SEED, target_type=["class"],
                       normalize_bbox=True, data_directory="oxford_pet_data", use_augmentation=False):
    """Create PyTorch DataLoaders for training, validation, and testing.

    Args:
        dataset: OxfordPetDataset instance
        batch_size: Batch size for DataLoaders
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Seed for reproducible splitting
        target_type: Target type for the model ("class", "species", "bbox", or "segmentation")
        normalize_bbox: Whether to normalize bounding box coordinates
        data_directory: Directory containing the dataset files (default: "oxford_pet_data")

    Returns:
        tuple: (train_loader, val_loader, test_loader) DataLoader instances
    """

    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define target transform for segmentation masks
    target_transform = None
    if "segmentation" in target_type if isinstance(target_type, list) else target_type == "segmentation":
        target_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])

    # split dataset
    train_data, val_data, test_data = split_dataset(
        dataset, train_ratio, val_ratio, test_ratio, random_seed
    )

    # Create PyTorch datasets and dataloaders
    train_dataset = OxfordPetTorchAdapter(
        train_data, dataset, transform=train_transform, target_type=target_type, 
        normalize_bbox=normalize_bbox, target_transform=target_transform, data_directory=data_directory
    )
    val_dataset = OxfordPetTorchAdapter(
        val_data, dataset, transform=val_test_transform, target_type=target_type, 
        normalize_bbox=normalize_bbox, target_transform=target_transform, data_directory=data_directory
    )
    test_dataset = OxfordPetTorchAdapter(
        test_data, dataset, transform=val_test_transform, target_type=target_type, 
        normalize_bbox=normalize_bbox, target_transform=target_transform, data_directory=data_directory
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        # num_workers=8,
        pin_memory=True,
        # persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size
    )

    return train_loader, val_loader, test_loader


# Example usage:
if __name__ == "__main__":

    # Prepare dataset
    dataset = OxfordPetDataset(root_dir="oxford_pet_data").prepare_dataset()
    all_data = dataset.get_all_data()
    adapter = OxfordPetTorchAdapter(all_data, dataset)

    assert len(adapter) == len(all_data), "Length test failed"
    print(f"Length test passed: {len(adapter)} items")


    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset, target_type=["class", "species", "bbox", "segmentation"]
    )

    print(f"Training images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")
    print(f"Testing images: {len(test_loader.dataset)}")

    images, targets = next(iter(train_loader))

    print(targets)


    id = 5

    print(images[id])
    print("First item targets:")
    for key, value in targets.items():
        print(f"  {key}: {value[5]}")

    img = images[id].permute(1, 2, 0).cpu().numpy()

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)

    plt.figure(figsize=(8, 8))
    plt.imshow(img)

    if 'bbox' in targets:
        bbox = targets['bbox'][id].cpu().numpy()

        # If bbox is normalized (values between 0-1), convert to pixel coordinates
        if bbox.max() <= 1.0:
            h, w = img.shape[0:2]
            xmin, ymin, xmax, ymax = bbox
            xmin, xmax = xmin * w, xmax * w
            ymin, ymax = ymin * h, ymax * h
        else:
            xmin, ymin, xmax, ymax = bbox

        # Create rectangle patch
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)

        # Retrieve image path
        img_path = train_loader.dataset.data_items[id][0]

    # Add labels if available
    title = f"Image: {img_path.name}"
    if 'class' in targets:
        class_idx = targets['class'][id].item()
        class_title = train_loader.dataset.oxford_dataset.class_names[class_idx]
        title += f"\nClass: {class_idx} - {class_title}"
    if 'species' in targets:
        species_idx = targets['species'][id].item()
        species_name = "Cat" if species_idx == 0 else "Dog"
        title += f"\nSpecies: {species_idx} - {species_name}"
    if 'segmentation' in targets:
        seg_mask_tensor = targets['segmentation'][id]
        # If the segmentation mask has an extra channel dimension, remove it
        if seg_mask_tensor.ndim > 2:
            seg_mask = seg_mask_tensor.squeeze().cpu().numpy()
        else:
            seg_mask = seg_mask_tensor.cpu().numpy()
        plt.imshow(seg_mask, cmap='jet', alpha=0.5)



    plt.title(title)
    plt.axis('off')
    plt.show()



