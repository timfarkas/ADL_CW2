import os
import random
import re
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

RANDOM_SEED = 27


class OxfordPetDataset(Dataset):
    """
    Oxford-IIIT Pet Dataset handler for downloading, extracting, processing images,
    and providing PyTorch-compatible dataset access.
    """

    IMAGES_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    ANNOTATIONS_URL = (
        "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
    )

    CAT_BREEDS = [
        "abyssinian",
        "bengal",
        "birman",
        "bombay",
        "british",
        "egyptian",
        "maine",
        "persian",
        "ragdoll",
        "russian",
        "siamese",
        "sphynx",
    ]

    DOG_BREEDS = [
        "american",
        "basset",
        "beagle",
        "boxer",
        "chihuahua",
        "english",
        "german",
        "great",
        "japanese",
        "keeshond",
        "leonberger",
        "miniature",
        "newfoundland",
        "pomeranian",
        "pug",
        "saint",
        "samoyed",
        "scottish",
        "shiba",
        "staffordshire",
        "wheaten",
        "yorkshire",
        "shih",
        "havanese",
        "golden",
    ]

    def __init__(
        self,
        root_dir="oxford_pet_data",
        transform=None,
        target_type=["class"],
        normalize_bbox=True,
        target_transform=None,
        cache_in_memory=False,
        split="train",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=RANDOM_SEED,
        resize_size=64,
    ):
        """Initialize dataset with directory structure and PyTorch adapter settings.

        Args:
            root_dir: Root directory for storing dataset files
            transform: Optional torchvision transforms to apply to images
            target_type: Type of target ("class", "species", "bbox", or "segmentation")
            normalize_bbox: Whether to normalize bounding box coordinates to [0,1]
            target_transform: Optional transforms to apply to targets (for segmentation masks)
            cache_in_memory: Whether to cache images in memory for faster access
            split: Which split to use ("train", "val", or "test")
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            random_seed: Seed for reproducible splitting
        """
        # Set up paths and create directories
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "images"
        self.annotations_dir = self.root_dir / "annotations"

        self.root_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        self.annotations_dir.mkdir(exist_ok=True)

        # Paths to downloaded files
        self.images_tar_path = self.root_dir / "images.tar.gz"
        self.annotations_tar_path = self.root_dir / "annotations.tar.gz"
        self.resize_size = resize_size
        # Class mappings
        self.class_names = None
        self.class_to_idx = None
        self.class_to_species = None
        # PyTorch adapter settings
        self.transform = transform
        if isinstance(target_type, str):
            self.target_type = [target_type]
        else:
            self.target_type = target_type
        self.normalize_bbox = normalize_bbox
        self.target_transform = target_transform

        # Cache settings
        self.cache_in_memory = cache_in_memory
        self.cached_images = {}
        self.cached_masks = {}

        # Prepare the dataset (download, extract, setup mappings)
        self.prepare_dataset()

        # Get all data
        data_labels = self.get_all_data_labels()

        # Validate split parameter
        if split not in ["train", "val", "test"]:
            raise ValueError(
                f"Invalid split: {split}. Must be 'train', 'val', or 'test'"
            )

        # Split the dataset
        train_data, val_data, test_data = self._split_dataset(
            data_labels, train_ratio, val_ratio, test_ratio, random_seed
        )

        # Select the appropriate split
        if split == "train":
            self.data_items = train_data
        elif split == "val":
            self.data_items = val_data
        elif split == "test":
            self.data_items = test_data

        # If caching is enabled, preload images for the selected split only
        if self.cache_in_memory:
            print(f"Preloading images for {split} split into memory cache...")
            for img_path, _, _, _ in self.data_items:
                if str(img_path) not in self.cached_images:
                    self.cached_images[str(img_path)] = self.load_image(img_path)
            print(f"Cached {len(self.cached_images)} images in memory")

    def prepare_dataset(self):
        """Download, extract, and setup class mappings for the dataset.

        Returns:
            self: The initialized dataset object
        """
        self.download_files()
        self.extract_files()
        self.setup_class_mappings()
        print(f"Dataset prepared with {len(self.class_names)} classes.")
        return self

    def download_files(self):
        """Download images and annotations if directory not already present."""
        # Check if images and annotations are already downloaded
        if self.images_tar_path.exists():
            print(f"Images already downloaded: {self.images_tar_path}")
        else:
            print("Downloading images...")
            urllib.request.urlretrieve(self.IMAGES_URL, self.images_tar_path)

        if self.annotations_tar_path.exists():
            print(f"Annotations already downloaded: {self.annotations_tar_path}")
        else:
            print("Downloading annotations...")
            urllib.request.urlretrieve(self.ANNOTATIONS_URL, self.annotations_tar_path)

    def extract_files(self):
        """Extract downloaded tar files if not already extracted."""
        # Extract images and annotations
        if not list(self.images_dir.glob("*.jpg")):
            print("Extracting images...")
            with tarfile.open(self.images_tar_path, "r:gz") as tar:
                tar.extractall(path=self.root_dir)

        if not (self.annotations_dir / "xmls").exists():
            print("Extracting annotations...")
            with tarfile.open(self.annotations_tar_path, "r:gz") as tar:
                tar.extractall(path=self.root_dir)

    def setup_class_mappings(self):
        """Create mapping between class names, indices, and species."""
        # Get all image files
        image_files = list(self.images_dir.glob("*.jpg"))

        # Extract class names from filenames
        class_names = sorted(
            list(set(re.sub(r"_\d+$", "", img.stem.lower()) for img in image_files))
        )

        self.class_names = class_names
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}

        # Map classes to species (cat or dog)
        self.class_to_species = {}
        for cls in class_names:
            if any(cls.startswith(cat_breed) for cat_breed in self.CAT_BREEDS):
                self.class_to_species[cls] = "cat"
            elif any(cls.startswith(dog_breed) for dog_breed in self.DOG_BREEDS):
                self.class_to_species[cls] = "dog"
            else:
                self.class_to_species[cls] = "unknown"

    def get_all_data_labels(self):
        """Get all dataset entries.

        Returns:
            list: List of tuples containing (image_path, class_idx, species_idx, bbox)
        """
        image_files = sorted(list(self.images_dir.glob("*.jpg")))
        dataset = []
        species_to_idx = {"cat": 0, "dog": 1, "unknown": 2}

        for img_path in image_files:
            # Get class info
            class_name = re.sub(r"_\d+$", "", img_path.stem.lower())
            class_idx = self.class_to_idx[class_name]
            species = self.class_to_species[class_name]
            species_idx = species_to_idx[species]

            # Get bounding box
            bbox = self.get_head_bbox(img_path.stem)

            dataset.append((img_path, class_idx, species_idx, bbox))

        return dataset

    def load_image(self, image_path):
        """Load an image from path and convert to RGB.

        Args:
            image_path: Path to the image file

        Returns:
            PIL.Image: Loaded RGB image
        """
        return Image.open(image_path).convert("RGB")

    def get_head_bbox(self, image_name):
        """Get pet head bounding box from annotation XML file.

        Args:
            image_name: Base name of the image without extension

        Returns:
            tuple: (xmin, ymin, xmax, ymax) or None if not found
        """
        xml_path = self.annotations_dir / "xmls" / f"{image_name}.xml"

        if not xml_path.exists():
            return None

        # Parse the XML file
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Find and extract bounding box
        bbox_elem = root.find(".//bndbox")
        if bbox_elem is None:
            return None

        # extracts coordinates
        xmin = int(bbox_elem.find("xmin").text)
        ymin = int(bbox_elem.find("ymin").text)
        xmax = int(bbox_elem.find("xmax").text)
        ymax = int(bbox_elem.find("ymax").text)

        return (xmin, ymin, xmax, ymax)

    def _split_dataset(
        self,
        dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=RANDOM_SEED,
    ):
        """Split the dataset into training, validation, and test sets.

        Args:
            dataset: List of data items
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            random_seed: Seed for reproducible splitting

        Returns:
            tuple: (train_data, val_data, test_data) lists
        """
        random.seed(random_seed)

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

    def check_all_segmentation_masks(self):
        print("Checking for segmentation masks with case sensitivity testing...")

        image_files = list(self.images_dir.glob("*.jpg"))
        total_images = len(image_files)

        original_case_matches = 0
        lowercase_only_matches = 0
        no_matches = 0
        missing_masks = set()

        for img_path in image_files:
            base_name = os.path.splitext(os.path.basename(img_path))[0]

            # Try original case
            orig_seg_path = os.path.join(
                self.root_dir, "annotations/trimaps", base_name + ".png"
            )

            # Then try lowercase
            lower_seg_path = os.path.join(
                self.root_dir, "annotations/trimaps", base_name.lower() + ".png"
            )

            if os.path.exists(orig_seg_path):
                original_case_matches += 1
            elif os.path.exists(lower_seg_path):
                lowercase_only_matches += 1
            else:
                no_matches += 1
                missing_masks.add(base_name)

        print(f"\nTotal images: {total_images}")
        print(
            f"Original case matches: {original_case_matches} ({original_case_matches / total_images:.1%})"
        )
        print(
            f"Lowercase-only matches: {lowercase_only_matches} ({lowercase_only_matches / total_images:.1%})"
        )
        print(f"No matches: {no_matches} ({no_matches / total_images:.1%})")

        return missing_masks

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

        # Load image from cache if available, otherwise load from disk
        if self.cache_in_memory and str(img_path) in self.cached_images:
            image = self.cached_images[str(img_path)]
        else:
            image = self.load_image(img_path)
            if self.cache_in_memory:
                self.cached_images[str(img_path)] = image

        original_width, original_height = image.size

        if self.transform:
            image = self.transform(image)

        if len(self.target_type) == 1:
            target = self._get_target(
                self.target_type[0],
                img_path,
                class_idx,
                species_idx,
                bbox,
                original_width,
                original_height,
            )
            return image, target
        else:
            targets = {}
            for t_type in self.target_type:
                targets[t_type] = self._get_target(
                    t_type,
                    img_path,
                    class_idx,
                    species_idx,
                    bbox,
                    original_width,
                    original_height,
                )
            return image, targets

    def _get_target(
        self,
        target_type,
        img_path,
        class_idx,
        species_idx,
        bbox,
        original_width,
        original_height,
    ):
        """Get the target based on the specified target type.

        Args:
            target_type: Type of target to retrieve
            img_path: Path to the image file
            class_idx: Class index
            species_idx: Species index
            bbox: Bounding box coordinates
            original_width: Original image width
            original_height: Original image height

        Returns:
            Target based on the specified type

        Raises:
            ValueError: If target_type is not recognized
        """
        if target_type == "is_animal":
            return 1  # All images in OxfordPetDataset are animals
        elif target_type in ["class", "breed"]:
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
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                    orig_w=original_width,
                    orig_h=original_height,
                    final_size=self.resize_size,
                )

                if self.normalize_bbox:
                    new_xmin /= self.resize_size
                    new_ymin /= self.resize_size
                    new_xmax /= self.resize_size
                    new_ymax /= self.resize_size

                bbox_tensor = torch.tensor(
                    [new_xmin, new_ymin, new_xmax, new_ymax], dtype=torch.float32
                )
            return bbox_tensor
        elif target_type == "segmentation":
            # Check if mask is in cache
            cache_key = str(img_path) + "_segmentation"
            if self.cache_in_memory and cache_key in self.cached_masks:
                return self.cached_masks[cache_key]

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            seg_path = os.path.join(
                self.root_dir, "annotations/trimaps", base_name + ".png"
            )

            try:
                mask = Image.open(seg_path)

                # Apply target transform if provided
                if self.target_transform:
                    mask = self.target_transform(mask)
                else:
                    # Convert to tensor by default if no transform provided
                    mask = transforms.ToTensor()(mask)

                # Cache the mask if caching is enabled
                if self.cache_in_memory:
                    self.cached_masks[cache_key] = mask
                return mask
            except FileNotFoundError:
                # If segmentation mask is not found, return a blank mask
                print(f"Warning: Segmentation mask not found for {base_name}")
                blank_mask = torch.zeros(
                    (1, original_height, original_width), dtype=torch.float32
                )
                if self.target_transform:
                    # Resize blank mask to match target transform expectations
                    blank_mask = torch.zeros(
                        (1, self.resize_size, self.resize_size), dtype=torch.float32
                    )
                return blank_mask
        else:
            raise ValueError(f"Unknown target_type: {target_type}")


class SegmentationToTensor:
    def __call__(self, pic):
        tensor = torch.as_tensor(np.array(pic), dtype=torch.long)
        return tensor


def split_dataset(
    dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=RANDOM_SEED
):
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
    val_data = all_data[train_size : train_size + val_size]
    test_data = all_data[train_size + val_size :]

    print(
        f"Dataset split complete: training ({len(train_data) / dataset_size:.1%}), "
        f"validation ({len(val_data) / dataset_size:.1%}), testing ({len(test_data) / dataset_size:.1%})"
    )

    return train_data, val_data, test_data


def adjust_bbox_for_center_crop(xmin, ymin, xmax, ymax, orig_w, orig_h, final_size):
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


def create_dataloaders(
    batch_size=32,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    resize_size=64,
    random_seed=RANDOM_SEED,
    target_type=["class"],
    normalize_bbox=True,
    data_directory="oxford_pet_data",
    use_augmentation=False,
    lazy_loading=True,
    shuffle=True,
):
    """Create PyTorch DataLoaders for training, validation, and testing.


        Args:
            batch_size: Batch size for DataLoaders
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
        random_seed: Seed for reproducible splitting
        target_type: Target type for the model ("class", "species", "bbox", or "segmentation")
        normalize_bbox: Whether to normalize bounding box coordinates
        data_directory: Directory containing the dataset files (default: "oxford_pet_data")
        use_augmentation: Whether to use data augmentation for training
        lazy_loading: Whether to load images on-demand (True) or preload into memory (False)

    Returns:
        tuple: (train_loader, val_loader, test_loader) DataLoader instances
    """

    if use_augmentation:
        train_transform = transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=Image.NEAREST),
                transforms.CenterCrop(resize_size),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=Image.NEAREST),
                transforms.CenterCrop(resize_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    val_test_transform = transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=Image.NEAREST),
            transforms.CenterCrop(resize_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Define target transform for segmentation masks
    target_transform = None
    if (
        "segmentation" in target_type
        if isinstance(target_type, list)
        else target_type == "segmentation"
    ):
        target_transform = transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=InterpolationMode.NEAREST),
                transforms.CenterCrop(resize_size),
                SegmentationToTensor(),
            ]
        )

    # Create dataset with all splits
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
        resize_size=resize_size,
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
        resize_size=resize_size,
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
        resize_size=resize_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        # num_workers=8,
        pin_memory=True,
        # persistent_workers=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def create_sample_loader_from_existing_loader(
    loader, num_samples=20, batch_size=20, shuffle=False, seed=42
):
    """
    Create a new DataLoader containing a subset of (image, segmentation_mask) pairs,
    extracted from a dataset where targets are dicts containing the 'segmentation' key.

    Args:
        loader (DataLoader): Existing PyTorch DataLoader.
        num_samples (int): Number of samples to extract.
        batch_size (int): Batch size for the new DataLoader.
        shuffle (bool): Whether to shuffle before selecting samples.
        seed (int): Random seed for reproducibility.

    Returns:
        DataLoader: New DataLoader yielding (image, segmentation_mask) batches.
    """
    dataset = loader.dataset
    dataset_size = len(dataset)

    if num_samples > dataset_size:
        raise ValueError(
            f"Requested {num_samples} samples, but dataset only has {dataset_size} items."
        )

    # Choose random or sequential indices
    indices = list(range(dataset_size))
    if shuffle:
        random.seed(seed)
        random.shuffle(indices)

    selected_indices = indices[:num_samples]

    # Extract (image, segmentation) pairs and store as tensors
    image_list = []
    mask_list = []
    for idx in selected_indices:
        image, target = dataset[idx]
        mask = target["segmentation"]
        image_list.append(image.unsqueeze(0))
        mask_list.append(mask.unsqueeze(0))

    # Stack to tensors
    images_tensor = torch.cat(image_list, dim=0)
    masks_tensor = torch.cat(mask_list, dim=0)

    # Wrap into a TensorDataset
    tensor_dataset = TensorDataset(images_tensor, masks_tensor)

    # Return standard DataLoader
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)


# Example usage:
if __name__ == "__main__":
    print("\nChecking all segmentation masks in dataset:")
    dataset = OxfordPetDataset()
    dataset.check_all_segmentation_masks()

    print("\nTESTING DATASET AND DATALOADER")
    print("Testing labels...")
    all_data = dataset.get_all_data_labels()
    print(f"Dataset contains {len(all_data)} images")

    # test example
    if all_data:
        print("Test example assessing and processing item:")

        img_path, class_idx, species_idx, bbox = all_data[0]
        print(f"First image: {img_path.name}")
        print(f"Class: {dataset.class_names[class_idx]}")
        print(f"Species: {'cat' if species_idx == 0 else 'dog'}")

        # Load and process the image
        image = dataset.load_image(img_path)
        print(f"Image size (x, y): {image.size}")
        print(f"Bounding box (xmin, ymin, xmax, ymax): {bbox}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        target_type=["class", "species", "bbox", "segmentation"], lazy_loading=False
    )

    print(f"\nTraining images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")
    print(f"Testing images: {len(test_loader.dataset)}")

    images, targets = next(iter(train_loader))

    print(targets)

    # Display 10 images in a grid
    plt.figure(figsize=(20, 15))

    for i in range(10):
        plt.subplot(2, 5, i + 1)

        # Get and process image
        img = images[i].permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        plt.imshow(img)

        if "bbox" in targets:
            bbox = targets["bbox"][i].cpu().numpy()

            # If bbox normalized, convert to pixel coordinates
            if bbox.max() <= 1.0:
                h, w = img.shape[0:2]
                xmin, ymin, xmax, ymax = bbox
                xmin, xmax = xmin * w, xmax * w
                ymin, ymax = ymin * h, ymax * h
            else:
                xmin, ymin, xmax, ymax = bbox

            rect = plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                edgecolor="red",
                linewidth=2,
            )
            plt.gca().add_patch(rect)

        title = ""
        if "class" in targets:
            class_idx = targets["class"][i].item()
            class_title = train_loader.dataset.class_names[class_idx]
            title += f"{class_title}"
        if "species" in targets:
            species_idx = targets["species"][i].item()
            species_name = "Cat" if species_idx == 0 else "Dog"
            title += f" ({species_name})"

        if "segmentation" in targets:
            seg_mask_tensor = targets["segmentation"][i]
            if seg_mask_tensor.ndim > 2:
                seg_mask = seg_mask_tensor.squeeze().cpu().numpy()
            else:
                seg_mask = seg_mask_tensor.cpu().numpy()
            plt.imshow(seg_mask, cmap="jet", alpha=0.5)

        plt.title(title, fontsize=8)
        plt.axis("off")

    plt.title(title)
    plt.axis("off")
    plt.show()
