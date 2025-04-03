# AI Usage Statement: AI assistance was used to help
# assist docstrings for this code.

import tarfile
import urllib.request
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image
import re


class OxfordPetDataset:
    """
    Oxford-IIIT Pet Dataset handler for downloading, extracting and processing images.
    """

    IMAGES_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    ANNOTATIONS_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

    CAT_BREEDS = [
        'abyssinian', 'bengal', 'birman', 'bombay', 'british', 'egyptian', 'maine',
        'persian', 'ragdoll', 'russian', 'siamese', 'sphynx'
    ]

    DOG_BREEDS = [
        'american', 'basset', 'beagle', 'boxer', 'chihuahua', 'english', 'german',
        'great', 'japanese', 'keeshond', 'leonberger', 'miniature', 'newfoundland',
        'pomeranian', 'pug', 'saint', 'samoyed', 'scottish', 'shiba', 'staffordshire',
        'wheaten', 'yorkshire', 'shih', 'havanese', 'golden'
    ]

    def __init__(self, root_dir="oxford_pet_data"):
        """Initialize dataset with directory structure.

        Args:
            root_dir: Root directory for storing dataset files
        """
        # set paths and create directories for root, images, and annotations
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "images"
        self.annotations_dir = self.root_dir / "annotations"

        self.root_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        self.annotations_dir.mkdir(exist_ok=True)

        # paths to downloaded files
        self.images_tar_path = self.root_dir / "images.tar.gz"
        self.annotations_tar_path = self.root_dir / "annotations.tar.gz"

        self.class_names = None
        self.class_to_idx = None
        self.class_to_species = None

        self.setup_class_mappings()

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
        # check if images and annotations are already downloaded
        # if not download them

        if self.images_tar_path.exists():
            print(f"Images already downloaded: {self.images_tar_path}")
        else:
            print(f"Downloading images...")
            urllib.request.urlretrieve(self.IMAGES_URL, self.images_tar_path)

        if self.annotations_tar_path.exists():
            print(f"Annotations already downloaded: {self.annotations_tar_path}")
        else:
            print(f"Downloading annotations...")
            urllib.request.urlretrieve(self.ANNOTATIONS_URL, self.annotations_tar_path)

    def extract_files(self):
        """Extract downloaded tar files if not already extracted."""

        # Extract images and annotations
        if not list(self.images_dir.glob("*.jpg")):
            print("Extracting images...")
            with tarfile.open(self.images_tar_path, 'r:gz') as tar:
                tar.extractall(path=self.root_dir)

        if not (self.annotations_dir / "xmls").exists():
            print("Extracting annotations...")
            with tarfile.open(self.annotations_tar_path, 'r:gz') as tar:
                tar.extractall(path=self.root_dir)

    def setup_class_mappings(self):
        """Create mapping between class names, indices, and species."""

        # Get all image files
        image_files = list(self.images_dir.glob("*.jpg"))

        # Extract class names from filenames
        class_names = sorted(list(set(re.sub(r'_\d+$', '', img.stem.lower()) for img in image_files)))

        self.class_names = class_names
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}

        # Map classes to species (cat or dog)
        self.class_to_species = {}
        for cls in class_names:
            if any(cls.startswith(cat_breed) for cat_breed in self.CAT_BREEDS):
                self.class_to_species[cls] = 'cat'
            elif any(cls.startswith(dog_breed) for dog_breed in self.DOG_BREEDS):
                self.class_to_species[cls] = 'dog'
            else:
                self.class_to_species[cls] = 'unknown'

    def get_all_data(self):
        """Get all dataset entries.

        Returns:
            list: List of tuples containing (image_path, class_idx, species_idx, bbox)
        """
        image_files = sorted(list(self.images_dir.glob("*.jpg")))
        dataset = []
        species_to_idx = {'cat': 0, 'dog': 1, 'unknown': 2}

        for img_path in image_files:
            # Get class info
            class_name = re.sub(r'_\d+$', '', img_path.stem.lower())
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

    def __len__(self):
        return len(list(self.images_dir.glob("*.jpg")))


def main():
    # Prepare the dataset
    dataset = OxfordPetDataset().prepare_dataset()

    # Get all data for training
    all_data = dataset.get_all_data()
    print(f"Dataset contains {len(all_data)} images")

    # test example
    if all_data:
        print("\nTest example assessing and processing item:")

        img_path, class_idx, species_idx, bbox = all_data[0]
        print(f"First image: {img_path.name}")
        print(f"Class: {dataset.class_names[class_idx]}")
        print(f"Species: {'cat' if species_idx == 0 else 'dog'}")

        # Load and process the image
        image = dataset.load_image(img_path)
        print(f"Image size (x, y): {image.size}")
        print(f"Bounding box (xmin, ymin, xmax, ymax): {bbox}")

if __name__ == "__main__":
    main()