import random
from torch.utils.data import Dataset


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
