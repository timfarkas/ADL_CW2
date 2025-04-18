import os
import numpy as np
import torch
import random

from datasets.dataset_manager import DatasetManager

RANDOM_SEED = 27

if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    # Set device
    print(os.cpu_count())
    if torch.cuda.is_available():
        device = torch.device("cuda")
        workers = 12
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        workers = 6
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load the dataset
    dataset_manager = DatasetManager()
    print(dataset_manager.pet_datasets[0][0][0])

