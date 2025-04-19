import os
import numpy as np
import torch
import random

from datasets.dataset_manager import DatasetManager

# General run configuration
RANDOM_SEED = 27
TEST_MODELS_BEFORE_TRAINING = True


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
        workers = 2

    print(f"Using device: {device}")

    # Load the dataset
    dataset_manager = DatasetManager()

    if TEST_MODELS_BEFORE_TRAINING:
        from models.utils import test_models

        test_models(device)
