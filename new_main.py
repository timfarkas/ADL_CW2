import numpy as np
import torch
import random

from cam_generation.utils import get_best_cam
from new_runs_config import runs_config, model_names, cam_types
from cam_generation.cam_evaluation import evaluate_cams
from training.pretraining import run_pretraining_process

# General run configuration
RANDOM_SEED = 27
TEST_MODELS_BEFORE_TRAINING = False
PRETRAIN_MODELS = True
EVALUATE_CAMS = True
GENERATE_CAM_DATASET = True
PRETRAIN_LEARNING_RATE = 3e-4
PRETRAIN_WEIGHT_DECAY = 1e-4
PRETRAIN_NUM_EPOCHS = 2

if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        workers = 12
        persistent_workers = True
        batch_size = 64
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        workers = 6
        persistent_workers = True
        batch_size = 48
    else:
        device = torch.device("cpu")
        workers = 2
        persistent_workers = False
        batch_size = 32

    print(f"Using device: {device} with {workers} workers and batch size {batch_size}")

    if TEST_MODELS_BEFORE_TRAINING:
        from models.utils import test_models

        test_models(device)

    if PRETRAIN_MODELS:
        run_pretraining_process(
            runs_config=runs_config,
            model_names=model_names,
            device=device,
            batch_size=batch_size,
            workers=workers,
            persistent_workers=persistent_workers,
            learning_rate=PRETRAIN_LEARNING_RATE,
            weight_decay=PRETRAIN_WEIGHT_DECAY,
            num_epochs=PRETRAIN_NUM_EPOCHS,
        )

    if EVALUATE_CAMS:
        evaluate_cams(
            runs_config=runs_config,
            device=device,
            batch_size=batch_size,
            workers=workers,
            persistent_workers=persistent_workers,
            num_samples=50,
            cam_types=cam_types,
        )

    if GENERATE_CAM_DATASET:
        best_cam_dict = get_best_cam(
            runs_config=runs_config,
        )
