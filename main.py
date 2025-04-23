"""
AI usage statement:

AI was used to assist with researching and debugging, as well as helping
with creating docstrings. All code was writte, reviewed and/or modified by a human.
"""

import numpy as np
import torch
import random

from cam_generation.cam_dataset import generate_cam_dataset
from cam_generation.utils import get_best_cam
from config import (
    RUNS_CONFIG,
    MODEL_NAMES,
    CAM_TYPES,
    SELF_LEARNING_EXPERIMENTS_CONFIG,
)
from cam_generation.cam_evaluation import evaluate_cams
from training.pre_training import run_pretraining_process
from training.self_supervised_training import run_self_training_process
from training.supervised_training import run_supervised_training_process
from training.test import test_and_compare_to_baseline
from training.utils import get_best_self_training

# General run configuration
RANDOM_SEED = 27
TEST_MODELS_BEFORE_TRAINING = False
PRETRAIN_MODELS = True
EVALUATE_CAMS = True
GENERATE_CAM_DATASET = True
TRAIN_WEAKLY_SUPERVISED = True
TRAIN_SELF_SUPERVISED = True
TRAIN_FULLY_SUPERVISED = True
EVALUATE_MODELS = True
# Pretraining configuration
PRETRAIN_LEARNING_RATE = 3e-4
PRETRAIN_WEIGHT_DECAY = 1e-4
PRETRAIN_NUM_EPOCHS = 2
# Selftraining configuration
SELFTRAINING_NUM_EPOCHS = 1
SELTRAINING_BOOSTRAP_ROUNDS = 2

supervised_model = None
weakly_supervised_model = None
best_self_training = None

if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        workers = 24
        persistent_workers = True
        pin_memory = True
        batch_size = 100
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        workers = 6
        persistent_workers = True
        pin_memory = True
        batch_size = 48
    else:
        device = torch.device("cpu")
        workers = 2
        persistent_workers = False
        pin_memory = False
        batch_size = 32

    print(f"Using device: {device} with {workers} workers and batch size {batch_size}")

    if TEST_MODELS_BEFORE_TRAINING:
        from models.utils import test_models

        test_models(device)

    if PRETRAIN_MODELS:
        run_pretraining_process(
            runs_config=RUNS_CONFIG,
            model_names=MODEL_NAMES,
            device=device,
            batch_size=batch_size,
            workers=workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            learning_rate=PRETRAIN_LEARNING_RATE,
            weight_decay=PRETRAIN_WEIGHT_DECAY,
            num_epochs=PRETRAIN_NUM_EPOCHS,
        )

    if EVALUATE_CAMS:
        evaluate_cams(
            runs_config=RUNS_CONFIG,
            device=device,
            batch_size=3,
            workers=0,
            persistent_workers=False,
            pin_memory=False,
            num_samples=100,
            cam_types=CAM_TYPES,
        )

    if GENERATE_CAM_DATASET:
        best_cam_dict = get_best_cam(
            runs_config=RUNS_CONFIG,
        )
        generate_cam_dataset(
            runs_config=RUNS_CONFIG,
            cam_dict=best_cam_dict,
            device=device,
            batch_size=batch_size,
            workers=0,
            persistent_workers=False,
            pin_memory=False,
            visualize=5,
        )

    if TRAIN_FULLY_SUPERVISED:
        supervised_model = run_supervised_training_process(
            device=device,
            batch_size=batch_size,
            workers=workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            learning_rate=PRETRAIN_LEARNING_RATE,
            weight_decay=PRETRAIN_WEIGHT_DECAY,
            num_epochs=PRETRAIN_NUM_EPOCHS,
        )

    if TRAIN_WEAKLY_SUPERVISED:
        weakly_supervised_model = run_supervised_training_process(
            device=device,
            batch_size=batch_size,
            workers=0,
            persistent_workers=False,
            pin_memory=pin_memory,
            learning_rate=PRETRAIN_LEARNING_RATE,
            weight_decay=PRETRAIN_WEIGHT_DECAY,
            num_epochs=PRETRAIN_NUM_EPOCHS,
            use_cam_dataset=True,
            cam_threshold=0.2,
        )

    if TRAIN_SELF_SUPERVISED:
        run_self_training_process(
            runs_config=SELF_LEARNING_EXPERIMENTS_CONFIG,
            device=device,
            batch_size=batch_size,
            workers=0,
            persistent_workers=False,
            pin_memory=pin_memory,
            learning_rate=PRETRAIN_LEARNING_RATE,
            weight_decay=PRETRAIN_WEIGHT_DECAY,
            num_epochs=SELFTRAINING_NUM_EPOCHS,
            num_bootstrap_rounds=SELTRAINING_BOOSTRAP_ROUNDS,
            threshold=0.2,
            num_validation_samples=100,
        )

        best_self_training = get_best_self_training(
            runs_config=SELF_LEARNING_EXPERIMENTS_CONFIG,
        )

    if EVALUATE_MODELS:
        test_and_compare_to_baseline(
            device=device,
            batch_size=batch_size,
            workers=workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            self_training_dict=best_self_training,
            baseline_model=supervised_model,
            weakly_supervised_model=weakly_supervised_model,
        )
