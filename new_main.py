import numpy as np
import torch
import random

from cam_generation.cam_dataset import generate_cam_dataset
from cam_generation.utils import get_best_cam
from new_runs_config import (
    runs_config,
    model_names,
    cam_types,
    self_learning_experiments_config,
)
from cam_generation.cam_evaluation import evaluate_cams
from training.pre_training import run_pretraining_process
from training.self_training import run_self_training_process
from training.supervised_training import run_supervised_training_process
from training.test import test_and_compare_to_baseline
from training.utils import get_best_self_training

# General run configuration
RANDOM_SEED = 27
TEST_MODELS_BEFORE_TRAINING = False
PRETRAIN_MODELS = False
EVALUATE_CAMS = False
GENERATE_CAM_DATASET = False
TRAIN_SEMI_SUPERVISED = True
TRAIN_SELFTRAINING = False
TRAIN_FULLY_SUPERVISED = True
# Pretraining configuration
PRETRAIN_LEARNING_RATE = 3e-4
PRETRAIN_WEIGHT_DECAY = 1e-4
PRETRAIN_NUM_EPOCHS = 20
# Selftraining configuration
SELFTRAINING_NUM_EPOCHS = 5
SELTRAINING_BOOSTRAP_ROUNDS = 2

supervised_model = None

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
            runs_config=runs_config,
            model_names=model_names,
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
            runs_config=runs_config,
            device=device,
            batch_size=3,
            workers=0,
            persistent_workers=False,
            pin_memory=False,
            num_samples=50,
            cam_types=cam_types,
        )

    if GENERATE_CAM_DATASET:
        best_cam_dict = get_best_cam(
            runs_config=runs_config,
        )
        generate_cam_dataset(
            runs_config=runs_config,
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
            workers=0,
            persistent_workers=False,
            pin_memory=pin_memory,
            learning_rate=PRETRAIN_LEARNING_RATE,
            weight_decay=PRETRAIN_WEIGHT_DECAY,
            num_epochs=PRETRAIN_NUM_EPOCHS,
        )

    if TRAIN_SEMI_SUPERVISED:
        semi_supervised_model = run_supervised_training_process(
            device=device,
            batch_size=batch_size,
            workers=workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            learning_rate=PRETRAIN_LEARNING_RATE,
            weight_decay=PRETRAIN_WEIGHT_DECAY,
            num_epochs=PRETRAIN_NUM_EPOCHS,
            use_cam_dataset=True,
            cam_threshold=0.2,
        )

        test_and_compare_to_baseline(
            device=device,
            batch_size=batch_size,
            workers=workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            model_to_compare=semi_supervised_model,
            baseline_model=supervised_model,
        )

    if TRAIN_SELFTRAINING:
        run_self_training_process(
            runs_config=self_learning_experiments_config,
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
        )

        best_self_training = get_best_self_training(
            runs_config=self_learning_experiments_config,
        )

        test_and_compare_to_baseline(
            device=device,
            batch_size=batch_size,
            workers=workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            self_training_dict=best_self_training,
            baseline_model=supervised_model,
        )
