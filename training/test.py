import json
import os
import torch

from cam_generation.utils import get_best_cam_dataset_file
from datasets.dataloader_manager import DataloaderManager
from datasets.dataset_manager import DatasetManager
from models.u_net import UNet
from runs_config import (
    get_checkpoints_and_logs_dirs,
    baseline_model_folder,
    baseline_model_name,
    visualizations_folder,
    semi_supervised_model_folder,
)
from training.evaluations import evaluate_segmentation_model


def test_and_compare_to_baseline(
    device: torch.device,
    batch_size: int,
    workers: int,
    persistent_workers: bool,
    pin_memory: bool,
    self_training_dict: dict | None = None,
    weakly_supervised_model: torch.nn.Module | None = None,
    baseline_model: torch.nn.Module | None = None,
):
    dataset_manager = DatasetManager(
        target_type=["segmentation"],
        mixed=False,
        use_augmentation=False,
    )
    dataloader_manager = DataloaderManager(
        dataset_manager=dataset_manager,
        batch_size=batch_size,
        workers=workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )
    _, _, test_dataloader = dataloader_manager.create_dataloaders(shuffle_train=False)

    results = {}

    # Get model to compare
    if self_training_dict is not None:
        checkpoints_dir, logs_dir = get_checkpoints_and_logs_dirs(
            run_name=self_training_dict["dataset_name"],
            model_name=self_training_dict["run_name"],
        )
        model = UNet().to(device)
        model.load_state_dict(
            torch.load(
                os.path.join(checkpoints_dir, f"{self_training_dict['round_name']}.pt"),
                map_location="cpu",
            )
        )

        print("\nEvaluating self-training model")
        ioi_self_training, f1_self_training = evaluate_segmentation_model(
            model=model,
            test_loader=test_dataloader,
            device=device,
            storage_path=f"{visualizations_folder}/{self_training_dict['run_name']}-{self_training_dict['run_name']}.png",
        )

        results["self_training"] = {
            "dataset_name": self_training_dict["dataset_name"],
            "run_name": self_training_dict["run_name"],
            "round_name": self_training_dict["round_name"],
            "ioi": ioi_self_training,
            "f1": f1_self_training,
        }

    if weakly_supervised_model is None:
        dataset = get_best_cam_dataset_file()
        checkpoints_dir, logs_dir = get_checkpoints_and_logs_dirs(
            run_name=semi_supervised_model_folder,
            model_name=dataset.split("/")[-1].split(".")[0],
        )
        weakly_supervised_model = UNet().to(device)
        weakly_supervised_model.load_state_dict(
            torch.load(
                os.path.join(checkpoints_dir, f"{baseline_model_name}.pt"),
                map_location="cpu",
            )
        )

    print("\nEvaluating weakly-supervised model")
    ioi_weakly_supervised_training, f1_weakly_supervised_training = (
        evaluate_segmentation_model(
            model=weakly_supervised_model,
            test_loader=test_dataloader,
            device=device,
            storage_path=f"{visualizations_folder}/semi-supervised-training.png",
        )
    )
    results["weakly_supervised"] = {
        "ioi": ioi_weakly_supervised_training,
        "f1": f1_weakly_supervised_training,
    }

    if baseline_model is None:
        checkpoints_dir, logs_dir = get_checkpoints_and_logs_dirs(
            run_name=baseline_model_folder,
            model_name=baseline_model_name,
        )
        baseline_model = UNet().to(device)
        baseline_model.load_state_dict(
            torch.load(
                os.path.join(checkpoints_dir, f"{baseline_model_name}.pt"),
                map_location="cpu",
            )
        )

    print("\nEvaluating baseline model")
    ioi_baseline, f1_baseline = evaluate_segmentation_model(
        model=baseline_model,
        test_loader=test_dataloader,
        device=device,
        storage_path=f"{visualizations_folder}/{baseline_model_folder}-{baseline_model_name}.png",
    )

    results[baseline_model_folder] = (
        {
            "ioi": ioi_baseline,
            "f1": f1_baseline,
        },
    )

    # Log in the logs folder (only the first parth of the path) the results of both models as json

    comparison_json = os.path.join(logs_dir.split("/")[0], "comparison_json.json")
    with open(comparison_json, "w") as f:
        json.dump(results, f, indent=4)
