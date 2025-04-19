import json
import os
import torch

from datasets.dataloader_manager import DataloaderManager
from datasets.dataset_manager import DatasetManager
from models.utils import get_model_dict_by_name, get_pretrainer_by_config
from new_runs_config import get_checkpoints_and_logs_dirs


logs_file = "pretraining.json"


def run_pretraining_process(
    runs_config: dict,
    model_names: list[str],
    device: torch.device,
    batch_size: int,
    workers: int,
    persistent_workers: bool,
    learning_rate: float,
    weight_decay: float,
    num_epochs: int,
):
    for run_name, run_config in runs_config.items():
        dataset_manager = DatasetManager(
            use_augmentation=run_config["use_augmentation"],
            mixed=run_config["use_mixed_data"],
        )  # Please note this uses the default configuration
        dataloader_manager = DataloaderManager(
            dataset_manager=dataset_manager,
            batch_size=batch_size,
            workers=workers,
            persistent_workers=persistent_workers,
        )
        for model_name in model_names:
            model_config = get_model_dict_by_name(
                model_name=model_name,
                is_mixed_data=run_config["use_mixed_data"],
            )
            checkpoints_dir, logs_dir = get_checkpoints_and_logs_dirs(
                run_name=run_name,
                model_name=model_name,
            )
            pretrainer = get_pretrainer_by_config(
                model_config=model_config,
                checkpoints_dir=checkpoints_dir,
                logs_dir=logs_dir,
                logs_file=logs_file,
                device=device,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                dataloaders=dataloader_manager.create_dataloaders(
                    pet_only=run_config["use_mixed_data"]
                ),
            )
            pretrainer.fit_model(
                num_epochs=num_epochs,
            )


def get_best_epoch_per_model(
    run_name: str,
) -> dict[str, int]:
    """
    Get the best epoch for each model in a run.
    """
    _, logs_dir = get_checkpoints_and_logs_dirs(
        run_name=run_name,
        model_name="",
    )

    with open(os.path.join(logs_dir, logs_file), "r") as f:
        results_json = json.load(f)

    # Extract best epoch based on val_loss
    best_epochs = {}

    for model_name, epochs in results_json.items():
        best_epoch = None
        best_val_loss = float("inf")

        for epoch_str, metrics in epochs.items():
            val_loss = metrics.get("val_loss", float("inf"))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = int(epoch_str)

        best_epochs[model_name] = best_epoch

    return best_epochs
