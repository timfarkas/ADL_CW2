import torch

from datasets.dataloader_manager import DataloaderManager
from datasets.dataset_manager import DatasetManager
from models.utils import get_model_dict_by_name, get_pretrainer_by_config


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
            logs_dir = f"logs/{run_name}"
            checkpoint_dir = f"checkpoints/{run_name}/{model_name}"
            pretrainer = get_pretrainer_by_config(
                model_config=model_config,
                checkpoints_dir=checkpoint_dir,
                logs_dir=logs_dir,
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
