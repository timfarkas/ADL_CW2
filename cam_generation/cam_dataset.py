import os
import torch

from cam_generation.cam_manager import CAMManager
from cam_generation.utils import find_conv_layer_by_index
from datasets.dataloader_manager import DataloaderManager
from datasets.dataset_manager import DatasetManager
from models.pretrainer import PretrainedModel
from models.utils import get_model_dict_by_name, get_pretrainer_by_config
from new_runs_config import get_checkpoints_and_logs_dirs, cam_dataset_folder


def generate_cam_dataset(
    runs_config: dict[str, dict],
    cam_dict: dict[str, any],
    device: torch.device,
    batch_size: int,
    workers: int,
    persistent_workers: bool,
):
    print(
        f"\Generating {cam_dict['cam_type']} dataset for model {cam_dict['model_name']} for run {cam_dict['run_name']} head {cam_dict['head_target']}"
    )
    run_config = runs_config[cam_dict["run_name"]]

    # Get dataloaders
    dataset_manager = DatasetManager(
        use_augmentation=run_config["use_augmentation"],
        mixed=run_config["use_mixed_data"],
        mixing_ratio=9999,  # Generate the dataset needed, but actual images are never background only
    )  # Please note this uses the default configuration
    dataloader_manager = DataloaderManager(
        dataset_manager=dataset_manager,
        batch_size=batch_size,
        workers=workers,
        persistent_workers=persistent_workers,
    )
    dataloaders = dataloader_manager.create_dataloaders(
        shuffle_train=False
    )  # No need to shuffle for this

    # Get the pretrainer from the configuration
    model_config = get_model_dict_by_name(
        model_name=cam_dict["model_name"],
        is_mixed_data=run_config["use_mixed_data"],
    )
    checkpoints_dir, logs_dir = get_checkpoints_and_logs_dirs(
        run_name=cam_dict["run_name"],
        model_name=cam_dict["model_name"],
    )
    pretrainer = get_pretrainer_by_config(
        model_config=model_config,
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
        logs_file="",
        device=device,
        learning_rate=None,
        weight_decay=None,
        dataloaders=dataloaders,
    )
    checkpoint_file_path = pretrainer.checkpoint_path(cam_dict["best_epoch"])
    pretrainer.load_checkpoint(checkpoint_file_path)

    # Create the model
    head_index = pretrainer.target_names.index(cam_dict["head_target"])
    model = PretrainedModel(
        backbone=pretrainer.backbone, head=pretrainer.heads[head_index]
    )
    model.to(device)
    target_layer = find_conv_layer_by_index(model, cam_dict["layer_index"])
    manager = CAMManager(
        model=model,
        dataloader=dataloaders[0],
        target_layer=target_layer,
        target_type=cam_dict["head_target"],
        method=cam_dict["cam_type"],
    )
    dataset = manager.get_cam_dataset()
    os.makedirs(cam_dataset_folder, exist_ok=True) 
    target_path = os.path.join(
        cam_dataset_folder,
        cam_dict["model_name"]
        + "_"
        + "head"
        + "_"
        + cam_dict["head_target"]
        + f"_idx{cam_dict['layer_index']}_{cam_dict['cam_type']}.pt",
    )
    torch.save(dataset, target_path)
