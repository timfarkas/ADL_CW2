from typing import List, Tuple
import torch
from torch import nn

from cam_generation.cam_manager import CAMManager
from cam_generation.utils import (
    compute_cam_iou,
    get_conv_layers,
    save_model_cam_settings_to_json,
)
from datasets.dataloader_manager import DataloaderManager
from datasets.dataset_manager import DatasetManager
from models.pretrainer import PretrainedModel
from models.utils import get_model_dict_by_name, get_pretrainer_by_config
from new_runs_config import get_checkpoints_and_logs_dirs
from training.pretraining import get_best_epoch_per_model

cam_evaluation_json = "cam_evaluation.json"


def evaluate_cams(
    runs_config: dict,
    device: torch.device,
    batch_size: int,
    workers: int,
    persistent_workers: bool,
    cam_types: list,
    num_samples: int = 50,
):
    for run_name, run_config in runs_config.items():
        print(f"\nEvaluating CAMs for run {run_name}")
        models_dict = get_best_epoch_per_model(
            run_name=run_name,
        )
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
        for model_name, best_epoch in models_dict.items():
            print(f"\nEvaluating {model_name} for run {run_name} at epoch {best_epoch}")

            dataloaders = dataloader_manager.create_dataloaders(
                shuffle_train=False
            )  # No need to shuffle for this
            checkpoints_dir, logs_dir = get_checkpoints_and_logs_dirs(
                run_name=run_name,
                model_name=model_name,
            )
            model_config = get_model_dict_by_name(
                model_name=model_name,
                is_mixed_data=run_config["use_mixed_data"],
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
            checkpoint_file_path = pretrainer.checkpoint_path(best_epoch)
            pretrainer.load_checkpoint(checkpoint_file_path)

            # Iterate through the potential heads
            for head_index, head in enumerate(pretrainer.heads):
                if head.is_bbox:
                    # Skip the bbox head
                    continue

                model = PretrainedModel(backbone=pretrainer.backbone, head=head)
                model.to(device)
                target_type = model_config["loader_targets"][head_index]
                for cam_type in cam_types:
                    print(
                        f"Evaluating {cam_type} for {model_name} head {head_index} ({head.name})"
                    )
                    print(f"Target type: {target_type}")
                    iou_per_layer = find_ioi_per_layer(
                        model=model,
                        loader=dataloaders[1],
                        target_type=target_type,
                        cam_type=cam_type,
                        num_samples=num_samples,
                    )
                    save_model_cam_settings_to_json(
                        model_name=model_name,
                        settings_name=f"{cam_type}_{target_type}",
                        cam_settings=iou_per_layer,
                        json_path=f"{logs_dir}/{cam_evaluation_json}",
                    )


def find_ioi_per_layer(
    model: PretrainedModel,
    loader: torch.utils.data.DataLoader,
    target_type: str,
    cam_type: str,
    num_samples: int = 50,
) -> List[Tuple[int, float, float]]:
    """
    Find optimal CAM IoU for each convolutional layer in the model.

    Args:
        model (nn.Module): The model to analyze
        loader (torch.utils.data.DataLoader): DataLoader containing the dataset
        cam_type (str): The type of CAM to use (e.g., 'GradCAM', 'ScoreCAM')
        num_samples (int, optional): Number of samples to evaluate. Defaults to 50.

    Returns:
        List[Tuple[int, float, float]]: A list of tuples containing:
            - int: Layer index
            - float: IoU score
    """
    layers_iou = []
    # Classic CAM can only use one conv layer
    conv_layers = get_conv_layers(model)
    layers = [conv_layers[-1]] if cam_type == "ClassicCAM" else conv_layers
    for i, layer in enumerate(layers):
        iou = find_layer_cam_iou(
            model=model,
            layer=layer,
            loader=loader,
            target_type=target_type,
            cam_type=cam_type,
            num_samples=num_samples,
        )
        layers_iou.append((i, iou))
    return layers_iou


def find_layer_cam_iou(
    model: nn.Module,
    layer: nn.Conv2d,
    loader: torch.utils.data.DataLoader,
    target_type: str,
    cam_type: str,
    num_samples: int,
) -> Tuple[float, float]:
    """
    Find the optimal threshold and IoU for a specific layer using a specific CAM method.

    Args:
        model (nn.Module): The model to analyze
        layer (nn.Conv2d): The specific convolutional layer to target
        loader (torch.utils.data.DataLoader): DataLoader containing the dataset
        cam_type (str): The type of CAM to use (e.g., 'GradCAM', 'ScoreCAM')
        num_samples (int, optional): Number of samples to evaluate. Defaults to 50.

    Returns:
        Tuple[float, float]: A tuple containing:
            - float: The optimal threshold value
            - float: The IoU score at the optimal threshold
    """
    manager = CAMManager(
        model=model,
        dataloader=loader,
        target_type=target_type,
        target_layer=layer,
        method=cam_type,
    )
    dataset = manager.get_cam_dataset(num_samples=num_samples)

    iou_list = []
    generator = (data for data in dataset)
    for i, batch in enumerate(generator):
        img, cam, segment = batch

        segment = segment.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        iou = compute_cam_iou(cam, segment)
        iou_list.append(iou)
        if i + 1 == num_samples:
            break

    del dataset
    del manager
    return sum(iou_list) / len(iou_list)
