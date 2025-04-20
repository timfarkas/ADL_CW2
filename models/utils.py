import json
import os
import torch
from torch.utils.data import DataLoader

from models.backbones import CNNBackbone, ResNetBackbone
from models.heads import PretrainHead
from models.pretrainer import Pretrainer
from training.evaluations import compute_accuracy, convert_and_get_iou
from training.losses import cel_fn, mse_fn


def test_models(device: torch.device):
    print("\nTesting models...\n")
    cnn_backbone = CNNBackbone().to(device)
    res_backbone = ResNetBackbone().to(device)
    cnn_bbox_head = PretrainHead(adapter="cnn", output_type="bbox").to(device)
    res_bbox_head = PretrainHead(adapter="res18", output_type="bbox").to(device)
    res_class_head = PretrainHead(adapter="res18", output_type="breed").to(device)
    res_class_head2 = PretrainHead(adapter="res50", output_type="species").to(device)
    cnn_class_head = PretrainHead(adapter="cnn", output_type="is_animal").to(device)

    # Print the number of parameters in the CNN backbone
    cnn_params = sum(p.numel() for p in cnn_backbone.parameters())
    print(f"CNN Backbone parameters: {cnn_params:,}")

    # Print the number of parameters in the ResNet backbone
    res_params = sum(p.numel() for p in res_backbone.parameters())
    print(f"ResNet Backbone parameters: {res_params:,}")


def get_model_dict_by_name(model_name: str, is_mixed_data: bool):
    model_parts = model_name.split("_")

    # Get the backbone from the first item in the name
    backbone_name = model_parts[0]
    backbone = (
        CNNBackbone()
        if backbone_name == "cnn"
        else ResNetBackbone(pretrained=True, model_type=backbone_name)
    )

    # Get the heads from the rest of the items in the name
    heads = []
    loss_functions = []
    eval_functions = []
    eval_names = []
    loader_targets = model_parts[1:]
    for part in loader_targets:
        heads.append(PretrainHead(adapter=backbone_name, output_type=part))
        if part == "bbox":
            loss_functions.append(mse_fn)
            eval_functions.append(convert_and_get_iou)
            eval_names.append("IoU")
        else:
            loss_functions.append(cel_fn)
            eval_functions.append(compute_accuracy)
            eval_names.append("Acc")

    if is_mixed_data:
        # Add the mixed data head
        loader_targets.append("is_animal")
        heads.append(PretrainHead(adapter=backbone_name, output_type="is_animal"))
        loss_functions.append(cel_fn)
        eval_functions.append(compute_accuracy)
        eval_names.append("Acc")

    # Create the model dictionary
    return {
        "heads": heads,
        "backbone": backbone,
        "eval_functions": eval_functions,
        "eval_function_names": eval_names,
        "loss_functions": loss_functions,
        "loader_targets": loader_targets,
    }


def get_pretrainer_by_config(
    model_config: dict,
    checkpoints_dir: str,
    logs_dir: str,
    logs_file: str,
    device: torch.device,
    learning_rate: float | None,
    weight_decay: float | None,
    dataloaders: tuple[DataLoader] | None,
) -> Pretrainer:
    """
    Create a pretrainer object based on the model configuration.
    """

    pretrainer = Pretrainer(device=device, log_dir=logs_dir, log_file=logs_file)
    pretrainer.set_model(
        backbone=model_config["backbone"],
        heads=model_config["heads"],
        model_path=checkpoints_dir,
    )
    pretrainer.set_eval_functions(
        model_config["eval_functions"],
        model_config["eval_function_names"],
    )
    pretrainer.set_loss_functions(
        model_config["loss_functions"],
    )
    pretrainer.set_target_names(
        model_config["loader_targets"],
    )
    if learning_rate is not None and weight_decay is not None:
        pretrainer.set_optimizer(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

    if dataloaders is not None:
        train_dataloader, val_dataloader, _ = dataloaders
        pretrainer.set_loaders(
            train_loader=train_dataloader,
            val_loader=val_dataloader,
        )

    return pretrainer


def log_self_training_performance(
    log_dir: str, run_name: str, round_name: str, ioi: float, f1: float
):
    """
    Log performance metrics to a JSON file.
    """

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "self_training_log.json")

    # Create or load existing log
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            log_data = json.load(f)
    else:
        log_data = {}

    if run_name not in log_data:
        log_data[run_name] = {}

    if round_name not in log_data[run_name]:
        log_data[run_name][round_name] = {}

    log_data[run_name][round_name]["ioi"] = ioi
    log_data[run_name][round_name]["f1"] = f1

    # Save updated log
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=4)
