import json
import os
from matplotlib import pyplot as plt
import torch

from torch import nn

from training.evaluations import compute_iou
from new_runs_config import cam_evaluation_json, get_checkpoints_and_logs_dirs
from training.pretraining import get_best_epoch_per_model


def get_conv_layers(model: nn.Module) -> list[nn.Conv2d]:
    return [m for m in model.modules() if isinstance(m, nn.Conv2d)]


def find_conv_layer_by_index(model: nn.Module, index: int = -1) -> nn.Conv2d | None:
    conv_layers = get_conv_layers(model)
    if conv_layers:
        return conv_layers[index]
    return None


def get_best_cam(runs_config: dict[str, any]):
    """
    Get the best CAM settings for each model in the given runs and stores them in a JSON file.
    Args:
        runs_config (dict): A dictionary containing the configuration for each run.
    Returns:
        best_overall (dict): A dictionary containing the best CAM settings across all runs.
    """
    results = {}
    best_overall = {}
    for run_name, _ in runs_config.items():
        _, logs_dir = get_checkpoints_and_logs_dirs(
            run_name=run_name,
            model_name="",
        )
        best_epoch_dict = get_best_epoch_per_model(
            run_name=run_name,
        )

        cam_json_path = os.path.join(logs_dir, cam_evaluation_json)
        if os.path.exists(cam_json_path):
            with open(cam_json_path, "r") as f:
                cam_data = json.load(f)

            run_best_results = []
            for model_name, model_data in cam_data.items():
                for settings_name, settings in model_data.items():
                    best_iou = max(
                        settings,
                        key=lambda x: x["iou"],
                    )
                    run_best_results.append(
                        {
                            "model_name": model_name,
                            "cam_type": settings_name.split("_")[0],
                            "head_target": settings_name.split("_")[1],
                            "layer_index": best_iou["layer_index"],
                            "iou": best_iou["iou"],
                            "best_epoch": best_epoch_dict.get(model_name, 0),
                        }
                    )
            run_best_results.sort(key=lambda x: x["iou"], reverse=True)
            results[run_name] = run_best_results
            if run_best_results[0]["iou"] > best_overall.get("iou", 0):
                best_overall = {
                    "run_name": run_name,
                    "model_name": run_best_results[0]["model_name"],
                    "cam_type": run_best_results[0]["cam_type"],
                    "head_target": run_best_results[0]["head_target"],
                    "layer_index": run_best_results[0]["layer_index"],
                    "iou": run_best_results[0]["iou"],
                    "best_epoch": best_epoch_dict.get(model_name, 0),
                }
    # Save the best overall settings to a JSON file
    best_cam_settings_per_run = os.path.join(
        logs_dir.split("/")[0], "best_cam_settings_per_run.json"
    )
    with open(best_cam_settings_per_run, "w") as f:
        json.dump(results, f, indent=4)

    print(
        f"Best CAM settings overall: run: {best_overall['run_name']}, model: {best_overall['model_name']}, cam type: {best_overall['cam_type']}, target: {best_overall['head_target']}, layer: {best_overall['layer_index']}, iou: {best_overall['iou']}"
    )
    return best_overall


def compute_cam_iou(cam: torch.Tensor, segment: torch.Tensor) -> float:
    """
    Compute the Intersection over Union (IoU) between a thresholded CAM and a segmentation mask.

    Args:
        cam (torch.Tensor): The CAM tensor of shape (B, H, W)
        segment (torch.Tensor): The segmentation mask tensor of shape (B, 1, H, W)
        threshold (float): The threshold value to apply to the CAM
        visualize (bool, optional): Whether to visualize the results. Defaults to False.

    Returns:
        float: The IoU score
    """
    binary_cam = (cam > 0).float()
    unique_values = torch.unique(segment)

    assert len(binary_cam.shape) == 3, (
        f"Expected cam to be of shape B, H, W (got {binary_cam.shape})."
    )
    assert len(segment.shape) == 4, (
        f"Expected segment to be of shape B, 1, H, W (got {segment.shape})."
    )
    # Assert that the height and width of cam and segment are identical
    assert binary_cam.shape[1:] == segment.shape[2:], (
        f"Height and width of cam {binary_cam.shape[1:]} and segment {segment.shape[2:]} must be identical"
    )

    if len(unique_values) == 3:
        segment = segment == unique_values[:, None, None]  # (3, 256, 256)
        segment[:, 0] += segment[:, 2]
        segment = segment[:, 0]
    elif len(unique_values) == 1:
        segment = segment[:, 0]
    else:
        raise ValueError(
            f"Unexpected segmentation mask value count {len(unique_values)}"
        )

    return compute_iou(binary_cam, segment)


def save_model_cam_settings_to_json(
    model_name: str,
    settings_name: str,
    cam_settings: list[tuple[int, float]],
    json_path: str = "logs/cam_stats.json",
) -> None:
    """
    Save the CAM settings for a model to a JSON file.

    Args:
        model_name (str): The name of the model
        cam_settings (List[Tuple[int, float]]): A list of tuples containing layer index and IoU
        json_path (str, optional): Path to the JSON file. Defaults to "logs/cam_stats.json".
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # Load existing data if file exists
    data = {}
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)

    # Add or update the model's CAM settings
    if model_name not in data:
        data[model_name] = {}

    data[model_name][settings_name] = [
        {"layer_index": layer_idx, "iou": iou} for layer_idx, iou in cam_settings
    ]

    # Save the updated data
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"CAM settings for {model_name} saved to {json_path}")


def visualize_cam_samples(dataloader, num_samples=4, storage_path: str | None = None):
    """
    Visualize image, CAM, and GT mask for a few samples from a DataLoader.

    Args:
        dataloader (DataLoader): DataLoader yielding (image, CAM, GT mask) batches
        num_samples (int): Number of samples to visualize
    """
    # Get one batch
    batch = next(iter(dataloader))
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        images, cams, masks = batch
    else:
        raise ValueError("Expected DataLoader to return (image, cam, mask) tuples.")

    # Slice to desired number of samples
    images = images[:num_samples].cpu()
    cams = cams[:num_samples].cpu()
    masks = masks[:num_samples].cpu()

    # Unnormalize images for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    unnormalized_images = images * std + mean
    unnormalized_images = unnormalized_images.clamp(0, 1)

    # Setup plot
    fig, axs = plt.subplots(3, num_samples, figsize=(num_samples * 3, 9))

    for i in range(num_samples):
        # Image
        img = unnormalized_images[i].permute(1, 2, 0).cpu().numpy()
        axs[0, i].imshow(img)
        axs[0, i].set_title(f"Image {i + 1}")
        axs[0, i].axis("off")

        # CAM (assumes shape (1, H, W))
        cam = cams[i].squeeze().cpu().numpy()
        axs[1, i].imshow(cam, cmap="jet")
        axs[1, i].set_title(f"CAM {i + 1}")
        axs[1, i].axis("off")

        # Ground Truth Mask (assumes shape (1, H, W))
        mask = masks[i].squeeze().cpu().numpy()
        axs[2, i].imshow(mask, cmap="gray")
        axs[2, i].set_title(f"GT Mask {i + 1}")
        axs[2, i].axis("off")

    plt.tight_layout()
    if storage_path:
        dir_name = os.path.dirname(storage_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(storage_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
