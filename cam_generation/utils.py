import json
import os
import torch

from torch import nn

from training.evaluations import compute_iou


def get_conv_layers(model: nn.Module) -> list[nn.Conv2d]:
    return [m for m in model.modules() if isinstance(m, nn.Conv2d)]


def find_conv_layer_by_index(model: nn.Module, index: int = -1) -> nn.Conv2d | None:
    conv_layers = get_conv_layers(model)
    if conv_layers:
        return conv_layers[index]
    return None


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
        {"layer_index": layer_idx, "iou": iou}
        for layer_idx, iou in cam_settings
    ]

    # Save the updated data
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"CAM settings for {model_name} saved to {json_path}")
