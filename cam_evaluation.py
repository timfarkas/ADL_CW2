from pre_training import (
    Trainer,
    NUM_SPECIES,
    NUM_BREEDS,
    checkpoints_dir,
)
from models import (
    BboxHead,
    CAMManager,
    CNNBackbone,
    ClassifierHead,
    ResNetBackbone,
    TrainedModel,
)
import torch
import torch.nn as nn
import data
import matplotlib.pyplot as plt
import json
import os
from typing import List, Tuple, Optional

# TODO: Set the right list of checkpoints to generate
checkpoint_dicts = [
    # --- CNN-based models ---
    {"model_path": "cnn_bbox", "heads": [BboxHead(adapter="CNN")], "epoch": 5},
    {
        "model_path": "cnn_breed",
        "heads": [ClassifierHead(NUM_BREEDS, adapter="CNN")],
        "epoch": 20,
    },
    {
        "model_path": "cnn_breed_bbox",
        "heads": [ClassifierHead(NUM_BREEDS, adapter="CNN"), BboxHead(adapter="CNN")],
        "epoch": 20,
    },
    {
        "model_path": "cnn_breed_species",
        "heads": [
            ClassifierHead(NUM_BREEDS, adapter="CNN"),
            ClassifierHead(NUM_SPECIES, adapter="CNN"),
        ],
        "epoch": 20,
    },
    {
        "model_path": "cnn_species",
        "heads": [ClassifierHead(NUM_SPECIES, adapter="CNN")],
        "epoch": 10,
    },
    {
        "model_path": "cnn_species_bbox",
        "heads": [ClassifierHead(NUM_SPECIES, adapter="CNN"), BboxHead(adapter="CNN")],
        "epoch": 20,
    },
    {
        "model_path": "cnn_species_breed_bbox",
        "heads": [
            ClassifierHead(NUM_SPECIES, adapter="CNN"),
            ClassifierHead(NUM_BREEDS, adapter="CNN"),
            BboxHead(adapter="CNN"),
        ],
        "epoch": 15,
    },
    # --- ResNet-based models (sizes: 18, 50, 101) ---
    # res_breed_species
    {
        "model_path": "res_breed_species",
        "heads": [
            ClassifierHead(NUM_BREEDS, adapter="res18"),
            ClassifierHead(NUM_SPECIES, adapter="res18"),
        ],
        "epoch": 20,
        "size": "18",
    },
    {
        "model_path": "res_breed_species",
        "heads": [
            ClassifierHead(NUM_BREEDS, adapter="res50"),
            ClassifierHead(NUM_SPECIES, adapter="res50"),
        ],
        "epoch": 20,
        "size": "50",
    },
    {
        "model_path": "res_breed_species",
        "heads": [
            ClassifierHead(NUM_BREEDS, adapter="res101"),
            ClassifierHead(NUM_SPECIES, adapter="res101"),
        ],
        "epoch": 10,
        "size": "101",
    },
    # res_breed
    {
        "model_path": "res_breed",
        "heads": [ClassifierHead(NUM_BREEDS, adapter="res18")],
        "epoch": 15,
        "size": "18",
    },
    {
        "model_path": "res_breed",
        "heads": [ClassifierHead(NUM_BREEDS, adapter="res50")],
        "epoch": 5,
        "size": "50",
    },
    {
        "model_path": "res_breed",
        "heads": [ClassifierHead(NUM_BREEDS, adapter="res101")],
        "epoch": 10,
        "size": "101",
    },
    # res_breed_bbox
    {
        "model_path": "res_breed_bbox",
        "heads": [
            ClassifierHead(NUM_BREEDS, adapter="res18"),
            BboxHead(adapter="res18"),
        ],
        "epoch": 15,
        "size": "18",
    },
    {
        "model_path": "res_breed_bbox",
        "heads": [
            ClassifierHead(NUM_BREEDS, adapter="res50"),
            BboxHead(adapter="res50"),
        ],
        "epoch": 15,
        "size": "50",
    },
    {
        "model_path": "res_breed_bbox",
        "heads": [
            ClassifierHead(NUM_BREEDS, adapter="res101"),
            BboxHead(adapter="res101"),
        ],
        "epoch": 10,
        "size": "101",
    },
    # res_species
    {
        "model_path": "res_species",
        "heads": [ClassifierHead(NUM_SPECIES, adapter="res18")],
        "epoch": 10,
        "size": "18",
    },
    {
        "model_path": "res_species",
        "heads": [ClassifierHead(NUM_SPECIES, adapter="res50")],
        "epoch": 15,
        "size": "50",
    },
    {
        "model_path": "res_species",
        "heads": [ClassifierHead(NUM_SPECIES, adapter="res101")],
        "epoch": 20,
        "size": "101",
    },
    # res_species_bbox
    {
        "model_path": "res_species_bbox",
        "heads": [
            ClassifierHead(NUM_SPECIES, adapter="res18"),
            BboxHead(adapter="res18"),
        ],
        "epoch": 10,
        "size": "18",
    },
    {
        "model_path": "res_species_bbox",
        "heads": [
            ClassifierHead(NUM_SPECIES, adapter="res50"),
            BboxHead(adapter="res50"),
        ],
        "epoch": 10,
        "size": "50",
    },
    {
        "model_path": "res_species_bbox",
        "heads": [
            ClassifierHead(NUM_SPECIES, adapter="res101"),
            BboxHead(adapter="res101"),
        ],
        "epoch": 20,
        "size": "101",
    },
    # res_species_breed_bbox
    {
        "model_path": "res_species_breed_bbox",
        "heads": [
            ClassifierHead(NUM_SPECIES, adapter="res18"),
            ClassifierHead(NUM_BREEDS, adapter="res18"),
            BboxHead(adapter="res18"),
        ],
        "epoch": 10,
        "size": "18",
    },
    {
        "model_path": "res_species_breed_bbox",
        "heads": [
            ClassifierHead(NUM_SPECIES, adapter="res50"),
            ClassifierHead(NUM_BREEDS, adapter="res50"),
            BboxHead(adapter="res50"),
        ],
        "epoch": 5,
        "size": "50",
    },
    {
        "model_path": "res_species_breed_bbox",
        "heads": [
            ClassifierHead(NUM_SPECIES, adapter="res101"),
            ClassifierHead(NUM_BREEDS, adapter="res101"),
            BboxHead(adapter="res101"),
        ],
        "epoch": 5,
        "size": "101",
    },
    # res_bbox
    {
        "model_path": "res_bbox",
        "heads": [BboxHead(adapter="res18")],
        "epoch": 10,
        "size": "18",
    },
    {
        "model_path": "res_bbox",
        "heads": [BboxHead(adapter="res50")],
        "epoch": 5,
        "size": "50",
    },
    {
        "model_path": "res_bbox",
        "heads": [BboxHead(adapter="res101")],
        "epoch": 5,
        "size": "101",
    },
]


def getConvLayers(model: nn.Module) -> List[nn.Conv2d]:
    return [m for m in model.modules() if isinstance(m, nn.Conv2d)]


def findConvLayerByIndex(model: nn.Module, index: int = -1) -> Optional[nn.Conv2d]:
    """
    Find a convolutional layer by index in a model.

    Args:
        model (nn.Module): The model to search through
        index (int, optional): The index of the convolutional layer to return (default is -1, the last layer)

    Returns:
        Optional[nn.Conv2d]: The convolutional layer at the specified index, or None if not found
    """
    conv_layers = getConvLayers(model)
    if conv_layers:
        return conv_layers[index]
    return None


def visualize_cam_and_segment(
    img: Optional[torch.Tensor],
    thresholded_cam: torch.Tensor,
    segment: torch.Tensor,
    iou: Optional[float] = None,
    threshold: Optional[float] = None,
) -> None:
    """
    Visualize the image, thresholded CAM, and segmentation mask side by side.

    Args:
        img (Optional[torch.Tensor]): The input image tensor of shape (C, H, W) or None
        thresholded_cam (torch.Tensor): The thresholded CAM tensor of shape (H, W)
        segment (torch.Tensor): The segmentation mask tensor of shape (H, W)
        iou (Optional[float], optional): The IoU score to display. Defaults to None.
        threshold (Optional[float], optional): The threshold value used. Defaults to None.
    """
    assert thresholded_cam.shape == segment.shape, (
        "CAM and segment must have the same dimensions (H, W)"
    )
    assert len(thresholded_cam.shape) == 2 and len(segment.shape) == 2, (
        "Both CAM and segment must be 2D arrays"
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    if iou is not None:
        fig.suptitle(f"IoU: {round(iou * 100, 2)}% Threshold: " + str(threshold))

    if img is not None:
        # Visualize the image
        axes[0].imshow(img.permute(1, 2, 0).cpu().numpy())
        axes[0].set_title("Image")
        axes[0].axis("off")

    # Visualize the CAM
    im = axes[1].imshow(thresholded_cam, cmap="jet")
    axes[1].set_title("CAM")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], orientation="vertical", fraction=0.046, pad=0.04)

    # Visualize the segment
    segment_np = segment.cpu().numpy()
    axes[2].imshow(segment_np, cmap="grey")
    axes[2].set_title("Segment")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_single_raw_cam_and_mask(
    img: Optional[torch.Tensor], cam: torch.Tensor, segment: torch.Tensor
) -> None:
    """
    Visualize the image, raw CAM, and segmentation mask side by side.

    Args:
        img (Optional[torch.Tensor]): The input image tensor of shape (C, H, W) or None
        cam (torch.Tensor): The raw CAM tensor of shape (H, W)
        segment (torch.Tensor): The segmentation mask tensor of shape (H, W)
    """
    assert cam.shape == segment.shape, (
        "CAM and segment must have the same dimensions (H, W)"
    )
    assert len(cam.shape) == 2 and len(segment.shape) == 2, (
        "Both CAM and segment must be 2D arrays (H,W)"
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    if img is not None:
        # Visualize the image
        axes[0].imshow(img.permute(1, 2, 0).cpu().numpy())
        axes[0].set_title("Image")
        axes[0].axis("off")

    # Visualize the CAM
    im = axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("CAM")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], orientation="vertical", fraction=0.046, pad=0.04)

    # Visualize the segment
    segment_np = segment.cpu().numpy()
    axes[2].imshow(segment_np, cmap="grey")
    axes[2].set_title("Segment")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def thresholdCAM(cam: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Apply a threshold to a CAM tensor.

    Args:
        cam (torch.Tensor): The CAM tensor of shape (H, W)
        threshold (float): The threshold value to apply

    Returns:
        torch.Tensor: A binary tensor where values > threshold are 1.0, else 0.0
    """
    return (cam > threshold).float()


def computeIoU(
    cam: torch.Tensor, segment: torch.Tensor, threshold: float, visualize: bool = False
) -> float:
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
    cam = thresholdCAM(cam, threshold)

    unique_values = torch.unique(segment)
    assert len(cam.shape) == 3, (
        f"Expected cam to be of shape B, H, W (got {cam.shape})."
    )
    assert len(segment.shape) == 4, (
        f"Expected segment to be of shape B, 1, H, W (got {segment.shape})."
    )

    if len(unique_values) == 3:
        segment = segment == unique_values[:, None, None]  # (3, 256, 256)
        segment[:, 0] += segment[:, 2]
        segment = segment[:, 0]
    elif len(unique_values) == 1:
        segment = segment[:, 0]

    intersection = torch.logical_and(cam, segment).sum().item()
    union = torch.logical_or(cam, segment).sum().item()
    iou = intersection / union if union != 0 else 0.0

    if visualize:
        visualize_cam_and_segment(
            None, cam[0], segment.squeeze(1)[0], iou, threshold.item()
        )
    return iou


def find_optimal_threshold(
    camDataset: torch.utils.data.TensorDataset, num_samples: int = 30
) -> Tuple[float, float]:
    """
    Find the optimal threshold for the CAM (Class Activation Map) by evaluating the Intersection over Union (IoU)
    across a range of threshold values.

    Args:
        camDataset (torch.utils.data.TensorDataset): A dataset containing tuples of (image, CAM, segment).
        num_samples (int, optional): The number of samples to evaluate for each threshold. Defaults to 30.

    Returns:
        Tuple[float, float]: A tuple containing:
            - float: The optimal threshold value that maximizes the mean IoU.
            - float: The IoU score at the optimal threshold.
    """
    assert type(camDataset) is torch.utils.data.TensorDataset, (
        f"Unexpected type for camDataset {type(camDataset)}"
    )

    optimal_threshold = 0
    max_iou_mean = 0
    ious = None

    for threshold in torch.arange(0, 1, 0.05):
        ious = []
        generator = (data for data in camDataset)
        for i, batch in enumerate(generator):
            img, cam, segment = batch
            segment = segment.unsqueeze(0)  # (1,1,H,W)

            iou = computeIoU(cam, segment, threshold, visualize=False)
            ious.append(iou)
            if i + 1 == num_samples:
                break

        ious = torch.tensor(ious)
        iou_mean = ious.mean()

        if iou_mean > max_iou_mean:
            max_iou_mean = iou_mean
            optimal_threshold = threshold

    ious = list(ious)
    print(f"Found optimal threshold: {optimal_threshold} with IoU mean: {max_iou_mean}")
    optimal_threshold_item = (
        optimal_threshold.item()
        if isinstance(optimal_threshold, torch.Tensor)
        else optimal_threshold
    )
    max_iou_mean_item = (
        max_iou_mean.item() if isinstance(max_iou_mean, torch.Tensor) else max_iou_mean
    )

    return round(optimal_threshold_item, 2), round(max_iou_mean_item, 2)


def findOptimalCAMSettings(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    cam_type: str,
    num_samples: int = 50,
) -> List[Tuple[int, float, float]]:
    """
    Find optimal CAM settings for each convolutional layer in the model.

    Args:
        model (nn.Module): The model to analyze
        loader (torch.utils.data.DataLoader): DataLoader containing the dataset
        cam_type (str): The type of CAM to use (e.g., 'GradCAM', 'ScoreCAM')
        num_samples (int, optional): Number of samples to evaluate. Defaults to 50.

    Returns:
        List[Tuple[int, float, float]]: A list of tuples containing:
            - int: Layer index
            - float: Optimal threshold
            - float: IoU score at the optimal threshold
    """
    optimal = []
    # Classic CAM can only use one conv layer
    conv_layers = getConvLayers(model)
    layers = [conv_layers[0]] if cam_type == "ClassicCAM" else conv_layers
    for i, layer in enumerate(layers):
        threshold, iou = findLayerCAMThresholdAndIOU(
            model, layer, loader, cam_type, num_samples=num_samples
        )
        optimal.append((i, threshold, iou))
    return optimal


def findLayerCAMThresholdAndIOU(
    model: nn.Module,
    layer: nn.Conv2d,
    loader: torch.utils.data.DataLoader,
    cam_type: str,
    num_samples: int = 50,
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
        model, loader, target_type="species", target_layer=layer, method=cam_type
    )

    dataset = manager.get_cam_dataset(num_samples=num_samples)
    threshold, iou = find_optimal_threshold(dataset, num_samples=num_samples)
    del dataset
    del manager
    return threshold, iou


def _saveModelCAMSettingsToJson(
    model_name: str,
    settings_name: str,
    cam_settings: List[Tuple[int, float, float]],
    json_path: str = "logs/cam_stats.json",
) -> None:
    """
    Save the CAM settings for a model to a JSON file.

    Args:
        model_name (str): The name of the model
        cam_settings (List[Tuple[int, float, float]]): A list of tuples containing layer index, threshold, and IoU
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
        {"layer_index": layer_idx, "threshold": threshold, "iou": iou}
        for layer_idx, threshold, iou in cam_settings
    ]

    # Save the updated data
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"CAM settings for {model_name} saved to {json_path}")


if __name__ == "__main__":
    _, _, loader = data.create_dataloaders(
        target_type=["species", "segmentation"], batch_size=32
    )

    cam_types = ["ClassicCAM"]

    print("\n------------------------ Generating CAMS ---------------------\n")
    print(f"Iterating through {len(checkpoint_dicts)} checkpoints...")
    ### Load model checkpoint
    for checkpoint in checkpoint_dicts:
        if not checkpoint.get("model_path"):
            print("Checkpoint does not have a model path.")
            continue

        path = checkpoint["model_path"]
        path_parts = path.split("_")
        checkpoint_path = os.path.join(checkpoints_dir, path)

        trainer = Trainer()
        if path_parts[0] == "cnn":
            backbone = CNNBackbone()
            trainer.set_model(backbone, checkpoint["heads"], checkpoint_path)
        elif path_parts[0] == "res":
            size = checkpoint["size"]
            backbone = ResNetBackbone(model_type=f"resnet" + size)
            [head.change_adapter("res" + size) for head in checkpoint["heads"]]
            trainer.set_model(
                backbone, checkpoint["heads"], checkpoint_path + "_" + size
            )

        checkpoint_file_path = trainer.checkpoint_path(checkpoint["epoch"])
        trainer.load_checkpoint(checkpoint_file_path)

        ### enumerate through heads
        for head_index, head in enumerate(trainer.heads):
            model = TrainedModel(backbone=trainer.backbone, head=head)
            target_type = path_parts[head_index + 1]

            """
            According to a quick research, and also the results of the CAMS
            generated, the bbox head is not compatible with the CAMs methods
            from the library; there are separate CAM methods for bounding boxes.

            Do note this line is only here for the multi-head case, we should
            just not include bbox single head models in checkpoint_dicts.
            """
            if target_type == "bbox":
                del model
                continue

            ### enumerate through cam types
            for cam in cam_types:
                model_name = (
                    checkpoint["model_path"] + "_" + checkpoint["size"]
                    if "size" in checkpoint
                    else checkpoint["model_path"]
                )
                settings_name = f"{head.name}_{cam}"

                print(f"Generating {cam} for {path} head {target_type}")
                cam_settings = findOptimalCAMSettings(
                    model, loader, cam, num_samples=100
                )
                _saveModelCAMSettingsToJson(model_name, settings_name, cam_settings)

            del model
