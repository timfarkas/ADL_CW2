from pre_training import Trainer
from models import CAMManager, CNNBackbone, ClassifierHead
import torch
import torch.nn as nn
import data
import matplotlib.pyplot as plt
import json
import os
from typing import List, Tuple, Optional, Union, Any

def getConvLayers(model: nn.Module) -> List[nn.Conv2d]:
    """
    Get a list of all convolutional layers in a model.
    
    Args:
        model (nn.Module): The model to search through
        
    Returns:
        List[nn.Conv2d]: A list of all convolutional layers found
    """
    conv_layers = []
    
    # Check if model is Sequential
    if isinstance(model, nn.Sequential):
        # Check each module in the sequential container
        for module in model:
            conv_layers.extend(getConvLayers(module))
    
    # Check if model has features attribute (like our backbone)
    elif hasattr(model, 'features') and isinstance(model.features, nn.Sequential):
        for module in model.features:
            if isinstance(module, nn.Conv2d):
                conv_layers.append(module)
    
    # Check if the model itself is a Conv2d
    elif isinstance(model, nn.Conv2d):
        conv_layers.append(model)
    
    return conv_layers

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

def visualize_cam_and_segment(img: Optional[torch.Tensor], thresholded_cam: torch.Tensor, 
                             segment: torch.Tensor, iou: Optional[float] = None, 
                             threshold: Optional[float] = None) -> None:
    """
    Visualize the image, thresholded CAM, and segmentation mask side by side.
    
    Args:
        img (Optional[torch.Tensor]): The input image tensor of shape (C, H, W) or None
        thresholded_cam (torch.Tensor): The thresholded CAM tensor of shape (H, W)
        segment (torch.Tensor): The segmentation mask tensor of shape (H, W)
        iou (Optional[float], optional): The IoU score to display. Defaults to None.
        threshold (Optional[float], optional): The threshold value used. Defaults to None.
    """
    assert thresholded_cam.shape == segment.shape, "CAM and segment must have the same dimensions (H, W)"
    assert len(thresholded_cam.shape) == 2 and len(segment.shape) == 2, "Both CAM and segment must be 2D arrays"
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    if iou is not None:
        fig.suptitle(f"IoU: {round(iou*100,2)}% Threshold: "+str(threshold))

    if img is not None:
        # Visualize the image
        axes[0].imshow(img.permute(1, 2, 0).cpu().numpy())
        axes[0].set_title("Image")
        axes[0].axis('off')

    # Visualize the CAM
    im = axes[1].imshow(thresholded_cam, cmap='jet')
    axes[1].set_title("CAM")
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)

    # Visualize the segment
    segment_np = segment.cpu().numpy()
    axes[2].imshow(segment_np, cmap='grey')
    axes[2].set_title("Segment")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

def visualize_single_raw_cam_and_mask(img: Optional[torch.Tensor], cam: torch.Tensor, 
                                     segment: torch.Tensor) -> None:
    """
    Visualize the image, raw CAM, and segmentation mask side by side.
    
    Args:
        img (Optional[torch.Tensor]): The input image tensor of shape (C, H, W) or None
        cam (torch.Tensor): The raw CAM tensor of shape (H, W)
        segment (torch.Tensor): The segmentation mask tensor of shape (H, W)
    """
    assert cam.shape == segment.shape, "CAM and segment must have the same dimensions (H, W)"
    assert len(cam.shape) == 2 and len(segment.shape) == 2, "Both CAM and segment must be 2D arrays (H,W)"
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    if img is not None:
        # Visualize the image
        axes[0].imshow(img.permute(1, 2, 0).cpu().numpy())
        axes[0].set_title("Image")
        axes[0].axis('off')

    # Visualize the CAM
    im = axes[1].imshow(cam, cmap='jet')
    axes[1].set_title("CAM")
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)

    # Visualize the segment
    segment_np = segment.cpu().numpy()
    axes[2].imshow(segment_np, cmap='grey')
    axes[2].set_title("Segment")
    axes[2].axis('off')

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

def computeIoU(cam: torch.Tensor, segment: torch.Tensor, threshold: float, 
              visualize: bool = False) -> float:
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
    assert len(cam.shape) == 3, f"Expected cam to be of shape B, H, W (got {cam.shape})."
    assert len(segment.shape) == 4, f"Expected segment to be of shape B, 1, H, W (got {segment.shape})."

    if len(unique_values) == 3:
        segment = (segment == unique_values[:, None, None])  # (3, 256, 256)
        segment[:, 0] += segment[:, 2]
        segment = segment[:, 0]
    elif len(unique_values)  == 1:
        segment = segment[:, 0] 

    intersection = torch.logical_and(cam, segment).sum().item()
    union = torch.logical_or(cam, segment).sum().item()
    iou = intersection / union if union != 0 else 0.0
    
    if visualize:
        visualize_cam_and_segment(None, cam[0], segment.squeeze(1)[0], iou, threshold.item())
    return iou

def find_optimal_threshold(camDataset: torch.utils.data.TensorDataset, 
                          num_samples: int = 30) -> Tuple[float, float]:
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
    assert type(camDataset) == torch.utils.data.TensorDataset, f"Unexpected type for camDataset {type(camDataset)}"
    

    optimal_threshold = 0
    max_iou_mean = 0
    ious = None
    
    for threshold in torch.arange(0, 1, 0.05):
        ious = []
        generator = (data for data in camDataset)
        for i, batch in enumerate(generator):
            img, cam, segment = batch
            segment = segment.unsqueeze(0) # (1,1,H,W)

            iou = computeIoU(cam, segment, threshold, visualize = False)
            ious.append(iou)
            if i+1 == num_samples:
                break
        
        ious = torch.tensor(ious)
        iou_mean = ious.mean()
        
        if iou_mean > max_iou_mean:
            max_iou_mean = iou_mean
            optimal_threshold = threshold

    ious = list(ious)
    print(f"Found optimal threshold: {optimal_threshold} with IoU mean: {max_iou_mean}")
    return round(optimal_threshold.item(), 2), round(max_iou_mean.item(),2)

def findOptimalCAMSettings(model: nn.Module, loader: torch.utils.data.DataLoader, 
                          cam_type: str, num_samples: int = 50) -> List[Tuple[int, float, float]]:
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
    for i, layer in enumerate(getConvLayers(model)):
        threshold, iou = findLayerCAMThresholdAndIOU(model, layer, loader, cam_type, num_samples=num_samples)
        optimal.append((i, threshold, iou))
    return optimal

def findLayerCAMThresholdAndIOU(model: nn.Module, layer: nn.Conv2d, 
                               loader: torch.utils.data.DataLoader, 
                               cam_type: str, num_samples: int = 50) -> Tuple[float, float]:
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
    manager = CAMManager(model, loader, target_type="species", target_layer=layer, method=cam_type)

    dataset = manager.get_cam_dataset(num_samples=num_samples)
    threshold, iou = find_optimal_threshold(dataset, num_samples=num_samples)
    return threshold, iou

def _saveModelCAMSettingsToJson(model_name: str, cam_settings: List[Tuple[int, float, float]], 
                               json_path: str = "logs/cam_stats.json") -> None:
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
        with open(json_path, 'r') as f:
            data = json.load(f)
    
    # Add or update the model's CAM settings
    data[model_name] = [
        {
            "layer_index": layer_idx,
            "threshold": threshold,
            "iou": iou
        } for layer_idx, threshold, iou in cam_settings
    ]
    
    # Save the updated data
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"CAM settings for {model_name} saved to {json_path}")


if __name__ == "__main__":
    trainer = Trainer()
    backbone = CNNBackbone()
    head = ClassifierHead()
    trainer.set_model(backbone, [head], "checkpoints/cnn_species_checkpoint_epoch10.pt")
    trainer.load_checkpoint("checkpoints/cnn_species_checkpoint_epoch10.pt")

    _, _ , loader  = data.create_dataloaders(target_type=["species", "segmentation"], batch_size=32)
    cam_type = "GradCAM" # Literal['GradCAM', 'ScoreCAM', 'AblationCAM']
    
    model = nn.Sequential(backbone, head)

    cam_settings = findOptimalCAMSettings(model, loader, cam_type, num_samples = 100)
      
    model_name = "cnn_species"
    
    _saveModelCAMSettingsToJson(model_name, cam_settings)