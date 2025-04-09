from pre_training import Trainer
from models import CAMManager, CNNBackbone, ClassifierHead
import torch
import torch.nn as nn
import data
import matplotlib.pyplot as plt

def getConvLayers(model):
    """
    Get a list of all convolutional layers in a model.
    
    Args:
        model: The model to search through
        
    Returns:
        A list of all convolutional layers found
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

def findConvLayerByIndex(model, index=-1):
    """
    Find a convolutional layer by index in a model.
    
    Args:
        model: The model to search through
        index: The index of the convolutional layer to return (default is -1, the last layer)
        
    Returns:
        The convolutional layer at the specified index, or None if not found
    """
    conv_layers = getConvLayers(model)
    if conv_layers:
        return conv_layers[index]
    return None

def visualize_cam_and_segment(img, thresholded_cam, segment, iou = None, threshold = None):
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

def visualize_single_raw_cam_and_mask(img, cam, segment):
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
    Takes in cam tensor (H, W) and a threshold value, returns tensor with values thresholded.
    """
    return (cam > threshold).float()

def computeIoU(cam : torch.Tensor, segment : torch.Tensor, threshold : float, visualize : bool = False) -> float:
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

def find_optimal_threshold(camDataset: torch.utils.data.TensorDataset, num_samples: int = 30) -> float:
    """
    Find the optimal threshold for the CAM (Class Activation Map) by evaluating the Intersection over Union (IoU)
    across a range of threshold values.

    Args:
        camDataset (torch.utils.data.TensorDataset): A dataset containing tuples of (image, CAM, segment).
        num_samples (int, optional): The number of samples to evaluate for each threshold. Defaults to 30.

    Returns:
        float: The optimal threshold value that maximizes the mean IoU.
    """
    assert type(camDataset) == torch.utils.data.TensorDataset, f"Unexpected type for camDataset {type(camDataset)}"
    generator = (data for data in camDataset)

    optimal_threshold = 0
    max_iou_mean = 0
    ious = None
    
    for threshold in torch.arange(0, 1, 0.05):
        print(f"Threshold: {threshold}")
        ious = []
        
        for i, batch in enumerate(generator):
            img, cam, segment = batch
            segment = segment.unsqueeze(0) # (1,1,H,W)

            iou = computeIoU(cam, segment, threshold, visualize = False)
            ious.append(iou)
            if i+1 == num_samples:
                break
        
        ious = torch.tensor(ious)
        iou_mean = ious.mean()
        print(f"Threshold {threshold}, iou {iou_mean}")
        if iou_mean > max_iou_mean:
            max_iou_mean = iou_mean
            optimal_threshold = threshold

    ious = list(ious)
    print(f"Found optimal threshold: {optimal_threshold} with IoU mean: {max_iou_mean}")
    return threshold, max_iou_mean


if __name__ == "__main__":
    trainer = Trainer()
    backbone = CNNBackbone()
    head = ClassifierHead()
    trainer.set_model(backbone, [head], "checkpoints/cnn_species_checkpoint_epoch10.pt")
    trainer.load_checkpoint("checkpoints/cnn_species_checkpoint_epoch10.pt")

    _, _ , loader  = data.create_dataloaders(target_type=["species", "segmentation"], batch_size=32)

    model = nn.Sequential(backbone, head)

    layer = findConvLayerByIndex(model, -1)
    
    cam_type = "GradCAM" # Literal['GradCAM', 'ScoreCAM', 'AblationCAM']
    
    manager = CAMManager(model, loader, target_type="species", target_layer=layer, method=cam_type)
    
    dataset = manager.get_cam_dataset()

    threshold, iou = find_optimal_threshold(dataset)
