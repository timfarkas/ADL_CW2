import torch
import torchvision
import cv2
import numpy as np

def store_images(
    images: torch.Tensor, file_name: str, is_segmentation: bool = False
) -> None:
    """
    Stores a list of images
    """

    if is_segmentation:
        images_tensor = (torch.cat(images.split(1, 0), 3).squeeze(0)) / 2
    else:
        images_tensor = torch.cat(images.split(1, 0), 3).squeeze()
        images_tensor = ((images_tensor)) * 255 # Review if we change data ingestion
        images_tensor = images_tensor.clamp(0, 255).byte()
    
    image = torchvision.transforms.ToPILImage()(images_tensor)
    image.save(file_name)

def resize_images(
        img_tensor, 
        H : int,
        W : int
) -> torch.Tensor:
    """
    Resize images to the specified dimensions, supporting arbitrary batch dimensions.
    
    Args:
        img_tensor: Input tensor or numpy array of images
        H: Target height
        W: Target width
        
    Returns:
        Resized tensor with the same batch dimensions as input
    """
    is_tensor = isinstance(img_tensor, torch.Tensor)
    
    if is_tensor:
        original_device = img_tensor.device
        img_np = img_tensor.detach().cpu().numpy()
    else:
        img_np = img_tensor
    
    # Get original shape to determine if we have a batch dimension
    original_shape = img_np.shape
    
    # Handle different input dimensions
    if len(original_shape) == 2:  # Single grayscale image
        img_resized = cv2.resize(img_np, (W, H))
    elif len(original_shape) == 3:
        if original_shape[2] == 3 or original_shape[2] == 4:  # Single RGB/RGBA image (H,W,C)
            img_resized = cv2.resize(img_np, (W, H))
        else:  # Single channel image with batch dimension (B,H,W)
            img_resized = np.stack([cv2.resize(img, (W, H)) for img in img_np])
    elif len(original_shape) == 4:  # Batch of images (B,H,W,C) or (B,C,H,W)
        if original_shape[1] == 3 or original_shape[1] == 4:  # (B,C,H,W)
            # Transpose to (B,H,W,C) for cv2, resize, then transpose back
            img_np = np.transpose(img_np, (0, 2, 3, 1))
            img_resized = np.stack([cv2.resize(img, (W, H)) for img in img_np])
            img_resized = np.transpose(img_resized, (0, 3, 1, 2))
        else:  # (B,H,W,C)
            img_resized = np.stack([cv2.resize(img, (W, H)) for img in img_np])
    else:
        raise ValueError(f"Unsupported tensor shape: {original_shape}")
    
    # Convert back to tensor if input was tensor
    if is_tensor:
        img_resized = torch.tensor(img_resized, device=original_device)
    
    return img_resized


#### EVAL FUNCTIONS FOR PRE-TRAINING TRAINER
def compute_accuracy(outputs, targets) -> float:
    """
    Calculate classification accuracy.
    
    Args:
        outputs: Model predictions (logits)
        targets: Ground truth labels
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    _, predicted = torch.max(outputs.data, 1)
    total = targets.size(0)
    correct = (predicted == targets).sum().item()
    return correct / total

def convertVOCBBoxFormatToAnchorFormat(boxes):
    """
    Takes in a tensor of bounding boxes in VOC format, of the form [x_min, y_min, x_max, y_max], or a single box.
    Converts it to a tensor of bounding boxes in anchor format, of the form [cx, cy, w, h], or a single box.
    """
    assert type(boxes) == torch.Tensor, f"Expected tensor, received {type(boxes)}"
    unsqueezed = False
    if len(boxes.shape) == 1:
        unsqueezed = True
        boxes.unsqueeze(0) ## if getting single box, unsqueeze first dim 
    assert len(boxes.shape) == 2, f"Expected two dimensions, received {len(boxes.shape)}"
    B, V = boxes.shape
    assert V == 4, f"Expected 2nd dim to have size 4, received {len(boxes.shape[1])}" 
    new_tensor = torch.Tensor(B,V)
    
    new_tensor[:, 0] = (boxes[:, 2] + boxes[:, 0]) / 2 # cx = (xmax + xmin) / 2
    new_tensor[:, 1] = (boxes[:, 3] + boxes[:, 1]) / 2 # cy = (ymax + ymin) / 2
    new_tensor[:, 2] = boxes[:, 2] - boxes[:, 0] # w = xmax - xmin
    new_tensor[:, 3] = boxes[:, 3] - boxes[:, 1] # h = ymax - ymin

    if unsqueezed: # revert to original state
        new_tensor.squeeze(0) 

    return new_tensor

def computeBBoxIoU(outputs, targets) -> float:
    """
    Calculate average bbox intersection over union.
    Assumes outputs and targets to be in anchor-box format ([cx, cy, w, h]).

     Args:
        outputs: Model predictions (logits)
        targets: Ground truth labels
        
    Returns:
        IoU of the two.
    """
    ### First, compute areas
    outputs_area = torch.mul(outputs[:,2], outputs[:,3]).sum() # w * h
    targets_area = torch.mul(targets[:,2], targets[:,3]).sum() # w * h
    
    ### Now, compute intersection rectangles
    #### x_max = cx + w/2 
    min_x_max = torch.min(outputs[:,0] + outputs[:,2]/2, targets[:,0] + targets[:,2]/2) 
    #### x_min = cx - w/2
    max_x_min = torch.max(outputs[:,0] - outputs[:,2]/2, targets[:,0] - targets[:,2]/2)
    ### width = min_x_max - max_x_min
    width = torch.torch.clamp(min_x_max - max_x_min, min=0)
    
    #### y_max = cy + h/2 
    min_y_max = torch.min(outputs[:,1] + outputs[:,3]/2, targets[:,1] + targets[:,3]/2)     
    #### y_min = cy - h/2
    max_y_min = torch.max(outputs[:,1] - outputs[:,3]/2, targets[:,1] - targets[:,3]/2)
    ### height = min_y_max - max_y_min
    height = torch.clamp(min_y_max - max_y_min, min=0)

    intersection_area = torch.mul(width, height).sum() # w * h

    ## Finally, compute IoU
    iou = 2 * intersection_area / (outputs_area+targets_area)
    return iou

def compute_IOULoss(outputs, targets) -> float:
    iou = computeBBoxIoU(outputs, targets) ### 1 = max overlap, 0 = zero overlap
    iou_loss = (1 - iou) ** 2 ### square to punish very bad predictions more
    return iou_loss


if __name__ == "__main__":
    import torch
    
    # Test cases for computeBBoxIoU function
    
    # Case 1: Zero overlap
    print("Testing with zero overlap:")
    outputs_zero = torch.tensor([[0.1, 0.1, 0.2, 0.2]])  # Box at (0.1, 0.1) with width=0.2, height=0.2
    targets_zero = torch.tensor([[0.8, 0.8, 0.2, 0.2]])  # Box at (0.8, 0.8) with width=0.2, height=0.2
    iou_zero = computeBBoxIoU(outputs_zero, targets_zero)
    print(f"IoU (zero overlap): {iou_zero}")
    print(f"IoU Loss (zero overlap): {compute_IOULoss(outputs_zero, targets_zero)}")
    
    # Case 2: Medium overlap
    print("\nTesting with medium overlap:")
    outputs_medium = torch.tensor([[0.5, 0.5, 0.4, 0.4]])  # Box at (0.5, 0.5) with width=0.4, height=0.4
    targets_medium = torch.tensor([[0.6, 0.6, 0.4, 0.4]])  # Box at (0.6, 0.6) with width=0.4, height=0.4
    iou_medium = computeBBoxIoU(outputs_medium, targets_medium)
    print(f"IoU (medium overlap): {iou_medium}")
    print(f"IoU Loss (medium overlap): {compute_IOULoss(outputs_medium, targets_medium)}")
    
    # Case 3: Maximum overlap (identical boxes)
    print("\nTesting with maximum overlap:")
    outputs_max = torch.tensor([[0.5, 0.5, 0.4, 0.4]])  # Box at (0.5, 0.5) with width=0.4, height=0.4
    targets_max = torch.tensor([[0.5, 0.5, 0.4, 0.4]])  # Identical box
    iou_max = computeBBoxIoU(outputs_max, targets_max)
    print(f"IoU (maximum overlap): {iou_max}")
    print(f"IoU Loss (maximum overlap): {compute_IOULoss(outputs_max, targets_max)}")
    
    # Testing with VOC format bboxes
    print("\nTesting with VOC format bboxes:")
    
    # Case 1: Zero overlap in VOC format [xmin, ymin, xmax, ymax]
    outputs_voc_zero = torch.tensor([[0.0, 0.0, 0.2, 0.2]])  # Box from (0,0) to (0.2,0.2)
    targets_voc_zero = torch.tensor([[0.7, 0.7, 0.9, 0.9]])  # Box from (0.7,0.7) to (0.9,0.9)
    
    # Convert to anchor format
    outputs_anchor_zero = convertVOCBBoxFormatToAnchorFormat(outputs_voc_zero)
    targets_anchor_zero = convertVOCBBoxFormatToAnchorFormat(targets_voc_zero)
    
    iou_voc_zero = computeBBoxIoU(outputs_anchor_zero, targets_anchor_zero)
    print(f"IoU (VOC format, zero overlap): {iou_voc_zero}")
    print(f"IoU Loss (VOC format, zero overlap): {compute_IOULoss(outputs_anchor_zero, targets_anchor_zero)}")
    
    # Case 2: Medium overlap in VOC format
    outputs_voc_medium = torch.tensor([[0.3, 0.3, 0.7, 0.7]])  # Box from (0.3,0.3) to (0.7,0.7)
    targets_voc_medium = torch.tensor([[0.5, 0.5, 0.9, 0.9]])  # Box from (0.5,0.5) to (0.9,0.9)
    
    # Convert to anchor format
    outputs_anchor_medium = convertVOCBBoxFormatToAnchorFormat(outputs_voc_medium)
    targets_anchor_medium = convertVOCBBoxFormatToAnchorFormat(targets_voc_medium)
    
    iou_voc_medium = computeBBoxIoU(outputs_anchor_medium, targets_anchor_medium)
    print(f"IoU (VOC format, medium overlap): {iou_voc_medium}")
    print(f"IoU Loss (VOC format, medium overlap): {compute_IOULoss(outputs_anchor_medium, targets_anchor_medium)}")
    
    # Case 3: Maximum overlap in VOC format (identical boxes)
    outputs_voc_max = torch.tensor([[0.3, 0.3, 0.7, 0.7]])  # Box from (0.3,0.3) to (0.7,0.7)
    targets_voc_max = torch.tensor([[0.3, 0.3, 0.7, 0.7]])  # Identical box
    
    # Convert to anchor format
    outputs_anchor_max = convertVOCBBoxFormatToAnchorFormat(outputs_voc_max)
    targets_anchor_max = convertVOCBBoxFormatToAnchorFormat(targets_voc_max)
    
    iou_voc_max = computeBBoxIoU(outputs_anchor_max, targets_anchor_max)
    print(f"IoU (VOC format, maximum overlap): {iou_voc_max}")
    print(f"IoU Loss (VOC format, maximum overlap): {compute_IOULoss(outputs_anchor_max, targets_anchor_max)}")

