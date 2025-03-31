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