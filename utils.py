import torch
import torchvision


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
