import torch
import torchvision

from utils import store_images


def get_categories_from_normalization(x: torch.Tensor) -> torch.Tensor:
    """
    Converts the input tensor, which is normalized between 0 and 1, to categories
    with values 0, 1 and 2.
    """
    categories = x * 255 - 1
    return torch.where(
        categories == 0,
        torch.ones_like(categories),
        torch.where(categories == 1, torch.zeros_like(categories), categories),
    )


def load_test_pet_data(batch_size: int) -> torch.utils.data.DataLoader:
    """
    Loads the test data from the Oxford Pets dataset.
    """
    test_set = torchvision.datasets.OxfordIIITPet(
        root="./data",
        split="test",
        target_types="segmentation",
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((256, 256)),
                torchvision.transforms.ToTensor(),
            ]
        ),
        target_transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((256, 256)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(get_categories_from_normalization),
            ]
        ),
        download=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return test_loader


def compute_iou_and_f1(predictions: torch.Tensor, true_masks: torch.Tensor) -> tuple:
    """
    Computes IoU and F1 score for a batch of predictions and true masks.
    
    Args:
        predictions: Tensor of shape (batch_size, C, H, W) with model predictions
        true_masks: Tensor of shape (batch_size, C, H, W) with ground truth masks
        
    Returns:
        tuple: (total_iou, total_f1) - sum of IoU and F1 scores for the batch
    """
    batch_size = predictions.shape[0]
    total_iou = 0
    total_f1 = 0
    
    for j in range(batch_size):
        prediction = predictions[j]
        true_mask = true_masks[j]
        
        # Calculate IoU
        intersection = torch.logical_and(prediction, true_mask)
        union = torch.logical_or(prediction, true_mask)
        if torch.sum(union) == 0:
            iou = torch.tensor(1.0)  # This seems to be a bug in the dataset
        else:
            iou = torch.sum(intersection) / torch.sum(union)
        total_iou += iou.item()
        
        # Calculate precision and recall
        true_positive = torch.sum(intersection)
        false_positive = (prediction > 0).sum() - true_positive
        false_negative = (true_mask > 0).sum() - true_positive
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        if precision.isnan() or recall.isnan():
            f1_score = torch.tensor(1.0)
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
        total_f1 += f1_score.item()
    
    return total_iou, total_f1

def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    image_number: int,
    image_name: str,
    device: str = None
) -> None:
    """
    Evaluates the model on the test data and stores the images.
    """
    print("Evaluating model...")
    total_IoU = 0
    total_f1_score = 0
    num_samples = 0
    
    for i, (images, masks) in enumerate(test_loader):
        # TODO: When we have a model, ensure the prediction results are actually comparable
        images = images.to(device)
        masks = masks.to(device)
        predictions = model(images)
        
        batch_iou, batch_f1 = compute_iou_and_f1(predictions, masks)
        total_IoU += batch_iou
        total_f1_score += batch_f1
        num_samples += images.size(0)

        if i == 0:
            images_to_store = images[:image_number]
            masks_to_store = masks[:image_number]
            predictions_to_store = predictions.unsqueeze(1)[:image_number]
            store_images(images_to_store, image_name + "_images.jpg")
            store_images(
                masks_to_store, image_name + "_true_masks.jpg", is_segmentation=True
            )
            store_images(
                predictions_to_store,
                image_name + "_predictions.jpg",
                is_segmentation=True,
            )
        else:
            break

    print(f"Mean IoU: {total_IoU / num_samples}")
    print(f"Mean F1 Score: {total_f1_score / num_samples}")



