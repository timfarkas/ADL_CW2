import torch
import torchvision
import os
from utils import store_images,unnormalize


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


def load_test_pet_data(batch_size: int,resize_size: int = 256) -> torch.utils.data.DataLoader:
    """
    Loads the test data from the Oxford Pets dataset.
    """
    test_set = torchvision.datasets.OxfordIIITPet(
        root="./data",
        split="test",
        target_types="segmentation",
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((resize_size, resize_size)),
                torchvision.transforms.ToTensor(),
            ]
        ),
        target_transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((resize_size, resize_size)),
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
    """
    eps = 1e-6
    batch_size = predictions.shape[0]
    total_iou = 0
    total_f1 = 0

    for j in range(batch_size):
        prediction = predictions[j].bool()
        true_mask = true_masks[j].bool()

        true_positive = (prediction & true_mask).sum().float()
        false_positive = (prediction & ~true_mask).sum().float()
        false_negative = (~prediction & true_mask).sum().float()
        union = (prediction | true_mask).sum().float()

        # Handle IoU
        iou = true_positive / (union + eps)
        total_iou += iou.item()

        # Handle F1 Score
        denominator = (2 * true_positive + false_positive + false_negative)
        if denominator == 0:
            # Case: both pred and gt are completely empty
            f1_score = 1.0 if true_mask.sum() == 0 else 0.0
        else:
            f1_score = 2 * true_positive / (denominator + eps)

        total_f1 += f1_score.item()

    return total_iou, total_f1



def binarise_predictions(predictions: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (predictions > threshold).bool()

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


def evaluate_dataset(dataset, image_number: int, image_name: str, device: str = None,threshold: float=0.5):
    """
    Evaluates a segmentation dataset using IoU and F1 metrics.

    Args:
        dataset (TensorDataset): Dataset containing (images, predicted_masks, ground_truth_masks)
        image_number (int): Number of samples to visualize
        image_name (str): Filename prefix for saved visualizations
        device (str): Device to use ('cuda', 'cpu', etc.)
    """
    print(f"Evaluating dataset with threshold:{threshold}" )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    total_IoU = 0
    total_f1_score = 0
    num_samples = 0

    output_dir = "evaluation_visualization"
    os.makedirs(output_dir, exist_ok=True)


    for i, (images, masks, masks_gt) in enumerate(dataloader):
        masks = masks.to(device)
        masks_gt = masks_gt.to(device)
        masks_gt_binary = get_categories_from_normalization(masks_gt)
        masks_binary=binarise_predictions(masks,threshold)
        batch_iou, batch_f1 = compute_iou_and_f1(masks_binary, masks_gt_binary)
        total_IoU += batch_iou
        total_f1_score += batch_f1
        num_samples += images.size(0)

        if i == 0:
            images_to_store = unnormalize(images[:image_number])
            masks_to_store = masks[:image_number]
            masks_binary_to_store = masks_binary[:image_number]
            gt_to_store = (masks_gt[:image_number] * 255).byte()


            store_images(images_to_store, os.path.join(output_dir, f"{image_name}_{threshold}_images.jpg"))
            store_images(masks_to_store, os.path.join(output_dir, f"{image_name}_{threshold}_masks.jpg"), is_segmentation=True)
            store_images(masks_binary_to_store, os.path.join(output_dir, f"{image_name}_{threshold}_masks_binary.jpg"), is_segmentation=True)
            store_images(gt_to_store, os.path.join(output_dir, f"{image_name}_{threshold}_gt.jpg"), is_segmentation=True)

    print(f"Mean IoU: {total_IoU / num_samples:.4f}")
    print(f"Mean F1 Score: {total_f1_score / num_samples:.4f}")

