import torch
import torchvision
import os
from utils import store_images,unnormalize,save_image_grid
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF



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

def get_binary_from_normalization(x: torch.Tensor) -> torch.Tensor:
    """
    Converts normalized [0,1] segmentation tensor into:
    - 0.0039 (label 1) → 1 (foreground)
    - 0.0078 (label 2) → 0 (background)
    - 0.0118 (label 3) → 1 (boundary → treated as foreground)
    """
    include_boundary=True
    categories = (x * 255 - 1).round().long()  # Convert to int labels: 0, 1, 2
    if include_boundary==True:
        # Map: 0 → 1, 1 → 0, 2 → 1
        categories = torch.where(
            categories == 0, torch.tensor(1, device=x.device),
            torch.where(categories == 1, torch.tensor(0, device=x.device), torch.tensor(1, device=x.device))
        )
    else:
        # Map: 0 → 1, 1 → 0, 2 → 0
        categories = torch.where(
            categories == 0, torch.tensor(1, device=x.device),  # 0 → 1 (foreground)
            torch.tensor(0, device=x.device)  # else → 0 (background)
        )
    return categories

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

    Args:
        predictions: Tensor of shape (batch_size, C, H, W) with model predictions
        true_masks: Tensor of shape (batch_size, C, H, W) with ground truth masks

    Returns:
        tuple: (total_iou, total_f1) - sum of IoU and F1 scores for the batch
    """
    eps = 1e-6
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
        precision = true_positive / (true_positive + false_positive+eps)
        recall = true_positive / (true_positive + false_negative+eps)
        if precision.isnan() or recall.isnan():
            f1_score = torch.tensor(0.0)
        else:
            f1_score = 2 * (precision * recall) / (precision + recall+eps)

        total_f1 += f1_score.item()

    return total_iou, total_f1



def binarise_predictions(predictions: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (predictions > threshold).bool()

def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    image_number: int,
    image_name: str,
    device: str = None,
    threshold:float = 0.5,
    output_dir = "evaluation_visualization"
) -> None:
    """
    Evaluates the model on the test data and stores the images.
    """
    print(f"Evaluating model with theshold:{threshold}")
    total_IoU = 0
    total_f1_score = 0
    num_samples = 0
    total_loss = 0.0
    output_dir=output_dir
    os.makedirs(output_dir, exist_ok=True)
    loss_funciton=nn.BCEWithLogitsLoss()
    for i, (images, masks_gt) in enumerate(test_loader):
        logits = model(images)  # raw output
        # print(f"[Logits] min: {logits.min().item():.4f}, max: {logits.max().item():.4f}")
        masks_pre = torch.sigmoid(logits)  # apply sigmoid to get probabilities
        # masks_pre=logits
        # print(f"[Sigmoid Output] min: {masks_pre.min().item():.4f}, max: {masks_pre.max().item():.4f}")
        batch_loss=loss_funciton(masks_pre,masks_gt)
        masks_pre_binary= binarise_predictions(masks_pre, threshold)
        masks_gt = masks_gt.to(device)
        masks_gt_binary = get_binary_from_normalization(masks_gt)

        batch_iou, batch_f1 = compute_iou_and_f1(masks_pre_binary, masks_gt_binary)
        total_IoU += batch_iou
        total_f1_score += batch_f1
        total_loss += batch_loss.item() * images.size(0)
        num_samples += images.size(0)

        if i == 0:
            images_to_store = unnormalize(images[:image_number])
            masks_pre_to_store = masks_pre[:image_number]
            masks_pre_binary_to_store = masks_pre_binary[:image_number]
            masks_gt_to_store = masks_gt_binary[:image_number]

            save_image_grid(images_to_store,masks_pre_to_store,masks_pre_binary_to_store,masks_gt_to_store,os.path.join(output_dir, f"{image_name}_thres_{threshold}.jpg"))
            #
            # store_images(images_to_store, os.path.join(output_dir, f"{image_name}_{threshold}_images.jpg"))
            # store_images(masks_pre_to_store, os.path.join(output_dir, f"{image_name}_{threshold}_masks.jpg"), is_segmentation=True)
            # store_images(masks_pre_binary_to_store, os.path.join(output_dir, f"{image_name}_{threshold}_masks_binary.jpg"), is_segmentation=True)
            # store_images(masks_gt_to_store, os.path.join(output_dir, f"{image_name}_{threshold}_gt.jpg"), is_segmentation=True)

        else:
            break
    mean_loss = total_loss / num_samples
    print(f"Mean IoU: {total_IoU / num_samples}, F1 Score: {total_f1_score / num_samples}, CE Loss: {mean_loss}")

def evaluate_model_with_grabcut(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    image_number: int,
    image_name: str,
    device: str = None,
    threshold:float = 0.5,
    output_dir = "evaluation_visualization",
    grabcut_thres:float=0.2,
) -> None:
    """
    Evaluates the model on the test data and stores the images.
    """
    from models import SelfTraining
    print(f"Evaluating model with theshold:{threshold}")
    total_IoU = 0
    total_f1_score = 0
    num_samples = 0
    total_loss = 0.0
    output_dir=output_dir
    os.makedirs(output_dir, exist_ok=True)
    loss_funciton=nn.BCEWithLogitsLoss()
    for i, (images, masks_gt) in enumerate(test_loader):
        logits = model(images)
        masks_pre = torch.sigmoid(logits)
        batch_loss = loss_funciton(masks_pre, masks_gt)

        images = images.to(device)
        masks_gt = masks_gt.to(device)

        batch_size = images.size(0)
        batch_iou, batch_f1 = 0.0, 0.0

        for j in range(batch_size):
            img = images[j]  # [C, H, W]
            prob = masks_pre[j] # [1, H, W]
            gt = masks_gt[j]  # [1, H, W]
            pred_binary = SelfTraining.grabcut_from_mask(img, prob, grabcut_thres)
            pred_binary = torch.from_numpy(pred_binary).to(device)
            gt_binary = get_binary_from_normalization(gt)

            iou, f1 = compute_iou_and_f1(
                pred_binary.unsqueeze(0),  # [1, H, W]
                gt_binary.unsqueeze(0)  # [1, H, W]
            )
            batch_iou += iou
            batch_f1 += f1

        total_IoU += batch_iou
        total_f1_score += batch_f1
        total_loss += batch_loss.item() * batch_size
        num_samples += batch_size


    mean_loss = total_loss / num_samples
    print(f"Mean IoU: {total_IoU / num_samples}, F1 Score: {total_f1_score / num_samples}, CE Loss: {mean_loss}")


def evaluate_dataset(dataset, image_number: int, image_name: str, device: str = None,threshold: float=0.5,output_dir = "evaluation_visualization"):
    """
    Evaluates a segmentation dataset using IoU and F1 metrics.

    Args:
        dataset (TensorDataset): Dataset containing (images, predicted_masks, ground_truth_masks)
        image_number (int): Number of samples to visualize
        image_name (str): Filename prefix for saved visualizations
        device (str): Device to use ('cuda', 'cpu', etc.)
    """
    print(f"Evaluating dataset with threshold:{threshold}" )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    total_IoU = 0
    total_f1_score = 0
    num_samples = 0
    total_loss = 0.0
    output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)
    loss_funciton=nn.BCEWithLogitsLoss()

    for i, (images, masks_pre, masks_gt) in enumerate(dataloader):
        masks_pre = masks_pre.to(device)
        masks_pre_binary=binarise_predictions(masks_pre,threshold)
        masks_gt = masks_gt.to(device)
        masks_gt_binary = get_binary_from_normalization(masks_gt)
        batch_loss=loss_funciton(masks_pre,masks_gt)

        batch_iou, batch_f1 = compute_iou_and_f1(masks_pre_binary, masks_gt_binary)
        total_IoU += batch_iou
        total_f1_score += batch_f1
        total_loss += batch_loss.item() * images.size(0)
        num_samples += images.size(0)

        if i == 0:
            images_to_store = unnormalize(images[:image_number])
            masks_pre_to_store = masks_pre[:image_number]
            masks_pre_binary_to_store = masks_pre_binary[:image_number]
            masks_gt_to_store = masks_gt_binary[:image_number]
            save_image_grid(images_to_store,masks_pre_to_store,masks_pre_binary_to_store,masks_gt_to_store,os.path.join(output_dir, f"{image_name}_thres_{threshold}.jpg"))
            #
            # store_images(images_to_store, os.path.join(output_dir, f"{image_name}_{threshold}_images.jpg"))
            # store_images(masks_pre_to_store, os.path.join(output_dir, f"{image_name}_{threshold}_masks.jpg"), is_segmentation=True)
            # store_images(masks_pre_binary_to_store, os.path.join(output_dir, f"{image_name}_{threshold}_masks_binary.jpg"), is_segmentation=True)
            # store_images(masks_gt_to_store, os.path.join(output_dir, f"{image_name}_{threshold}_gt.jpg"), is_segmentation=True)
    mean_loss = total_loss / num_samples
    print(f"Mean IoU: {total_IoU / num_samples}, F1 Score: {total_f1_score / num_samples}, CE Loss: {mean_loss}")