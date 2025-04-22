"""
AI usage statement:

AI was used to assist with researching and debugging, as well as helping
with creating docstrings. All code was writte, reviewed and/or modified by a human.
"""

import torch

from config import segmentation_output_threshold
from training.utils import (
    get_binary_masks_from_trimap,
    unnormalize,
    visualize_predicted_masks,
)


def compute_iou(outputs, targets) -> float:
    """
    Calculate average intersection over union.

    Args:
        outputs: Model predictions (logits)
        targets: Ground truth labels

    Returns:
        IoU of the two.
    """
    intersection = torch.logical_and(outputs, targets).sum().item()
    union = torch.logical_or(outputs, targets).sum().item()
    iou = intersection / union if union != 0 else 0.0
    return iou


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


def convert_voc_bbox_format_to_anchor_format(boxes):
    """
    Takes in a tensor of bounding boxes in VOC format, of the form [x_min, y_min, x_max, y_max], or a single box.
    Converts it to a tensor of bounding boxes in anchor format, of the form [cx, cy, w, h], or a single box.
    """
    assert isinstance(boxes, torch.Tensor), f"Expected tensor, received {type(boxes)}"
    unsqueezed = False
    if len(boxes.shape) == 1:
        unsqueezed = True
        boxes.unsqueeze(0)  ## if getting single box, unsqueeze first dim
    assert len(boxes.shape) == 2, (
        f"Expected two dimensions, received {len(boxes.shape)}"
    )
    B, V = boxes.shape
    assert V == 4, f"Expected 2nd dim to have size 4, received {len(boxes.shape[1])}"
    new_tensor = torch.Tensor(B, V)

    new_tensor[:, 0] = (boxes[:, 2] + boxes[:, 0]) / 2  # cx = (xmax + xmin) / 2
    new_tensor[:, 1] = (boxes[:, 3] + boxes[:, 1]) / 2  # cy = (ymax + ymin) / 2
    new_tensor[:, 2] = boxes[:, 2] - boxes[:, 0]  # w = xmax - xmin
    new_tensor[:, 3] = boxes[:, 3] - boxes[:, 1]  # h = ymax - ymin

    if unsqueezed:  # revert to original state
        new_tensor.squeeze(0)

    return new_tensor


def compute_bbox_iou(outputs, targets) -> float:
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
    outputs_area = torch.mul(outputs[:, 2], outputs[:, 3]).sum()  # w * h
    targets_area = torch.mul(targets[:, 2], targets[:, 3]).sum()  # w * h

    #### x_max = cx + w/2
    min_x_max = torch.min(
        outputs[:, 0] + outputs[:, 2] / 2, targets[:, 0] + targets[:, 2] / 2
    )
    #### x_min = cx - w/2
    max_x_min = torch.max(
        outputs[:, 0] - outputs[:, 2] / 2, targets[:, 0] - targets[:, 2] / 2
    )

    width = torch.torch.clamp(min_x_max - max_x_min, min=0)

    #### y_max = cy + h/2
    min_y_max = torch.min(
        outputs[:, 1] + outputs[:, 3] / 2, targets[:, 1] + targets[:, 3] / 2
    )
    #### y_min = cy - h/2
    max_y_min = torch.max(
        outputs[:, 1] - outputs[:, 3] / 2, targets[:, 1] - targets[:, 3] / 2
    )
    ### height = min_y_max - max_y_min
    height = torch.clamp(min_y_max - max_y_min, min=0)

    intersection_area = torch.mul(width, height).sum()  # w * h

    iou = 2 * intersection_area / (outputs_area + targets_area)
    return iou


def convert_and_get_iou(outputs, targets) -> float:
    outputs = convert_voc_bbox_format_to_anchor_format(outputs)
    targets = convert_voc_bbox_format_to_anchor_format(targets)
    iou = compute_bbox_iou(outputs, targets)
    return iou


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
        precision = true_positive / (true_positive + false_positive + eps)
        recall = true_positive / (true_positive + false_negative + eps)
        if precision.isnan() or recall.isnan():
            f1_score = torch.tensor(0.0)
        else:
            f1_score = 2 * (precision * recall) / (precision + recall + eps)

        total_f1 += f1_score.item()

    return total_iou, total_f1


def evaluate_segmentation_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    image_number: int = 5,
    device: str = None,
    storage_path: str = None,
) -> tuple[float, float]:
    """
    Evaluates the model on the test data and stores the images.

    Returns:
        tuple: (average_IoU, average_f1_score)
    """
    print(f"Evaluating model with threshold: {segmentation_output_threshold}")
    total_IoU = 0
    total_f1_score = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for i, (images, masks_gt) in enumerate(test_loader):
            images = images.to(device)
            masks_gt = masks_gt.to(device)

            logits = model(images)  # raw output
            masks_pre = torch.sigmoid(logits)  # apply sigmoid to get probabilities
            masks_pre_binary = (
                masks_pre > segmentation_output_threshold
            ).long()  # apply threshold to get binary masks
            masks_gt_binary = get_binary_masks_from_trimap(masks_gt)

            batch_iou, batch_f1 = compute_iou_and_f1(masks_pre_binary, masks_gt_binary)
            total_IoU += batch_iou
            total_f1_score += batch_f1
            num_samples += images.size(0)

            if storage_path and i == 0:
                visualize_predicted_masks(
                    images[:image_number],
                    masks_pre_binary[:image_number],
                    masks_gt_binary[:image_number],
                    storage_path=storage_path,
                    device=device,
                )

    average_IoU = total_IoU / num_samples
    average_f1_score = total_f1_score / num_samples

    print(f"Mean IoU: {average_IoU}, F1 Score: {average_f1_score}")

    return (average_IoU, average_f1_score)
