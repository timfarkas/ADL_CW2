import torch

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
