import json
import os
from matplotlib import pyplot as plt
import torch


def unnormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (img_tensor * std + mean).clamp(0, 1)


def get_binary_from_normalization(
    x: torch.Tensor, include_boundary: bool = True
) -> torch.Tensor:
    """
    Converts normalized [0,1] segmentation tensor into:

    - 0.0039 (label 1) → 1 (foreground)
    - 0.0078 (label 2) → 0 (background)
    - 0.0118 (label 3) → 1 (boundary → treated as foreground)
    """
    categories = (x * 255 - 1).round().long()  # Convert to int labels: 0, 1, 2
    if include_boundary:
        # Map: 0 → 1, 1 → 0, 2 → 1
        categories = torch.where(
            categories == 0,
            torch.tensor(1, device=x.device),
            torch.where(
                categories == 1,
                torch.tensor(0, device=x.device),
                torch.tensor(1, device=x.device),
            ),
        )
    else:
        # Map: 0 → 1, 1 → 0, 2 → 0
        categories = torch.where(
            categories == 0,
            torch.tensor(1, device=x.device),  # 0 → 1 (foreground)
            torch.tensor(0, device=x.device),  # else → 0 (background)
        )
    return categories


def visualize_predicted_masks(images, masks, masks_gt, storage_path=None):
    """
    Visualize predicted masks and ground-truth masks with 3xN layout:
    Row 1: Input images
    Row 2: Predicted masks
    Row 3: Ground truth masks
    """

    num_samples = min(images.size(0), masks.size(0), masks_gt.size(0))
    fig, axs = plt.subplots(3, num_samples, figsize=(num_samples * 3, 3 * 3))

    for i in range(num_samples):
        img = unnormalize(images[i]).permute(1, 2, 0).cpu().numpy()
        pred_mask = masks[i][0].cpu().numpy()
        gt_mask = masks_gt[i][0].cpu().numpy()

        axs[0, i].imshow(img)
        axs[0, i].set_title(f"Image {i + 1}")
        axs[1, i].imshow(pred_mask, cmap="gray")
        axs[1, i].set_title("Predicted")
        axs[2, i].imshow(gt_mask, cmap="gray")
        axs[2, i].set_title("Ground Truth")

        for row in range(3):
            axs[row, i].axis("off")

    plt.tight_layout()

    if storage_path:
        dir_name = os.path.dirname(storage_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(storage_path)
        print(f"Saved visualization to {storage_path}")

    plt.close()


def log_self_training_performance(
    log_dir: str, run_name: str, round_name: str, ioi: float, f1: float
):
    """
    Log performance metrics to a JSON file.
    """

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "self_training_log.json")

    # Create or load existing log
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            log_data = json.load(f)
    else:
        log_data = {}

    if run_name not in log_data:
        log_data[run_name] = {}

    if round_name not in log_data[run_name]:
        log_data[run_name][round_name] = {}

    log_data[run_name][round_name]["ioi"] = ioi
    log_data[run_name][round_name]["f1"] = f1

    # Save updated log
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=4)
