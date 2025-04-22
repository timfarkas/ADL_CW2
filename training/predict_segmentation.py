"""
AI usage statement:

AI was used to assist with researching and debugging, as well as helping
with creating docstrings. All code was writte, reviewed and/or modified by a human.
"""

from typing import Literal
import cv2
import numpy as np
import torch


def predict_segmentation_dataset(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    predictions_transform: Literal["filter", "grabcut", "mixlabel"],
    threshold: float,
):
    model.eval()
    image_list = []
    prob_mask_list = []
    gt_mask_list = []

    with torch.no_grad():
        for images, former_probs, gt_masks in dataloader:
            images = images.to(device)
            former_probs = former_probs.to(device)
            logits = model(images)  # [B, 1, H, W]
            probs = torch.sigmoid(logits)  # ∈ [0,1]
            batch_size = images.size(0)

            if predictions_transform == "filter":
                output_segmentation = filter_probs(
                    probs=probs,
                    threshold=threshold,
                )

            elif predictions_transform == "grabcut":
                refined_masks = []

                # Convert images and probs to CPU for OpenCV
                images = images.cpu().detach()
                probs = probs.cpu().detach()

                for i in range(batch_size):
                    refined_mask_np = grabcut_from_mask(images[i], probs[i], threshold)
                    refined_mask_tensor = torch.from_numpy(
                        refined_mask_np
                    ).float()  # [H, W]

                    final_mask = (
                        0.5 * probs[i].squeeze(0) + 0.5 * refined_mask_tensor
                    )  # blend the grabcut result with the original mask
                    refined_masks.append(final_mask.unsqueeze(0))  # [1, H, W]

                output_segmentation = torch.stack(refined_masks, dim=0)  # [B, 1, H, W]

            elif predictions_transform == "mixlabel":
                threshold_value_low = threshold
                threshold_value_high = 1 - threshold
                output_segmentation = refine_probs_with_mixlabel(
                    model_probs=probs,
                    weak_probs=former_probs,
                    low_thresh=threshold_value_low,
                    high_thresh=threshold_value_high,
                )

            elif predictions_transform is None:
                output_segmentation = probs

            else:
                raise ValueError(
                    f"Unknown prediction transform: {predictions_transform}. "
                    "Choose from ['filter', 'grabcut', 'mixlabel']."
                )

            image_list.append(images.cpu())
            prob_mask_list.append(output_segmentation.cpu())
            gt_mask_list.append(gt_masks.cpu())

    all_images = torch.cat(image_list, dim=0)
    all_probs = torch.cat(prob_mask_list, dim=0)
    all_gts = torch.cat(gt_mask_list, dim=0)
    return torch.utils.data.TensorDataset(all_images, all_probs, all_gts)


def filter_probs(
    probs: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """
    Filter probabilities based on a threshold.
    Args:
        probs: Tensor of probabilities.
        threshold: Minimum confidence threshold; values below will be set to 0.
    Returns:
        Tensor with filtered probabilities.
    """
    filtered_probs = probs * (probs >= threshold).float()
    return filtered_probs


def grabcut_from_mask(
    image: torch.Tensor, prob: torch.Tensor, threshold: float, iter_count: int = 5
):
    """
    Applies GrabCut using an initial seed mask.
    Returns binary mask after GrabCut, or fallback based on confident seeds.
    """

    img_np = image.cpu().permute(1, 2, 0).numpy()  # [H, W, C], float32
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)  # [H, W, C], uint8

    # Convert prob to NumPy mask
    prob = prob.squeeze().detach().cpu().numpy()  # [H, W]

    low_thres = np.percentile(prob, threshold * 100)
    high_thres = np.percentile(prob, (1 - threshold) * 100)

    init_mask = np.full_like(prob, 3, dtype=np.uint8)

    # Assign confident FG and BG based on thresholds
    init_mask[prob >= high_thres] = 1  # definite foreground
    init_mask[prob <= low_thres] = 0  # definite background

    # print(grabcut_mask)

    num_fg = np.sum(init_mask == cv2.GC_FGD)
    num_bg = np.sum(init_mask == cv2.GC_BGD)

    has_fg = num_fg > 0
    has_bg = num_bg > 0

    if not has_fg or not has_bg:
        # print("⚠️ GrabCut skipped (no confident FG/BG seeds) → using fallback init_mask.")
        # # Convert: 1 (FG) and 3 (probable FG) → 1, everything else → 0
        return ((init_mask == cv2.GC_FGD) | (init_mask == cv2.GC_PR_FGD)).astype(
            np.uint8
        )

    # Proceed with GrabCut if valid seeds exist
    mask = init_mask.copy().astype("uint8")
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        img_np,
        mask,
        None,
        bgdModel,
        fgdModel,
        iter_count,
        mode=cv2.GC_INIT_WITH_MASK,
    )

    binary_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(
        "uint8"
    )
    refined_mask = binary_mask.astype(np.float32)  #
    return refined_mask


def refine_probs_with_mixlabel(
    model_probs: torch.Tensor,
    weak_probs: torch.Tensor,
    low_thresh: float,
    high_thresh: float,
):
    """
    Refine weak CAM/prob labels using model's predictions.

    Args:
        model_probs (Tensor): [B, 1, H, W] - Model sigmoid output.
        weak_probs (Tensor): [B, 1, H, W] - Initial weak CAM labels.
        low_thresh (float): Below this, predictions are ignored.
        high_thresh (float): Above this, predictions overwrite weak labels.

    Returns:
        Tensor: [B, 1, H, W] - Refined probability mask.
    """
    high_conf_fg = (model_probs >= high_thresh).float()
    merged = torch.maximum(high_conf_fg, weak_probs)
    confident = (model_probs >= low_thresh).float() * merged
    return confident
