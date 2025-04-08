import torch
import torchvision
import os
from utils import store_images,unnormalize
import matplotlib.pyplot as plt


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
    threshold:float = 0.5
) -> None:
    """
    Evaluates the model on the test data and stores the images.
    """
    print(f"Evaluating model with theshold:{threshold}")
    total_IoU = 0
    total_f1_score = 0
    num_samples = 0
    output_dir = "evaluation_visualization"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (images, masks_gt) in enumerate(test_loader):
        # TODO: When we have a model, ensure the prediction results are actually comparable
        images = images.to(device)
        masks_pre=torch.sigmoid(model(images))
        masks_pre_binary= binarise_predictions(masks_pre, threshold)
        masks_gt = masks_gt.to(device)
        masks_gt_binary = get_categories_from_normalization(masks_gt)

        # masks_pre_vis = torch.sigmoid(masks_pre[:4].detach().cpu())
        # images_vis = images[:4].cpu()  # (B, 3, H, W)
        #
        # for idx in range(images_vis.size(0)):
        #     fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        #
        #     # Show image
        #     img = images_vis[idx]
        #     img = img.permute(1, 2, 0).numpy()  # CHW → HWC
        #     img = (img - img.min()) / (img.max() - img.min())  # Normalize for display
        #     axs[0].imshow(img)
        #     axs[0].set_title("Image")
        #     axs[0].axis("off")
        #
        #     # Show predicted mask
        #     mask = masks_pre_vis[idx].squeeze().detach().numpy()
        #     axs[1].imshow(mask, cmap='gray')
        #     axs[1].set_title("Predicted Mask")
        #     axs[1].axis("off")
        #
        #     plt.tight_layout()
        #     plt.show()


        batch_iou, batch_f1 = compute_iou_and_f1(masks_pre_binary, masks_gt_binary)
        total_IoU += batch_iou
        total_f1_score += batch_f1
        num_samples += images.size(0)

        if i == 0:
            images_to_store = unnormalize(images[:image_number])
            masks_pre_to_store = masks_pre[:image_number]
            masks_pre_binary_to_store = masks_pre_binary[:image_number]
            masks_gt_to_store = masks_gt_binary[:image_number]
            # print("Shape:", masks_pre_to_store.shape)
            # print("Dtype:", masks_pre_to_store.dtype)
            # print("Min value:", masks_pre_to_store.min().item())
            # print("Max value:", masks_pre_to_store.max().item())


            # n = min(image_number, images_to_store.size(0))
            #
            # for i in range(n):
            #     fig, axs = plt.subplots(1, 4, figsize=(16, 4))
            #
            #     # Unnormalize and convert to NumPy image
            #     image = images_to_store[i].permute(1, 2, 0).cpu().numpy()
            #     image = (image * 255).astype('uint8')  # If already unnormalized
            #
            #     pred_mask = masks_pre_to_store[i].squeeze().detach().cpu().numpy()
            #     pred_binary = masks_pre_binary_to_store[i].squeeze().detach().cpu().numpy()
            #     gt_mask = masks_gt_to_store[i].squeeze().detach().cpu().numpy()
            #
            #     axs[0].imshow(image)
            #     axs[0].set_title("Image")
            #     axs[1].imshow(pred_mask, cmap='jet')
            #     axs[1].set_title("Predicted Mask")
            #     axs[2].imshow(pred_binary, cmap='gray')
            #     axs[2].set_title("Binary Prediction")
            #     axs[3].imshow(gt_mask, cmap='gray')
            #     axs[3].set_title("Ground Truth")
            #
            #     for ax in axs:
            #         ax.axis('off')
            #     plt.tight_layout()
            #     plt.show()
            #


            store_images(images_to_store, os.path.join(output_dir, f"{image_name}_{threshold}_images.jpg"))
            store_images(masks_pre_to_store, os.path.join(output_dir, f"{image_name}_{threshold}_masks.jpg"), is_segmentation=True)
            store_images(masks_pre_binary_to_store, os.path.join(output_dir, f"{image_name}_{threshold}_masks_binary.jpg"), is_segmentation=True)
            store_images(masks_gt_to_store, os.path.join(output_dir, f"{image_name}_{threshold}_gt.jpg"), is_segmentation=True)

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


    for i, (images, masks_pre, masks_gt) in enumerate(dataloader):
        masks_pre = masks_pre.to(device)
        masks_pre_binary=binarise_predictions(masks_pre,threshold)
        masks_gt = masks_gt.to(device)
        masks_gt_binary = get_categories_from_normalization(masks_gt)
        batch_iou, batch_f1 = compute_iou_and_f1(masks_pre_binary, masks_gt_binary)
        total_IoU += batch_iou
        total_f1_score += batch_f1
        num_samples += images.size(0)

        if i == 0:
            images_to_store = unnormalize(images[:image_number])
            masks_pre_to_store = masks_pre[:image_number]
            masks_pre_binary_to_store = masks_pre_binary[:image_number]
            masks_gt_to_store = masks_gt_binary[:image_number]

            store_images(images_to_store, os.path.join(output_dir, f"{image_name}_{threshold}_images.jpg"))
            store_images(masks_pre_to_store, os.path.join(output_dir, f"{image_name}_{threshold}_masks.jpg"), is_segmentation=True)
            store_images(masks_pre_binary_to_store, os.path.join(output_dir, f"{image_name}_{threshold}_masks_binary.jpg"), is_segmentation=True)
            store_images(masks_gt_to_store, os.path.join(output_dir, f"{image_name}_{threshold}_gt.jpg"), is_segmentation=True)

    print(f"Mean IoU: {total_IoU / num_samples:.4f}")
    print(f"Mean F1 Score: {total_f1_score / num_samples:.4f}")

