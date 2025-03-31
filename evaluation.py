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
    for i, (images, masks) in enumerate(test_loader):
        # TODO: When we have a model, ensure the prediction results are actually comparable
        images = images.to(device)
        masks = masks.to(device)
        predictions = model(images)

        for j, prediction in enumerate(predictions):
            true_mask = masks[j]

            # Calculate IoU
            intersection = torch.logical_and(prediction, true_mask)
            union = torch.logical_or(prediction, true_mask)
            if torch.sum(union) == 0:
                iou = torch.tensor(1.0)  # This seems to be a bug in the dataset
            else:
                iou = torch.sum(intersection) / torch.sum(union)
            total_IoU += iou.item()

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
            total_f1_score += f1_score.item()

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

    print(f"Mean IoU: {total_IoU / len(test_loader)}")
    print(f"Mean F1 Score: {total_f1_score / len(test_loader)}")

