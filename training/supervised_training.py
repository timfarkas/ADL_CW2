import os
import torch

from datasets.dataloader_manager import DataloaderManager
from datasets.dataset_manager import DatasetManager
from models.u_net import UNet
from training.utils import get_binary_masks_from_trimap
from training.losses import bce_fn
from new_runs_config import (
    get_checkpoints_and_logs_dirs,
    baseline_model_name,
    baseline_model_folder,
    segmentation_output_threshold,
)


def run_supervised_training_process(
    device: torch.device,
    batch_size: int,
    workers: int,
    persistent_workers: bool,
    learning_rate: float,
    weight_decay: float,
    num_epochs: int,
):
    dataset_manager = DatasetManager(
        target_type=["segmentation"],
        mixed=False,
        use_augmentation=False,
    )
    dataloader_manager = DataloaderManager(
        dataset_manager=dataset_manager,
        batch_size=batch_size,
        workers=workers,
        persistent_workers=persistent_workers,
    )
    train_dataloader, _, _ = dataloader_manager.create_dataloaders(
        shuffle_train=True
    )

    print(f"\nTraining fully supervised model (baseline)")
    model = UNet(3, 1).to(device)
    all_params = list(model.parameters())
    optimizer = torch.optim.AdamW(
        all_params, lr=learning_rate, weight_decay=weight_decay
    )

    checkpoints_dir, logs_dir = get_checkpoints_and_logs_dirs(
        run_name=baseline_model_folder,
        model_name=baseline_model_name,
    )

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    model.train()
    for epoch in range(num_epochs):
        correct_pixels = 0
        total_pixels = 0
        total_loss = 0

        for images, masks in train_dataloader:
            images = images.to(device)
            masks = masks.to(device)
            masks_bin = get_binary_masks_from_trimap(masks)
            masks_bin = masks_bin.unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = bce_fn(outputs, masks_bin)

            loss.backward()
            optimizer.step()

            # Calculate pixel-wise accuracy
            probs = torch.sigmoid(outputs)
            predictions = probs > segmentation_output_threshold  # threshold logits
            correct_pixels += (predictions == masks_bin.bool()).sum().item()
            total_pixels += masks_bin.numel()
            total_loss += loss.item() * images.size(0)

        avg_loss = total_loss / len(train_dataloader.dataset)
        pixel_accuracy = correct_pixels / total_pixels

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Pixel Accuracy: {pixel_accuracy:.4f}, Loss: {avg_loss:.4f}"
        )

    print("Saving model...")
    torch.save(model.state_dict(), f"{checkpoints_dir}/epoch_{num_epochs}.pt")

    return model
