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
    cam_dataset_folder,
    semi_supervised_model_folder,
)


def run_supervised_training_process(
    device: torch.device,
    batch_size: int,
    workers: int,
    persistent_workers: bool,
    pin_memory: bool,
    learning_rate: float,
    weight_decay: float,
    num_epochs: int,
    use_cam_dataset: bool = False,
    cam_threshold: float = 0.5,
):
    if use_cam_dataset:
        dataset = [
            os.path.join(cam_dataset_folder, f)
            for f in os.listdir(cam_dataset_folder)
            if f.endswith(".pt")
        ][0] # Expects to only have one
        initial_dataset_raw = torch.load(
            dataset, weights_only=False, map_location="cpu"
        )
        binarized_data = []
        for image, cam, mask in initial_dataset_raw:
            cam_binary = (cam > cam_threshold).float()
            binarized_data.append((image, cam_binary, mask))
        train_dataloader = torch.utils.data.TensorDataset(
            torch.stack([x[0] for x in binarized_data]),
            torch.stack([x[1] for x in binarized_data]),
            torch.stack([x[2] for x in binarized_data]),
        )
        checkpoints_dir, logs_dir = get_checkpoints_and_logs_dirs(
            run_name=semi_supervised_model_folder,
            model_name=dataset.split("/")[-1].split(".")[0],
        )
    else:
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
            pin_memory=pin_memory,
        )
        train_dataloader, _, _ = dataloader_manager.create_dataloaders(
            shuffle_train=True
        )
        checkpoints_dir, logs_dir = get_checkpoints_and_logs_dirs(
            run_name=baseline_model_folder,
            model_name=baseline_model_name,
        )

    print("\nTraining fully supervised model (baseline)")
    model = UNet().to(device)
    all_params = list(model.parameters())
    optimizer = torch.optim.AdamW(
        all_params, lr=learning_rate, weight_decay=weight_decay
    )

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    model.train()
    for epoch in range(num_epochs):
        correct_pixels = 0
        total_pixels = 0
        total_loss = 0

        for images, masks, *rest in train_dataloader:
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
    torch.save(model.state_dict(), f"{checkpoints_dir}/{baseline_model_name}.pt")

    return model
