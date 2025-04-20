import os
import torch

from torch.utils.data import DataLoader

from datasets.dataloader_manager import DataloaderManager
from datasets.dataset_manager import DatasetManager
from models.u_net import UNet
from training.utils import log_self_training_performance
from training.evaluations import evaluate_segmentation_model
from training.losses import bce_fn
from new_runs_config import get_checkpoints_and_logs_dirs, cam_dataset_folder
from training.predict_segmentation import predict_segmentation_dataset


def run_self_training_process(
    runs_config: dict,
    device: torch.device,
    batch_size: int,
    workers: int,
    persistent_workers: bool,
    learning_rate: float,
    weight_decay: float,
    num_epochs: int,
    num_bootstrap_rounds: int,
    threshold: float,
):
    # TODO: still missing validation logic
    # Open the datasets folder and get each file, then load each dataset if its a .pt file
    datasets = [
        os.path.join(cam_dataset_folder, f)
        for f in os.listdir(cam_dataset_folder)
        if f.endswith(".pt")
    ]

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
    _, val_dataloader, _ = dataloader_manager.create_dataloaders(
        shuffle_train=False
    )  # No need to shuffle for this

    for dataset in datasets:
        dataset_name = os.path.basename(dataset).split(".")[0]
        print(f"Loading dataset: {dataset_name}")

        # Load the dataset with binarized CAMs
        initial_dataset_raw = torch.load(
            dataset, weights_only=False, map_location="cpu"
        )
        binarized_data = []
        for image, cam, mask in initial_dataset_raw:
            cam_binary = (cam > threshold).float()
            binarized_data.append((image, cam_binary, mask))

        for run_name, run_config in runs_config.items():
            print(
                f"\nRunning self-training for experiment {run_name} with dataset {dataset_name}"
            )
            model = UNet(3, 1).to(device)
            boostrap_dataset = torch.utils.data.TensorDataset(
                torch.stack([x[0] for x in binarized_data]),
                torch.stack([x[1] for x in binarized_data]),
                torch.stack([x[2] for x in binarized_data]),
            )
            all_params = list(model.parameters())
            optimizer = torch.optim.AdamW(
                all_params, lr=learning_rate, weight_decay=weight_decay
            )

            checkpoints_dir, logs_dir = get_checkpoints_and_logs_dirs(
                run_name=dataset_name,
                model_name=run_name,
            )  # Create one folder per dataset even though we expect only one, so run info is at the same level

            for rounds_num in range(num_bootstrap_rounds):
                print(
                    f"Running bootstrap round {rounds_num + 1} on a dataset length of {len(boostrap_dataset)}"
                )
                round_name = f"round_{rounds_num + 1}"
                dataloader = DataLoader(
                    boostrap_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=workers,
                    persistent_workers=persistent_workers,
                )

                os.makedirs(checkpoints_dir, exist_ok=True)
                os.makedirs(logs_dir, exist_ok=True)

                model.train()
                for epoch in range(num_epochs):
                    correct_pixels = 0
                    total_pixels = 0
                    total_loss = 0

                    for i, (images, masks, _) in enumerate(dataloader):
                        images = images.to(device)
                        masks = masks.to(device).float()

                        optimizer.zero_grad()
                        outputs = model(images)

                        if run_config["seed_loss"]:
                            threshold_bg = 0.05  # Always
                            threshold_fg = 1 - threshold
                            seed_mask = (
                                (masks >= threshold_fg) | (masks <= threshold_bg)
                            ).float()
                            masks_bin = torch.zeros_like(masks)
                            masks_bin[masks >= threshold_fg] = 1  # Foreground
                            masks_bin[masks <= threshold_bg] = 0  # Background
                            loss_raw = bce_fn(outputs, masks_bin)
                            loss = (loss_raw * seed_mask).sum() / (
                                seed_mask.sum() + 1e-6
                            )
                        else:
                            masks_bin = (masks >= threshold) * masks
                            loss = bce_fn(outputs, masks_bin)

                        loss.backward()
                        optimizer.step()

                        # Calculate pixel-wise accuracy
                        probs = torch.sigmoid(outputs)
                        predictions = probs > 0.5  # threshold logits
                        correct_pixels += (predictions == masks.bool()).sum().item()
                        total_pixels += masks.numel()
                        total_loss += loss.item() * images.size(0)

                    avg_loss = total_loss / len(dataloader.dataset)
                    pixel_accuracy = correct_pixels / total_pixels

                    print(
                        f"Epoch {epoch + 1}/{num_epochs}, Pixel Accuracy: {pixel_accuracy:.4f}, Loss: {avg_loss:.4f}"
                    )

                torch.save(model.state_dict(), f"{checkpoints_dir}/{round_name}.pt")

                ioi, f1 = evaluate_segmentation_model(
                    model=model,
                    test_loader=val_dataloader,
                    device=device,
                )

                log_self_training_performance(
                    log_dir=logs_dir,
                    run_name=run_name,
                    round_name=round_name,
                    ioi=ioi,
                    f1=f1,
                )

                if rounds_num + 1 < num_bootstrap_rounds:
                    new_dataset = predict_segmentation_dataset(
                        model=model,
                        dataloader=dataloader,
                        device=device,
                        predictions_transform=run_config["predictions_transform"],
                        threshold=threshold,
                    )

                    if run_config["dataset_management"] == "add":
                        boostrap_dataset = torch.utils.data.ConcatDataset(
                            [boostrap_dataset, new_dataset]
                        )
                    elif run_config["dataset_management"] == "replace":
                        boostrap_dataset = new_dataset
                    else:
                        raise ValueError(
                            f"Unknown dataset management method: {run_config['dataset_management']}"
                        )
