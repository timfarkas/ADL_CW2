from data import OxfordPetDataset
import data
import json
import os
import torch
import torch.nn as nn
import sys
import io
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import compute_segmentation_accuracy

import torch.nn.common_types
from models import UNetBackbone, SegmentationHead

# from baseline_model_loader import test_loader, val_loader, train_loader

import matplotlib.pyplot as plt
from pre_training import Trainer, convert_and_get_IoU


class TrainingBaseline:
    def initModel(self):
        segmentation_model = baseline_model.UNetWrapper(
            in_channels=3,  # 3 channels for RGB images
            n_classes=3,  # 3 class for segmentation - foreground, background, unknown
            depth=3,  # 3 encoding layers, 2 decoding layers
            wf=5,  # 2^5 = 32 channels
            padding=True,  # equivalent to padding=1 in Bruce's implementation
            batch_norm=True,  # use batch normalization after layers with activation function
            up_mode="upconv",
        )
        return segmentation_model

    # TODO: Use Trainer class

    def initOptimizer(self, model):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-5)
        return optimizer

    def train(
        self, model: torch.nn.Module, train_loader, optimizer, loss_fn, epochs=10
    ):
        # training loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            total_train_correct = 0
            total_train_pixels = 0

            for batch_ndx, batch in enumerate(train_loader):
                images, targets = batch

                images = images.to("cuda" if torch.cuda.is_available() else "cpu")
                targets = targets.to("cuda" if torch.cuda.is_available() else "cpu")

                targets = (
                    targets.squeeze(1) - 1
                )  # Remove the channel dimension from targets

                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                total_train_correct += (
                    (torch.argmax(outputs, dim=1) == targets).sum().item()
                )
                total_train_pixels += targets.numel()

                if batch_ndx % 10 == 0:
                    print(
                        f"Epoch {epoch + 1}/{epochs}, Batch {batch_ndx}, Loss: {loss.item():.4f}"
                    )
            epoch_loss /= len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
            train_accuracy = total_train_correct / total_train_pixels
            print(f"Training Accuracy: {train_accuracy:.4f}")

            # Test the model on the validation set
            with torch.no_grad():
                val_loss = 0.0
                total_val_correct = 0
                total_val_pixels = 0
                for val_batch in val_loader:
                    val_images, val_targets = val_batch
                    val_images = val_images.to(
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )
                    val_targets = val_targets.to(
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )
                    val_targets = val_targets.squeeze(1) - 1
                    val_outputs = model(val_images)
                    val_loss += loss_fn(val_outputs, val_targets).item()
                    _, predicted = torch.max(val_outputs, 1)
                    total_val_correct += (predicted == val_targets).sum().item()
                    total_val_pixels += val_targets.numel()
                val_accuracy = total_val_correct / total_val_pixels
                val_loss /= len(val_loader)
                print(
                    f"Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {val_loss:.4f}"
                )
                print(f"Validation Loss: {val_loss:.4f}")

        print("Training complete.")
        torch.save(model.state_dict(), "baseline_model.pth")
        print("Model training complete and saved as baseline_model.pth")

    def visualisation(self, images, masks):
        # Print shape information
        print(f"Image batch shape: {images.shape}")
        print(f"Mask batch shape: {masks.shape}")
        print(f"Unique values in masks: {torch.unique(masks)}")

        # Display a few samples
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for i in range(min(4, len(images))):
            # Show image (denormalize if needed)
            img = images[i].cpu().permute(1, 2, 0)  # CHW -> HWC
            axes[0, i].imshow(img)
            axes[0, i].set_title(f"Image {i}")
            axes[0, i].axis("off")

            # Show segmentation mask
            mask = masks[i].cpu()[0]  # Remove channel dimension
            axes[1, i].imshow(mask, cmap="tab20")
            axes[1, i].set_title(f"Mask {i} - Classes: {torch.unique(mask).tolist()}")
            axes[1, i].axis("off")

        plt.tight_layout()
        plt.savefig("sample_batch.png")
        plt.close()
        print("Sample visualization saved to 'sample_batch.png'")

    def main(self):

        # Initialize dataset, dataloaders, optimiser and loss fn
        model = self.initModel()
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = self.initOptimizer(model)
        loss = torch.nn.CrossEntropyLoss()

        # Visualise sample data
        print("Visualizing sample data...")
        sample_batch = next(iter(train_loader))
        images, masks = sample_batch
        self.visualisation(images, masks)

        # Train the model
        self.train(model, train_loader, optimizer, loss, epochs=10)
        print("Training complete.")

        # Evaluate the model
        # TODO: Implement evaluation logic


class SegCrossEntropyLoss(nn.Module):
    """
    Adapts CrossEntropyLoss to work with the 4D segmentation targets. Removes the channel dimension to keep comptability
    with pre_training format. Allows for ignoring the background class (2) in the loss calculation by inheriting ignore_index.
    """

    def __init__(self, ignore_index=2, weight=None):
        """Set the ignore_index to 2 (background) by default for our task."""
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weight)
        self.ignore_index = ignore_index

    def forward(self, inputs, targets: torch.Tensor):
        """
        Computes the CEL. Reshapes the targets to remove the channel dimension and then computes the loss.
        """
        targets = targets.squeeze(1) - 1
        targets = targets.long()
        return self.loss_fn(inputs, targets)


if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using {device}")

    print("Testing models, loader and devices.")
    backbone = UNetBackbone().to(device)
    seg_head = SegmentationHead(in_channels=backbone.out_channels).to(device)
    print(backbone.out_channels)

    # Print the number of parameters in the UNet backbone
    unet_params = sum(p.numel() for p in backbone.parameters())
    print(f"Unet Backbone parameters: {unet_params:,}")

    # loader, _, _ = data.create_dataloaders()

    # ### First, test whether models, loader & device work
    # for images, labels in loader:
    #     images = images.to(device)
    #     labels = labels.to(device)
    #     features = backbone(images)
    #     predictions = seg_head(features)
    #     print("All tests passed.")
    #     break

    # print("\n\nPreparing pre-training sweep...")

    # Define loss functions
    cel_fn = SegCrossEntropyLoss(ignore_index=2)

    checkpoints_dir = "checkpoints"

    run_dicts = [
        {  # baseline_segmentation
            "model_path": os.path.join(checkpoints_dir, "segmentation_baseline"),
            "heads": [seg_head],
            "backbone": "unet",
            "eval_functions": [compute_segmentation_accuracy],
            "eval_function_names": ["accuracy"],
            "loss_functions": [cel_fn],
            "loader_targets": ["segmentation"],
        },
    ]

    print(f"Starting {len(run_dicts)} runs....\n")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    dataset = OxfordPetDataset()

    batch_size = 64
    learning_rate = 3e-4
    weight_decay = 1e-4
    num_epochs = 15

    print(f"Using {device}")

    for i, run_dict in enumerate(run_dicts):
        model_path = run_dict["model_path"]
        print(f"Starting training run {i+1}, {os.path.basename(model_path)}...")

        print("Setting up trainer...")

        trainer = Trainer(log_dir="logs", log_file="baseline_training.json")

        trainer.set_eval_functions(
            run_dict["eval_functions"], run_dict["eval_function_names"]
        )

        train_loader, val_loader, _ = data.create_dataloaders(
            batch_size,
            target_type=run_dict["loader_targets"],
            lazy_loading=False,
        )
        trainer.set_loaders(train_loader, val_loader)
        trainer.set_loss_functions(run_dict["loss_functions"])

        backbone = None

        if run_dict["backbone"] == "unet":
            backbone = UNetBackbone()
            trainer.set_model(backbone, run_dict["heads"], model_path)
            trainer.set_optimizer(learning_rate, weight_decay)
            print("Trainer set up successfully!")
            trainer.fit_sgd(device=device, num_epochs=100)
