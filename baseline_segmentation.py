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

class SegmentationTrainer(Trainer):
    """Adapted from pre_training.py with some segmentation-specific changes."""
    def __init__(self, log_dir="logs", log_file="segmentation_training.json"):
        super().__init__(log_dir, log_file)
        
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

    def visualize_predictions(self, backbone: nn.Module, head: nn.Module, loader, device, save_path):
        """Visualize model predictions alongside ground truth masks"""
        backbone.eval()
        head.eval()
        images, masks = next(iter(loader))
        images = images.to(device)
        
        # Generate predictions
        with torch.no_grad():
            features = backbone(images)
            predictions = head(features)
            pred_masks = predictions.argmax(dim=1).cpu()
        
        # Move everything to CPU for visualization
        images = images.cpu()
        masks = masks.cpu()
        
        # Create visualization with three columns: image, ground truth, prediction
        fig, axes = plt.subplots(min(4, len(images)), 3, figsize=(12, 10))
        for i in range(min(4, len(images))):
            # Original image
            axes[i, 0].imshow(images[i].permute(1, 2, 0))
            axes[i, 0].set_title("Input Image")
            axes[i, 0].axis("off")
            
            # Ground truth
            axes[i, 1].imshow(masks[i].squeeze(0), cmap="tab20")
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis("off")
            
            # Prediction
            axes[i, 2].imshow(pred_masks[i], cmap="tab20")
            axes[i, 2].set_title("Prediction")
            axes[i, 2].axis("off")
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Predictions visualization saved to '{save_path}'")    
    

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

    """Adapted from pre-training.py"""
    
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

        trainer = SegmentationTrainer(log_dir="logs", log_file="baseline_training.json")

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
            
            trainer.fit_sgd(device=device, num_epochs=1)
            trainer.visualize_predictions(
                trainer.backbone, trainer.heads[0], val_loader, device, save_path="baseline_predictions.png"
            )
