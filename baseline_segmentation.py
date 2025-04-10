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
    def __init__(self, log_dir="logs", log_file="segmentation_training.json"):
        super().__init__(log_dir, log_file)


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
            trainer.fit_sgd(device=device, num_epochs=100)
