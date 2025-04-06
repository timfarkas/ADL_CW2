import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from utils import resize_images

# import os
# import torch.nn.functional as F
# import torch.optim as optim

### Num Inputs:
#       Breed:                  37
#       Species:                2
#       Bbox:                   256x256
#       Breed + Species??:      39
#       Breed + Bbox:           37 + 256x256
#       Species + Bbox:         2 + 256x256
#       Breed+Species+Bbox?:    39 + 256x256

# Backbone:
#   CNN
#   ResNet
# Head:
#   Bbox Head
#   Classifier Head


class BasicCAMWrapper(nn.Module):
    def __init__(self, model):
        """
        Wrapper class for models to generate Class Activation Maps (CAM) during forward pass

        Args:
            model: The base model to wrap (CNN or ResNetBackbone)
        """
        super().__init__()
        self.model = model
        self.hooks = []

    def forward(self, x, target_class=None, return_cam=False):
        """
        Forward pass that can optionally return CAM visualization

        Args:
            x: Input tensor (images)
            target_class: Optional class index to generate CAM for. If None and return_cam=True,
                          uses predicted class
            return_cam: If True, returns CAM along with model output

        Returns:
            - If return_cam=False: model output (logits)
            - If return_cam=True: tuple of (logits, cam_image, predicted_class)
        """
        # Get model outputs and feature maps
        logits, feature_maps = self.model(x, return_features=True)

        batch_size = x.size(0)
        cams = []

        for b in range(batch_size):
            feature_map = feature_maps[b]  # (C, H, W) for this batch item

            # Determine class for CAM visualization
            if target_class is None:
                # Use predicted class if target not specified
                _, predicted_class = torch.max(logits[b], 0)
                current_target = predicted_class.item()
            else:
                current_target = target_class

            # Compute CAM for this batch item
            classifier_weights = self.model.classifier.weight.data
            cam = self._compute_cam(feature_map, classifier_weights, current_target)
            cam = resize_images(cam, 256, 256)
            cams.append(torch.tensor(cam))

        # Stack all CAMs into a batch
        batch_cams = torch.stack(cams)

        return batch_cams

    def _compute_cam(self, feature_map, classifier_weights, class_idx):
        """Compute class activation map"""
        weights = (
            classifier_weights[class_idx].unsqueeze(1).unsqueeze(2)
        )  # Shape: (C, 1, 1)
        cam = torch.sum(weights * feature_map, dim=0)  # Shape: (H, W)
        cam = cam.detach().cpu().numpy()
        cam = np.maximum(cam, 0)  # ReLU
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)  # Normalize to [0, 1]
        return cam

    def _generate_cam_image(self, img_tensor, cam):
        """Generate CAM overlay on the original image"""
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        h, w = img_np.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))

        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        heatmap = heatmap.astype(np.float32) / 255

        # Create overlay
        overlay = 0.5 * img_np + 0.5 * heatmap
        overlay = np.clip(overlay, 0, 1)

        return overlay

    def visualize_batch(self, images, labels=None, num_images=4):
        """
        Visualize CAM for a batch of images

        Args:
            images: Batch of images
            labels: Optional ground truth labels
            num_images: Number of images to visualize
        """
        # Limit to specified number of images
        images = images[:num_images]

        # Prepare figure
        fig, axes = plt.subplots(nrows=2, ncols=num_images, figsize=(4 * num_images, 8))

        with torch.no_grad():
            logits, feature_maps = self.model(images, return_features=True)

            for i in range(num_images):
                img = images[i]
                fmap = feature_maps[i]

                # Determine target class
                if labels is not None:
                    target_class = labels[i].item()
                else:
                    _, target_class = torch.max(logits[i], 0)
                    target_class = target_class.item()

                # Compute CAM
                cam = self._compute_cam(
                    fmap, self.model.classifier.weight.data, target_class
                )
                img_np = img.permute(1, 2, 0).cpu().numpy()

                # Original image on first row
                axes[0][i].imshow(img_np)
                axes[0][i].set_title(f"Original - Class {target_class}")
                axes[0][i].axis("off")

                # CAM overlay on second row
                cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
                cam_normalized = np.clip(cam_resized, 0, 1)

                heatmap = cv2.applyColorMap(
                    np.uint8(255 * cam_normalized), cv2.COLORMAP_JET
                )
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                heatmap = heatmap.astype(np.float32) / 255.0

                overlay = 0.5 * img_np + 0.5 * heatmap
                overlay = np.clip(overlay, 0, 1)

                axes[1][i].imshow(overlay)
                axes[1][i].set_title(f"CAM - Class {target_class}")
                axes[1][i].axis("off")

        plt.tight_layout()
        plt.show()


class CAMManager:
    """
    Class that manages different CAM methods based on the grad-cam package. It allows to visualise CAMs and can prepare outputs for self-training.
    """

    def __init__(self, model, method="gradCAM", target_layer=None):
        """

        Args:
            model: The model to generate CAMs for
            method: CAM method ('GradCAM', 'HiResCAM', etc. naming based on grad-cam package)
            target_layer: Target layer for CAM methods, if None will be auto-detected
        """
        self.model = model
        self.method = method

        # Auto-detect target layers if not specified
        if target_layer is None:
            if hasattr(model, "features") and hasattr(model.features, "__getitem__"):
                if isinstance(model, ResNetBackbone):
                    self.target_layers = [
                        model.features[-1][-1]
                    ]  # Last block of layer4 in ResNet
                else:  # CNN
                    self.target_layers = [
                        model.features[-2]
                    ]  # Last conv layer before GAP in CNN
            else:
                raise ValueError(
                    "Unsupported model type for CAM, could not detect target layers."
                )
        else:
            self.target_layers = [target_layer]

        # Initialise the appropriate CAM method
        if method == "GradCAM":
            from pytorch_grad_cam import GradCAM

            self.cam = GradCAM(model=model, target_layers=self.target_layers)
        # TO-DO: add other CAM methods
        else:
            raise ValueError(f"Unsupported CAM method: {method}")

    def generate_cams(self, images, labels=None, threshold=0.2):
        """
        Generate CAMs for input images.

        Args:
            images: Input images tensor [B, C, H, W]
            labels: Optional target labels
            threshold: Threshold to filter out low confidence areas

        Returns:
            Dictionary containing:
            - cam_maps: Processed CAM maps [B, 1, H, W]. (CAM masks are typically grayscale (1 channel). This format matches the expected input shape for UNet) 
            - images: Original images
        """
        device = next(self.model.parameters()).device
        images = images.to(device)

        # prepare targets if labels are provided
        if labels is not None:
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

            targets = [ClassifierOutputTarget(label.item()) for label in labels]
        else:
            targets = None

        # Generate CAM maps
        grayscale_cams = self.cam(input_tensor=images, targets=targets)

        # Process CAMs to match what (i think) self-training expects:
        # - Convert to torch tensor
        # - Add channel dimension
        # - we also apply a threshold to filter low-confidence regions
        processed_cams = torch.from_numpy(grayscale_cams).float().unsqueeze(1)
        filtered_cams = processed_cams * (processed_cams > threshold).float()

        return {"images": images.cpu(), "cam_maps": filtered_cams}

    def visualize_batch(self, images, labels=None, num_images=4):
        """
        Visualise CAMs for a batch of images

        Args:
            images: Batch of images [B, C, H, W]
            labels: Optional ground truth labels
            num_images: Number of images to visualise
        """
        # Limit number of images
        images = images[:num_images]
        labels = labels[:num_images] if labels is not None else None

        # Generate CAMs
        result = self.generate_cams(images, labels)
        cam_maps = result["cam_maps"].squeeze(1).numpy()  # [B, H, W]

        # figure
        _, axes = plt.subplots(nrows=2, ncols=num_images, figsize=(4 * num_images, 8))

        for i in range(num_images):
            # Original image
            img = images[i].permute(1, 2, 0).cpu().numpy()
            img = np.clip(img, 0, 1)
            axes[0][i].imshow(img)
            class_label = labels[i].item() if labels is not None else "predicted"
            axes[0][i].set_title(f"Original - Class {class_label}")
            axes[0][i].axis("off")

            # CAM visualisation. Create heatmap overlay
            cam_resized = cam_maps[i]

            # Create heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            heatmap = heatmap.astype(np.float32) / 255

            # Create overlay
            overlay = 0.5 * img + 0.5 * heatmap
            overlay = np.clip(overlay, 0, 1)

            axes[1][i].imshow(overlay)
            axes[1][i].set_title(
                f"{self.method.capitalize()}-CAM - Class {class_label}"
            )
            axes[1][i].axis("off")

        plt.tight_layout()
        plt.show()

    def generate_cam_dataset(self, dataloader, threshold=0.2):
        """
        Generate a dataset with CAM masks for self-training.
        (TO-DO: The output of this needs to exactly agree with format expected by the recent version of thed self-training pipeline.)

        Args:
            dataloader: DataLoader with images
            threshold: Threshold for filtering low confidence areas

        Returns:
            TensorDataset: Contains (images, masks) where masks are [B, 1, H, W].
        """
        from torch.utils.data import TensorDataset

        all_images = []
        all_masks = []

        for batch_images, batch_labels in dataloader:
            result = self.generate_cams(batch_images, batch_labels, threshold)
            all_images.append(result["images"])
            all_masks.append(result["cam_maps"])

        # Concatenate all batches
        images_tensor = torch.cat(all_images, dim=0)
        masks_tensor = torch.cat(all_masks, dim=0)

        return TensorDataset(images_tensor, masks_tensor)


class BboxHead(nn.Module):
    def __init__(self, adapter="CNN"):
        super().__init__()
        num_inputs = 256 if adapter.lower() == "cnn" else 512
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # (C,H,W) → (C,1,1)
            nn.Flatten(),  # → (C,)
            nn.Linear(num_inputs, 4),
            nn.Sigmoid(),  ### [cx, cy, w, h]
        )

    def forward(self, z):
        return self.head(z)


class ClassifierHead(nn.Module):
    def __init__(self, num_classes=2, adapter="CNN"):
        super().__init__()
        num_inputs = 256 if adapter.lower() == "cnn" else 512
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # (C,H,W) → (C,1,1)
            nn.Flatten(),  # → (C,)
            nn.Linear(num_inputs, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.head(z)


class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Input: (B, 3, 256, 256)
            nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=3, padding=1
            ),  # (B, 32, 256, 256)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (B, 32, 128, 128)
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, padding=1
            ),  # (B, 64, 128, 128)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (B, 64, 64, 64)
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, padding=1
            ),  # (B, 128, 64, 64)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (B, 128, 32, 32)
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, padding=1
            ),  # (B, 256, 32, 32)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (B, 256, 16, 16)
        )

    def forward(self, img):
        return self.features(img)


class ResNetBackbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        if pretrained:
            base_model = resnet18(
                weights=ResNet18_Weights.IMAGENET1K_V1
            )  # With pretrained weights
        else:
            base_model = resnet18(weights=None)  # No pretrained weights

        self.features = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
        )

    def forward(self, img):
        return self.features(img)
